"""Behaviour: B1 tool calls, B2 truncation, B3 stop sequences vs hidden reasoning, B5 context window,
B6 streaming parity, B7 error semantics."""

from __future__ import annotations

import random
import re

from ..runner import FAIL, NA, PASS, WARN, Incomplete, check_result
from ..schema import validate
from ..thresholds import THRESHOLDS
from .common import WEATHER_SCHEMA, first_number, is_claude, require, transient, weather_result, weather_tool

WEATHER_ASK = "What's the weather in Lisbon right now? Use the get_weather tool."
TRUNC_ASK = "Count from 1 to 300, writing every number in words, separated by commas. Do not stop early."
ECHO_LINE = "The quick brown fox jumps over the lazy dog 0123456789."


def _tool_problem(r, name="get_weather") -> str:
    if not r.tool_calls:
        return f"no tool call (finish {r.finish_raw!r}; text {r.text[:60]!r})"
    tc = r.tool_calls[0]
    if tc.get("json_error"):
        return f"arguments are not valid JSON: {str(tc.get('arguments_raw'))[:80]!r}"
    if tc.get("name") != name:
        return f"called {tc.get('name')!r}, not {name!r}"
    errs = validate(tc.get("input"), WEATHER_SCHEMA)
    return f"arguments don't match the schema: {'; '.join(errs)}" if errs else ""


# ---------------------------------------------------------------------------------------------------- B1
def b1(ctx) -> dict:
    p = ctx.proto
    tools = [weather_tool(ctx)]
    parts, fails, warns = {}, [], []
    msgs = [p.user(WEATHER_ASK)]
    r = ctx.call(p.request(msgs, max_tokens=2048, tools=tools, tool_choice=p.tool_choice("auto")), label="auto")
    require(r, "tool call (auto)")
    prob = _tool_problem(r) if r.ok else r.short_error()
    parts["auto"] = prob or f"get_weather({r.tool_calls[0]['arguments_raw']})"
    if prob:
        fails.append(f"auto: {prob}")
    else:
        hist = msgs + [p.assistant_echo(r)] + p.tool_results(r, weather_result("Lisbon"))
        r2 = ctx.call(p.request(hist, max_tokens=2048, tools=tools, tool_choice=p.tool_choice("auto")), label="tool-result-turn")
        require(r2, "tool result turn")
        if not r2.ok:
            fails.append(f"multi-turn: sending the tool result back failed: {r2.short_error()}")
        elif not (r2.text.strip() or r2.tool_calls):
            fails.append(f"multi-turn: no answer after the tool result (finish {r2.finish_raw!r})")
        parts["multi_turn"] = r2.text[:80] if r2.ok else r2.short_error()
        if r.reasoning_text or r.thinking:
            parts["multi_turn"] += " (reasoning returned and echoed)"
    rf = ctx.call(p.request([p.user("Tell me one fun fact about Oslo.")], max_tokens=2048, tools=tools,
                            tool_choice=p.tool_choice("forced", "get_weather")), label="forced")
    require(rf, "tool call (forced)")
    if rf.ok:
        prob = _tool_problem(rf)
        parts["forced"] = prob or f"get_weather({rf.tool_calls[0]['arguments_raw']})"
        if prob:
            fails.append("forced tool_choice: " + ("ignored, answered in text instead (a silent text answer)"
                                                   if not rf.tool_calls else prob))
    elif rf.status == 404:
        parts["forced"] = rf.short_error()
        fails.append(f"forced tool_choice: HTTP 404 ({rf.error_message[:100]})")
    elif rf.status and 400 <= rf.status < 500 and rf.error_message:
        parts["forced"] = rf.short_error()
        warns.append(f"forced tool_choice not supported, reported as an error: {rf.error_message[:120]}")
    else:
        parts["forced"] = rf.short_error()
        fails.append(f"forced tool_choice: {rf.short_error()}")
    i3 = ctx.shared.get("I3")
    if i3:
        if not i3["intact"].ok:
            fails.append("thinking: the turn with the returned thinking block was rejected (see I3)")
        elif i3["forwarded"] is False:
            fails.append("thinking: the returned thinking block did not reach the model (billed prompt equal to a turn "
                         "without it), so the model loses its reasoning between tool calls")
        parts["thinking_multi_turn"] = ("accepted" if i3["intact"].ok else "rejected") + \
            ("" if i3["forwarded"] is None else ("; block forwarded" if i3["forwarded"] else "; block dropped"))
    d = {"parts": parts, "failures": fails, "documented_deviations": warns}
    if fails:
        return check_result("B1", FAIL, "; ".join(fails), d)
    if warns:
        return check_result("B1", WARN, "; ".join(warns), d)
    return check_result("B1", PASS, "; ".join(f"{k}: {v}" for k, v in parts.items())[:300], d)


# ---------------------------------------------------------------------------------------------------- B2
def b2(ctx) -> dict:
    p, mt = ctx.proto, THRESHOLDS["B2"]["max_tokens"]
    seen, bad = {}, []
    for stream in (False, True):
        r = ctx.call(p.request([p.user(TRUNC_ASK)], max_tokens=mt, stream=stream), stream=stream,
                     label="stream" if stream else "non-stream")
        require(r, "truncation request")
        where = "stream" if stream else "non-stream"
        if not r.ok:
            bad.append(f"{where}: {r.short_error()}")
            continue
        seen[where] = {"finish": r.finish_raw, "text_chars": len(r.text), "output_tokens": (r.usage or {}).get("output")}
        if r.finish != "length":
            bad.append(f"{where}: a max_tokens={mt} cut reported as {r.finish_raw!r}, want "
                       f"{'max_tokens' if p.api == 'anthropic' else 'length'}")
    d = {"max_tokens": mt, "seen": seen}
    over = [w for w, v in seen.items() if (v.get("output_tokens") or 0) > mt]
    if bad:
        return check_result("B2", FAIL, "; ".join(bad), d)
    note = f"; output tokens above max_tokens on {', '.join(over)}" if over else ""
    return check_result("B2", WARN if over else PASS, f"cut reported as {seen['non-stream']['finish']!r} streamed and not"
                        + note, d)


# ---------------------------------------------------------------------------------------------------- B3
def b3(ctx) -> dict:
    p = ctx.proto
    a, b, c = ctx.rng.randint(3001, 9999), ctx.rng.randint(1001, 9999), ctx.rng.randint(10001, 99999)
    want = str(a * b - c)
    ask = (f"What is {a} × {b} − {c}? Work it out carefully, then reply with only the final number "
           "(digits only, no words, no symbols).")
    thinking = ctx.thinking_param() if p.api == "anthropic" and is_claude(ctx.opts.model) else None
    r = ctx.call(p.request([p.user(ask)], max_tokens=8192, stop=["="], thinking=thinking), label="stop-eq")
    require(r, "stop-sequence request")
    if r.ok and thinking and not r.reasoning_observed:   # adaptive thinking chose not to think: ask for more effort
        r = ctx.call(p.request([p.user(ask)], max_tokens=8192, stop=["="], thinking=thinking,
                               extra={"output_config": {"effort": "high"}}), label="stop-eq-effort-high")
        require(r, "stop-sequence request (effort high)")
    d = {"stop": ["="], "expected": want, "text": r.text[:120], "finish": r.finish_raw,
         "reasoning_observed": r.reasoning_observed, "reasoning_tokens": (r.usage or {}).get("reasoning")}
    if not r.ok:
        if r.status and 400 <= r.status < 500:
            return check_result("B3", WARN, f"stop sequences rejected with this request: {r.short_error()}", d)
        return check_result("B3", FAIL, r.short_error(), d)
    got = first_number(r.text)
    if not r.reasoning_observed:
        where = (f"the stop fired in the visible text ({r.text.strip()[:30]!r})" if r.finish == "stop_sequence"
                 else f"answer {r.text.strip()[:30]!r}")
        return check_result("B3", NA, "no hidden reasoning on this request (no thinking blocks or reasoning tokens), "
                            f"so there was nothing for the stop to wrongly cut; {where}", d)
    if got == want:
        return check_result("B3", PASS, f"stop sequence '=' left the hidden reasoning alone: answer {got} correct", d)
    if not r.text.strip() and r.finish in ("stop", "stop_sequence"):
        return check_result("B3", FAIL, "empty answer ending in a stop: the stop sequence '=' fired inside the hidden "
                            "reasoning", d)
    if r.finish == "length":
        raise Incomplete("the reasoning used the whole max_tokens; no answer to grade")
    return check_result("B3", WARN, f"answer {r.text.strip()[:40]!r} (want {want}); the stop may have fired in the "
                        "visible text, or the model erred", d)


# ---------------------------------------------------------------------------------------------------- B5
LEVELS = ["INFO", "INFO", "INFO", "DEBUG", "WARN"]
SERVICES = ["api", "worker", "scheduler", "billing", "auth", "search", "ingest"]
MSGS = ["processed job {n} in {ms} ms", "cache refresh for shard {n} took {ms} ms", "request {n} completed with status 200",
        "retrying upload {n} after {ms} ms", "queue depth {n}, oldest message {ms} ms", "healthcheck ok, latency {ms} ms"]
WORDS = ["amber", "falcon", "cobalt", "willow", "granite", "harbor", "lantern", "meadow", "quartz", "saffron", "tundra",
         "violet", "ember", "juniper", "orchid", "pebble"]


def _filler(rng, n_chars: int) -> list:
    lines, size, sec = [], 0, 0
    while size < n_chars:
        sec += rng.randint(1, 4)
        m = rng.choice(MSGS).format(n=rng.randint(1000, 99999), ms=rng.randint(2, 900))
        line = (f"2026-11-05T{(sec // 3600) % 24:02d}:{(sec // 60) % 60:02d}:{sec % 60:02d}Z {rng.choice(LEVELS):<5} "
                f"{rng.choice(SERVICES)}[{rng.randint(100, 999)}]: {m}\n")
        lines.append(line)
        size += len(line)
    return lines


def _needle_prompt(rng, density: float, prompt_tokens: int, positions_tokens: list):
    chars = int(prompt_tokens / density)
    lines = _filler(rng, chars)
    needles = []
    for pos in positions_tokens:
        name = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}"
        num = str(rng.randint(1_000_000, 9_999_999))
        needles.append((name, num))
        target_char, acc, idx = int(pos / density), 0, 0
        for idx, ln in enumerate(lines):
            acc += len(ln)
            if acc >= target_char:
                break
        lines.insert(idx + 1, f"2026-11-05T12:00:00Z NOTICE deploy[1]: release {name} build number is {num}\n")
    q = ("\nAbove is a service log. Exactly three NOTICE lines in it state a build number for a release. Reply with "
         "exactly three lines, in the order they appear, each in the form <release>: <build number>, and nothing else.")
    return "".join(lines) + q, needles


def _density(ctx) -> tuple[float, int]:
    """Tokens per character of the filler on this endpoint (one small calibration request)."""
    p = ctx.proto
    sample = "".join(_filler(random.Random(7), 12000))
    r0 = ctx.call(p.request([p.user("x")], max_tokens=16), label="calibrate-empty")
    r1 = ctx.call(p.request([p.user(sample)], max_tokens=16), label="calibrate-filler")
    require(r1, "calibration")
    if not (r0.ok and r1.ok and r0.usage and r1.usage):
        raise Incomplete("no usage on the calibration request; can't size the prompt")
    return max(0.05, (r1.usage["prompt_total"] - r0.usage["prompt_total"]) / len(sample)), r0.usage["prompt_total"]


def _recall(ctx, prompt_tokens: int, depths: list, window: int | None, label: str, stream: bool):
    p = ctx.proto
    density, overhead = ctx.shared.setdefault("B5_density", _density(ctx))
    ref = window or prompt_tokens
    prompt, needles = _needle_prompt(ctx.rng, density, prompt_tokens, [d * ref for d in depths])
    est = int(len(prompt) * density) + overhead
    ok, cost = ctx.can_afford(est, 300)
    if not ok:
        raise Incomplete(f"{label}: a ~{est:,}-token prompt would exceed the spend cap (est. ${cost or 0:.3f})")
    ctx.log(f"    B5 {label}: ~{est:,} prompt tokens" + (f", est. ${cost:.3f}" if cost else ""))
    r = ctx.call(p.request([p.user(prompt)], max_tokens=512, stream=stream), stream=stream, label=label,
                 timeout=max(ctx.opts.timeout, 600))
    got = {n: (num in (r.text or "")) for n, num in needles}
    reported = (r.usage or {}).get("prompt_total")
    return r, {"estimated_prompt_tokens": est, "reported_prompt_tokens": reported, "depths": depths,
               "recalled": got, "answer": (r.text or "")[:200], "status": r.status, "finish": r.finish_raw}


def b5(ctx) -> dict:
    th = THRESHOLDS["B5"]
    o = ctx.opts
    d, fails, notes = {}, [], []

    def judge(part: str, r, info: dict):
        d[part] = info
        if transient(r):
            raise Incomplete(f"{part}: {r.short_error()}")
        if not r.ok:
            fails.append(f"{part}: a ~{info['estimated_prompt_tokens']:,}-token prompt was rejected: {r.short_error()}")
            return
        rep = info["reported_prompt_tokens"]
        if rep is not None and rep < th["min_reported_share"] * info["estimated_prompt_tokens"]:
            fails.append(f"{part}: {rep:,} prompt tokens billed for a ~{info['estimated_prompt_tokens']:,}-token prompt "
                         "(content was cut or compressed before the model)")
        missing = [k for k, v in info["recalled"].items() if not v]
        if missing:
            fails.append(f"{part}: {len(missing)} of 3 facts not recalled")

    r, info = _recall(ctx, th["short_prompt_tokens"], th["depths"], None, "short", False)
    judge("short", r, info)
    if o.long_context:
        if not o.context_window:
            notes.append("--long-context needs --context-window (the advertised window); long part skipped")
        else:
            target = int(o.context_window * th["long_fill"]) - 600
            r, info = _recall(ctx, target, th["depths"], o.context_window, "long", True)
            info["advertised_window"] = o.context_window
            judge("long", r, info)
    if o.max_output:
        p = ctx.proto
        r = ctx.call(p.request([p.user("Reply with the word OK.")], max_tokens=o.max_output, stream=True), stream=True,
                     label="max-output")
        d["max_output"] = {"max_tokens": o.max_output, "status": r.status, "ok": r.ok}
        if not r.ok and not transient(r):
            fails.append(f"advertised max output {o.max_output} rejected as max_tokens: {r.short_error()}")
    d["notes"] = notes
    if fails:
        return check_result("B5", FAIL, "; ".join(fails), d)
    done = "short (~{:,} tokens)".format(d["short"]["estimated_prompt_tokens"])
    if "long" in d:
        return check_result("B5", PASS, f"facts at 25/50/90% recalled at ~{d['long']['estimated_prompt_tokens']:,} tokens "
                            f"(window {o.context_window:,}) and in the {done} prompt", d)
    return check_result("B5", NA, f"3/3 facts recalled in a {done} prompt; the advertised window was not tested "
                        "(opt in with --long-context --context-window N)" + (f"; {notes[0]}" if notes else ""), d)


# ---------------------------------------------------------------------------------------------------- B6
def b6(ctx) -> dict:
    p = ctx.proto
    ask = f"Repeat exactly this line and nothing else:\n{ECHO_LINE}"
    rs = ctx.call(p.request([p.user(ask)], max_tokens=1024), label="non-stream")
    require(rs, "non-streamed request")
    rt = ctx.call(p.request([p.user(ask)], max_tokens=1024, stream=True), stream=True, label="stream")
    require(rt, "streamed request")
    fails, warns = [], []
    d = {"non_stream": {"text": rs.text[:120], "finish": rs.finish_raw, "usage": _u(rs)},
         "stream": {"text": rt.text[:120], "finish": rt.finish_raw, "usage": _u(rt), "final_event": rt.final_event,
                    "usage_in_stream": rt.usage_in_stream, "error": rt.stream_error, "is_stream": rt.stream}}
    if not rs.ok or not rt.ok:
        return check_result("B6", FAIL, f"non-stream {rs.short_error() if not rs.ok else 'ok'}; stream "
                            f"{rt.short_error() if not rt.ok else 'error event: ' + str(rt.stream_error)[:100]}", d)
    if not rt.stream:
        fails.append("stream=true did not return an event stream")
    if not rt.final_event:
        fails.append("no final event (" + ("message_stop" if p.api == "anthropic" else "[DONE]") + ")")
    if not rt.usage_in_stream:
        fails.append("no usage in the stream")
    if rs.finish != rt.finish:
        fails.append(f"finish differs: {rs.finish_raw!r} vs streamed {rt.finish_raw!r}")
    ps, pt = (rs.usage or {}).get("prompt_total"), (rt.usage or {}).get("prompt_total")
    if ps is not None and pt is not None and abs(ps - pt) > 2:
        fails.append(f"prompt tokens differ: {ps} vs streamed {pt}")
    if rs.text.strip() != rt.text.strip():
        warns.append("text differs between streamed and non-streamed (sampling is not deterministic): "
                     f"{rs.text.strip()[:40]!r} vs {rt.text.strip()[:40]!r}")
    tools = [weather_tool(ctx)]
    rtool = ctx.call(p.request([p.user(WEATHER_ASK)], max_tokens=2048, tools=tools, tool_choice=p.tool_choice("auto"),
                               stream=True), stream=True, label="stream-tool")
    require(rtool, "streamed tool call")
    prob = _tool_problem(rtool) if rtool.ok else rtool.short_error()
    d["stream_tool"] = prob or f"get_weather({rtool.tool_calls[0]['arguments_raw']})"
    if prob:
        fails.append(f"streamed tool call: {prob}")
    if fails:
        return check_result("B6", FAIL, "; ".join(fails), d)
    if warns:
        return check_result("B6", WARN, "; ".join(warns), d)
    return check_result("B6", PASS, f"same text and finish {rs.finish_raw!r}; final event and usage present; streamed "
                        "tool call assembles to valid JSON", d)


def _u(r):
    return {k: v for k, v in (r.usage or {}).items() if k != "raw"} or None


# ---------------------------------------------------------------------------------------------------- B7
LEAK = [
    (re.compile(r"\b(org|proj|project|ak|acct|account|user|team|workspace|ws)[-_][A-Za-z0-9]{8,}"), "account / project id"),
    (re.compile(r"\b(sk|rk)-[A-Za-z0-9_\-]{12,}"), "API key"),
    (re.compile(r"arn:aws:[a-z0-9\-]+:[a-z0-9\-]*:\d{12}:"), "AWS account"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+\.[A-Za-z0-9.\-]+"), "email address"),
    (re.compile(r"(organi[sz]ation|account|project|tenant)[^\n\"]{0,24}[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                re.I), "organization / account UUID"),
]


def _leaks(text: str) -> list:
    out = []
    for rx, what in LEAK:
        for m in rx.finditer(text or ""):
            s = m.group(0)
            out.append({"kind": what, "masked": s[:4] + "…" + s[-2:] if len(s) > 8 else "…"})
    return out


def b7(ctx) -> dict:
    p, o = ctx.proto, ctx.opts
    probes = [("invalid max_tokens", p.request([p.user("Reply with the word OK.")], max_tokens=-1), None, (400, 422)),
              ("unknown model", p.request([p.user("Reply with the word OK.")], max_tokens=16,
                                          model="tokenwatt-verify-no-such-model-0"), None, (400, 404))]
    if o.key:
        probes.append(("invalid key", p.request([p.user("Reply with the word OK.")], max_tokens=16),
                       "invalid-key-tokenwatt-verify-000000", (401, 403)))
    fails, warns, parts = [], [], {}
    for what, body, key, want in probes:
        r = ctx.call(body, label=what, key=key, retry=False)
        parts[what] = {"status": r.status, "error_type": r.error_type, "message": r.error_message[:200],
                       "error_in_200": r.error_in_200}
        if r.transport_error:
            warns.append(f"{what}: {r.transport_error[:80]}")
        elif r.status == 200:
            fails.append(f"{what}: " + ("error returned with HTTP 200" if r.error_in_200 else
                                        "accepted with HTTP 200" + (" and answered (a model that doesn't exist was served)"
                                                                    if what == "unknown model" else "")))
        elif r.status >= 500:
            fails.append(f"{what}: client error reported as HTTP {r.status}")
        elif r.status not in want:
            warns.append(f"{what}: HTTP {r.status} (expected {'/'.join(map(str, want))})")
        if r.status and r.status >= 400 and not r.error_message:
            warns.append(f"{what}: HTTP {r.status} without an error message")
        elif r.status and r.status >= 400 and p.api == "anthropic" and not r.error_type:
            warns.append(f"{what}: the error body is not an Anthropic error object (no error.type)")
    # every exchange of the reading so far
    e200 = [ob for ob in ctx.obs if ob["error_in_200"]]
    if e200:
        fails.append(f"{len(e200)} response(s) in this reading were errors with HTTP 200 "
                     f"(checks {sorted({ob['check'] for ob in e200})})")
    leaks = []
    for ob in ctx.obs:
        txt = " ".join(str(x) for x in (ob.get("error_message"), ob.get("stream_error")) if x)
        for lk in _leaks(txt):
            leaks.append({**lk, "check": ob["check"], "seq": ob["seq"]})
    if leaks:
        fails.append(f"error messages contain identifiers: {', '.join(sorted({lk['kind'] for lk in leaks}))} "
                     f"({len(leaks)} occurrence(s))")
    r429 = ctx.session.observed_429
    no_ra = [x for x in r429 if not x.get("retry_after")]
    if no_ra:
        warns.append(f"{len(no_ra)} of {len(r429)} HTTP 429 responses had no Retry-After")
    d = {"probes": parts, "errors_in_200": [{"check": ob["check"], "seq": ob["seq"]} for ob in e200], "leaks": leaks,
         "http_429": r429, "note": "429 handling is observed only if the endpoint rate-limited this reading"}
    if fails:
        return check_result("B7", FAIL, "; ".join(fails), d)
    if warns:
        return check_result("B7", WARN, "; ".join(warns), d)
    return check_result("B7", PASS, "; ".join(f"{k} → HTTP {v['status']}" for k, v in parts.items())
                        + ("; no 429 seen" if not r429 else f"; {len(r429)} × 429 with Retry-After"), d)
