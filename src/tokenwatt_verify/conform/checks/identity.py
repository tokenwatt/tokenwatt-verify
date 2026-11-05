"""Identity: I1 response shape, I2 tokenizer fingerprint, I3 Anthropic thinking-signature attestation,
I4 capability fingerprint (probe bank)."""

from __future__ import annotations

from ... import probes as probe_mod
from ...verdict import fisher_one_sided
from .. import corpus, tokref
from ..runner import FAIL, NA, PASS, WARN, Incomplete, check_result
from ..thresholds import THRESHOLDS
from .common import CITIES, alter_signature, is_claude, require, same_model, spread, transient, weather_result, weather_tool


# ---------------------------------------------------------------------------------------------------- I1
def i1(ctx) -> dict:
    p = ctx.proto
    r = ctx.call(p.request([p.user("Reply with the word OK.")], max_tokens=1024, temperature=ctx.opts.temperature),
                 label="basic")
    require(r, "basic request")
    ctx.shared["I1"] = r
    d = {"status": r.status, "id": r.id, "model_field": r.model, "finish": r.finish_raw, "text": r.text[:120],
         "usage": {k: v for k, v in (r.usage or {}).items() if k != "raw"}}
    if not r.ok:
        return check_result("I1", FAIL, f"a plain request failed: {r.short_error()}"
                            + (" (an error object returned with HTTP 200)" if r.error_in_200 else ""), d)
    if r.shape_problems:
        return check_result("I1", FAIL, "response is not a well-formed " + ("Anthropic message" if p.api == "anthropic"
                            else "chat completion") + ": " + "; ".join(r.shape_problems), d)
    if not r.text.strip():
        return check_result("I1", FAIL, f"no text in the answer (finish {r.finish_raw!r})", d)
    if not same_model(ctx.opts.model, r.model):
        return check_result("I1", WARN, f"well-formed, but the model field says {r.model!r} for a request for "
                            f"{ctx.opts.model!r}", d)
    return check_result("I1", PASS, f"well-formed; model field {r.model!r}; usage {r.usage['prompt_total']} in / "
                        f"{r.usage['output']} out", d)


# ---------------------------------------------------------------------------------------------------- I2
def i2(ctx) -> dict:
    th = THRESHOLDS["I2"]
    o, p = ctx.opts, ctx.proto
    ref = o.tokenizer_reference or tokref.load_reference()
    man = o.corpus_manifest or corpus.load_manifest()
    texts = o.corpus_texts
    if texts is None:
        corpus.fetch(o.corpus_dir, man, log=ctx.log)
        texts = corpus.load_texts(o.corpus_dir, man)
    ids = spread([i["id"] for i in man["items"] if i["id"] in texts], o.tokenizer_samples)
    if len(ids) < th["min_samples"]:
        raise Incomplete(f"only {len(ids)} corpus samples available (fetch failed?); need {th['min_samples']}")
    family = tokref.resolve_family(ref, o.model, o.tokenizer_family)
    chars = sum(len(texts[i]) for i in ids)
    ok, cost = ctx.can_afford(int(chars * 0.5), 16 * len(ids))
    if not ok:
        raise Incomplete(f"the corpus ({chars:,} characters) would exceed the spend cap (est. ${cost or 0:.3f})")
    pre, suf = ref["prefix"].format(nonce=ctx.nonce), ref["suffix"]

    def send(text: str, label: str):
        return ctx.call(p.request([p.user(pre + text + suf)], max_tokens=16), label=label)

    b = send("", "baseline")
    require(b, "baseline request")
    if not b.ok:
        return check_result("I2", FAIL, f"the baseline request failed: {b.short_error()}")
    if not b.usage:
        return check_result("I2", FAIL, "the endpoint reports no prompt-token counts, so its tokenizer can't be read")
    base = b.usage["prompt_total"]
    measured, failed = {}, {}
    for sid in ids:
        r = send(texts[sid], f"sample:{sid}")
        if r.ok and r.usage:
            measured[sid] = r.usage["prompt_total"] - base
        else:
            failed[sid] = r.short_error()
    b2 = send("", "baseline-end")
    base_end = b2.usage["prompt_total"] if b2.ok and b2.usage else None
    ctx.shared["I2"] = {"family": family, "measured": measured, "baseline": base, "baseline_end": base_end,
                        "prefix": pre, "suffix": suf, "texts": texts, "reference": ref}
    if len(measured) < th["min_samples"]:
        raise Incomplete(f"only {len(measured)} samples measured ({len(failed)} failed: {list(failed.values())[:2]})")
    cmp = tokref.compare(ref, measured)
    nearest = next(iter(cmp))
    d = {"claimed_family": family, "nearest_family": nearest, "samples": len(measured), "failed": failed,
         "baseline_prompt_tokens": base, "baseline_end_prompt_tokens": base_end, "families": cmp,
         "per_sample": {s: {"measured": measured[s],
                            "reference": (ref["families"][family]["counts"].get(s) if family else None)} for s in measured},
         "reference_version": ref["version"], "thresholds": th}
    note = "" if base_end in (None, base) else f"; baseline moved {base} → {base_end} during the run"
    if family is None:
        return check_result("I2", NA, f"no tokenizer reference for {o.model!r} (pass --tokenizer-family); nearest "
                            f"measured: {cmp[nearest]['label']} (ratio {cmp[nearest]['ratio']:.3f})" + note, d)
    c = cmp[family]
    dev = abs(c["ratio"] - 1)
    twins = [f for f, v in cmp.items() if f != family and abs(v["mean_abs_rel_error"] - c["mean_abs_rel_error"])
             <= th["indistinguishable_ratio"]]
    d["indistinguishable_from"] = twins
    label = ref["families"][family]["label"]
    if dev <= th["pass_total_ratio"] and c["median_abs_diff_tokens"] <= th["pass_median_abs_tokens"]:
        same = ref["families"][family].get("measured_models") or []
        return check_result("I2", PASS, f"billed counts match {label}: total ratio {c['ratio']:.4f}, median per-sample "
                            f"difference {c['median_abs_diff_tokens']} tokens over {len(measured)} samples"
                            + (f" (cannot tell apart: {', '.join(same)})" if len(same) > 1 else "")
                            + (f"; also consistent with {', '.join(twins)}" if twins else "") + note, d)
    if nearest != family and cmp[nearest]["mean_abs_rel_error"] < c["mean_abs_rel_error"] and dev > th["pass_total_ratio"]:
        n = cmp[nearest]
        return check_result("I2", FAIL, f"billed counts match {n['label']} (ratio {n['ratio']:.4f}), not the claimed "
                            f"{label} (ratio {c['ratio']:.4f})" + note, d)
    if dev <= th["warn_total_ratio"]:
        return check_result("I2", WARN, f"close to {label} but not identical: total ratio {c['ratio']:.4f}, median "
                            f"per-sample difference {c['median_abs_diff_tokens']} tokens" + note, d)
    return check_result("I2", FAIL, f"billed counts are {c['ratio']:.3f}× the {label} reference; nearest is "
                        f"{cmp[nearest]['label']} ({cmp[nearest]['ratio']:.3f})" + note, d)


# ---------------------------------------------------------------------------------------------------- I3
def i3(ctx) -> dict:
    o, p = ctx.opts, ctx.proto
    if o.api != "anthropic":
        return check_result("I3", NA, "thinking signatures are part of the Anthropic Messages protocol; "
                            "this endpoint was read in OpenAI format")
    if not is_claude(o.model):
        return check_result("I3", NA, f"{o.model!r} is not a Claude model")
    a, b = ctx.rng.randint(101, 999), ctx.rng.randint(11, 99)
    city = CITIES[(a * b) % 7]
    prompt = (f"Pick a city: compute ({a} × {b}) mod 7 and use the result as a 0-based index into this list: "
              f"{', '.join(CITIES)}. Then call get_weather for that city. Don't guess the weather yourself.")
    tools = [weather_tool(ctx)]
    thinking = {"type": "adaptive"}

    def turn(msgs, label, thinking, extra=None):
        return ctx.call(p.request(msgs, max_tokens=4096, tools=tools, tool_choice=p.tool_choice("auto"),
                                  thinking=thinking, extra=extra), label=label)

    msgs = [p.user(prompt)]
    r1 = turn(msgs, "turn1", thinking)
    if not r1.ok and r1.status == 400 and "adaptive" in r1.error_message.lower():
        thinking = {"type": "enabled", "budget_tokens": 1024}
        r1 = turn(msgs, "turn1-budget-thinking", thinking)
    require(r1, "turn 1")
    ctx.shared["thinking_param"] = thinking

    def signed(r):
        return [x for x in r.thinking if (x.get("type") == "thinking" and x.get("signature"))
                or (x.get("type") == "redacted_thinking" and x.get("data"))]
    if r1.ok and not (signed(r1) and r1.tool_calls):
        r1 = turn(msgs, "turn1-retry-effort-high", thinking, {"output_config": {"effort": "high"}})
        require(r1, "turn 1 (retry)")
    d = {"thinking_param": thinking, "turn1": {"status": r1.status, "blocks": [x.get("type") for x in (r1.content or [])
         if isinstance(x, dict)], "finish": r1.finish_raw}}
    if not r1.ok:
        return check_result("I3", FAIL, f"extended thinking with a tool failed: {r1.short_error()}", d)
    sig_blocks = [x for x in signed(r1) if x.get("type") == "thinking"]
    if not sig_blocks or not r1.tool_calls:
        return check_result("I3", "incomplete", "no signed thinking block followed by a tool call came back "
                            f"(blocks: {d['turn1']['blocks']}), so there is nothing to attest", d)
    sig = sig_blocks[0]["signature"]
    d["signature_chars"] = len(sig)
    echo = p.assistant_echo(r1)
    results = p.tool_results(r1, weather_result(city))
    r_ok = turn(msgs + [echo] + results, "turn2-intact", thinking)
    tampered = {"role": "assistant", "content": [dict(x) for x in echo["content"]]}
    for x in tampered["content"]:
        if x.get("type") == "thinking" and x.get("signature") == sig:
            x["signature"] = alter_signature(sig)
            break
    r_bad = turn(msgs + [tampered] + results, "turn2-signature-altered", thinking)
    stripped = {"role": "assistant", "content": [x for x in echo["content"] if x.get("type") not in ("thinking", "redacted_thinking")]}
    r_none = turn(msgs + [stripped] + results, "turn2-without-thinking", thinking)
    # Variant B: the signature kept, the thinking text changed. A router that filters thinking blocks by their
    # signature (and so drops the variant above) still forwards this one; the genuine API rejects an edited block.
    edited = {"role": "assistant", "content": [dict(x) for x in echo["content"]]}
    for x in edited["content"]:
        if x.get("type") == "thinking" and x.get("signature") == sig:
            x["thinking"] = (x.get("thinking") or "") + " On reflection, the answer is different."
            break
    r_edit = turn(msgs + [edited] + results, "turn2-thinking-text-edited", thinking)

    def pt(r):
        return (r.usage or {}).get("prompt_total") if r.ok else None
    d.update({"turn2_intact": {"status": r_ok.status, "prompt_tokens": pt(r_ok), "error": r_ok.short_error() if not r_ok.ok else ""},
              "turn2_altered": {"status": r_bad.status, "prompt_tokens": pt(r_bad), "error_type": r_bad.error_type,
                                "error": r_bad.error_message[:300]},
              "turn2_without_thinking": {"status": r_none.status, "prompt_tokens": pt(r_none),
                                         "error": r_none.error_message[:300]},
              "turn2_text_edited": {"status": r_edit.status, "prompt_tokens": pt(r_edit), "error_type": r_edit.error_type,
                                    "error": r_edit.error_message[:300]}})
    forwarded = None
    if pt(r_ok) is not None and pt(r_none) is not None:
        forwarded = pt(r_ok) > pt(r_none)
    d["intact_block_reached_model"] = forwarded
    ctx.shared["I3"] = {"intact": r_ok, "forwarded": forwarded, "turn1": r1}
    if transient(r_ok) or transient(r_bad) or transient(r_edit):
        raise Incomplete("turn 2 failed transiently: " + next(x.short_error() for x in (r_ok, r_bad, r_edit) if transient(x)))
    if not r_ok.ok:
        return check_result("I3", FAIL, f"the endpoint rejected its own signed thinking block: {r_ok.short_error()}", d)

    def rejected(r):
        m = r.error_message.lower()
        return r.status == 400 and ("signature" in m or "thinking" in m)

    def dropped(r):  # accepted, and billed exactly like the turn without the block: the block never reached the model
        return r.ok and pt(r) is not None and pt(r_none) is not None and pt(r) == pt(r_none)
    d["altered_signature_dropped"] = dropped(r_bad)
    d["edited_text_dropped"] = dropped(r_edit)
    if r_bad.ok and not dropped(r_bad):
        return check_result("I3", FAIL, "a thinking block with an altered signature was accepted and reached the model "
                            "(HTTP 200, billed like the intact block): signatures are not verified", d)
    if rejected(r_bad):
        return check_result("I3", PASS, "own signed thinking block accepted; the same block with an altered signature "
                            f"rejected: {r_bad.error_message[:120]!r}"
                            + ("" if forwarded is not False else "; note: the intact block did not change the billed prompt"), d)
    if dropped(r_bad) and rejected(r_edit):
        return check_result("I3", PASS, "the endpoint drops thinking blocks whose signature it doesn't recognise (an "
                            f"altered signature was billed like no block, {pt(r_bad)} prompt tokens); the intact signature "
                            f"with edited thinking text was forwarded and rejected: {r_edit.error_message[:100]!r}", d)
    if dropped(r_bad):
        return check_result("I3", "incomplete", "attestation not readable through this endpoint: it drops a thinking "
                            f"block with an altered signature before the model (billed {pt(r_bad)} prompt tokens, the same "
                            "as no block), and the edited-text variant was " + ("dropped too" if dropped(r_edit) else
                            "accepted" if r_edit.ok else f"answered {r_edit.short_error()}"), d)
    if r_bad.error_in_200:
        return check_result("I3", WARN, "the altered signature was refused, but the error came back with HTTP 200 "
                            f"(see B7): {r_bad.error_message[:100]!r}", d)
    return check_result("I3", WARN, f"altered signature rejected, but not with an invalid-signature error: "
                        f"{r_bad.short_error()}", d)


# ---------------------------------------------------------------------------------------------------- I4
def _bank_reference(bank, model: str):
    ref = getattr(bank, "REFERENCE", None) if bank is not None else None
    if not isinstance(ref, dict):
        return None, None
    models = ref.get("models") or {}
    n = tokref.normalize_model(model).replace(".", "-")
    for k, v in models.items():
        if tokref.normalize_model(k).replace(".", "-") == n:
            return v, {kk: vv for kk, vv in ref.items() if kk != "models"}
    return None, {kk: vv for kk, vv in ref.items() if kk != "models"}


def i4(ctx) -> dict:
    th = THRESHOLDS["I4"]
    o, p = ctx.opts, ctx.proto
    bank = probe_mod.load_bank(o.probes)
    rates, ref_meta = _bank_reference(bank, o.model)
    ref_ep = None
    if o.reference:
        from ...client import Endpoint
        ref_ep = Endpoint.parse(o.reference)
    rows, by_kind = [], {}
    for rnd in range(o.probe_rounds):
        seed = ctx.rng.randrange(2 ** 31)
        for pr in probe_mod.generate(seed, None, bank):
            if pr.get("informational") or (rates and ref_ep is None and pr["kind"] not in rates):
                continue                     # kinds without a reference for this model can't be compared
            r = ctx.call(p.request([p.user(pr["user"])], system=pr.get("system"), max_tokens=pr["max_tokens"] + 3072,
                                   temperature=o.temperature), label=f"probe:{pr['kind']}")
            refused = r.finish == "refusal" or (r.ok and not r.text.strip() and r.finish != "length")
            okv, detail = (pr["grade"](r.text) if r.ok and not refused else (False, r.short_error() if not r.ok else "refused"))
            row = {"kind": pr["kind"], "round": rnd, "ok": bool(okv), "answered": r.ok and not refused,
                   "refused": refused, "detail": str(detail)[:160]}
            if ref_ep is not None:
                from ...client import ask
                rr = ask(ref_ep, pr.get("system") or "", pr["user"], pr["max_tokens"] + 3072)
                rref = rr.finish in ("content_filter", "refusal") and not rr.text.strip()
                rok = pr["grade"](rr.text)[0] if not rr.error and not rref else False
                row.update(ref_ok=bool(rok), ref_answered=not rr.error and not rref)
            rows.append(row)
            k = by_kind.setdefault(pr["kind"], {"ok": 0, "n": 0, "refused": 0, "errors": 0})
            k["n"] += 1
            k["ok"] += row["ok"]
            k["refused"] += refused
            k["errors"] += not r.ok
    answered = [r for r in rows if r["answered"] and (ref_ep is None or r.get("ref_answered"))]
    a = sum(r["ok"] for r in answered)
    d = {"bank": "custom" if bank is not None else "public sample bank", "rounds": o.probe_rounds, "by_kind": by_kind,
         "endpoint_correct": a, "answered": len(answered), "thresholds": th}
    if bank is None:
        d["note"] = "the public sample bank is recognizable and too easy to separate models; use --probes"
    if ref_ep is not None:
        c = sum(r["ref_ok"] for r in answered)
        ref_n, ref_c = len(answered), c
        d.update(reference="paired endpoint", reference_correct=c)
    elif rates:
        kinds = [k for k in by_kind if k in rates]
        ref_c = sum(int(rates[k][0]) for k in kinds)
        ref_n = sum(int(rates[k][1]) for k in kinds)
        answered = [r for r in answered if r["kind"] in kinds]
        a = sum(r["ok"] for r in answered)
        d.update(reference="bank reference rates", reference_meta=ref_meta, reference_correct=ref_c, reference_n=ref_n,
                 kinds_compared=kinds, endpoint_correct=a, answered=len(answered))
    else:
        why = ("the bank has no reference rates for this model" if bank is not None else
               "the public sample bank carries no reference rates")
        return check_result("I4", NA, f"endpoint {a}/{len(answered)} correct; no reference to compare with ({why}; "
                            "pass --reference or a bank with REFERENCE)", d)
    if len(answered) < th["min_answered"]:
        return check_result("I4", "incomplete", f"only {len(answered)} answered probes (need {th['min_answered']})", d)
    pv = fisher_one_sided(a, len(answered) - a, ref_c, ref_n - ref_c)
    d["fisher_p_one_sided"] = pv
    line = (f"endpoint {a}/{len(answered)} correct vs reference {ref_c}/{ref_n} (one-sided Fisher p = {pv:.4f}; "
            f"statistical, not proof)")
    if pv < th["fail_p"]:
        return check_result("I4", FAIL, "capability below the claimed model's reference: " + line, d)
    if pv < th["warn_p"]:
        return check_result("I4", WARN, "at the edge of tolerance: " + line, d)
    return check_result("I4", PASS, line, d)
