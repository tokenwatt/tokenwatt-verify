"""Billing: P1 billed tokens vs the reference tokenizer, P2 cost vs listed price, P3 cache discount honoured,
P4 failed requests not billed."""

from __future__ import annotations

import time
from pathlib import Path

from .. import pricing
from ..runner import FAIL, NA, PASS, WARN, check_result
from ..thresholds import THRESHOLDS
from .common import require


def _snippet(texts: dict, nbytes: int = 1200) -> tuple[str, str]:
    sid = "py-requests-auth" if "py-requests-auth" in texts else next(iter(texts))
    t = texts[sid]
    cut = t.encode()[:nbytes].decode("utf-8", "ignore")
    return sid, cut[: cut.rfind("\n") + 1] if "\n" in cut else cut


def _strip_fences(s: str) -> str:
    s = s.strip("\n")
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip("\n")


# ---------------------------------------------------------------------------------------------------- P1
def p1(ctx) -> dict:
    th = THRESHOLDS["P1"]
    i2 = ctx.shared.get("I2")
    if not i2:
        return check_result("P1", NA, "needs the corpus measurements of I2 (run I2 in the same reading)")
    p = ctx.proto
    fails, warns, passes, d = [], [], [], {"thresholds": th}
    fam = i2["family"]
    if fam:
        counts = i2["reference"]["families"][fam]["counts"]
        diffs = {s: m - counts[s] for s, m in i2["measured"].items() if s in counts}
        worst = max(diffs, key=lambda s: abs(diffs[s]))
        ratio = sum(i2["measured"][s] for s in diffs) / sum(counts[s] for s in diffs)
        d["text"] = {"family": fam, "per_sample_diff": diffs, "suite_ratio": round(ratio, 5), "worst": worst}
        if abs(diffs[worst]) > th["per_request_tokens"]:
            fails.append(f"{worst}: {i2['measured'][worst]} tokens billed vs {counts[worst]} reference "
                         f"(Δ {diffs[worst]:+d}, limit ±{th['per_request_tokens']})")
        if abs(ratio - 1) > th["suite_ratio"]:
            fails.append(f"suite total {ratio:.4f}× the reference (limit ±{th['suite_ratio']:.0%})")
        if not fails:
            passes.append(f"{len(diffs)} corpus samples within ±{max(abs(v) for v in diffs.values())} tokens of the "
                          f"reference (suite {ratio:.4f}×)")
    else:
        d["text"] = "no reference tokenizer for this model: input counts not compared"
    base = i2["baseline"]
    d["baseline_prompt_tokens"] = base
    if base > th["baseline_warn_tokens"]:
        warns.append(f"a one-line message is billed {base} prompt tokens: something is added to every prompt")
    # output side: a verbatim echo should bill about as many completion tokens as the same text costs as input
    sid, snip = _snippet(i2["texts"])
    r_in = ctx.call(p.request([p.user(i2["prefix"] + snip + i2["suffix"])], max_tokens=16), label="echo-input-count")
    require(r_in, "echo input count")
    ask = ("Repeat the text between the markers exactly, character for character. Output only that text: nothing "
           f"before or after it, no code fences.\n<<<BEGIN>>>\n{snip}<<<END>>>")
    r = ctx.call(p.request([p.user(ask)], max_tokens=4096), label="echo")
    require(r, "echo request")
    if r_in.ok and r_in.usage and r.ok and r.usage:
        n_in = r_in.usage["prompt_total"] - base
        out, reasoning = r.usage.get("output") or 0, r.usage.get("reasoning")
        verbatim = _strip_fences(r.text) == snip.strip("\n")
        e = {"sample": sid, "chars": len(snip), "input_count": n_in, "completion_tokens": out, "reasoning_tokens": reasoning,
             "verbatim": verbatim, "thinking_blocks": len(r.thinking)}
        d["echo"] = e
        if not verbatim:
            e["note"] = "echo not verbatim; output side not compared"
        elif (r.thinking or r.reasoning_text) and reasoning is None:
            e["note"] = "the answer included reasoning that usage does not itemize; output side not compared"
        else:
            diff = out - (reasoning or 0) - n_in
            e["difference"] = diff
            if abs(diff) > th["output_echo_tokens"]:
                fails.append(f"a verbatim echo of {n_in} input tokens billed {out - (reasoning or 0)} completion tokens "
                             f"(Δ {diff:+d}, limit ±{th['output_echo_tokens']})")
            else:
                passes.append(f"echo of {n_in} tokens billed {out - (reasoning or 0)} completion tokens")
    if fails:
        return check_result("P1", FAIL, "; ".join(fails), d)
    if not passes:
        return check_result("P1", NA, "nothing comparable: " + "; ".join(warns + [str(d.get("text")),
                            str((d.get("echo") or {}).get("note", ""))]), d)
    if warns:
        return check_result("P1", WARN, "; ".join(passes + warns), d)
    return check_result("P1", PASS, "; ".join(passes), d)


# ---------------------------------------------------------------------------------------------------- P2
def p2(ctx) -> dict:
    th = THRESHOLDS["P2"]
    o = ctx.opts
    if not ctx.prices:
        return check_result("P2", NA, "no listed price supplied (--prices)")
    if not (o.cost_header or o.cost_field):
        return check_result("P2", NA, "the endpoint's per-request cost is not observable here (no --cost-header / "
                            f"--cost-field); expected cost at list for this reading: ${ctx.spend_est:.4f}")
    obs = [ob for ob in ctx.obs if ob["observed_cost"] is not None]
    comp = [ob for ob in obs if ob["expected_cost"] is not None and ob["ok"]]
    if not comp:
        return check_result("P2", NA, f"no request had both a reported cost and a computable list cost "
                            f"({len(obs)} with a reported cost)")
    over, under = [], []
    for ob in comp:
        lim = ob["expected_cost"] * (1 + th["max_over_list_ratio"]) + th["abs_rounding_usd"]
        if ob["observed_cost"] > lim:
            over.append(ob)
        elif ob["observed_cost"] < ob["expected_cost"] * (1 - th["max_over_list_ratio"]) - th["abs_rounding_usd"]:
            under.append(ob)
    tot_o, tot_e = sum(ob["observed_cost"] for ob in comp), sum(ob["expected_cost"] for ob in comp)

    def row(ob):
        return {k: ob[k] for k in ("seq", "check", "label", "observed_cost", "expected_cost", "usage")}
    d = {"compared": len(comp), "observed_total_usd": round(tot_o, 6), "expected_total_usd": round(tot_e, 6),
         "above_list": [row(ob) for ob in over], "below_list": [row(ob) for ob in under][:20],
         "not_comparable": len(obs) - len(comp), "prices": ctx.prices, "price_source": ctx.price_meta,
         "cost_source": o.cost_header or o.cost_field, "thresholds": th}
    if over:
        w = max(over, key=lambda ob: ob["observed_cost"] / max(1e-12, ob["expected_cost"]))
        return check_result("P2", FAIL, f"{len(over)} of {len(comp)} requests billed above list; worst "
                            f"${w['observed_cost']:.6f} vs ${w['expected_cost']:.6f} ({w['check']} {w['label']})", d)
    return check_result("P2", PASS, f"{len(comp)} requests at or below list (reported ${tot_o:.4f} vs list ${tot_e:.4f})"
                        + (f"; {len(under)} below list" if under else ""), d)


# ---------------------------------------------------------------------------------------------------- P3
def _prefix_text(n_chars: int) -> str:
    src = Path(__file__).resolve().parents[2]
    files = sorted(src.rglob("*.py"))
    body = "\n\n".join(f"# file: {f.relative_to(src)}\n{f.read_text(encoding='utf-8')}" for f in files)
    while len(body) < n_chars:
        body += "\n\n" + body
    return body[:n_chars]


def p3(ctx) -> dict:
    th = THRESHOLDS["P3"]
    p = ctx.proto
    system = (f"Session {ctx.nonce}. For reference, this is the source code of the tool we are working on:\n\n"
              + _prefix_text(th["prefix_chars"]))
    rows, hit = [], None
    for i in range(th["repeats"]):
        if i:
            time.sleep(3 if i == 1 else 10)
        r = ctx.call(p.request([p.user("Reply with the word OK.")], system=system, max_tokens=16,
                               cache_system=p.api == "anthropic"), label=f"call{i + 1}")
        require(r, f"cache call {i + 1}")
        if not r.ok:
            return check_result("P3", FAIL, f"call {i + 1} failed: {r.short_error()}", {"calls": rows})
        u = r.usage or {}
        rows.append({"call": i + 1, "prompt_total": u.get("prompt_total"), "cache_read": u.get("cache_read"),
                     "cache_write": u.get("cache_write"), "observed_cost": r.observed_cost,
                     "expected_cost": pricing.expected_cost(ctx.prices, u)})
        first = rows[0]["prompt_total"] or 0
        if i and (u.get("cache_read") or 0) >= th["min_cached_share"] * first:
            hit = (r, rows[-1])
            break
    d = {"calls": rows, "prefix_chars": len(system), "thresholds": th}
    listed = bool(ctx.prices and "cache_read" in ctx.prices)
    if not hit:
        if listed:
            return check_result("P3", FAIL, f"a cached-input price is listed, but {len(rows) - 1} repeats of a "
                                f"{rows[0]['prompt_total']}-token prefix within ~15 s reported no cached tokens", d)
        return check_result("P3", NA, f"no cached tokens reported on {len(rows) - 1} repeats, and no cached-input price "
                            "is listed (--prices)", d)
    r, row = hit
    msg = f"repeat {row['call']}: {row['cache_read']} of {row['prompt_total']} prompt tokens reported as cached"
    if row["observed_cost"] is None or not ctx.prices:
        return check_result("P3", PASS, msg + "; discount not verified (no per-request cost observable)", d)
    if row["expected_cost"] is None:
        return check_result("P3", PASS, msg + "; discount not verified (no cached price in --prices)", d)
    full = pricing.expected_cost(ctx.prices, {**(r.usage or {}), "cache_read": 0, "cache_write": 0,
                                              "input_uncached": (r.usage or {}).get("prompt_total")})
    d.update(full_price_cost=full)
    if row["observed_cost"] > row["expected_cost"] * (1 + th["max_over_cached_price_ratio"]) + 2e-6:
        return check_result("P3", FAIL, msg + f", but billed ${row['observed_cost']:.6f} vs ${row['expected_cost']:.6f} at "
                            f"the cached price (${full:.6f} at full price)", d)
    return check_result("P3", PASS, msg + f"; billed ${row['observed_cost']:.6f} = the cached price "
                        f"(full price would be ${full:.6f})", d)


# ---------------------------------------------------------------------------------------------------- P4
def p4(ctx) -> dict:
    o = ctx.opts
    failed = [ob for ob in ctx.obs if not ob["ok"] and not ob["output_started"] and (ob["status"] or 0) >= 400]
    if not failed:
        return check_result("P4", NA, "no request failed in this reading")
    if not (o.cost_header or o.cost_field):
        return check_result("P4", NA, f"{len(failed)} requests failed; per-request cost is not observable (no "
                            "--cost-header / --cost-field): confirm on the invoice")
    with_cost = [ob for ob in failed if ob["observed_cost"] is not None]
    d = {"failed": [{k: ob[k] for k in ("seq", "check", "label", "status", "observed_cost")} for ob in failed]}
    if not with_cost:
        return check_result("P4", NA, f"{len(failed)} failed requests carried no cost; not billed as far as the "
                            "responses show (confirm on the invoice)", d)
    billed = [ob for ob in with_cost if ob["observed_cost"] > 0]
    if billed:
        return check_result("P4", FAIL, f"{len(billed)} failed requests reported a cost (e.g. ${billed[0]['observed_cost']:.6f}"
                            f" on {billed[0]['check']} {billed[0]['label']}, HTTP {billed[0]['status']})", d)
    return check_result("P4", PASS, f"{len(with_cost)} failed requests reported cost 0"
                        + (f"; {len(failed) - len(with_cost)} carried no cost" if len(with_cost) < len(failed) else ""), d)
