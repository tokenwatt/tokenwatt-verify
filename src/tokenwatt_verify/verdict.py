"""Turn paired results (endpoint vs reference, same prompts) into findings and an overall verdict.

Signals, from strongest to weakest:
  capability   graded accuracy on the same fresh prompts, endpoint vs reference (Fisher exact test)
  tokenizer    input-token counts the endpoint reports vs the reference for identical prompts
  refusals     how often each side refuses or is filtered (a model family's safety behavior is a fingerprint)
  latency      time to first token and generation speed (relays add latency; informational only)
  errors       share of requests that fail (a sign of multiple upstream channels or instability)

Verdict: "inconsistent" if capability or tokenizer evidence says the endpoint isn't the reference model;
"consistent" if there are enough answered requests and no such evidence; otherwise "inconclusive".
"""

from __future__ import annotations

import statistics
from math import comb

MIN_PAIRS = 20          # answered prompt pairs needed before calling anything "consistent"
TOKEN_TOL = 0.03        # input-token count difference that counts as a mismatch
TOKEN_SHARE = 0.10      # share of mismatching requests that counts as evidence
ALPHA = 0.01            # significance for the capability gap


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(endpoint correct ≤ a) under H0 of equal accuracy. Table: endpoint a ok / b wrong; reference c ok / d wrong."""
    n1, n2, k = a + b, c + d, a + c
    total = comb(n1 + n2, k)
    return sum(comb(n1, i) * comb(n2, k - i) for i in range(max(0, k - n2), a + 1)) / total


def evaluate(records: list[dict]) -> dict:
    graded = [r for r in records if not r.get("informational") and "ref_ok" in r]
    answered = [r for r in graded if not r["reply"].get("error") and not r["ref_reply"].get("error")
                and not r.get("refused") and not r.get("ref_refused")]
    findings: list[dict] = []

    a = sum(r["ok"] for r in answered); b = len(answered) - a
    c = sum(r["ref_ok"] for r in answered); d = len(answered) - c
    p = fisher_one_sided(a, b, c, d) if answered else 1.0
    cap_bad = len(answered) >= 10 and p < ALPHA
    findings.append({"signal": "capability", "level": "FAIL" if cap_bad else "PASS" if answered else "INFO",
                     "message": f"endpoint {a}/{len(answered)} correct vs reference {c}/{len(answered)} on the same prompts"
                                + (f" (one-sided Fisher p = {p:.4f})" if answered else "")})

    ratios = [r["input_token_ratio"] for r in records if r.get("input_token_ratio")]
    off = [x for x in ratios if abs(x - 1) > TOKEN_TOL]
    tok_bad = len(ratios) >= 10 and len(off) / len(ratios) > TOKEN_SHARE
    findings.append({"signal": "tokenizer", "level": "FAIL" if tok_bad else "PASS" if ratios else "INFO",
                     "message": (f"{len(off)}/{len(ratios)} requests report input-token counts more than {TOKEN_TOL:.0%} "
                                 f"away from the reference" + (f" (median ratio {statistics.median(ratios):.3f})" if ratios else ""))
                     if ratios else "no comparable token counts (endpoint or reference didn't report usage)"})

    tr = sum(bool(r.get("refused")) for r in graded); rr = sum(bool(r.get("ref_refused")) for r in graded)
    findings.append({"signal": "refusals", "level": "INFO",
                     "message": f"endpoint refused/filtered {tr}/{len(graded)}, reference {rr}/{len(graded)}"})

    def med(side, key):
        v = [r[side].get(key) for r in records if r.get(side) and r[side].get(key)]
        return statistics.median(v) if v else None
    t1, t2 = med("reply", "ttft_s"), med("ref_reply", "ttft_s")
    if t1 and t2:
        findings.append({"signal": "latency", "level": "WARN" if t1 > 3 * t2 else "INFO",
                         "message": f"median time to first token {t1:.2f} s vs reference {t2:.2f} s ({t1 / t2:.1f}×)"})

    errs = sum(bool(r["reply"].get("error")) for r in records)
    if records:
        findings.append({"signal": "errors", "level": "WARN" if errs / len(records) > 0.05 else "INFO",
                         "message": f"{errs}/{len(records)} endpoint requests failed"})

    if cap_bad or tok_bad:
        overall = "inconsistent"
    elif len(answered) >= MIN_PAIRS:
        overall = "consistent"
    else:
        overall = "inconclusive"
    return {"overall": overall, "answered_pairs": len(answered), "findings": findings}
