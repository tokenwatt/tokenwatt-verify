"""Service: S1 time to first token and decode speed (reported, not graded)."""

from __future__ import annotations

import concurrent.futures as cf
import math
import statistics

from ..runner import check_result
from ..thresholds import THRESHOLDS

TOPICS = ["a lighthouse on a rocky coast", "a night train crossing a desert", "a busy fish market at dawn",
          "an old library during a storm", "a mountain village in winter", "a harbour at low tide",
          "a botanical greenhouse", "a city rooftop garden", "a river ferry crossing", "a windmill on a plain"]


def pct(xs: list, q: float):
    """Nearest-rank percentile (with few samples, p99 is the maximum)."""
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, math.ceil(q / 100 * len(s)) - 1))]


def s1(ctx) -> dict:
    p, o = ctx.proto, ctx.opts
    mt = THRESHOLDS["S1"]["max_tokens"]

    def one(i: int):
        ask = f"[{ctx.nonce}-{i}] Write about 150 words describing {TOPICS[i % len(TOPICS)]}. Plain prose, one paragraph."
        return ctx.call(p.request([p.user(ask)], max_tokens=mt, stream=True), stream=True, label=f"sample{i + 1}")

    if o.concurrency > 1:
        with cf.ThreadPoolExecutor(max_workers=o.concurrency) as ex:
            rs = list(ex.map(one, range(o.samples)))
    else:
        rs = [one(i) for i in range(o.samples)]
    ok = [r for r in rs if r.ok and r.t_first_token is not None]
    ttft = [r.t_first_token for r in ok]
    ttft_text = [r.t_first_text for r in ok if r.t_first_text is not None]
    dec = []
    for r in ok:
        out = (r.usage or {}).get("output")
        if out and r.t_last_token and r.t_last_token > r.t_first_token and out > 1:
            dec.append((out - 1) / (r.t_last_token - r.t_first_token))
    prompt = statistics.median([(r.usage or {}).get("prompt_total") or 0 for r in ok]) if ok else None

    def stats(xs, nd, slow_is_low=False):
        tail = pct(xs, 1 if slow_is_low else 99)  # p99 = the slow tail: high latency, low decode speed
        return {"p50": round(pct(xs, 50), nd) if xs else None, "p99": round(tail, nd) if xs else None,
                "mean": round(statistics.mean(xs), nd) if xs else None, "n": len(xs)}
    d = {"samples": o.samples, "concurrency": o.concurrency, "succeeded": len(ok), "prompt_tokens_median": prompt,
         "max_tokens": mt, "ttft_s": stats(ttft, 3), "ttft_first_text_s": stats(ttft_text, 3),
         "decode_tokens_per_s": stats(dec, 1, slow_is_low=True),
         "definitions": "TTFT = first streamed token (text, reasoning or tool arguments) after the request was sent; "
                        "decode = (completion tokens - 1) / (last token - first token). Hidden reasoning generated "
                        "before the first visible token inflates decode. p99 is the slow tail (highest TTFT, lowest "
                        "decode speed); with a few samples it is the most extreme sample.",
         "errors": [r.short_error() for r in rs if not r.ok]}
    if not ok:
        return check_result("S1", "incomplete", f"no streamed sample succeeded ({d['errors'][:1]})", d)
    t, s = d["ttft_s"], d["decode_tokens_per_s"]
    return check_result("S1", "reported", f"TTFT p50 {t['p50']} s, p99 {t['p99']} s; decode p50 {s['p50']} tok/s, "
                        f"p99 {s['p99']} tok/s ({len(ok)} streamed samples, concurrency {o.concurrency}, "
                        f"~{prompt} prompt tokens)", d)
