"""tokenwatt-verify: is the model behind an API endpoint the one it claims to be, request by request?

    export OPENROUTER_API_KEY=...                       # reference: the genuine model through OpenRouter
    export EP_URL=https://api.example.com/v1 EP_KEY=... EP_MODEL=claude-sonnet-4-6 EP_API=anthropic
    tokenwatt-verify run --target env:EP --reference openrouter:anthropic/claude-sonnet-4.6 --rounds 5
    tokenwatt-verify run --target env:EP --reference openrouter:… --every 1800 --hours 24     # monitor
    tokenwatt-verify run … --probes my_private_bank.py                                         # your own prompts
    tokenwatt-verify summary results/*.jsonl

Each round sends one fresh, randomly generated instance of every probe to the endpoint and the same prompt
to the reference, then compares graded answers, reported token counts, refusals and latency. See README.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

from . import __version__
from .client import Endpoint, ask
from .probes import generate, load_bank
from .verdict import evaluate

ICON = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "INFO": "ℹ️ "}


def run(a) -> int:
    target = Endpoint.parse(a.target)
    ref = Endpoint.parse(a.reference)
    bank = load_bank(a.probes)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"verify-{target.name.replace(':', '_').replace('/', '_')}-{dt.datetime.now():%Y%m%d-%H%M%S}.jsonl"
    kinds = a.kinds.split(",") if a.kinds else None
    deadline = time.time() + a.hours * 3600 if a.every else None
    print(f"tokenwatt-verify {__version__} · endpoint {target.model} ({target.api}) · reference {ref.name} · "
          f"probes {'custom bank' if bank else 'public sample bank'} → {out}", flush=True)
    if not bank:
        print("  note: the sample bank is public and recognizable; use --probes with your own bank for real checks", flush=True)
    rnd = 0
    while True:
        rnd += 1
        seed = random.SystemRandom().randrange(2 ** 31)
        for p in generate(seed, kinds, bank):
            rec = {"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "round": rnd, "kind": p["kind"],
                   "informational": bool(p.get("informational"))}
            for side, ep in (("reply", target), ("ref_reply", ref)):
                r = ask(ep, p["system"], p["user"], p["max_tokens"], temperature=a.temperature)
                ok, detail = p["grade"](r.text) if not r.error else (False, r.error)
                refused = r.finish in ("content_filter", "refusal") and not r.text.strip()
                d = r.as_dict()
                d["text"] = d["text"][-800:]
                prefix = "" if side == "reply" else "ref_"
                rec.update({f"{prefix}ok": ok, f"{prefix}detail": detail, f"{prefix}refused": refused, side: d})
            if rec["reply"].get("input_tokens") and rec["ref_reply"].get("input_tokens"):
                rec["input_token_ratio"] = round(rec["reply"]["input_tokens"] / rec["ref_reply"]["input_tokens"], 4)
            with out.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mark = lambda o, rf, e: "error" if e else "refused" if rf else ("PASS" if o else "fail")  # noqa: E731
            tr = rec.get("input_token_ratio")
            print(f"  r{rnd} {p['kind']:<18} {mark(rec['ok'], rec['refused'], rec['reply'].get('error')):<7} | reference "
                  f"{mark(rec['ref_ok'], rec['ref_refused'], rec['ref_reply'].get('error')):<7}"
                  + (f" | tokens ×{tr:.3f}" if tr else "")
                  + f" | ttft {rec['reply'].get('ttft_s') or 0:.2f}s", flush=True)
        if not a.every and rnd >= a.rounds:
            break
        if deadline and time.time() > deadline:
            break
        if a.every:
            time.sleep(a.every)
    return report([out], as_json=a.json)


def report(paths: list[Path], as_json: str | None = None) -> int:
    recs = [json.loads(line) for p in paths for line in Path(p).read_text().splitlines() if line.strip()]
    v = evaluate(recs)
    print(f"\n{len(recs)} requests · {v['answered_pairs']} answered prompt pairs")
    for f in v["findings"]:
        print(f"  {ICON[f['level']]} {f['signal']:<11} {f['message']}")
    ids = [r["detail"] for r in recs if r["kind"] == "identity" and r["detail"]][:3]
    for i in ids:
        print(f"  ℹ️  identity    endpoint said: {i}")
    label = {"consistent": "CONSISTENT with the reference model", "inconsistent": "INCONSISTENT with the reference model",
             "inconclusive": "INCONCLUSIVE (not enough answered requests yet)"}[v["overall"]]
    print(f"\n  VERDICT: {label}\n")
    if as_json:
        Path(as_json).write_text(json.dumps(v, indent=2))
    return {"consistent": 0, "inconclusive": 0, "inconsistent": 2}[v["overall"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tokenwatt-verify", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"tokenwatt-verify {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="send probes to an endpoint and a reference, compare")
    r.add_argument("--target", required=True, help="env:<PREFIX>, openrouter:<model>, or a file with URL/KEY/MODEL/API lines")
    r.add_argument("--reference", required=True, help="the genuine model, usually openrouter:<vendor/model>")
    r.add_argument("--probes", help="path to your own probe bank (Python file defining SUITE)")
    r.add_argument("--kinds", help="comma list of probe kinds to run")
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--every", type=float, help="seconds between rounds (monitor mode)")
    r.add_argument("--hours", type=float, default=24, help="with --every: stop after this many hours")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--out", default="tokenwatt-verify-results")
    r.add_argument("--json", help="write the verdict as JSON")
    s = sub.add_parser("summary", help="verdict from earlier result files")
    s.add_argument("files", nargs="+")
    s.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    return report([Path(f) for f in a.files], as_json=a.json)


if __name__ == "__main__":
    sys.exit(main())
