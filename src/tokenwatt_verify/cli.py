"""tokenwatt-verify: is the model behind an API endpoint the one it claims to be, request by request?

    export OPENROUTER_API_KEY=...                       # reference: the genuine model through OpenRouter
    export EP_URL=https://api.example.com/v1 EP_KEY=... EP_MODEL=claude-sonnet-4-6 EP_API=anthropic
    tokenwatt-verify run --target env:EP --reference openrouter:anthropic/claude-sonnet-4.6 --rounds 5
    tokenwatt-verify run --target env:EP --reference openrouter:… --every 1800 --hours 24     # monitor
    tokenwatt-verify run … --probes my_private_bank.py                                         # your own prompts
    tokenwatt-verify summary results/*.jsonl

Each round sends one fresh, randomly generated instance of every probe to the endpoint and the same prompt
to the reference, then compares graded answers, reported token counts, refusals and latency. See README.

    tokenwatt-verify conform --base-url https://api.example.com --api anthropic --model claude-sonnet-5-5 \
        --key-file ~/.keys/example --prices prices.json [--probes my_bank.py] [--long-context --context-window N]

`conform` grades one endpoint against the TokenWatt Inference Endpoint Conformance draft (identity, behaviour,
billing, service checks) and writes report.json (with a content hash), report.md and evidence.jsonl.
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
    r.add_argument("--reference", required=True, help="the genuine model: openrouter:<vendor/model>, env:<PREFIX> or a file (any OpenAI-compatible server)")
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
    add_conform_parser(sub)
    v = sub.add_parser("verify-report", help="check a conform report's content hash and its evidence hashes")
    v.add_argument("report")
    c = sub.add_parser("corpus", help="download and verify the tokenizer corpus used by check I2")
    c.add_argument("--dir", help="corpus directory (default ~/.cache/tokenwatt-verify/corpus)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "conform":
        return conform(a)
    if a.cmd == "verify-report":
        from .conform.report import verify_report
        ok, msgs = verify_report(Path(a.report))
        print("\n".join(msgs))
        return 0 if ok else 1
    if a.cmd == "corpus":
        from .conform.corpus import fetch
        r = fetch(a.dir)
        return 0 if not r["skipped"] else 1
    return report([Path(f) for f in a.files], as_json=a.json)


def add_conform_parser(sub) -> None:
    c = sub.add_parser("conform", help="grade an endpoint against the Inference Endpoint Conformance draft",
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description="Run the conformance checks against one endpoint; see README for what each proves.")
    c.add_argument("--base-url", required=True, help="Anthropic: the host (…/v1/messages is appended); "
                   "OpenAI: the base ending in /v1 (…/chat/completions is appended)")
    c.add_argument("--api", required=True, choices=["anthropic", "openai"])
    c.add_argument("--model", required=True, help="model name exactly as the endpoint lists it")
    k = c.add_mutually_exclusive_group()
    k.add_argument("--key-file", help="file holding the API key (never printed or written to the report)")
    k.add_argument("--key-env", help="environment variable holding the API key")
    c.add_argument("--auth", choices=["auto", "x-api-key", "bearer", "both"], default="auto",
                   help="how to send the key (auto: x-api-key for anthropic, bearer for openai)")
    c.add_argument("--header", action="append", default=[], metavar="'Name: value'",
                   help="extra request header (repeatable); values are redacted from the evidence")
    c.add_argument("--prices", help="JSON file with the listed prices (USD per 1M tokens); see README")
    c.add_argument("--cost-header", help="response header that carries the billed cost of a request")
    c.add_argument("--cost-field", help="dotted JSON path of the billed cost in the response, e.g. usage.cost")
    c.add_argument("--disclosure-header", action="append", default=[], metavar="NAME",
                   help="response header in which the endpoint discloses parameters it adjusted (param=value, …); a "
                        "deviation it names is graded warn (documented), not fail. Repeatable")
    c.add_argument("--probes", help="probe bank for I4 (Python file defining SUITE, optionally REFERENCE)")
    c.add_argument("--probe-rounds", type=int, default=2)
    c.add_argument("--reference", help="I4: a reference endpoint answering the same probes (env:/file spec, as in run)")
    c.add_argument("--long-context", action="store_true", help="B5: fill the advertised window (costly; needs --context-window)")
    c.add_argument("--context-window", type=int, help="advertised context window in tokens")
    c.add_argument("--max-output", type=int, help="advertised max output tokens (B5 checks it is accepted)")
    c.add_argument("--samples", type=int, default=5, help="S1: streamed samples (default 5)")
    c.add_argument("--concurrency", type=int, default=1, help="S1: concurrent requests (default 1)")
    c.add_argument("--tokenizer-family", help="I2: tokenizer family to compare with, if the model name doesn't resolve")
    c.add_argument("--tokenizer-samples", type=int, help="I2: use only this many corpus samples (cheaper, less precise)")
    c.add_argument("--corpus-dir", help="I2: corpus directory (default ~/.cache/tokenwatt-verify/corpus)")
    c.add_argument("--checks", help="comma list of check ids to run (default: all)")
    c.add_argument("--skip", help="comma list of check ids to skip")
    c.add_argument("--max-spend", type=float, default=2.0, help="stop sending once the estimated spend at --prices "
                   "reaches this (USD; default 2)")
    c.add_argument("--max-tokens-total", type=int, default=3_000_000, help="token cap when no prices are given")
    c.add_argument("--timeout", type=float, default=300)
    c.add_argument("--seed", type=int)
    c.add_argument("--run-by", default="self-run", help="who ran the reading (written into the report)")
    c.add_argument("--out", default="tokenwatt-verify-results")


def conform(a) -> int:
    import os

    from .conform.runner import Options, run as run_conform
    key = ""
    if a.key_file:
        key = Path(a.key_file).expanduser().read_text().strip()
    elif a.key_env:
        key = os.environ.get(a.key_env, "")
        if not key:
            raise SystemExit(f"environment variable {a.key_env} is empty")
    headers = {}
    for h in a.header:
        if ":" not in h:
            raise SystemExit(f"--header {h!r}: use 'Name: value'")
        n, v = h.split(":", 1)
        headers[n.strip()] = v.strip()
    if a.long_context and not a.context_window:
        raise SystemExit("--long-context needs --context-window (the advertised window, in tokens)")
    opts = Options(base_url=a.base_url, api=a.api, model=a.model, key=key, auth=a.auth, headers=headers,
                   prices_path=a.prices, cost_header=a.cost_header, cost_field=a.cost_field,
                   disclosure_headers=a.disclosure_header, probes=a.probes,
                   probe_rounds=a.probe_rounds, reference=a.reference, long_context=a.long_context,
                   context_window=a.context_window, max_output=a.max_output, samples=a.samples,
                   concurrency=a.concurrency, tokenizer_family=a.tokenizer_family,
                   tokenizer_samples=a.tokenizer_samples, corpus_dir=a.corpus_dir,
                   checks=a.checks.split(",") if a.checks else None, skip=a.skip.split(",") if a.skip else [],
                   max_spend=a.max_spend, max_tokens_total=a.max_tokens_total, out=a.out, run_by=a.run_by,
                   timeout=a.timeout, seed=a.seed)
    rep, out_dir = run_conform(opts)
    g = rep["grade"]
    print(f"\n  GRADE: {g['grade']}" + (f"  ({'; '.join(g['failed'] or g['incomplete'] or g['warned'])})"
                                       if g["grade"] != "Conforming" else ""))
    s = rep["spend"]
    if s["estimated_usd_at_list"] is not None:
        print(f"  spend: ${s['estimated_usd_at_list']:.4f} at list" + (f", ${s['endpoint_reported_usd']:.4f} reported"
                                                                      if s["endpoint_reported_usd"] is not None else ""))
    print(f"  report: {out_dir / 'report.md'} · {out_dir / 'report.json'} (content sha256 {rep['content_sha256'][:16]}…)")
    print("  note: check the endpoint's terms before publishing results about it")
    return {"Conforming": 0, "Conforming with notes": 0, "Nonconforming": 2, "Incomplete": 3}[g["grade"]]


if __name__ == "__main__":
    sys.exit(main())
