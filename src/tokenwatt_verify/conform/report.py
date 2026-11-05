"""The conformance report: grade (draft §8), JSON with a content hash, Markdown summary, and verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .. import __version__
from .runner import FAIL, INCOMPLETE, NA, PASS, WARN
from .thresholds import THRESHOLDS, VERSION
from .transport import canonical, sha256_text, verify_evidence

STANDARD = {"name": "TokenWatt Inference Endpoint Conformance", "version": "v0 (draft, 2026-11-05)",
            "note": "draft standard: checks and thresholds will change as more endpoints are measured"}
TERMS_NOTE = ("Check the endpoint's terms of service before publishing results about it: some providers bar "
              "benchmarking or publishing performance results without written permission.")
DISPLAY = ["I1", "I2", "I3", "I4", "I5", "I6", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "P1", "P2", "P3", "P4",
           "S1", "S2", "S3"]


def grade(results: dict) -> dict:
    graded = {cid: r for cid, r in results.items() if cid[0] in "IBP"}
    by = {s: sorted(c for c, r in graded.items() if r["status"] == s) for s in (PASS, WARN, FAIL, INCOMPLETE, NA)}
    not_run = sorted(c for c, r in graded.items() if r["status"] == "not_checked")
    if by[FAIL]:
        g, why = "Nonconforming", [f"{c} failed: {graded[c]['summary']}" for c in by[FAIL]]
    elif by[INCOMPLETE]:
        g, why = "Incomplete", [f"{c} could not be completed: {graded[c]['summary']}" for c in by[INCOMPLETE]]
    elif by[WARN]:
        g, why = "Conforming with notes", [f"{c}: {graded[c]['summary']}" for c in by[WARN]]
    else:
        g, why = "Conforming", []
    return {"grade": g, "reasons": why, "passed": by[PASS], "warned": by[WARN], "failed": by[FAIL],
            "incomplete": by[INCOMPLETE], "not_applicable": by[NA], "not_checked": not_run,
            "service": "not graded: S1 is reported, not graded (draft §8); S2 availability and S3 accuracy need "
                       "continuous probing and accuracy sets, which a single reading does not provide",
            "scope": "the grade covers the identity, behaviour and billing checks run in this reading; checks marked "
                     "not_applicable or not_checked are not covered"}


def build_report(ctx, started, finished) -> dict:
    o = ctx.opts
    results = {cid: ctx.results[cid] for cid in DISPLAY if cid in ctx.results}
    ev = ctx.session.evidence_path
    ev_hash = hashlib.sha256(ev.read_bytes()).hexdigest() if ev and ev.exists() else None
    from .tokref import load_reference
    from .corpus import load_manifest
    tref = o.tokenizer_reference or load_reference()
    man = o.corpus_manifest or load_manifest()
    rep = {
        "schema": "tokenwatt-verify/conform-report/v1",
        "tool": {"name": "tokenwatt-verify", "version": __version__},
        "standard": STANDARD,
        "thresholds": {"version": VERSION, "values": THRESHOLDS},
        "reference_data": {"tokenizer_reference": tref.get("version"), "tokenizer_reference_measured": tref.get("measured"),
                           "corpus": man.get("version")},
        "run": {"started_utc": started.isoformat(timespec="seconds"), "finished_utc": finished.isoformat(timespec="seconds"),
                "run_by": o.run_by, "seed": ctx.seed, "nonce": ctx.nonce,
                "options": {"long_context": o.long_context, "probe_rounds": o.probe_rounds, "probes": "custom bank" if o.probes
                            else "public sample bank", "reference": bool(o.reference), "samples": o.samples,
                            "concurrency": o.concurrency, "tokenizer_samples": o.tokenizer_samples,
                            "max_spend_usd": o.max_spend, "checks": o.checks, "skip": o.skip,
                            "extra_headers": sorted(o.headers)}},
        "endpoint": {"base_url": o.base_url, "api": o.api, "model": o.model,
                     "listing": {"context_window": o.context_window, "max_output": o.max_output,
                                 "prices_usd_per_1m": ctx.prices, "prices_source": ctx.price_meta or None}},
        "grade": grade(results),
        "checks": [results[c] for c in DISPLAY if c in results],
        "spend": {"estimated_usd_at_list": round(ctx.spend_est, 6) if ctx.prices else None,
                  "endpoint_reported_usd": round(ctx.spend_observed, 6) if (o.cost_header or o.cost_field) else None,
                  "tokens": ctx.tokens_used, "requests": len(ctx.session.records)},
        "evidence": {"file": "evidence.jsonl", "sha256": ev_hash, "exchanges": len(ctx.session.records),
                     "note": "one JSON line per HTTP attempt (request and response, keys removed), each with its own "
                             "sha256; check evidence lists those hashes"},
        "terms_note": TERMS_NOTE,
    }
    rep["content_sha256"] = sha256_text(canonical(rep))
    return rep


def write_outputs(rep: dict, out_dir: Path) -> None:
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(markdown(rep), encoding="utf-8")


ICON = {PASS: "pass", WARN: "warn", FAIL: "FAIL", NA: "n/a", INCOMPLETE: "incomplete", "reported": "reported",
        "not_checked": "not checked"}


def markdown(rep: dict) -> str:
    e, g = rep["endpoint"], rep["grade"]
    lines = [f"# Endpoint conformance reading: {e['model']}", "",
             f"- Endpoint: `{e['base_url']}` ({e['api']} format), model `{e['model']}`",
             f"- Standard: {rep['standard']['name']} {rep['standard']['version']}; thresholds `{rep['thresholds']['version']}`",
             f"- Run: {rep['run']['started_utc']} → {rep['run']['finished_utc']}, by {rep['run']['run_by']}, "
             f"tokenwatt-verify {rep['tool']['version']}",
             f"- **Grade: {g['grade']}**" + (f" ({'; '.join(g['failed'] or g['incomplete'] or g['warned'])})"
                                              if g["grade"] != "Conforming" else ""), "",
             "| Check | | Status | Finding |", "|---|---|---|---|"]
    for c in rep["checks"]:
        lines.append(f"| {c['id']} | {c['name']} | {ICON.get(c['status'], c['status'])} | "
                     f"{c['summary'].replace('|', '/')[:400]} |")
    s = rep["spend"]
    lines += ["", f"Grade scope: {g['scope']}. Service: {g['service']}.", ""]
    spend = []
    if s["estimated_usd_at_list"] is not None:
        spend.append(f"${s['estimated_usd_at_list']:.4f} at the listed prices")
    if s["endpoint_reported_usd"] is not None:
        spend.append(f"${s['endpoint_reported_usd']:.4f} reported by the endpoint")
    lines.append(f"Spend: {', '.join(spend) or 'not priced (no --prices)'}; {s['requests']} HTTP requests, "
                 f"{s['tokens']:,} tokens.")
    lines += ["", f"Evidence: `{rep['evidence']['file']}`, {rep['evidence']['exchanges']} exchanges, sha256 "
              f"`{rep['evidence']['sha256']}`. Report content sha256 `{rep['content_sha256']}` "
              "(check with `tokenwatt-verify verify-report report.json`).", "", f"_{rep['terms_note']}_", ""]
    return "\n".join(lines)


def verify_report(path: Path) -> tuple[bool, list[str]]:
    """Recompute the report's content hash and, if the evidence file sits next to it, every exchange hash."""
    path = Path(path)
    rep = json.loads(path.read_text(encoding="utf-8"))
    msgs, ok = [], True
    claimed = rep.pop("content_sha256", None)
    actual = sha256_text(canonical(rep))
    if claimed != actual:
        ok = False
        msgs.append(f"content hash mismatch: report says {claimed}, content is {actual}")
    else:
        msgs.append(f"content hash ok ({actual})")
    ev = path.parent / rep["evidence"]["file"]
    if ev.exists():
        file_hash = hashlib.sha256(ev.read_bytes()).hexdigest()
        if file_hash != rep["evidence"]["sha256"]:
            ok = False
            msgs.append(f"evidence file hash mismatch: {file_hash}")
        n, bad = verify_evidence(ev)
        if bad:
            ok = False
            msgs.append(f"{len(bad)} of {n} exchanges fail their own hash (seq {bad[:10]})")
        else:
            msgs.append(f"evidence ok: {n} exchanges, every hash matches")
        hashes = {json.loads(line)["sha256"] for line in ev.read_text(encoding="utf-8").splitlines() if line.strip()}
        missing = [h for c in rep["checks"] for h in c.get("evidence", []) if h not in hashes]
        if missing:
            ok = False
            msgs.append(f"{len(missing)} evidence hashes cited by checks are not in the evidence file")
    else:
        msgs.append("evidence file not found next to the report; only the content hash was checked")
    return ok, msgs
