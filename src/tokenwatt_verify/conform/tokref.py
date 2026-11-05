"""Tokenizer reference data (data/tokenizer_reference.json): which tokenizer family a model name belongs to,
and how a set of measured per-sample counts compares with every family."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from . import data_path


def load_reference(path: Path | None = None) -> dict:
    return json.loads((path or data_path("tokenizer_reference.json")).read_text(encoding="utf-8"))


def normalize_model(name: str) -> str:
    """'anthropic/claude-sonnet-5.5' -> 'claude-sonnet-5.5'; Bedrock / Vertex decorations removed."""
    n = name.strip().lower().replace("_", "-").replace(" ", "-")
    n = n.rsplit("/", 1)[-1]
    n = re.sub(r"^(us|eu|apac|global)\.", "", n)
    n = re.sub(r"^anthropic\.", "", n)
    n = re.sub(r"@.*$", "", n)
    n = re.sub(r"-v\d+(:\d+)?$", "", n)
    return n


def resolve_family(ref: dict, model: str, override: str | None = None) -> str | None:
    if override:
        if override not in ref["families"]:
            raise SystemExit(f"unknown tokenizer family {override!r}; known: {', '.join(ref['families'])}")
        return override
    n = normalize_model(model)
    for fid, fam in ref["families"].items():
        if any(re.match(p, n) for p in fam.get("match", [])):
            return fid
    return None


def compare(ref: dict, measured: dict) -> dict:
    """Per family: totals over the samples measured, ratio measured/reference, mean |relative error| per sample,
    median |difference| in tokens. Sorted best match first."""
    out = {}
    for fid, fam in ref["families"].items():
        common = [s for s in measured if s in fam["counts"]]
        if not common:
            continue
        diffs = [measured[s] - fam["counts"][s] for s in common]
        rel = [abs(d) / max(1, fam["counts"][s]) for d, s in zip(diffs, common)]
        tot_ref = sum(fam["counts"][s] for s in common)
        tot = sum(measured[s] for s in common)
        out[fid] = {"label": fam["label"], "samples": len(common), "reference_total": tot_ref, "measured_total": tot,
                    "ratio": round(tot / tot_ref, 5) if tot_ref else None,
                    "mean_abs_rel_error": round(statistics.mean(rel), 5),
                    "median_abs_diff_tokens": statistics.median(abs(d) for d in diffs),
                    "max_abs_diff_tokens": max(abs(d) for d in diffs)}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["mean_abs_rel_error"]))
