"""Listed prices, supplied by the user (`--prices prices.json`). tokenwatt-verify ships no prices: a reading
is checked against the listing you saw, on the day you saw it.

    {
      "source": "https://example.com/pricing, read 2026-11-05",
      "models": {
        "my-model": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5,
                     "tiers": [{"above_prompt_tokens": 200000, "input": 4.0, "output": 15.0}]}
      }
    }

Prices are USD per 1M tokens. `cache_read` / `cache_write` are needed only if the endpoint reports cached
tokens; a tier applies when the request's prompt (cache reads and writes included) is above its threshold.
A file with top-level "input"/"output" is taken as the price of the model under test.
"""

from __future__ import annotations

import json
from pathlib import Path

from .tokref import normalize_model


def load_prices(path: str | None, model: str) -> tuple[dict | None, dict]:
    """(price entry for `model` or None, file metadata)."""
    if not path:
        return None, {}
    d = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    meta = {k: v for k, v in d.items() if k not in ("models",) and not isinstance(v, (int, float))}
    if "input" in d and "output" in d:
        return _check({k: d[k] for k in ("input", "output", "cache_read", "cache_write", "tiers") if k in d}), meta
    models = d.get("models") or {}
    if model in models:
        return _check(models[model]), meta
    n = normalize_model(model)
    for k, v in models.items():
        if normalize_model(k) == n:
            return _check(v), meta
    return None, meta


def _check(p: dict) -> dict:
    for k in ("input", "output"):
        if not isinstance(p.get(k), (int, float)):
            raise SystemExit(f"prices: '{k}' (USD per 1M tokens) is required")
    return p


def tier_for(p: dict, prompt_total: int) -> dict:
    best, best_at = p, -1
    for t in p.get("tiers") or []:
        at = t.get("above_prompt_tokens", 0)
        if prompt_total > at and at > best_at:
            best, best_at = {**p, **t}, at
    return best


def expected_cost(p: dict | None, usage: dict | None) -> float | None:
    """Tokens x listed price in USD, or None when a needed price is missing or usage wasn't reported."""
    if not p or not usage or usage.get("output") is None:
        return None
    t = tier_for(p, usage.get("prompt_total") or 0)
    read, write = usage.get("cache_read") or 0, usage.get("cache_write") or 0
    if (read and "cache_read" not in t) or (write and "cache_write" not in t):
        return None
    c = (usage.get("input_uncached") or 0) * t["input"] + read * t.get("cache_read", 0) + \
        write * t.get("cache_write", 0) + (usage.get("output") or 0) * t["output"]
    return c / 1e6


def rough_cost(p: dict | None, prompt_tokens: int, output_tokens: int) -> float | None:
    if not p:
        return None
    t = tier_for(p, prompt_tokens)
    return (prompt_tokens * t["input"] + output_tokens * t["output"]) / 1e6
