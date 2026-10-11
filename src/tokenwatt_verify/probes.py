"""Probe banks: how tokenwatt-verify gets its test prompts.

A probe is a dict with: kind, system, user, max_tokens, and grade(text) -> (ok: bool, detail: str).
A bank is any Python module (or file) that defines

    SUITE = [(kind, make_probe(rng) -> dict), ...]

Every probe should be generated fresh from a random seed and graded by program, so no fixed question can
be memorized, and should look like an ordinary request.

**The sample bank below is public, which means an endpoint operator can recognize it and route it to a
genuine model.** It's here so the tool works out of the box and shows the format. For real checks, write
your own bank (keep it private and change it often) and pass it with `--probes path/to/bank.py`.
"""

from __future__ import annotations

import importlib.util
import random
import re
import string
from pathlib import Path
from types import ModuleType

SYSTEMS = ["You are a helpful assistant.", "You are a concise coding assistant.", "Answer precisely."]
ANSWER_RULE = "\n\nPut only the final answer on the last line, in the form `ANSWER: <value>`."


def final_answer(text: str) -> str:
    m = re.findall(r"ANSWER:\s*`?([^`\n]*)`?", text)
    return m[-1].strip().strip(".").strip() if m else ""


# --------------------------------------------------------------------------- sample probes (public)
def sample_arithmetic(rng: random.Random) -> dict:
    a, b, c = rng.randint(100, 999), rng.randint(10, 99), rng.randint(2, 9)
    expected = a * b - c
    user = f"Compute {a} × {b} − {c}." + ANSWER_RULE

    def grade(text: str):
        got = re.sub(r"[^\d-]", "", final_answer(text))
        return got == str(expected), f"expected {expected}, got {got!r}"
    return {"kind": "sample_arithmetic", "system": rng.choice(SYSTEMS), "user": user, "max_tokens": 600, "grade": grade}


def sample_reverse(rng: random.Random) -> dict:
    s = "".join(rng.choices(string.ascii_lowercase + string.digits, k=rng.randint(14, 20)))
    user = f"Reverse this string exactly: {s}" + ANSWER_RULE

    def grade(text: str):
        got = final_answer(text).replace(" ", "").strip("`")
        return got == s[::-1], f"expected {s[::-1]!r}, got {got!r}"
    return {"kind": "sample_reverse", "system": rng.choice(SYSTEMS), "user": user, "max_tokens": 600, "grade": grade}


def sample_lookup(rng: random.Random, n: int = 300) -> dict:
    keys = list(dict.fromkeys(f"k{rng.randint(1000, 9999)}" for _ in range(n)))
    vals = {k: rng.randint(10, 99999) for k in keys}
    target = rng.choice(keys)
    user = ("Here is a table:\n" + "\n".join(f"{k}: {v}" for k, v in vals.items())
            + f"\n\nWhat is the value for {target}?" + ANSWER_RULE)

    def grade(text: str):
        got = re.sub(r"\D", "", final_answer(text))
        return got == str(vals[target]), f"expected {vals[target]}, got {got!r}"
    return {"kind": "sample_lookup", "system": rng.choice(SYSTEMS), "user": user, "max_tokens": 300, "grade": grade}


def sample_identity(rng: random.Random) -> dict:
    """Recorded, not trusted: any model can be told to claim any name."""
    def grade(text: str):
        return True, text.strip()[:160]
    return {"kind": "identity", "system": rng.choice(SYSTEMS), "user": "Which model are you? One line.",
            "max_tokens": 100, "grade": grade, "informational": True}


SUITE = [("sample_arithmetic", sample_arithmetic), ("sample_reverse", sample_reverse),
         ("sample_lookup", sample_lookup), ("identity", sample_identity)]


# --------------------------------------------------------------------------- loading
def load_bank(path: str | None) -> ModuleType | None:
    if not path:
        return None
    p = Path(path).expanduser()
    spec = importlib.util.spec_from_file_location("tokenwatt_verify_bank", p)
    if not spec or not spec.loader:
        raise SystemExit(f"can't load probe bank {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "SUITE"):
        raise SystemExit(f"{p} must define SUITE = [(kind, make_probe), ...]")
    return mod


def generate(seed: int, kinds: list[str] | None = None, bank: ModuleType | None = None) -> list[dict]:
    suite = bank.SUITE if bank is not None else SUITE
    rng = random.Random(seed)
    out = []
    for kind, make in suite:
        if kinds and kind not in kinds:
            continue
        p = dict(make(random.Random(rng.random())))
        p.setdefault("kind", kind)
        p["seed"] = seed
        out.append(p)
    return out
