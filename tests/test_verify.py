import random

from tokenwatt_verify.client import Endpoint
from tokenwatt_verify.probes import SUITE, generate, load_bank
from tokenwatt_verify.verdict import evaluate, fisher_one_sided


def rec(ok, ref_ok, ratio=1.0, ttft=1.0, ref_ttft=1.0, err="", refused=False):
    return {"kind": "x", "ok": ok, "ref_ok": ref_ok, "refused": refused, "ref_refused": False,
            "input_token_ratio": ratio,
            "reply": {"error": err, "ttft_s": ttft}, "ref_reply": {"error": "", "ttft_s": ref_ttft}}


def test_genuine_endpoint_is_consistent():
    v = evaluate([rec(True, True) for _ in range(25)])
    assert v["overall"] == "consistent"


def test_too_few_requests_is_inconclusive():
    assert evaluate([rec(True, True) for _ in range(5)])["overall"] == "inconclusive"


def test_capability_gap_is_inconsistent():
    rows = [rec(i % 3 == 0, True) for i in range(30)]          # endpoint right 1 in 3, reference always
    v = evaluate(rows)
    assert v["overall"] == "inconsistent"
    assert next(f for f in v["findings"] if f["signal"] == "capability")["level"] == "FAIL"


def test_tokenizer_mismatch_is_inconsistent():
    rows = [rec(True, True, ratio=1.18) for _ in range(25)]   # another model's tokenizer
    assert evaluate(rows)["overall"] == "inconsistent"


def test_errors_and_refusals_are_excluded_from_capability():
    rows = [rec(True, True) for _ in range(20)] + [rec(False, True, err="HTTP 400") for _ in range(10)]
    v = evaluate(rows)
    assert v["overall"] == "consistent" and v["answered_pairs"] == 20
    assert next(f for f in v["findings"] if f["signal"] == "errors")["level"] == "WARN"


def test_slow_relay_is_flagged_but_not_a_substitution():
    v = evaluate([rec(True, True, ttft=6.6, ref_ttft=1.1) for _ in range(25)])
    assert v["overall"] == "consistent"
    assert next(f for f in v["findings"] if f["signal"] == "latency")["level"] == "WARN"


def test_fisher_bounds():
    assert fisher_one_sided(10, 0, 10, 0) == 1.0
    assert fisher_one_sided(0, 20, 20, 0) < 1e-6


def test_sample_probes_grade_their_own_answers():
    for kind, make in SUITE:
        p = make(random.Random(1))
        if p.get("informational"):
            continue
        expected = p["grade"]("")[1].split("expected ")[1].split(", got")[0].strip("'")
        assert p["grade"](f"ANSWER: {expected}")[0], kind


def test_custom_bank(tmp_path):
    bank = tmp_path / "bank.py"
    bank.write_text("SUITE = [('mine', lambda rng: {'system': 's', 'user': 'u', 'max_tokens': 5, "
                    "'grade': lambda t: (t == 'ok', '')})]\n")
    probes = generate(3, bank=load_bank(str(bank)))
    assert [p["kind"] for p in probes] == ["mine"]


def test_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("EP_URL", "https://api.example.com/v1/")
    monkeypatch.setenv("EP_KEY", "k")
    monkeypatch.setenv("EP_MODEL", "claude-x")
    monkeypatch.setenv("EP_API", "anthropic")
    ep = Endpoint.parse("env:EP")
    assert (ep.url, ep.model, ep.api) == ("https://api.example.com/v1", "claude-x", "anthropic")
