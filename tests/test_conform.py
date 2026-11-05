import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_endpoint import PRICES, State, serve  # noqa: E402

from tokenwatt_verify.conform import corpus, pricing, tokref  # noqa: E402
from tokenwatt_verify.conform.checks.common import alter_signature, same_model, spread  # noqa: E402
from tokenwatt_verify.conform.protocol import Protocol, norm_usage_anthropic, norm_usage_openai  # noqa: E402
from tokenwatt_verify.conform.report import grade, verify_report  # noqa: E402
from tokenwatt_verify.conform.runner import Options, run  # noqa: E402
from tokenwatt_verify.conform.schema import validate  # noqa: E402
from tokenwatt_verify.conform.transport import Response  # noqa: E402

# ---- a small synthetic corpus and reference (no network in tests) -------------------------------------------
_rng = random.Random(5)
TEXTS = {f"s{i}": "".join(_rng.choice("abcdefgh ijk\n(){}=;") for _ in range(400 + 120 * i)) for i in range(6)}
GPT = {k: len(v) // 4 for k, v in TEXTS.items()}
REF = {"version": "test", "measured": "test", "prefix": "[calibration {nonce}]\n",
       "suffix": "\n\nReply with the single word OK and nothing else.",
       "families": {
           "claude-5.5": {"label": "Claude 5.5", "measured_models": ["claude-opus-5.5", "claude-sonnet-5.5"],
                          "match": [r"^claude-(opus|sonnet|haiku)-5[-.]5(-\d{8})?$"],
                          "counts": {k: int(v * 1.6) for k, v in GPT.items()}},
           "gpt-5.x-o200k": {"label": "GPT-5.x", "measured_models": ["gpt-5.5"], "match": [r"^gpt-5[-.]5$"], "counts": GPT}}}
MANIFEST = {"version": "test", "items": [{"id": k} for k in TEXTS]}


@pytest.fixture
def mock(tmp_path):
    servers = []

    def start(faults=(), model="claude-sonnet-5-5"):
        family = "gpt-5.x-o200k" if model.startswith("gpt") else "claude-5.5"   # the mock bills its own model's counts
        st = State(faults, model=model, key="test-key", reference=REF, texts=TEXTS, family=family)
        srv, url = serve(st)
        servers.append(srv)
        return url
    yield start
    for s in servers:
        s.shutdown()


def reading(url, tmp_path, api="anthropic", model="claude-sonnet-5-5", **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"source": "mock", "models": {model: PRICES}}))
    opts = Options(base_url=url if api == "anthropic" else url + "/v1", api=api, model=model, key="test-key",
                   prices_path=str(prices), cost_header="X-Mock-Cost", samples=3, out=str(tmp_path / "out"), seed=1,
                   tokenizer_reference=REF, corpus_manifest=MANIFEST, corpus_texts=TEXTS, **kw)
    rep, out_dir = run(opts, log=lambda *a: None)
    return rep, {c["id"]: c for c in rep["checks"]}, out_dir


# ---- integration: the mock ------------------------------------------------------------------------------------
def test_genuine_mock_conforms(mock, tmp_path):
    rep, c, out_dir = reading(mock(), tmp_path)
    assert rep["grade"]["grade"] == "Conforming", rep["grade"]
    for cid in ("I1", "I2", "I3", "B1", "B2", "B3", "B6", "B7", "P1", "P2", "P3"):
        assert c[cid]["status"] == "pass", (cid, c[cid]["summary"])
    assert c["I3"]["details"]["intact_block_reached_model"] is True
    assert c["S1"]["status"] == "reported"
    assert c["I4"]["status"] == "not_applicable"            # public sample bank: no reference rates
    assert c["B5"]["status"] == "not_applicable"            # long context not requested
    ok, msgs = verify_report(out_dir / "report.json")
    assert ok, msgs


def test_gpt_counts_behind_a_claude_name_fail_i2(mock, tmp_path):
    rep, c, _ = reading(mock(["gpt_tokens"]), tmp_path)
    assert c["I2"]["status"] == "fail" and "GPT-5.x" in c["I2"]["summary"]
    assert c["P1"]["status"] == "fail"
    assert rep["grade"]["grade"] == "Nonconforming"


def test_forged_signatures_fail_i3(mock, tmp_path):
    rep, c, _ = reading(mock(["forge_signature"]), tmp_path, checks=["I1", "I3", "B1"])
    assert c["I3"]["status"] == "fail" and "altered signature was accepted" in c["I3"]["summary"]
    assert rep["grade"]["grade"] == "Nonconforming"


def test_router_that_drops_unknown_thinking_is_incomplete_not_fail(mock, tmp_path):
    rep, c, _ = reading(mock(["drop_unknown_thinking"]), tmp_path, checks=["I1", "I3"])
    assert c["I3"]["status"] == "incomplete" and "not readable" in c["I3"]["summary"]
    assert c["I3"]["details"]["altered_signature_dropped"] is True
    assert rep["grade"]["grade"] == "Incomplete"


def test_undisclosed_adjustments_fail(mock, tmp_path):
    for api, model in (("anthropic", "claude-sonnet-5-5"), ("openai", "gpt-5.5")):
        rep, c, _ = reading(mock(["adjust_tool_choice", "clamp_max_tokens"], model=model), tmp_path / api, api=api,
                            model=model, checks=["B1", "B7"])
        assert c["B1"]["status"] == "fail" and "silent text answer" in c["B1"]["summary"]
        assert c["B7"]["status"] == "fail" and "invalid max_tokens: accepted with HTTP 200" in c["B7"]["summary"]


def test_disclosed_adjustments_are_documented_deviations(mock, tmp_path):
    for api, model in (("anthropic", "claude-sonnet-5-5"), ("openai", "gpt-5.5")):
        rep, c, _ = reading(mock(["adjust_tool_choice", "clamp_max_tokens", "disclose"], model=model), tmp_path / api,
                            api=api, model=model, checks=["B1", "B7"], disclosure_headers=["X-Mock-Adjusted"])
        assert c["B1"]["status"] == "warn" and "X-Mock-Adjusted: tool_choice=auto" in c["B1"]["summary"]
        assert c["B7"]["status"] == "warn" and "X-Mock-Adjusted: max_tokens=1" in c["B7"]["summary"]
        assert rep["grade"]["grade"] == "Conforming with notes"


def test_disclosure_counts_only_when_the_header_is_configured_and_names_the_parameter(mock, tmp_path):
    url = mock(["adjust_tool_choice", "clamp_max_tokens", "disclose"])
    _, c, _ = reading(url, tmp_path / "a", checks=["B1", "B7"])                  # header sent, option not given
    assert c["B1"]["status"] == "fail" and c["B7"]["status"] == "fail"
    _, c, _ = reading(url, tmp_path / "b", checks=["B1"], disclosure_headers=["X-Other-Adjusted"])
    assert c["B1"]["status"] == "fail"


def test_cut_reported_as_stop_fails_b2(mock, tmp_path):
    for api in ("anthropic", "openai"):
        rep, c, _ = reading(mock(["cut_as_stop"]), tmp_path / api, api=api, checks=["B2"])
        assert c["B2"]["status"] == "fail", c["B2"]["summary"]
        assert rep["grade"]["grade"] == "Nonconforming"


def test_errors_as_200_fail_b7(mock, tmp_path):
    rep, c, _ = reading(mock(["errors_as_200"]), tmp_path, checks=["I1", "B7"])
    assert c["B7"]["status"] == "fail" and "HTTP 200" in c["B7"]["summary"]


def test_openai_mock(mock, tmp_path):
    rep, c, _ = reading(mock(model="gpt-5.5"), tmp_path, api="openai", model="gpt-5.5")
    assert c["I3"]["status"] == "not_applicable"
    for cid in ("I1", "I2", "B1", "B2", "B6", "B7", "P1", "P2", "P3"):
        assert c[cid]["status"] == "pass", (cid, c[cid]["summary"])
    assert rep["grade"]["grade"] == "Conforming"


def test_long_context_opt_in(mock, tmp_path):
    rep, c, _ = reading(mock(), tmp_path, checks=["B5"], long_context=True, context_window=20000)
    assert c["B5"]["status"] == "pass", c["B5"]["summary"]
    assert c["B5"]["details"]["long"]["estimated_prompt_tokens"] > 15000


def test_tampered_report_is_detected(mock, tmp_path):
    _, _, out_dir = reading(mock(), tmp_path, checks=["I1"])
    p = out_dir / "report.json"
    d = json.loads(p.read_text())
    d["grade"]["grade"] = "Conforming (edited)"
    p.write_text(json.dumps(d))
    assert not verify_report(p)[0]


def test_budget_cap_stops_requests(mock, tmp_path):
    rep, c, _ = reading(mock(), tmp_path, checks=["I2", "S1"], max_spend=0.0)
    assert c["I2"]["status"] == "incomplete"
    assert rep["spend"]["requests"] == 0


def test_key_never_written(mock, tmp_path):
    _, _, out_dir = reading(mock(), tmp_path, checks=["I1", "B7"])
    for f in out_dir.iterdir():
        assert "test-key" not in f.read_text(), f.name


# ---- units ----------------------------------------------------------------------------------------------------
def test_family_resolution():
    ref = tokref.load_reference()
    assert tokref.resolve_family(ref, "claude-sonnet-5-5") == "claude-5.5"
    assert tokref.resolve_family(ref, "anthropic/claude-haiku-5.5") == "claude-5.5"
    assert tokref.resolve_family(ref, "claude-haiku-4-5-20251001") == "claude-previous"
    assert tokref.resolve_family(ref, "deepseek-chat") == "deepseek-v4"
    assert tokref.resolve_family(ref, "z-ai/glm-5.3") == "glm-5.3"
    assert tokref.resolve_family(ref, "some-unknown-model") is None


def test_packaged_reference_matches_manifest():
    ref, man = tokref.load_reference(), corpus.load_manifest()
    ids = {i["id"] for i in man["items"]}
    assert len(ids) == 16
    for fam in ref["families"].values():
        assert set(fam["counts"]) == ids
    # families that share a tokenizer are one entry; different families differ on most samples
    a, b = ref["families"]["glm-5.3"]["counts"], ref["families"]["gpt-5.x-o200k"]["counts"]
    assert sum(a[s] != b[s] for s in ids) >= 8


def test_compare_picks_the_measured_tokenizer():
    cmp = tokref.compare(REF, dict(GPT))
    assert next(iter(cmp)) == "gpt-5.x-o200k" and cmp["gpt-5.x-o200k"]["ratio"] == 1.0


def test_usage_normalization():
    a = norm_usage_anthropic({"input_tokens": 10, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 5,
                              "output_tokens": 3, "output_tokens_details": {"thinking_tokens": 2}})
    assert (a["prompt_total"], a["input_uncached"], a["reasoning"]) == (105, 10, 2)
    o = norm_usage_openai({"prompt_tokens": 100, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 80}})
    assert (o["prompt_total"], o["input_uncached"], o["cache_read"]) == (100, 20, 80)
    x = norm_usage_openai({"prompt_tokens": 10, "completion_tokens": 4, "cache_read_input_tokens": 90})
    assert x["prompt_total"] == 100     # Anthropic semantics behind an OpenAI shape


def test_prices_and_tiers(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"models": {"claude-sonnet-5.5": {"input": 2, "output": 10, "cache_read": 0.2,
                                                              "tiers": [{"above_prompt_tokens": 200000, "input": 4}]}}}))
    p, _ = pricing.load_prices(str(f), "claude-sonnet-5.5")
    u = {"prompt_total": 1000, "input_uncached": 900, "cache_read": 100, "cache_write": 0, "output": 50}
    assert abs(pricing.expected_cost(p, u) - (900 * 2 + 100 * 0.2 + 50 * 10) / 1e6) < 1e-12
    big = {**u, "prompt_total": 300000, "input_uncached": 300000, "cache_read": 0}
    assert abs(pricing.expected_cost(p, big) - (300000 * 4 + 50 * 10) / 1e6) < 1e-12
    assert pricing.expected_cost(p, {**u, "cache_write": 10}) is None   # no cache_write price listed


def test_schema_validator():
    s = {"type": "object", "properties": {"city": {"type": "string"}, "n": {"type": "integer", "minimum": 1}},
         "required": ["city"], "additionalProperties": False}
    assert validate({"city": "Oslo"}, s) == []
    assert validate({"n": 0}, s) and validate({"city": 1}, s) and validate({"city": "x", "z": 1}, s)


def test_signature_alteration_keeps_length():
    sig = "EqQBCkYIBxgCKkB" + "A" * 200 + "=="
    alt = alter_signature(sig)
    assert alt != sig and len(alt) == len(sig) and alt.endswith("==")


def test_model_name_matching():
    assert same_model("claude-sonnet-5-5", "claude-sonnet-5-5-20261001")
    assert same_model("anthropic/claude-sonnet-5.5", "claude-sonnet-5-5")
    assert not same_model("claude-opus-5-5", "gpt-4o")


def test_spread_keeps_type_mix():
    assert spread(list("abcdefgh"), 4) == ["a", "c", "e", "g"]


def test_disclosure_header_parsing():
    from types import SimpleNamespace

    from tokenwatt_verify.conform.runner import Ctx
    ctx = Ctx.__new__(Ctx)
    ctx.opts = SimpleNamespace(disclosure_headers=["X-Router-Adjusted"])
    res = SimpleNamespace(headers={"x-router-adjusted": "tool_choice=auto, max_tokens=8192"})
    assert ctx.documented(res, "max_tokens") == "X-Router-Adjusted: tool_choice=auto, max_tokens=8192"
    assert ctx.documented(res, "TOOL_CHOICE")
    assert ctx.documented(res, "stop") is None
    assert ctx.documented(SimpleNamespace(headers={"x-router-adjusted": "tool_choice"}), "tool_choice") is None  # no =
    assert ctx.documented(SimpleNamespace(headers={}), "tool_choice") is None


def test_grade_table():
    def r(s):
        return {"status": s, "summary": "x"}
    assert grade({"I1": r("pass"), "B1": r("warn")})["grade"] == "Conforming with notes"
    assert grade({"I1": r("pass"), "B2": r("fail")})["grade"] == "Nonconforming"
    assert grade({"I1": r("pass"), "P3": r("incomplete")})["grade"] == "Incomplete"
    assert grade({"I1": r("pass"), "S1": r("reported"), "I5": r("not_checked")})["grade"] == "Conforming"


def test_anthropic_stream_reassembly():
    ev = [{"type": "message_start", "message": {"id": "msg_1", "model": "m", "usage": {"input_tokens": 5}}},
          {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
          {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "abc"}},
          {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "f", "input": {}}},
          {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"a": '}},
          {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "1}"}},
          {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
          {"type": "message_stop"}]
    resp = Response(status=200, events=[{"t": i / 10, "event": e["type"], "data": e} for i, e in enumerate(ev)])
    r = Protocol("anthropic", "m").parse(resp)
    assert r.final_event and r.usage_in_stream and r.finish == "tool_use"
    assert r.tool_calls[0]["input"] == {"a": 1} and r.thinking[0]["signature"] == "abc"
    assert r.content[0] == {"type": "thinking", "thinking": "", "signature": "abc"}
