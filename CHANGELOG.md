# Changelog

## 0.2.0 (2026-11-10)

- New `conform` command: one reading of an endpoint against the TokenWatt Inference Endpoint Conformance
  draft v0. Checks I1 response shape, I2 tokenizer fingerprint, I3 Anthropic thinking-signature attestation,
  I4 capability fingerprint (probe banks with optional `REFERENCE` rates, or a paired `--reference`),
  B1 tool calls, B2 truncation, B3 stop sequences vs hidden reasoning, B5 context window (long context
  opt-in), B6 streaming parity, B7 error semantics, P1 billed tokens, P2 cost vs listed price, P3 cache,
  P4 failed requests, S1 latency (reported, not graded). Each is pass / warn / fail / not applicable /
  incomplete with evidence; grade per the draft's §8.
- Evidence: every HTTP attempt kept in `evidence.jsonl` (keys removed) with its own sha256; `report.json`
  carries a content hash; `verify-report` re-checks both. `report.md` summary.
- Versioned thresholds (`conform/thresholds.py`), written into every report.
- Tokenizer reference data for 9 tokenizer families and a 16-sample public corpus manifest (texts fetched
  from their pinned sources and hash-checked, not shipped); `corpus` command to prefetch.
- Prices are supplied by the user (`--prices`); none are shipped.
- A local mock endpoint (`tests/mock_endpoint.py`, 127.0.0.1 only) with faults; the tests require every
  fault to be caught.
- Still standard library only; Python 3.9+.

## 0.1.1

- Key optional for local servers (vLLM, SGLang, Ollama, llama.cpp); `--reference` accepts env:/file endpoints.

## 0.1.0

- `run` / `summary`: request-by-request comparison of an endpoint with the genuine model (capability,
  tokenizer, refusal, latency fingerprints), monitor mode, pluggable private probe banks.
