# tokenwatt-verify

**Does the API endpoint serve the model it names, honour the parameters it accepts, and bill what it serves?**

An endpoint is listed by model name and price, but the same name can mean different products: a cheaper model
behind an expensive name, a stop sequence applied to hidden reasoning, a cut-off answer reported as finished,
cache discounts that never arrive, errors returned as HTTP 200. Asking the model "who are you?" proves nothing.

`tokenwatt-verify` checks behaviour instead. Standard library only, MIT licensed. Two commands:

- **`conform`** (new in 0.2): one *reading* of an endpoint against the TokenWatt Inference Endpoint
  Conformance draft (v0): identity, behaviour, billing and service checks, each reported as pass / warn /
  fail / not applicable with the evidence kept, plus a grade and a hashed JSON report.
- **`run`** (since 0.1): the same freshly generated prompts to the endpoint and to the genuine model, compared
  request by request, optionally for hours (catches intermittent swaps). See [Paired comparison](#paired-comparison-run).

```
pip install tokenwatt-verify

tokenwatt-verify conform --base-url https://api.example.com --api anthropic --model claude-sonnet-5-5 \
    --key-file ~/.keys/example --prices prices.json --cost-header X-Request-Cost
```

```
  I1 Response shape                          PASS     well-formed; model field 'claude-sonnet-5-5'; usage 17 in / 4 out
  I2 Tokenizer fingerprint                   PASS     billed counts match Claude 5.5: total ratio 1.0000, median per-sample difference 0 tokens over 16 samples
  I3 Backend attestation (thinking signature) PASS    own signed thinking block accepted; the same block with an altered signature rejected
  B1 Tool calls                              WARN     forced tool_choice not supported, reported as an error: …
  B2 Truncation reported as a cut            PASS     cut reported as 'max_tokens' streamed and not
  …
  GRADE: Conforming with notes  (B1)
  report: tokenwatt-verify-results/conform-claude-sonnet-5-5-…/report.md · report.json (content sha256 …)
```

(Illustrative output.) Every reading writes `report.json` (with a content hash), `report.md` and `evidence.jsonl`.

## What each check proves, and what it doesn't

| ID | Check | How | Proves | Does **not** prove |
|---|---|---|---|---|
| I1 | Response shape | One plain request; protocol fields, ids, `model` field, usage object | The endpoint speaks the protocol it claims | Anything about the model: trivially faked |
| I2 | Tokenizer fingerprint | 16 public texts (code, diffs, Markdown, English and Chinese prose, ~190K characters); billed prompt tokens of each, minus a baseline request, against the claimed model family's reference counts | Which tokenizer family bills the text: rules out swaps across model families (Claude 5.5 bills ~1.6× GPT-5.x on code) and old vs new tokenizers | Which model *within* a family: Opus, Sonnet and Haiku 5.5 bill identically. Nor that the counts are honest if a relay computes them itself |
| I3 | Backend attestation (Anthropic) | Extended thinking plus one tool call; send the turn back with the signed thinking block intact (must be accepted), with its signature altered (must be rejected with an invalid-signature error), and without it (to see whether the block reached the model) | That a backend verifying Anthropic thinking signatures answered: relays to other models can't produce or check them | **The tier**: an Opus request downgraded to Sonnet passes. Also not readable through a router that drops thinking blocks it doesn't recognise (reported as incomplete, not as a fail) |
| I4 | Capability fingerprint | A probe bank (`--probes`), scored against the bank's reference rates or a reference endpoint (`--reference`), one-sided Fisher exact test | Statistically, that the endpoint does as well as the claimed model on those probes | Identity: it's statistical, needs enough probes, and the built-in sample probes are public and easy (use your own bank) |
| B1 | Tool calls | `tool_choice` auto and forced, arguments validated against the JSON schema, the tool result sent back; with I3, whether returned thinking blocks reach the model | Tool calling works as the protocol says; a forced call is either honoured or refused with a documented error, never silently answered in text | Tool-use quality on real tasks |
| B2 | Truncation | A long answer asked for with `max_tokens` 16, streamed and not | A cut is reported as a cut (`max_tokens` / `length`), so clients can tell it from a finished answer | — |
| B3 | Stop sequences vs hidden reasoning | A multiplication with stop sequence `=`, which reasoning writes and the digits-only answer doesn't | The stop sequence is applied to the visible output only. Not applicable when the model didn't reason on the request | — |
| B5 | Context window | Three facts in a log at 25/50/90% depth; short (~8K tokens) by default, the advertised window with `--long-context --context-window N`; `--max-output N` checks the advertised max output is accepted | The window is accepted, the prompt reaches the model uncut (billed tokens ≈ sent), facts are recalled | Long-context *quality* beyond three facts |
| B6 | Streaming parity | The same deterministic request streamed and not; a streamed tool call | Same finish reason and usage, a final event, tool-call JSON that assembles | Identical text (sampling may differ: a warning, not a failure) |
| B7 | Error semantics | Invalid `max_tokens`, an unknown model, a bad key; every response of the reading scanned | Client errors are 4xx, nothing is an error inside an HTTP 200, error messages carry no account identifiers; any 429 has `Retry-After` | Behaviour under load (429s are reported only if the reading met one) |
| P1 | Billed tokens | I2's per-sample counts (±10 tokens per request, ±1% over the suite); a one-line message's billed size; a verbatim echo's completion tokens vs the same text's input count | Input and output tokens are counted with the claimed tokenizer, no hidden prompt is added | Reasoning tokens (not text, so not checkable this way) |
| P2 | Prices | Each request's reported cost (`--cost-header` / `--cost-field`) vs tokens × *your* listed prices (`--prices`) | No request billed above list | Your invoice, if the endpoint doesn't report per-request cost (then: not applicable) |
| P3 | Cache | A ~24K-character prefix sent 2–3 times within ~15 s | Cached tokens are reported and (with a reported cost) billed at the listed cached price | The cache TTL |
| P4 | Failed requests | Every failed request of the reading | Failures report no cost | Billing you can't see in responses |
| S1 | Speed | 5 streamed requests (configurable), p50 / p99 TTFT and decode | Reported, not graded | Behaviour at other prompt lengths or loads |

Not in this version (listed in every report as not checked): I5 reference agreement for open weights, I6
cryptographic proofs, B4 reasoning controls, S2 availability (needs probes over the whole term, e.g. with
`run --every`), S3 accuracy sets.

**Grade** (draft §8): *Nonconforming* if any identity, behaviour or billing check fails; *Incomplete* if one
could not be completed (outages, an attestation that can't be read); *Conforming with notes* if only warnings
(e.g. forced `tool_choice` refused with a documented error); else *Conforming*. The grade covers only the checks
that ran; speed is reported, not graded; availability and accuracy are not graded from a single reading.
Thresholds are versioned (`conform/thresholds.py`) and written into every report.

## Cost of a reading

The tool enforces a cap (`--max-spend`, default $2, priced from your `--prices`; without prices a token cap
applies) and reports the spend of every check. In our test readings, all default checks (~55 requests, ~120K
tokens, two thirds of them I2's corpus) cost about **$0.01 at $0.10 / $0.50 per 1M tokens** and about
**$0.26 at $2 / $10** (I2 alone $0.16). A private probe bank of 5 kinds × 3 rounds added $0.14 at $2 / $10;
`--long-context` adds about one context window of input tokens ($0.012 for a 100K window at $0.10 / 1M).
`--tokenizer-samples 6` cuts I2's cost to about a third, with less precision.

## Before you publish a result

**Check the endpoint's terms first.** Some providers' terms bar benchmarking or publishing performance
results without written permission, and an aggregator's terms can make each upstream provider's terms
binding on you. A reading of your own endpoint, or one you have permission to test, is the safe case.

## Options

```
--base-url URL          Anthropic: the host (…/v1/messages is appended). OpenAI: the base ending in /v1.
--api anthropic|openai
--model NAME            exactly as the endpoint lists it
--key-file F | --key-env VAR     the key is never printed and never written to the report or evidence
--auth auto|x-api-key|bearer|both
--header 'Name: value'  extra request header (repeatable); values are redacted from the evidence
--prices prices.json    your listed prices (below); P2, P3 and the spend cap use them
--cost-header NAME | --cost-field usage.cost   where the endpoint reports a request's billed cost
--probes bank.py [--probe-rounds 2] [--reference ENDPOINT]   I4
--long-context --context-window N [--max-output N]           B5
--samples 5 --concurrency 1                                  S1
--tokenizer-family ID --tokenizer-samples N --corpus-dir DIR  I2
--checks I1,I2,… / --skip …   --max-spend 2.0   --out DIR   --run-by "who ran it"   --seed N
```

**Prices** (USD per 1M tokens; tokenwatt-verify ships none, because a reading is checked against the listing
you saw on the day you saw it):

```json
{
  "source": "https://example.com/pricing, read 2026-11-05",
  "models": {
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5,
                          "tiers": [{"above_prompt_tokens": 200000, "input": 4.0, "output": 15.0}]}
  }
}
```

## Evidence and verification

`evidence.jsonl` has one line per HTTP attempt: the request (auth headers removed), the response (status,
headers, body or every stream event with its timestamp), timings, and that line's own sha256. Each check in
`report.json` lists the hashes of its exchanges; the report carries the tool version, the thresholds version,
the reference-data versions, who ran it, and a `content_sha256` over everything else.

```
tokenwatt-verify verify-report tokenwatt-verify-results/conform-…/report.json
```

re-computes the content hash and, if `evidence.jsonl` is next to it, every exchange hash and the file hash. The
evidence holds the endpoint's answers and the corpus texts (third-party, see below): treat it as private data.

## Tokenizer reference data (I2, P1)

`src/tokenwatt_verify/data/tokenizer_reference.json` holds, per tokenizer family, the billed count of each
public corpus sample, measured on 2026-11-05 with the same differential method (`count(text) =
prompt_tokens(prefix + text + suffix) − prompt_tokens(prefix + suffix)`); every model of a family billed
identical counts. Families: Claude 5.5, Claude's previous tokenizer (Haiku 4.5), GPT-5.x (o200k_base), Gemini 3,
DeepSeek V4, Kimi K2.6, GLM-5.3, Qwen3 2507 (Qwen Plus / Max), and Qwen3.8 (from its public tokenizer file). Where
a public tokenizer exists, it was run locally and matched the billed count within ±1 token per sample. Model
names resolve to a family only for models that were measured; for others pass `--tokenizer-family`, or read the
nearest family from the report.

The corpus texts are **not** in this package: `src/tokenwatt_verify/data/corpus_manifest.json` pins each one
(file at a commit, commit diffs, Wikipedia revision) with its license and sha256, and `tokenwatt-verify corpus`
(or the first `conform`) downloads them to `~/.cache/tokenwatt-verify/corpus` and keeps only those whose hash
matches. Licenses: MIT, BSD-3-Clause, Apache-2.0, Unlicense, CC BY-SA 4.0 (Wikipedia).

## Probe banks (I4)

The built-in probes are **public**: an endpoint operator can recognize them and route them to a genuine
model. They show the format; for real checks write your own bank and keep it private:

```python
# my_bank.py
import random

def my_probe(rng: random.Random) -> dict:
    a, b = rng.randint(1000, 9999), rng.randint(100, 999)
    return {
        "system": "You are a helpful coding assistant.",
        "user": f"A cache key is ({a} * {b}) mod 9973. What is it? End with ANSWER: <value>",
        "max_tokens": 800,
        "grade": lambda text: (text.strip().endswith(str((a * b) % 9973)), f"expected {(a * b) % 9973}"),
    }

SUITE = [("cache_key", my_probe)]

# optional, for `conform`: correct / answered counts per kind, measured on the genuine model
REFERENCE = {"source": "model maker's API, 2026-11-01", "models": {"claude-sonnet-5-5": {"cache_key": [38, 40]}}}
```

Good probes are generated fresh every time, graded by a program, look like ordinary requests, and separate the
claimed model from its cheaper neighbours. Telling models of the same family apart needs harder probes and
more of them.

## Paired comparison (`run`)

```
export OPENROUTER_API_KEY=...            # the reference: the genuine model, through OpenRouter
export EP_URL=https://api.example.com/v1 EP_KEY=... EP_MODEL=claude-sonnet-4-6 EP_API=anthropic

tokenwatt-verify run --target env:EP --reference openrouter:anthropic/claude-sonnet-4.6 --rounds 5
tokenwatt-verify run --target env:EP --reference openrouter:… --every 1800 --hours 24     # monitor
tokenwatt-verify summary results/*.jsonl
```

Each round sends one fresh instance of every probe to the endpoint and to the reference and compares graded
answers (one-sided Fisher exact test), reported input-token counts, refusals and latency → *consistent*,
*inconsistent* or *inconclusive*. Either side can be a local OpenAI-compatible server (vLLM, SGLang, Ollama,
llama.cpp) without a key, e.g. a quantized build against the original:

```
export EP_URL=http://localhost:8000/v1 EP_MODEL=Qwen/Qwen3-8B-FP8 REF_URL=http://localhost:8001/v1 REF_MODEL=Qwen/Qwen3-8B
tokenwatt-verify run --target env:EP --reference env:REF --rounds 5
```

## Limits

- **Identity within a family is statistical** (I4), never proof. I2 can't separate models that share a
  tokenizer; I3 proves an Anthropic backend, not which Claude.
- **An endpoint can treat the checks specially** if it recognizes them. The corpus and the sample probes are
  public. Private probes, fresh accounts and repeats over time make that harder.
- **References drift** when a model maker updates a model under the same name; every report records the
  reference-data version and date.
- **Prices change**: P2 checks against the prices you supply for the day of the reading.
- **The draft standard will change**: thresholds and checks are versioned, and old reports keep the version
  they were graded with.

## Privacy

Requests go only to the endpoint (and, for I4, a reference) you configure; the corpus is fetched from GitHub
and Wikipedia. Nothing is sent to TokenWatt. Keys are read from a file or an environment variable, never from
the command line, and are removed from the evidence.

## Development

```
pip install -e ".[dev]" && pytest -q
python tests/mock_endpoint.py --port 8765 --faults gpt_tokens     # a local misbehaving endpoint (127.0.0.1)
```

The tests run every check against a local mock endpoint, genuine and with each fault (GPT counts behind a
Claude name, forged thinking signatures, a cut reported as a stop, errors returned as HTTP 200), and require
each fault to be caught.

## About

Built by [TokenWatt](https://www.tokenwatt.io), which measures what GPUs and inference services really deliver.
See also [tokenwatt-check](https://github.com/tokenwatt/tokenwatt-check), our GPU health checker. MIT licensed.
