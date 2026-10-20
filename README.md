# tokenwatt-verify

**Is the model behind that API endpoint the one it claims to be, on every request?**

API resellers and relays are cheap and convenient. Some of them quietly route part of the traffic to a cheaper
model while still answering under the expensive model's name, and not always: genuine one hour, swapped the
next. Asking the model "who are you?" proves nothing, because any model can be told to say any name.

`tokenwatt-verify` checks behavior instead. It sends the same freshly generated prompts to the endpoint and to
the genuine model (the *reference*), and compares what comes back, request by request.

```
pip install tokenwatt-verify

export OPENROUTER_API_KEY=...            # the reference: the genuine model, through OpenRouter
export EP_URL=https://api.example.com/v1 EP_KEY=... EP_MODEL=claude-sonnet-4-6 EP_API=anthropic

tokenwatt-verify run --target env:EP --reference openrouter:anthropic/claude-sonnet-4.6 --rounds 5
```

```
  r1 sample_arithmetic  PASS    | reference PASS    | tokens ×1.000 | ttft 1.31s
  r1 sample_reverse     PASS    | reference PASS    | tokens ×1.000 | ttft 1.18s
  …
  ✅ capability  endpoint 14/15 correct vs reference 15/15 on the same prompts (one-sided Fisher p = 0.5000)
  ✅ tokenizer   0/20 requests report input-token counts more than 3% away from the reference (median ratio 1.000)
  ℹ️  refusals    endpoint refused/filtered 0/15, reference 0/15
  ℹ️  latency     median time to first token 1.24 s vs reference 1.09 s (1.1×)

  VERDICT: CONSISTENT with the reference model
```

## Local and self-hosted endpoints

Either side can be any OpenAI-compatible server, including one you run yourself (vLLM, SGLang, Ollama,
llama.cpp). Leave the key unset if the server doesn't require one. For example, to check that a quantized
build behaves like the full-precision model it was made from:

```
export EP_URL=http://localhost:8000/v1 EP_MODEL=Qwen/Qwen3-8B-FP8      # the build under test
export REF_URL=http://localhost:8001/v1 REF_MODEL=Qwen/Qwen3-8B         # the original, on another port

tokenwatt-verify run --target env:EP --reference env:REF --rounds 5
```

With the same model family on both sides the tokenizer signal stays equal, so the verdict rests on graded
answers and refusals; see [Limits](#limits).

## What it compares

| Signal | How | Why it's hard to fake |
|---|---|---|
| **Capability** | Graded answers to the same fresh prompts, endpoint vs reference (one-sided Fisher exact test) | A cheaper model fails prompts the genuine one passes. Prompts are generated from a random seed, so they can't be memorized |
| **Tokenizer** | Input-token counts the endpoint reports for identical prompts vs the reference | Each model family has its own tokenizer. A different model behind the endpoint reports different counts, unless the relay fakes them |
| **Refusals** | How often each side refuses or is filtered | A model family's safety behavior is a fingerprint |
| **Latency** | Time to first token, generation speed | Informational: relays add latency. A big jump in speed can hint at a smaller model |
| **Errors** | Share of failed requests | Bursts of errors often mean several upstream channels behind one model name |

**Verdict:** *inconsistent* if capability or tokenizer evidence says the endpoint isn't serving the reference
model; *consistent* once enough prompts have been answered with no such evidence; otherwise *inconclusive*.

## Catching intermittent swaps

One check is a snapshot. To catch an endpoint that switches models, monitor it:

```
tokenwatt-verify run --target env:EP --reference openrouter:anthropic/claude-sonnet-4.6 --every 1800 --hours 24
```

Every request is logged (JSONL), so you can see *when* answers stopped matching. `tokenwatt-verify summary
results/*.jsonl` re-computes the verdict over any set of runs.

## Use your own probes (important)

The built-in probes are **public**. Anyone running an endpoint can recognize them and send them to the genuine
model. They're here to show the format and make the tool work out of the box.

For real checks, write your own probe bank and keep it private:

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
```

```
tokenwatt-verify run --target env:EP --reference openrouter:… --probes my_bank.py
```

Good probes are generated fresh every time, graded by a program, look like ordinary requests, and are
things the genuine model gets right and cheaper models often don't. To tell models of the *same family* apart
(a smaller sibling standing in for the flagship), you need harder probes, since they share a tokenizer.

## Limits

- **Statistical, not proof.** One wrong answer proves nothing; a consistent gap over many prompts does.
- **Same-family swaps are harder** to detect than cross-family ones: the tokenizer is shared and capability
  gaps are smaller.
- **The reference path matters.** A model reached through a different provider can have different safety
  filters or settings, so compare refusal rates with care.
- **An endpoint can treat probes specially** if it can recognize them. Private, changing probes and probes that
  look like real traffic make that impractical.

## Privacy

Prompts go only to the endpoint and the reference you configure. Nothing is sent to TokenWatt. Keys are read
from environment variables or a file, never from the command line.

## About

Built by [TokenWatt](https://www.tokenwatt.io), which measures what GPUs and inference services really deliver.
See also [tokenwatt-check](https://github.com/tokenwatt/tokenwatt-check), our GPU health checker. MIT licensed.
