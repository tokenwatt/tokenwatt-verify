"""Minimal streaming client for OpenAI-compatible and Anthropic Messages endpoints, with timing.

Standard library only. Every call returns the text, the usage the endpoint reported, the model name it
claimed, and time to first token / generation speed, which are themselves fingerprints.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path



@dataclass
class Endpoint:
    name: str
    url: str            # base URL, e.g. https://openrouter.ai/api/v1 or https://host/v1 (anthropic: base without /messages)
    key: str = field(repr=False)
    model: str
    api: str = "openai"  # openai | anthropic

    @classmethod
    def from_file(cls, name: str, path: Path) -> "Endpoint":
        """Read `URL: …`, `KEY: …`, `MODEL: …`, optional `API: openai|anthropic` lines."""
        c = {}
        for line in path.read_text().splitlines():
            m = re.match(r"\s*(\w+)\s*[:=]\s*(.*)", line)
            if m:
                c[m.group(1).upper()] = m.group(2).strip()
        api = c.get("API", "anthropic" if "anthropic" in c.get("URL", "") else "openai").lower()
        return cls(name, c["URL"].rstrip("/"), c["KEY"], c["MODEL"], api)

    @classmethod
    def from_env(cls, prefix: str) -> "Endpoint":
        """Read <PREFIX>_URL, <PREFIX>_KEY, <PREFIX>_MODEL and optional <PREFIX>_API from the environment."""
        import os
        p = prefix.upper()
        try:
            url, key, model = os.environ[f"{p}_URL"], os.environ[f"{p}_KEY"], os.environ[f"{p}_MODEL"]
        except KeyError as e:
            raise SystemExit(f"missing environment variable {e.args[0]}") from None
        api = os.environ.get(f"{p}_API", "anthropic" if "anthropic" in url else "openai").lower()
        return cls(prefix.lower(), url.rstrip("/"), key, model, api)

    @classmethod
    def openrouter(cls, model: str) -> "Endpoint":
        """A reference through OpenRouter (key in OPENROUTER_API_KEY), e.g. anthropic/claude-sonnet-4.6."""
        import os
        key = os.environ.get("OPENROUTER_API_KEY") or ""
        if not key:
            raise SystemExit("set OPENROUTER_API_KEY to use an openrouter: reference")
        return cls(f"openrouter:{model}", "https://openrouter.ai/api/v1", key, model, "openai")

    @classmethod
    def parse(cls, spec: str) -> "Endpoint":
        """`openrouter:<model>`, `env:<PREFIX>`, or a path to a file with URL/KEY/MODEL/API lines."""
        if spec.startswith("openrouter:"):
            return cls.openrouter(spec.split(":", 1)[1])
        if spec.startswith("env:"):
            return cls.from_env(spec.split(":", 1)[1])
        p = Path(spec).expanduser()
        if not p.is_file():
            raise SystemExit(f"no endpoint file {p} (use openrouter:<model>, env:<PREFIX> or a file)")
        return cls.from_file(p.stem, p)


@dataclass
class Reply:
    text: str = ""
    model_claimed: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_s: float | None = None
    total_s: float = 0.0
    chunks: int = 0
    error: str = ""
    finish: str = ""          # stop / end_turn / max_tokens / content_filter / refusal …

    @property
    def gen_tok_s(self) -> float | None:
        if self.output_tokens and self.ttft_s is not None and self.total_s > self.ttft_s:
            return self.output_tokens / (self.total_s - self.ttft_s)
        return None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["gen_tok_s"] = self.gen_tok_s
        return d


def _request(ep: Endpoint, system: str, user: str, max_tokens: int, temperature: float):
    if ep.api == "anthropic":
        url = ep.url + ("/messages" if ep.url.endswith("/v1") else "/v1/messages")
        body = {"model": ep.model, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
                "system": system, "messages": [{"role": "user", "content": user}]}
        headers = {"x-api-key": ep.key, "Authorization": f"Bearer {ep.key}", "anthropic-version": "2023-06-01"}
    else:
        url = ep.url + "/chat/completions"
        body = {"model": ep.model, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {ep.key}"}
    headers.update({"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (compatible; tokenwatt-verify)"})
    return urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")


def ask(ep: Endpoint, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0,
        timeout: float = 300) -> Reply:
    r = Reply()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(_request(ep, system, user, max_tokens, temperature), timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                piece = _consume(ep.api, ev, r)
                if piece:
                    if r.ttft_s is None:
                        r.ttft_s = time.perf_counter() - t0
                    r.text += piece
                    r.chunks += 1
    except urllib.error.HTTPError as e:
        r.error = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
    except Exception as e:  # noqa: BLE001
        r.error = f"{type(e).__name__}: {e}"[:300]
    r.total_s = time.perf_counter() - t0
    return r


def _consume(api: str, ev: dict, r: Reply) -> str:
    if api == "anthropic":
        t = ev.get("type")
        if t == "message_start":
            m = ev.get("message") or {}
            r.model_claimed = m.get("model", "")
            r.input_tokens = (m.get("usage") or {}).get("input_tokens")
        elif t == "content_block_delta":
            d = ev.get("delta") or {}
            return d.get("text", "") if d.get("type") == "text_delta" else ""
        elif t == "message_delta":
            r.output_tokens = (ev.get("usage") or {}).get("output_tokens", r.output_tokens)
            r.finish = (ev.get("delta") or {}).get("stop_reason") or r.finish
        elif t == "error":
            r.error = json.dumps(ev.get("error"))[:300]
        return ""
    r.model_claimed = ev.get("model") or r.model_claimed
    if ev.get("usage"):
        r.input_tokens = ev["usage"].get("prompt_tokens", r.input_tokens)
        r.output_tokens = ev["usage"].get("completion_tokens", r.output_tokens)
    ch = ev.get("choices") or []
    if ch:
        r.finish = ch[0].get("finish_reason") or r.finish
        return (ch[0].get("delta") or {}).get("content") or ""
    return ""
