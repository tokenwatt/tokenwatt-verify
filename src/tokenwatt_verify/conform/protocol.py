"""The two wire protocols `conform` speaks (Anthropic Messages, OpenAI Chat Completions): request builders and
one normalized view of a response, streamed or not.

The normalized view keeps what the checks compare: visible text, tool calls (raw arguments and parsed input),
thinking blocks with their signatures, the finish reason (raw and normalized), usage (prompt tokens including
cache reads and writes, plus each part), and for streams the final event, usage and token timestamps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .transport import Response

FINISH_ANTHROPIC = {"end_turn": "stop", "stop_sequence": "stop_sequence", "max_tokens": "length", "tool_use": "tool_use",
                    "refusal": "refusal", "pause_turn": "pause", "model_context_window_exceeded": "length"}
FINISH_OPENAI = {"stop": "stop", "length": "length", "tool_calls": "tool_use", "function_call": "tool_use",
                 "content_filter": "refusal"}


@dataclass
class Result:
    api: str
    status: int | None = None
    seq: int = 0
    sha256: str = ""
    ok: bool = False                      # HTTP 200 and a model response (not an error object)
    transport_error: str = ""
    error_type: str = ""
    error_message: str = ""
    error_in_200: bool = False            # HTTP 200 whose body is an error object
    id: str = ""
    model: str = ""
    text: str = ""
    tool_calls: list = field(default_factory=list)   # {"id","name","arguments_raw","input","json_error"}
    thinking: list = field(default_factory=list)     # anthropic thinking / redacted_thinking blocks, as returned
    reasoning_text: str = ""                         # openai reasoning_content / reasoning
    content: Any = None                              # anthropic: content blocks as returned; openai: message
    finish_raw: str | None = None
    finish: str | None = None
    usage: dict | None = None
    shape_problems: list = field(default_factory=list)
    # streams
    stream: bool = False
    final_event: bool = False
    usage_in_stream: bool = False
    stream_error: Any = None
    t_first_token: float | None = None
    t_last_token: float | None = None
    t_first_text: float | None = None
    token_events: int = 0
    total_s: float = 0.0
    headers: dict = field(default_factory=dict)
    observed_cost: float | None = None

    @property
    def reasoning_observed(self) -> bool:
        u = self.usage or {}
        return bool(self.thinking or self.reasoning_text or (u.get("reasoning") or 0) > 0)

    def short_error(self) -> str:
        if self.transport_error:
            return self.transport_error[:160]
        return f"HTTP {self.status}" + (f" {self.error_type}" if self.error_type else "") + \
            (f": {self.error_message[:160]}" if self.error_message else "")


def _int(v) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def norm_usage_anthropic(u: dict | None) -> dict | None:
    if not isinstance(u, dict):
        return None
    inp, out = _int(u.get("input_tokens")), _int(u.get("output_tokens"))
    read = _int(u.get("cache_read_input_tokens")) or 0
    write = _int(u.get("cache_creation_input_tokens")) or 0
    if inp is None and out is None:
        return None
    thinking = _int((u.get("output_tokens_details") or {}).get("thinking_tokens"))
    return {"prompt_total": (inp or 0) + read + write, "input_uncached": inp or 0, "cache_read": read,
            "cache_write": write, "output": out, "reasoning": thinking, "raw": u}


def cached_tokens(u: dict) -> int:
    """Cached prompt tokens however an OpenAI-compatible server reports them."""
    cands = [(u.get("prompt_tokens_details") or {}).get("cached_tokens"), u.get("cache_read_tokens"),
             u.get("cache_read_input_tokens"), u.get("prompt_cache_hit_tokens"), u.get("cached_tokens"),
             (u.get("input_tokens_details") or {}).get("cached_tokens")]
    return max([int(v) for v in cands if isinstance(v, (int, float)) and v > 0] or [0])


def norm_usage_openai(u: dict | None) -> dict | None:
    if not isinstance(u, dict):
        return None
    p, c = _int(u.get("prompt_tokens")), _int(u.get("completion_tokens"))
    if p is None and c is None:
        return None
    read = cached_tokens(u)
    write = _int(u.get("cache_creation_input_tokens")) or _int(u.get("cache_creation_tokens")) or 0
    p = p or 0
    if read + write > p:          # Anthropic semantics behind an OpenAI shape: prompt_tokens excludes the cache
        total, uncached = p + read + write, p
    else:
        total, uncached = p, p - read - write
    reasoning = _int((u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    return {"prompt_total": total, "input_uncached": uncached, "cache_read": read, "cache_write": write,
            "output": c, "reasoning": reasoning, "raw": u}


class Protocol:
    def __init__(self, api: str, model: str):
        if api not in ("anthropic", "openai"):
            raise ValueError("api must be anthropic or openai")
        self.api, self.model = api, model

    @property
    def path(self) -> str:
        return "/v1/messages" if self.api == "anthropic" else "/chat/completions"

    # ---- building ------------------------------------------------------------------------------------
    def user(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def tool(self, name: str, description: str, schema: dict) -> dict:
        if self.api == "anthropic":
            return {"name": name, "description": description, "input_schema": schema}
        return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}

    def tool_choice(self, kind: str, name: str | None = None):
        if self.api == "anthropic":
            return {"type": "auto"} if kind == "auto" else {"type": "tool", "name": name}
        return "auto" if kind == "auto" else {"type": "function", "function": {"name": name}}

    def request(self, messages: list, *, system: str | None = None, max_tokens: int = 64, tools: list | None = None,
                tool_choice=None, stream: bool = False, stop: list | None = None, thinking: dict | None = None,
                temperature: float | None = None, cache_system: bool = False, model: str | None = None,
                extra: dict | None = None) -> dict:
        body: dict = {"model": model or self.model, "max_tokens": max_tokens}
        if self.api == "anthropic":
            if system is not None:
                body["system"] = ([{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                                  if cache_system else system)
            body["messages"] = messages
            if stop:
                body["stop_sequences"] = stop
            if thinking:
                body["thinking"] = thinking
        else:
            body["messages"] = ([{"role": "system", "content": system}] if system is not None else []) + messages
            if stop:
                body["stop"] = stop
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if temperature is not None:
            body["temperature"] = temperature
        if stream:
            body["stream"] = True
            if self.api == "openai":
                body["stream_options"] = {"include_usage": True}
        if extra:
            body.update(extra)
        return body

    def assistant_echo(self, res: Result) -> dict:
        """The assistant turn exactly as returned (what a client appends to its history)."""
        if self.api == "anthropic":
            return {"role": "assistant", "content": [dict(b) for b in (res.content or [])]}
        msg = {k: v for k, v in (res.content or {}).items() if v is not None}
        msg["role"] = "assistant"
        return msg

    def tool_results(self, res: Result, content: str) -> list:
        if self.api == "anthropic":
            return [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": tc["id"], "content": content}
                                                 for tc in res.tool_calls]}]
        return [{"role": "tool", "tool_call_id": tc["id"], "content": content} for tc in res.tool_calls]

    # ---- parsing -------------------------------------------------------------------------------------
    def parse(self, resp: Response) -> Result:
        res = Result(api=self.api, status=resp.status, seq=resp.seq, sha256=resp.sha256, headers=resp.headers,
                     total_s=resp.total_s)
        if resp.error and resp.status is None:
            res.transport_error = resp.error
            return res
        if resp.events:
            res.stream = True
            (self._anthropic_stream if self.api == "anthropic" else self._openai_stream)(resp, res)
            return res
        j = resp.json
        err = _error_of(j)
        if resp.status != 200 or err is not None or not isinstance(j, dict):
            res.ok = False
            if err is not None:
                res.error_type, res.error_message = err
            elif not isinstance(j, dict):
                res.error_message = (resp.text or "")[:300]
            res.error_in_200 = resp.status == 200
            return res
        (self._anthropic_message if self.api == "anthropic" else self._openai_message)(j, res)
        return res

    # anthropic
    def _blocks(self, content: list, res: Result) -> None:
        res.content = content
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                res.text += b.get("text") or ""
            elif t == "tool_use":
                inp = b.get("input")
                raw = b.get("_raw_json")
                je = b.get("_json_error")
                res.tool_calls.append({"id": b.get("id"), "name": b.get("name"), "input": inp if je is None else None,
                                       "arguments_raw": raw if raw is not None else json.dumps(inp),
                                       "json_error": je})
            elif t in ("thinking", "redacted_thinking"):
                res.thinking.append(b)

    def _anthropic_message(self, j: dict, res: Result) -> None:
        res.ok = True
        res.id, res.model = str(j.get("id") or ""), str(j.get("model") or "")
        self._blocks(j.get("content") or [], res)
        res.finish_raw = j.get("stop_reason")
        res.finish = FINISH_ANTHROPIC.get(res.finish_raw, res.finish_raw)
        res.usage = norm_usage_anthropic(j.get("usage"))
        p = res.shape_problems
        if not isinstance(j.get("id"), str) or not j.get("id"):
            p.append("no id")
        if j.get("type") != "message":
            p.append(f"type is {j.get('type')!r}, not 'message'")
        if j.get("role") != "assistant":
            p.append(f"role is {j.get('role')!r}")
        if not isinstance(j.get("content"), list) or any(not isinstance(b, dict) or "type" not in b for b in j.get("content") or []):
            p.append("content is not a list of typed blocks")
        if not isinstance(j.get("model"), str) or not j.get("model"):
            p.append("no model field")
        if j.get("stop_reason") not in FINISH_ANTHROPIC:
            p.append(f"stop_reason {j.get('stop_reason')!r} is not an Anthropic stop reason")
        u = j.get("usage")
        if not isinstance(u, dict) or _int(u.get("input_tokens")) is None or _int(u.get("output_tokens")) is None:
            p.append("usage.input_tokens / usage.output_tokens missing")

    def _anthropic_stream(self, resp: Response, res: Result) -> None:
        blocks: dict = {}
        usage: dict = {}
        for ev in resp.events:
            d, t = ev["data"], ev["t"]
            if not isinstance(d, dict):
                continue
            typ = d.get("type")
            if typ == "message_start":
                m = d.get("message") or {}
                res.id, res.model = str(m.get("id") or ""), str(m.get("model") or "")
                usage.update(m.get("usage") or {})
            elif typ == "content_block_start":
                b = dict(d.get("content_block") or {})
                if b.get("type") == "tool_use":
                    b["_json"] = ""
                blocks[d.get("index", len(blocks))] = b
            elif typ == "content_block_delta":
                b = blocks.setdefault(d.get("index", 0), {"type": "text", "text": ""})
                x = d.get("delta") or {}
                k = x.get("type")
                piece = ""
                if k == "text_delta":
                    piece = x.get("text") or ""
                    b["text"] = b.get("text", "") + piece
                    if piece and res.t_first_text is None:
                        res.t_first_text = t
                elif k == "input_json_delta":
                    piece = x.get("partial_json") or ""
                    b["_json"] = b.get("_json", "") + piece
                elif k == "thinking_delta":
                    piece = x.get("thinking") or ""
                    b["thinking"] = b.get("thinking", "") + piece
                elif k == "signature_delta":
                    b["signature"] = b.get("signature", "") + (x.get("signature") or "")
                if piece:
                    res.token_events += 1
                    res.t_first_token = t if res.t_first_token is None else res.t_first_token
                    res.t_last_token = t
            elif typ == "message_delta":
                dd = d.get("delta") or {}
                res.finish_raw = dd.get("stop_reason") or res.finish_raw
                if d.get("usage"):
                    usage.update({k: v for k, v in d["usage"].items() if v is not None})
                    res.usage_in_stream = True
            elif typ == "message_stop":
                res.final_event = True
            elif typ == "error":
                res.stream_error = d.get("error") or d
        content = []
        for i in sorted(blocks):
            b = blocks[i]
            if b.get("type") == "tool_use" and "_json" in b:
                raw = b.pop("_json")
                b["_raw_json"] = raw
                try:
                    b["input"] = json.loads(raw) if raw else b.get("input", {})
                except ValueError as e:
                    b["_json_error"] = str(e)
            content.append(b)
        self._blocks(content, res)
        res.content = [{k: v for k, v in b.items() if not k.startswith("_")} for b in content]
        res.finish = FINISH_ANTHROPIC.get(res.finish_raw, res.finish_raw)
        res.usage = norm_usage_anthropic(usage) if usage else None
        res.usage_in_stream = res.usage_in_stream and res.usage is not None and res.usage.get("output") is not None
        res.ok = resp.status == 200 and res.stream_error is None

    # openai
    def _openai_message(self, j: dict, res: Result) -> None:
        res.ok = True
        res.id, res.model = str(j.get("id") or ""), str(j.get("model") or "")
        ch = (j.get("choices") or [{}])[0] if isinstance(j.get("choices"), list) else {}
        msg = ch.get("message") or {}
        res.content = msg
        c = msg.get("content")
        if isinstance(c, list):
            c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
        res.text = c or ""
        res.reasoning_text = msg.get("reasoning_content") or msg.get("reasoning") or ""
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw = fn.get("arguments")
            inp, je = None, None
            try:
                inp = json.loads(raw) if isinstance(raw, str) else raw
            except ValueError as e:
                je = str(e)
            res.tool_calls.append({"id": tc.get("id"), "name": fn.get("name"), "arguments_raw": raw, "input": inp,
                                   "json_error": je})
        res.finish_raw = ch.get("finish_reason")
        res.finish = FINISH_OPENAI.get(res.finish_raw, res.finish_raw)
        res.usage = norm_usage_openai(j.get("usage"))
        p = res.shape_problems
        if not isinstance(j.get("id"), str) or not j.get("id"):
            p.append("no id")
        if j.get("object") != "chat.completion":
            p.append(f"object is {j.get('object')!r}, not 'chat.completion'")
        if _int(j.get("created")) is None:
            p.append("no integer 'created'")
        if not isinstance(j.get("model"), str) or not j.get("model"):
            p.append("no model field")
        if not isinstance(j.get("choices"), list) or not j.get("choices"):
            p.append("no choices")
        elif msg.get("role") != "assistant":
            p.append(f"choices[0].message.role is {msg.get('role')!r}")
        if ch.get("finish_reason") not in FINISH_OPENAI:
            p.append(f"finish_reason {ch.get('finish_reason')!r} is not an OpenAI finish reason")
        u = j.get("usage")
        if not isinstance(u, dict) or _int(u.get("prompt_tokens")) is None or _int(u.get("completion_tokens")) is None:
            p.append("usage.prompt_tokens / usage.completion_tokens missing")
        elif _int(u.get("total_tokens")) is not None and u["total_tokens"] != u["prompt_tokens"] + u["completion_tokens"]:
            p.append(f"usage.total_tokens {u['total_tokens']} != prompt + completion")

    def _openai_stream(self, resp: Response, res: Result) -> None:
        calls: dict = {}
        msg: dict = {"role": "assistant", "content": ""}
        reasoning = ""
        usage = None
        for ev in resp.events:
            d, t = ev["data"], ev["t"]
            if d == "[DONE]":
                res.final_event = True
                continue
            if not isinstance(d, dict):
                continue
            if d.get("error"):
                res.stream_error = d["error"]
            if d.get("usage"):
                usage = d["usage"]
            res.id = res.id or str(d.get("id") or "")
            res.model = d.get("model") or res.model
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or ""
                rpiece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if piece:
                    msg["content"] += piece
                    if res.t_first_text is None:
                        res.t_first_text = t
                reasoning += rpiece
                for tc in delta.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", len(calls)), {"id": None, "type": "function",
                                                                       "function": {"name": "", "arguments": ""}})
                    c["id"] = tc.get("id") or c["id"]
                    fn = tc.get("function") or {}
                    c["function"]["name"] += fn.get("name") or ""
                    c["function"]["arguments"] += fn.get("arguments") or ""
                    piece = piece or (fn.get("arguments") or "")
                if piece or rpiece:
                    res.token_events += 1
                    res.t_first_token = t if res.t_first_token is None else res.t_first_token
                    res.t_last_token = t
                res.finish_raw = ch.get("finish_reason") or res.finish_raw
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = [calls[i] for i in sorted(calls)]
        if not msg["content"] and calls:
            msg["content"] = None
        self._openai_message({"choices": [{"message": msg, "finish_reason": res.finish_raw}], "usage": usage,
                              "id": res.id, "model": res.model}, res)
        res.shape_problems = []
        res.usage_in_stream = usage is not None and res.usage is not None
        res.ok = resp.status == 200 and res.stream_error is None


def _error_of(j) -> tuple[str, str] | None:
    """(type, message) when a body is an error object, else None."""
    if not isinstance(j, dict):
        return None
    e = j.get("error")
    if j.get("type") == "error" or (e is not None and not (j.get("choices") or j.get("content"))):
        if isinstance(e, dict):
            return str(e.get("type") or e.get("code") or ""), str(e.get("message") or "")
        return "", str(e or "")
    return None
