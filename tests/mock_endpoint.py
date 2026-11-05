"""A local mock inference endpoint (Anthropic Messages and OpenAI Chat Completions) for tests and negative
controls. It binds to 127.0.0.1 only.

By default it behaves like a conforming endpoint for a Claude 5.5 model: signed thinking blocks that are
verified when they come back, cuts reported as cuts, errors as 4xx, prompt tokens counted with the claimed
model's reference counts. Faults make it misbehave the way a substituting relay would:

  gpt_tokens        bills GPT (o200k_base) counts while claiming Claude
  forge_signature   returns made-up thinking signatures and accepts any signature it is sent
  cut_as_stop       reports a max_tokens cut as a normal stop
  errors_as_200     returns every error with HTTP 200
  drop_unknown_thinking  a router in front of a genuine backend that silently drops thinking blocks whose
                    signature it didn't issue (so an altered signature never reaches the backend)

    python tests/mock_endpoint.py --port 8765 --faults gpt_tokens,forge_signature
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import re
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FAULTS = ("gpt_tokens", "forge_signature", "cut_as_stop", "errors_as_200", "drop_unknown_thinking")
CAL = re.compile(r"^\[calibration [0-9a-f]+\]\n(.*)\n\nReply with the single word OK and nothing else\.$", re.S)
NUMBERS = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
           "seventeen eighteen nineteen twenty").split()
LOREM = ("the light moves slowly across the water while the old stones keep their silence and the wind carries "
         "small sounds from far away ").split()
PRICES = {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5}
HIDDEN_THINKING_TOKENS = 50


def h(s: str) -> int:
    """The mock's tokenizer for text that isn't a corpus sample."""
    return (len(s.encode()) + 3) // 4


class State:
    def __init__(self, faults=(), model="claude-sonnet-5-5", key="test-key", reference=None, texts=None,
                 family="claude-5.5", gpt_family="gpt-5.x-o200k"):
        self.faults = set(faults)
        bad = self.faults - set(FAULTS)
        if bad:
            raise ValueError(f"unknown faults {bad}")
        self.model, self.key = model, key
        self.secret = secrets.token_bytes(32)
        self.cache = set()
        self.issued = set()
        self.lock = threading.Lock()
        self.counts = {}
        fam = gpt_family if "gpt_tokens" in self.faults else family
        if reference and texts:
            for sid, text in texts.items():
                c = reference["families"].get(fam, {}).get("counts", {}).get(sid)
                if c is not None:
                    self.counts[hashlib.sha256(text.encode()).hexdigest()] = c

    # tokens -----------------------------------------------------------------------------------------
    def count_text(self, s: str) -> int:
        m = CAL.match(s)
        if m:
            inner = m.group(1)
            return 20 + (self.counts.get(hashlib.sha256(inner.encode()).hexdigest()) or (h(inner) if inner else 0))
        return h(s)

    # signatures -------------------------------------------------------------------------------------
    def sign(self) -> str:
        if "forge_signature" in self.faults:
            return base64.b64encode(secrets.token_bytes(96)).decode()
        payload = secrets.token_bytes(64)
        sig = base64.b64encode(payload + hmac.new(self.secret, payload, hashlib.sha256).digest()).decode()
        self.issued.add(sig)
        return sig

    def valid(self, sig: str) -> bool:
        if "forge_signature" in self.faults:
            return True
        try:
            raw = base64.b64decode(sig, validate=True)
        except Exception:  # noqa: BLE001
            return False
        payload, mac = raw[:-32], raw[-32:]
        return len(raw) > 32 and hmac.compare_digest(mac, hmac.new(self.secret, payload, hashlib.sha256).digest())


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text") or "")
    return "".join(out)


def respond(st: State, last_user: str, tool_result: bool, tools: list, forced: bool, thinking: bool):
    """(blocks, natural stop) for a request. Blocks: ("text", s) / ("tool", name, input) / ("thinking",)."""
    blocks = [("thinking",)] if thinking else []
    t = last_user
    if tool_result:
        return blocks + [("text", "It is 18 °C and sunny there.")]
    if tools and (forced or "get_weather" in t or "weather" in t.lower()):
        city = "Paris"
        m = re.search(r"compute \((\d+) × (\d+)\) mod 7", t)
        if m:
            from_list = re.search(r"list: ([^.]+)\.", t).group(1).split(", ")
            city = from_list[(int(m.group(1)) * int(m.group(2))) % 7]
        elif "Lisbon" in t:
            city = "Lisbon"
        return blocks + [("tool", "get_weather", {"city": city})]
    if t.startswith("Count from 1 to 300"):
        return blocks + [("text", ", ".join(NUMBERS * 30))]
    if t.startswith("Repeat exactly this line"):
        return blocks + [("text", t.split("\n", 1)[1])]
    m = re.search(r"<<<BEGIN>>>\n(.*)<<<END>>>", t, re.S)
    if m:
        return blocks + [("text", m.group(1))]
    m = re.search(r"What is (\d+) × (\d+) − (\d+)\?", t)
    if m:
        a, b, c = map(int, m.groups())
        return blocks + [("text", str(a * b - c))]
    if "NOTICE lines" in t:
        found = re.findall(r"release (\S+) build number is (\d+)", t)
        return blocks + [("text", "\n".join(f"{n}: {v}" for n, v in found))]
    m = re.search(r"Compute (\d+) × (\d+) − (\d+)\.", t)
    if m:
        a, b, c = map(int, m.groups())
        return blocks + [("text", f"ANSWER: {a * b - c}")]
    m = re.search(r"Reverse this string exactly: (\S+)", t)
    if m:
        return blocks + [("text", f"ANSWER: {m.group(1)[::-1]}")]
    m = re.search(r"What is the value for (k\d+)\?", t)
    if m:
        v = re.search(rf"^{m.group(1)}: (\d+)$", t, re.M)
        return blocks + [("text", f"ANSWER: {v.group(1) if v else '?'}")]
    if "Write about 150 words" in t:
        return blocks + [("text", " ".join((LOREM * 10)[:150]) + ".")]
    if "Which model are you" in t:
        return blocks + [("text", "I am Claude.")]
    return blocks + [("text", "OK")]


class Handler(BaseHTTPRequestHandler):
    st: State = None  # set per server

    def log_message(self, *a):  # quiet
        pass

    # ---- plumbing ------------------------------------------------------------------------------------
    def _send(self, status: int, obj, headers: dict | None = None):
        if status >= 400 and "errors_as_200" in self.st.faults:
            status = 200
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, api: str, status: int, typ: str, msg: str):
        if api == "anthropic":
            self._send(status, {"type": "error", "error": {"type": typ, "message": msg}})
        else:
            self._send(status, {"error": {"message": msg, "type": typ, "code": None}})

    def _sse(self, events: list, sse_names: bool):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for ev in events:
            if sse_names and isinstance(ev, dict):
                self.wfile.write(f"event: {ev['type']}\n".encode())
            payload = ev if isinstance(ev, str) else json.dumps(ev)
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path.endswith("/v1/messages"):
            return self.anthropic(body)
        if self.path.endswith("/chat/completions"):
            return self.openai(body)
        self._send(404, {"error": {"message": "not found"}})

    def _common_errors(self, api: str, body: dict) -> bool:
        key = self.headers.get("x-api-key") if api == "anthropic" else (self.headers.get("Authorization") or "")[7:]
        if key != self.st.key:
            self._error(api, 401, "authentication_error", "invalid x-api-key" if api == "anthropic" else "Invalid API key")
            return True
        if body.get("model") != self.st.model:
            self._error(api, 404, "not_found_error", f"model: {body.get('model')}")
            return True
        mt = body.get("max_tokens")
        if not isinstance(mt, int) or mt < 1:
            self._error(api, 400, "invalid_request_error", "max_tokens: must be greater than or equal to 1")
            return True
        return False

    # ---- anthropic -----------------------------------------------------------------------------------
    def anthropic(self, body: dict):
        st = self.st
        if self._common_errors("anthropic", body):
            return
        msgs = body.get("messages") or []
        prompt, hidden = 8, 0
        system = body.get("system")
        sys_text = system if isinstance(system, str) else _text_of(system) if system else ""
        cache_read = cache_write = 0
        if sys_text:
            n = st.count_text(sys_text)
            if isinstance(system, list) and any(b.get("cache_control") for b in system):
                k = hashlib.sha256(sys_text.encode()).hexdigest()
                with st.lock:
                    if k in st.cache:
                        cache_read = n
                    else:
                        st.cache.add(k)
                        cache_write = n
            else:
                prompt += n
        for i, m in enumerate(msgs):
            c = m.get("content")
            prompt += 4
            if isinstance(c, str):
                prompt += st.count_text(c)
                continue
            for j, b in enumerate(c or []):
                t = b.get("type")
                if t == "text":
                    prompt += st.count_text(b.get("text") or "")
                elif t == "tool_result":
                    prompt += h(json.dumps(b.get("content")))
                elif t == "tool_use":
                    prompt += h(json.dumps(b.get("input")))
                elif t == "thinking":
                    if "drop_unknown_thinking" in st.faults and b.get("signature") not in st.issued:
                        continue                     # the router drops it before the backend sees it
                    # like the real API for Claude 5.x: the reasoning is in the signature, the visible text is ignored
                    if not st.valid(b.get("signature") or ""):
                        return self._error("anthropic", 400, "invalid_request_error",
                                           f"messages.{i}.content.{j}: Invalid `signature` in `thinking` block.")
                    hidden += HIDDEN_THINKING_TOKENS
        prompt += hidden
        last = msgs[-1] if msgs else {"content": ""}
        lc = last.get("content")
        tool_result = isinstance(lc, list) and any(b.get("type") == "tool_result" for b in lc)
        tc = body.get("tool_choice") or {}
        blocks = respond(st, _text_of(lc), tool_result, body.get("tools") or [], tc.get("type") in ("tool", "any"),
                         bool(body.get("thinking")))
        content, out, stop = [], 0, "end_turn"
        budget = body["max_tokens"]
        for b in blocks:
            if b[0] == "thinking":
                content.append({"type": "thinking", "thinking": "", "signature": st.sign()})
                out += min(HIDDEN_THINKING_TOKENS, budget)
            elif b[0] == "tool":
                content.append({"type": "tool_use", "id": "toolu_" + secrets.token_hex(8), "name": b[1], "input": b[2]})
                out += h(json.dumps(b[2]))
                stop = "tool_use"
            else:
                text = b[1]
                for s in body.get("stop_sequences") or []:
                    if s in text:
                        text, stop = text[: text.index(s)], "stop_sequence"
                room = max(0, budget - out)
                if h(text) > room:
                    text = text.encode()[: room * 4].decode("utf-8", "ignore")
                    stop = "max_tokens"
                content.append({"type": "text", "text": text})
                out += h(text)
        if stop == "max_tokens" and "cut_as_stop" in st.faults:
            stop = "end_turn"
        usage = {"input_tokens": prompt, "cache_creation_input_tokens": cache_write, "cache_read_input_tokens": cache_read,
                 "output_tokens": out}
        cost = (prompt * PRICES["input"] + cache_write * PRICES["cache_write"] + cache_read * PRICES["cache_read"]
                + out * PRICES["output"]) / 1e6
        mid = "msg_" + secrets.token_hex(12)
        if not body.get("stream"):
            return self._send(200, {"id": mid, "type": "message", "role": "assistant", "model": st.model,
                                    "content": content, "stop_reason": stop, "stop_sequence": None, "usage": usage},
                              {"X-Mock-Cost": f"{cost:.6f}"})
        evs = [{"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant", "model": st.model,
                                                     "content": [], "stop_reason": None,
                                                     "usage": {**usage, "output_tokens": 1}}}]
        for i, b in enumerate(content):
            if b["type"] == "text":
                evs.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
                for k in range(0, len(b["text"]), 8):
                    evs.append({"type": "content_block_delta", "index": i,
                                "delta": {"type": "text_delta", "text": b["text"][k:k + 8]}})
            elif b["type"] == "tool_use":
                evs.append({"type": "content_block_start", "index": i,
                            "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}})
                js = json.dumps(b["input"])
                for k in range(0, len(js), 5):
                    evs.append({"type": "content_block_delta", "index": i,
                                "delta": {"type": "input_json_delta", "partial_json": js[k:k + 5]}})
            else:
                evs.append({"type": "content_block_start", "index": i,
                            "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
                evs.append({"type": "content_block_delta", "index": i,
                            "delta": {"type": "signature_delta", "signature": b["signature"]}})
            evs.append({"type": "content_block_stop", "index": i})
        evs += [{"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                 "usage": {"output_tokens": out}}, {"type": "message_stop"}]
        self._sse(evs, True)

    # ---- openai --------------------------------------------------------------------------------------
    def openai(self, body: dict):
        st = self.st
        if self._common_errors("openai", body):
            return
        msgs = body.get("messages") or []
        prompt, cached = 7, 0
        for m in msgs:
            c = m.get("content")
            n = st.count_text(c) if isinstance(c, str) else h(json.dumps(m))
            if m.get("role") == "system" and isinstance(c, str) and n >= 1024:
                k = hashlib.sha256(c.encode()).hexdigest()
                with st.lock:
                    if k in st.cache:
                        cached = n
                    st.cache.add(k)
            prompt += n + 3
        last = msgs[-1] if msgs else {}
        tc = body.get("tool_choice")
        blocks = respond(st, last.get("content") if isinstance(last.get("content"), str) else "",
                         last.get("role") == "tool", body.get("tools") or [], isinstance(tc, dict), False)
        text, calls, out, finish = "", [], 0, "stop"
        for b in blocks:
            if b[0] == "tool":
                calls.append({"id": "call_" + secrets.token_hex(6), "type": "function",
                              "function": {"name": b[1], "arguments": json.dumps(b[2])}})
                out += h(json.dumps(b[2]))
                finish = "tool_calls"
            elif b[0] == "text":
                text = b[1]
                for s in body.get("stop") or []:
                    if s in text:
                        text = text[: text.index(s)]
                if h(text) > body["max_tokens"]:
                    text, finish = text.encode()[: body["max_tokens"] * 4].decode("utf-8", "ignore"), "length"
                out += h(text)
        if finish == "length" and "cut_as_stop" in st.faults:
            finish = "stop"
        usage = {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        cost = ((prompt - cached) * PRICES["input"] + cached * PRICES["cache_read"] + out * PRICES["output"]) / 1e6
        cid = "chatcmpl-" + secrets.token_hex(10)
        msg = {"role": "assistant", "content": text or None}
        if calls:
            msg["tool_calls"] = calls
        if not body.get("stream"):
            return self._send(200, {"id": cid, "object": "chat.completion", "created": 1762300000, "model": st.model,
                                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage},
                              {"X-Mock-Cost": f"{cost:.6f}"})
        base = {"id": cid, "object": "chat.completion.chunk", "created": 1762300000, "model": st.model}
        evs = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
        for k in range(0, len(text), 8):
            evs.append({**base, "choices": [{"index": 0, "delta": {"content": text[k:k + 8]}, "finish_reason": None}]})
        for i, c in enumerate(calls):
            evs.append({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": i, "id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": ""}}]},
                "finish_reason": None}]})
            a = c["function"]["arguments"]
            for k in range(0, len(a), 5):
                evs.append({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": i, "function": {"arguments": a[k:k + 5]}}]}, "finish_reason": None}]})
        evs.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        if (body.get("stream_options") or {}).get("include_usage"):
            evs.append({**base, "choices": [], "usage": usage})
        evs.append("[DONE]")
        self._sse(evs, False)


def serve(state: State, port: int = 0):
    """Start the mock on 127.0.0.1 in a background thread. Returns (server, base_url)."""
    handler = type("BoundHandler", (Handler,), {"st": state})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--faults", default="", help="comma list of: " + ", ".join(FAULTS))
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--key", default="test-key")
    ap.add_argument("--corpus-dir", help="tokenizer corpus (so the mock bills reference counts for it)")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from tokenwatt_verify.conform import corpus, tokref
    texts = corpus.load_texts(a.corpus_dir) if a.corpus_dir else {}
    st = State([f for f in a.faults.split(",") if f], a.model, a.key, tokref.load_reference(), texts)
    srv, url = serve(st, a.port)
    print(f"mock endpoint on {url} (faults: {a.faults or 'none'}; {len(st.counts)} corpus samples known)", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
