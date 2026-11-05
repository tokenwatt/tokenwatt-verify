"""HTTP for `conform`: one request at a time, every attempt kept as evidence (keys removed) and hashed.

Standard library only. Streams are read event by event with a timestamp per event, so the same record serves
the protocol checks (final event, usage) and the service measures (time to first token, decode speed).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UA = "tokenwatt-verify-conform"
TRANSIENT = {408, 425, 429, 500, 502, 503, 504, 529}


class BudgetExceeded(Exception):
    pass


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class Response:
    status: int | None = None
    headers: dict = field(default_factory=dict)       # lower-case names
    text: str = ""                                    # raw body (non-stream, or a stream that wasn't SSE)
    json: Any = None
    events: list = field(default_factory=list)        # stream: [{"t": s, "event": name, "data": obj|str}]
    is_stream: bool = False
    error: str = ""                                   # transport error (timeout, reset, ...)
    ttfb_s: float | None = None
    total_s: float = 0.0
    seq: int = 0
    sha256: str = ""
    attempts: int = 1
    retry_after: str | None = None

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())


class Session:
    """POSTs to one endpoint. `post()` retries transient failures (each attempt is its own evidence record)."""

    def __init__(self, base_url: str, api: str, key: str = "", *, auth: str = "auto", extra_headers: dict | None = None,
                 timeout: float = 300, evidence_path: Path | None = None, log=None):
        self.base = base_url.rstrip("/")
        self.api = api
        self.key = key or ""
        self.auth = auth
        self.extra_headers = dict(extra_headers or {})
        self.timeout = timeout
        self.evidence_path = Path(evidence_path) if evidence_path else None
        self.log = log or (lambda *_: None)
        self.lock = threading.Lock()
        self.seq = 0
        self.records: list[dict] = []          # light index of every exchange (no bodies)
        self.observed_429: list[dict] = []
        self.check = ""                        # set by the runner: which check is sending
        self.guard = None                      # callable raising BudgetExceeded before a request
        if self.evidence_path:
            self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
            self.evidence_path.write_text("")

    # ---- URLs and headers ----------------------------------------------------------------------------
    def url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        if self.api == "anthropic" and path == "/v1/messages" and self.base.endswith("/v1"):
            return self.base + "/messages"
        return self.base + path

    def auth_headers(self, key: str | None) -> dict:
        k = self.key if key is None else key
        if not k:
            return {}
        mode = self.auth if self.auth != "auto" else ("x-api-key" if self.api == "anthropic" else "bearer")
        h = {}
        if mode in ("x-api-key", "both"):
            h["x-api-key"] = k
        if mode in ("bearer", "both"):
            h["Authorization"] = f"Bearer {k}"
        return h

    def redact(self, s: str) -> str:
        if not isinstance(s, str):
            return s
        for secret in [self.key, *self.extra_headers.values()]:
            if secret and len(secret) >= 6:
                s = s.replace(secret, "[redacted]")
        return s

    # ---- requests --------------------------------------------------------------------------------------
    def post(self, path: str, body: dict, *, stream: bool = False, label: str = "", key: str | None = None,
             retry: bool = True, timeout: float | None = None, extra_headers: dict | None = None) -> Response:
        attempts, last = (3 if retry else 1), None
        for attempt in range(1, attempts + 1):
            if self.guard:
                self.guard()
            r = self._once(path, body, stream, label, key, timeout or self.timeout, extra_headers or {}, attempt)
            last = r
            transient = r.status in TRANSIENT or (r.status is None and r.error)
            if not retry or not transient or attempt == attempts:
                break
            wait = 3 if attempt == 1 else 10
            try:
                wait = max(wait, min(30.0, float(r.retry_after))) if r.retry_after else wait
            except ValueError:
                pass
            self.log(f"    retry {label or path} after HTTP {r.status or r.error[:60]} (wait {wait:.0f}s)")
            time.sleep(wait)
        last.attempts = attempt
        return last

    def _once(self, path, body, stream, label, key, timeout, extra, attempt) -> Response:
        headers = {**self.extra_headers, **extra, **self.auth_headers(key),
                   "Content-Type": "application/json", "User-Agent": UA}
        if self.api == "anthropic":
            headers.setdefault("anthropic-version", "2023-06-01")
        url = self.url(path)
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        r = Response(is_stream=stream)
        t0 = time.perf_counter()
        started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                r.status = resp.status
                r.headers = {k.lower(): v for k, v in resp.headers.items()}
                ctype = r.headers.get("content-type", "")
                if stream and "event-stream" in ctype:
                    self._read_sse(resp, r, t0)
                else:
                    raw = resp.read()
                    r.ttfb_s = round(time.perf_counter() - t0, 4)
                    r.text = raw.decode("utf-8", "replace")
                    r.is_stream = False
        except urllib.error.HTTPError as e:
            r.status = e.code
            r.headers = {k.lower(): v for k, v in (e.headers or {}).items()}
            try:
                r.text = e.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                r.text = ""
            r.is_stream = False
        except Exception as e:  # noqa: BLE001  timeouts, resets, DNS
            r.error = f"{type(e).__name__}: {e}"[:300]
        r.total_s = round(time.perf_counter() - t0, 4)
        if r.text:
            try:
                r.json = json.loads(r.text)
            except ValueError:
                r.json = None
        r.retry_after = r.headers.get("retry-after")
        self._record(r, url, headers, body, label, started, attempt)
        return r

    def _read_sse(self, resp, r: Response, t0: float) -> None:
        event = None
        for raw in resp:
            now = round(time.perf_counter() - t0, 4)
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if r.ttfb_s is None:
                r.ttfb_s = now
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
                try:
                    obj = json.loads(data) if data != "[DONE]" else "[DONE]"
                except ValueError:
                    obj = data
                r.events.append({"t": now, "event": event, "data": obj})
                event = None

    def _record(self, r: Response, url, headers, body, label, started, attempt) -> None:
        with self.lock:
            self.seq += 1
            r.seq = self.seq
        shown = {k: ("[redacted]" if k.lower() in ("authorization", "x-api-key") or k in self.extra_headers else v)
                 for k, v in headers.items()}
        resp_headers = {k: v for k, v in r.headers.items() if k != "set-cookie"}
        rec = {
            "seq": r.seq, "check": self.check, "label": label, "attempt": attempt, "time_utc": started,
            "request": {"method": "POST", "url": url, "headers": shown, "body": body},
            "response": {"status": r.status, "headers": resp_headers,
                         "body": self.redact(r.text) if not r.events else None,
                         "events": [[e["t"], e["event"], e["data"] if isinstance(e["data"], str) else
                                     json.loads(self.redact(canonical(e["data"])))] for e in r.events] or None},
            "timing": {"ttfb_s": r.ttfb_s, "total_s": r.total_s},
            "error": self.redact(r.error) if r.error else "",
        }
        rec["sha256"] = sha256_text(canonical(rec))
        r.sha256 = rec["sha256"]
        if r.status == 429:
            self.observed_429.append({"seq": r.seq, "retry_after": r.retry_after, "check": self.check})
        with self.lock:
            self.records.append({"seq": r.seq, "check": self.check, "label": label, "status": r.status,
                                 "sha256": rec["sha256"], "stream": bool(r.events)})
            if self.evidence_path:
                with self.evidence_path.open("a", encoding="utf-8") as f:
                    f.write(canonical(rec) + "\n")


def verify_evidence(path: Path) -> tuple[int, list[int]]:
    """Re-hash every exchange in an evidence file. Returns (count, [seq of records whose hash doesn't match])."""
    bad, n = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n += 1
        h = rec.pop("sha256", None)
        if sha256_text(canonical(rec)) != h:
            bad.append(rec.get("seq"))
    return n, bad

