"""Run the conformance checks against one endpoint and write report.json, report.md and evidence.jsonl."""

from __future__ import annotations

import datetime as dt
import random
import secrets
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .. import __version__
from . import pricing
from .protocol import Protocol, Result
from .transport import BudgetExceeded, Session

PASS, WARN, FAIL, NA, INCOMPLETE = "pass", "warn", "fail", "not_applicable", "incomplete"


class Incomplete(Exception):
    """A check could not be completed (transport errors, upstream outages): neither pass nor fail."""


@dataclass
class Options:
    base_url: str
    api: str
    model: str
    key: str = ""
    auth: str = "auto"
    headers: dict = field(default_factory=dict)
    prices_path: str | None = None
    cost_header: str | None = None
    cost_field: str | None = None
    disclosure_headers: list = field(default_factory=list)
    probes: str | None = None
    probe_rounds: int = 2
    reference: str | None = None
    long_context: bool = False
    context_window: int | None = None
    max_output: int | None = None
    samples: int = 5
    concurrency: int = 1
    tokenizer_family: str | None = None
    tokenizer_samples: int | None = None
    corpus_dir: str | None = None
    checks: list | None = None
    skip: list = field(default_factory=list)
    max_spend: float = 2.0
    max_tokens_total: int = 3_000_000
    out: str = "tokenwatt-verify-results"
    run_by: str = "self-run"
    timeout: float = 300
    seed: int | None = None
    temperature: float | None = None
    # test hooks: inject reference / manifest / texts instead of the packaged data and the network
    tokenizer_reference: dict | None = None
    corpus_manifest: dict | None = None
    corpus_texts: dict | None = None


def check_result(cid: str, status: str, summary: str, details: dict | None = None, evidence: list | None = None) -> dict:
    return {"id": cid, "status": status, "summary": summary, "details": details or {}, "evidence": evidence or []}


class Ctx:
    def __init__(self, opts: Options, session: Session, proto: Protocol, prices: dict | None, price_meta: dict, log):
        self.opts, self.session, self.proto, self.prices, self.price_meta, self.log = opts, session, proto, prices, price_meta, log
        self.results: dict = {}
        self.shared: dict = {}
        self.obs: list[dict] = []
        self.spend_est = 0.0
        self.spend_observed = 0.0
        self.tokens_used = 0
        self.seed = opts.seed if opts.seed is not None else secrets.randbits(31)
        self.rng = random.Random(self.seed)
        self.nonce = secrets.token_hex(8)
        self.check = ""
        self.check_evidence: list[str] = []
        session.guard = self.guard

    # ---- budget ----------------------------------------------------------------------------------------
    def guard(self) -> None:
        if self.prices and self.spend_est >= self.opts.max_spend:
            raise BudgetExceeded(f"estimated spend ${self.spend_est:.4f} reached --max-spend ${self.opts.max_spend:.2f}")
        if self.tokens_used >= self.opts.max_tokens_total:
            raise BudgetExceeded(f"{self.tokens_used} tokens used, the --max-tokens-total cap")

    def can_afford(self, prompt_tokens: int, output_tokens: int = 0) -> tuple[bool, float | None]:
        c = pricing.rough_cost(self.prices, prompt_tokens, output_tokens)
        if c is None:
            return self.tokens_used + prompt_tokens + output_tokens <= self.opts.max_tokens_total, None
        return self.spend_est + c <= self.opts.max_spend, c

    # ---- requests --------------------------------------------------------------------------------------
    def call(self, body: dict, *, stream: bool = False, label: str = "", key: str | None = None, retry: bool = True,
             timeout: float | None = None) -> Result:
        resp = self.session.post(self.proto.path, body, stream=stream, label=label, key=key, retry=retry, timeout=timeout)
        res = self.proto.parse(resp)
        res.observed_cost = self._observed_cost(resp, res)
        exp = pricing.expected_cost(self.prices, res.usage)
        u = res.usage or {}
        self.tokens_used += (u.get("prompt_total") or 0) + (u.get("output") or 0)
        if exp:
            self.spend_est += exp
        if res.observed_cost:
            self.spend_observed += res.observed_cost
        self.obs.append({"seq": res.seq, "sha256": res.sha256, "check": self.check, "label": label, "stream": res.stream,
                         "status": res.status, "ok": res.ok, "error_in_200": res.error_in_200,
                         "error": res.short_error() if not res.ok else "", "error_message": res.error_message,
                         "stream_error": res.stream_error, "usage": {k: v for k, v in u.items() if k != "raw"} or None,
                         "observed_cost": res.observed_cost, "expected_cost": exp,
                         "output_started": bool(res.text or res.tool_calls or res.thinking)})
        self.check_evidence.append(res.sha256)
        return res

    def _observed_cost(self, resp, res: Result) -> float | None:
        o = self.opts
        if o.cost_header and not res.stream:
            v = resp.header(o.cost_header)
            try:
                return float(v) if v is not None else None
            except ValueError:
                return None
        if o.cost_field:
            path = o.cost_field.split(".")
            cands = [resp.json] if not resp.events else [e["data"] for e in reversed(resp.events) if isinstance(e["data"], dict)]
            for c in cands:
                v = c
                for k in path:
                    v = v.get(k) if isinstance(v, dict) else None
                if isinstance(v, (int, float)):
                    return float(v)
        return None

    def documented(self, res: Result, *params: str) -> str | None:
        """'<Header>: <value>' when a configured disclosure header on this response names one of `params`
        (format `param=value[, param=value]`): the endpoint said it adjusted that parameter. None otherwise."""
        want = {p.lower() for p in params}
        for name in self.opts.disclosure_headers:
            v = (res.headers or {}).get(name.lower())
            if not v:
                continue
            named = {part.split("=", 1)[0].strip().lower() for part in v.replace(";", ",").split(",") if "=" in part}
            if named & want:
                return f"{name}: {v}"
        return None

    def thinking_param(self) -> dict:
        return self.shared.get("thinking_param") or {"type": "adaptive"}


# ---- registry ----------------------------------------------------------------------------------------------
NAMES = {
    "I1": "Response shape", "I2": "Tokenizer fingerprint", "I3": "Backend attestation (Anthropic thinking signature)",
    "I4": "Capability fingerprint", "I5": "Reference agreement (open weights)", "I6": "Cryptographic proof",
    "B1": "Tool calls", "B2": "Truncation reported as a cut", "B3": "Stop sequences not applied to hidden reasoning",
    "B4": "Reasoning controls", "B5": "Context window", "B6": "Streaming parity", "B7": "Error semantics",
    "P1": "Billed tokens vs reference tokenizer", "P2": "Cost vs listed price", "P3": "Cache discount honoured",
    "P4": "Failed requests not billed", "S1": "Time to first token and decode speed", "S2": "Availability",
    "S3": "Accuracy",
}
NOT_BUILT = {
    "I5": "needs the official open-weight checkpoint served by a reference deployment; not in this version",
    "I6": "no commercial endpoint offers cryptographic proofs of inference today",
    "B4": "not in this version (reasoning parameters differ per model; partly exercised by I3 and B3)",
    "S2": "needs probes over the whole term (e.g. every 5 minutes); a single reading can't measure it",
    "S3": "accuracy sets with intervals next to the reference are not in this version (I4 reports probe scores)",
}
ORDER = ["I1", "I2", "I3", "B1", "B2", "B3", "B6", "P1", "P3", "B5", "I4", "S1", "B7", "P4", "P2"]


def checks_table() -> dict[str, Callable]:
    from .checks import behaviour, billing, identity, service
    return {"I1": identity.i1, "I2": identity.i2, "I3": identity.i3, "I4": identity.i4,
            "B1": behaviour.b1, "B2": behaviour.b2, "B3": behaviour.b3, "B5": behaviour.b5, "B6": behaviour.b6,
            "B7": behaviour.b7, "P1": billing.p1, "P2": billing.p2, "P3": billing.p3, "P4": billing.p4,
            "S1": service.s1}


def run(opts: Options, log=print) -> tuple[dict, Path]:
    from .report import build_report, write_outputs
    started = dt.datetime.now(dt.timezone.utc)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in opts.model)
    out_dir = Path(opts.out) / f"conform-{safe}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    session = Session(opts.base_url, opts.api, opts.key, auth=opts.auth, extra_headers=opts.headers,
                      timeout=opts.timeout, evidence_path=out_dir / "evidence.jsonl", log=log)
    proto = Protocol(opts.api, opts.model)
    prices, meta = pricing.load_prices(opts.prices_path, opts.model)
    if opts.prices_path and prices is None:
        log(f"  note: {opts.prices_path} has no price for {opts.model}; P2 is not applicable and spend is capped by tokens")
    ctx = Ctx(opts, session, proto, prices, meta, log)
    table = checks_table()
    wanted = [c.upper() for c in (opts.checks or ORDER)]
    skip = {c.upper() for c in opts.skip}
    log(f"tokenwatt-verify {__version__} conform · {opts.api} · {opts.model} · {opts.base_url} → {out_dir}")
    for cid in ORDER:
        if cid not in wanted or cid in skip:
            continue
        ctx.check, session.check, ctx.check_evidence = cid, cid, []
        t0, spend0, tok0 = time.time(), ctx.spend_est, ctx.tokens_used
        try:
            r = table[cid](ctx)
        except BudgetExceeded as e:
            r = check_result(cid, INCOMPLETE, f"stopped: {e}")
        except Incomplete as e:
            r = check_result(cid, INCOMPLETE, str(e))
        except Exception as e:  # noqa: BLE001  a bug in one check must not stop the reading
            r = check_result(cid, INCOMPLETE, f"checker error: {type(e).__name__}: {str(e)[:200]}",
                             {"traceback": traceback.format_exc()[-1500:]})
        r["name"] = NAMES[cid]
        r["evidence"] = r.get("evidence") or list(ctx.check_evidence)
        r["duration_s"] = round(time.time() - t0, 1)
        r["tokens"] = ctx.tokens_used - tok0
        r["spend_usd_at_list"] = round(ctx.spend_est - spend0, 6) if ctx.prices else None
        ctx.results[cid] = r
        log(f"  {cid} {NAMES[cid]:<48} {r['status'].upper():<14} {r['summary'][:150]}")
    for cid in ORDER + list(NOT_BUILT):
        if cid in NOT_BUILT:
            ctx.results[cid] = {**check_result(cid, "not_checked", NOT_BUILT[cid]), "name": NAMES[cid]}
        elif cid not in ctx.results:
            ctx.results[cid] = {**check_result(cid, "not_checked", "not selected for this run"), "name": NAMES[cid]}
    finished = dt.datetime.now(dt.timezone.utc)
    report = build_report(ctx, started, finished)
    write_outputs(report, out_dir)
    return report, out_dir
