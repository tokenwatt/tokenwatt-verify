"""The tokenizer corpus for check I2: a manifest of public texts, fetched on demand and verified by sha256.

The texts are third-party (code at a pinned commit, commit diffs, Wikipedia revisions) and are not shipped
with the package. `fetch()` downloads each one from its pinned source, applies the recorded cut rule, and keeps
it only if the sha256 matches the manifest; a sample whose source has changed is skipped and reported.
"""

from __future__ import annotations

import hashlib
import html.parser
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from . import data_path

UA = "tokenwatt-verify (https://github.com/tokenwatt/tokenwatt-verify)"


def default_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tokenwatt-verify" / "corpus"


def load_manifest(path: Path | None = None) -> dict:
    return json.loads((path or data_path("corpus_manifest.json")).read_text(encoding="utf-8"))


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _get(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class _Paragraphs(html.parser.HTMLParser):
    """Text of <p> elements, skipping <sup> (reference markers), <style> and <script>."""

    def __init__(self):
        super().__init__()
        self.paras, self.buf, self.in_p, self.skip = [], [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.in_p += 1
        elif tag in ("sup", "style", "script"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag == "p" and self.in_p:
            self.in_p -= 1
            t = re.sub(r"[ \t]+", " ", "".join(self.buf)).strip()
            if t:
                self.paras.append(t)
            self.buf = []
        elif tag in ("sup", "style", "script") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if self.in_p and not self.skip:
            self.buf.append(data)


def _wiki_text(src: dict) -> str:
    q = {"action": "parse", "oldid": src["revid"], "prop": "text", "format": "json", "formatversion": 2}
    if src.get("variant"):
        q["variant"] = src["variant"]
    url = f"https://{src['lang']}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(q)
    target = int(re.search(r"until (\d+) bytes", src.get("rules", "")).group(1))
    p = _Paragraphs()
    p.feed(json.loads(_get(url))["parse"]["text"])
    out, size = [], 0
    for para in p.paras:  # whole paragraphs, in order, until the byte target is reached
        if size >= target:
            break
        out.append(para)
        size += len(para.encode()) + 2
    return "\n\n".join(out) + "\n"


def build_text(item: dict) -> str:
    """Fetch one sample from its pinned source and apply its cut rule (the rules the corpus was built with)."""
    src = item["source"]
    kind = src["kind"]
    if kind == "file":
        return _get(src["raw_url"]).decode()
    if kind == "commit_diffs":
        parts = [_get(u).decode() for u in src["urls"]]
        return "".join(p if p.endswith("\n") else p + "\n" for p in parts)
    if kind == "wikipedia":
        return _wiki_text(src)
    raise ValueError(f"sample {item['id']}: source kind {kind!r} can't be fetched")


def fetch(corpus_dir: Path | None = None, manifest: dict | None = None, log=print) -> dict:
    """Make sure every sample is on disk and verified. Returns {"ok": [ids], "skipped": {id: reason}}."""
    d = Path(corpus_dir or default_dir())
    d.mkdir(parents=True, exist_ok=True)
    m = manifest or load_manifest()
    ok, skipped = [], {}
    for item in m["items"]:
        f = d / f"{item['id']}.txt"
        if f.exists() and sha256(f.read_bytes()) == item["sha256"]:
            ok.append(item["id"])
            continue
        try:
            b = build_text(item).encode()
        except Exception as e:  # noqa: BLE001  network, moved source
            skipped[item["id"]] = f"fetch failed: {type(e).__name__}: {str(e)[:120]}"
            continue
        if sha256(b) != item["sha256"]:
            skipped[item["id"]] = "source changed: sha256 does not match the manifest"
            continue
        f.write_bytes(b)
        ok.append(item["id"])
    if log:
        log(f"corpus: {len(ok)}/{len(m['items'])} samples verified in {d}"
            + (f"; skipped {', '.join(f'{k} ({v})' for k, v in skipped.items())}" if skipped else ""))
    return {"ok": ok, "skipped": skipped, "dir": str(d)}


def load_texts(corpus_dir: Path | None = None, manifest: dict | None = None) -> dict:
    """{id: text} for every sample present and matching its sha256 (no network)."""
    d = Path(corpus_dir or default_dir())
    m = manifest or load_manifest()
    out = {}
    for item in m["items"]:
        f = d / f"{item['id']}.txt"
        if f.exists():
            b = f.read_bytes()
            if sha256(b) == item["sha256"]:
                out[item["id"]] = b.decode()
    return out
