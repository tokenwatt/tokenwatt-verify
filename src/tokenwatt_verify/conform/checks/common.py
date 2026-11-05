"""Shared pieces of the checks: tools, prompts, helpers."""

from __future__ import annotations

import re

from ..runner import Incomplete
from ..tokref import normalize_model

WEATHER_SCHEMA = {"type": "object", "properties": {
    "city": {"type": "string", "description": "City name"},
    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}},
    "required": ["city"], "additionalProperties": False}
WEATHER_DESC = "Get the current weather for a city."
CITIES = ["Paris", "Tokyo", "Lima", "Oslo", "Cairo", "Perth", "Quito"]
B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def weather_tool(ctx) -> dict:
    return ctx.proto.tool("get_weather", WEATHER_DESC, WEATHER_SCHEMA)


def weather_result(city: str) -> str:
    return '{"city": "%s", "temp_c": 18, "conditions": "sunny"}' % city


def transient(res) -> bool:
    return bool(res.transport_error) or res.status in (408, 425, 429, 500, 502, 503, 504, 529)


def require(res, what: str) -> None:
    """Raise Incomplete when a request failed for reasons that say nothing about conformance."""
    if transient(res):
        raise Incomplete(f"{what}: {res.short_error()} (after retries)")


def same_model(requested: str, returned: str) -> bool:
    def n(x: str) -> str:
        x = normalize_model(x).replace(".", "-")
        return re.sub(r"(-\d{8}|-\d{4}-\d\d-\d\d|-latest|-preview)$", "", x)
    a, b = n(requested), n(returned)
    return bool(a and b) and (a == b or a.startswith(b) or b.startswith(a))


def is_claude(model: str) -> bool:
    return "claude" in normalize_model(model)


def alter_signature(sig: str) -> str:
    """The same signature with 8 characters in the middle shifted to other base64 characters (length and
    padding unchanged), so it still decodes but no longer verifies."""
    mid = max(0, len(sig) // 2 - 4)
    out = list(sig)
    for i in range(mid, min(len(sig), mid + 8)):
        c = out[i]
        if c in B64:
            out[i] = B64[(B64.index(c) + 17) % 64]
    return "".join(out)


def spread(ids: list, n: int | None) -> list:
    """n ids spread evenly over the list (keeps the mix of content types)."""
    if not n or n >= len(ids):
        return list(ids)
    step = len(ids) / n
    return [ids[int(i * step)] for i in range(n)]


def first_number(text: str) -> str:
    m = re.findall(r"-?\d[\d,]*", text or "")
    return m[-1].replace(",", "") if m else ""
