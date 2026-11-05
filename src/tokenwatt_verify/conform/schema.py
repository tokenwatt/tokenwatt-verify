"""A small JSON Schema subset validator (type, properties, required, additionalProperties, enum, items,
minimum, maximum), enough for the tool schemas `conform` declares. Standard library only."""

from __future__ import annotations

TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is(value, typ: str) -> bool:
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, TYPES[typ])


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Return a list of problems (empty when the value matches)."""
    errs: list[str] = []
    typ = schema.get("type")
    if typ:
        types = typ if isinstance(typ, list) else [typ]
        if not any(_is(value, t) for t in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for k in schema.get("required") or []:
            if k not in value:
                errs.append(f"{path}: missing required '{k}'")
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected property '{k}'")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            errs += validate(v, schema["items"], f"{path}[{i}]")
    return errs
