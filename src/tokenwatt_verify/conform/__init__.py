"""`tokenwatt-verify conform`: grade an inference endpoint against the TokenWatt Inference Endpoint
Conformance draft (v0): identity (I1-I4), behaviour (B1-B3, B5-B7), billing (P1-P4) and service (S1).

Every request and response is kept as evidence (keys removed), each exchange is hashed, and the report
carries a content hash. Thresholds are versioned in thresholds.py.
"""

from __future__ import annotations

from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def data_path(name: str) -> Path:
    return DATA / name
