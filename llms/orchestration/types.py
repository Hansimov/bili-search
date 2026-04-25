from __future__ import annotations

from dataclasses import dataclass

from llms.contracts import ModelSpec


@dataclass(frozen=True, slots=True)
class ModelDecision:
    client: object
    spec: ModelSpec
    reason: str
    factors: tuple[str, ...]
