from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re


_ASSET_PATH = Path(__file__).with_name("assets") / "owner_intent_policy.json"


@dataclass(frozen=True, slots=True)
class OwnerIntentPolicy:
    resolve_size: int
    score_min: float
    filter_gap_min: float
    multi_owner_score_margin: float
    multi_owner_name_margin: float
    max_candidates: int
    spaced_rerank_score_gap: float
    query_min_length: int
    query_max_length: int
    prefix_candidate_max_extra_chars: int
    blocked_markers: tuple[str, ...]
    spaced_blocked_markers: tuple[str, ...]
    multi_owner_source_labels: frozenset[str]
    model_code_patterns: tuple[re.Pattern[str], ...]

    def has_blocked_marker(self, text: str, *, allow_whitespace: bool) -> bool:
        markers = (
            self.blocked_markers if allow_whitespace else self.spaced_blocked_markers
        )
        return any(marker in str(text or "") for marker in markers)

    def query_length_ok(self, text: str) -> bool:
        length = len(str(text or "").strip())
        return self.query_min_length <= length <= self.query_max_length

    def looks_like_model_code_query(self, text: str) -> bool:
        normalized = re.sub(r"[\s._-]+", "", str(text or "")).strip()
        if not normalized:
            return False
        if any("\u4e00" <= char <= "\u9fff" for char in normalized):
            return False
        if not any(char.isalpha() for char in normalized):
            return False
        if not any(char.isdigit() for char in normalized):
            return False
        return any(
            pattern.fullmatch(normalized) for pattern in self.model_code_patterns
        )

    def supports_multi_owner_sources(self, candidate: dict) -> bool:
        sources = set(candidate.get("sources") or [])
        return bool(sources.intersection(self.multi_owner_source_labels))


@lru_cache(maxsize=1)
def get_owner_intent_policy() -> OwnerIntentPolicy:
    payload = json.loads(_ASSET_PATH.read_text(encoding="utf-8"))
    query_length = payload.get("query_length") or {}
    return OwnerIntentPolicy(
        resolve_size=int(payload.get("resolve_size", 8)),
        score_min=float(payload.get("score_min", 180.0)),
        filter_gap_min=float(payload.get("filter_gap_min", 30.0)),
        multi_owner_score_margin=float(payload.get("multi_owner_score_margin", 50.0)),
        multi_owner_name_margin=float(payload.get("multi_owner_name_margin", 12.0)),
        max_candidates=int(payload.get("max_candidates", 5)),
        spaced_rerank_score_gap=float(payload.get("spaced_rerank_score_gap", 8.0)),
        query_min_length=int(query_length.get("min", 2)),
        query_max_length=int(query_length.get("max", 24)),
        prefix_candidate_max_extra_chars=int(
            payload.get("prefix_candidate_max_extra_chars", 3)
        ),
        blocked_markers=tuple(
            payload.get("blocked_markers") or [":", "=", '"', "'", "[", "]"]
        ),
        spaced_blocked_markers=tuple(
            payload.get("spaced_blocked_markers")
            or [":", "=", '"', "'", "[", "]", "(", ")", "|", "&"]
        ),
        multi_owner_source_labels=frozenset(
            payload.get("multi_owner_source_labels") or ["topic", "relation"]
        ),
        model_code_patterns=tuple(
            re.compile(pattern)
            for pattern in (
                payload.get("model_code_patterns")
                or [r"(?i)^[a-z]{1,2}\d{2,}[a-z0-9]*$"]
            )
        ),
    )


__all__ = ["OwnerIntentPolicy", "get_owner_intent_policy"]
