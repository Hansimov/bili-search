from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path


_ASSET_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "search_semantic_policy.json"
)


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class SearchSemanticRewritePolicy:
    enabled: bool
    alias_rewrite_enabled: bool
    relation_rewrite_enabled: bool
    relation_mode: str
    relation_fallback_mode: str
    relation_size: int
    relation_scan_limit: int
    query_min_length: int
    query_max_length: int
    keyword_min_count: int
    keyword_max_count: int
    blocked_markers: tuple[str, ...]
    candidate_blocked_markers: tuple[str, ...]
    min_option_score: float
    min_option_score_ratio: float
    max_rewrite_options: int
    require_same_keyword_count: bool
    trigger_mixed_script_keyword: bool
    trigger_alias_rewritten_query: bool
    trigger_model_code_attribute_keywords: bool

    def query_length_ok(self, text: str) -> bool:
        length = len(str(text or "").strip())
        return self.query_min_length <= length <= self.query_max_length

    def keyword_count_ok(self, keywords: list[str] | tuple[str, ...]) -> bool:
        count = len(list(keywords or []))
        return self.keyword_min_count <= count <= self.keyword_max_count

    def has_blocked_marker(self, text: str) -> bool:
        return any(marker in str(text or "") for marker in self.blocked_markers)

    def has_candidate_blocked_marker(self, text: str) -> bool:
        return any(
            marker in str(text or "") for marker in self.candidate_blocked_markers
        )


@lru_cache(maxsize=1)
def get_search_semantic_policy() -> SearchSemanticRewritePolicy:
    payload = json.loads(_ASSET_PATH.read_text(encoding="utf-8"))
    query_length = payload.get("query_length") or {}
    keyword_count = payload.get("keyword_count") or {}
    triggers = payload.get("triggers") or {}
    return SearchSemanticRewritePolicy(
        enabled=_truthy_env(
            "BILI_SEARCH_SEMANTIC_REWRITE_ENABLED",
            bool(payload.get("enabled", True)),
        ),
        alias_rewrite_enabled=_truthy_env(
            "BILI_SEARCH_ALIAS_REWRITE_ENABLED",
            bool(payload.get("alias_rewrite_enabled", True)),
        ),
        relation_rewrite_enabled=_truthy_env(
            "BILI_SEARCH_RELATION_REWRITE_ENABLED",
            bool(payload.get("relation_rewrite_enabled", True)),
        ),
        relation_mode=str(payload.get("relation_mode") or "semantic"),
        relation_fallback_mode=str(payload.get("relation_fallback_mode") or "auto"),
        relation_size=int(payload.get("relation_size", 6)),
        relation_scan_limit=int(payload.get("relation_scan_limit", 128)),
        query_min_length=int(query_length.get("min", 2)),
        query_max_length=int(query_length.get("max", 48)),
        keyword_min_count=int(keyword_count.get("min", 1)),
        keyword_max_count=int(keyword_count.get("max", 6)),
        blocked_markers=tuple(
            payload.get("blocked_markers")
            or [":", "=", '"', "'", "[", "]", "(", ")", "|", "&"]
        ),
        candidate_blocked_markers=tuple(
            payload.get("candidate_blocked_markers")
            or [":", "=", '"', "'", "[", "]", "(", ")", "|", "&"]
        ),
        min_option_score=float(payload.get("min_option_score", 80.0)),
        min_option_score_ratio=float(payload.get("min_option_score_ratio", 0.35)),
        max_rewrite_options=int(payload.get("max_rewrite_options", 3)),
        require_same_keyword_count=bool(
            payload.get("require_same_keyword_count", True)
        ),
        trigger_mixed_script_keyword=bool(triggers.get("mixed_script_keyword", True)),
        trigger_alias_rewritten_query=bool(triggers.get("alias_rewritten_query", True)),
        trigger_model_code_attribute_keywords=bool(
            triggers.get("model_code_attribute_keywords", True)
        ),
    )


__all__ = ["SearchSemanticRewritePolicy", "get_search_semantic_policy"]
