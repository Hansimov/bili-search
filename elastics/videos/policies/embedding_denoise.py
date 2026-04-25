from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path


_ASSET_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "search_embedding_denoise_policy.json"
)


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class SearchEmbeddingDenoisePolicy:
    enabled: bool
    model_code_attribute_retry_only: bool
    min_hits: int
    candidate_limit: int
    max_rerank_hits: int
    keyword_boost: float
    title_keyword_boost: float
    score_field: str
    attribute_evidence_gate_enabled: bool
    attribute_term_min_score: float
    attribute_cjk_term_min_score: float
    max_attribute_candidate_terms: int
    max_attribute_evidence_terms: int
    missing_attribute_penalty: float
    attribute_evidence_boost: float


@lru_cache(maxsize=1)
def get_search_embedding_denoise_policy() -> SearchEmbeddingDenoisePolicy:
    payload = json.loads(_ASSET_PATH.read_text(encoding="utf-8"))
    return SearchEmbeddingDenoisePolicy(
        enabled=_truthy_env(
            "BILI_SEARCH_EMBEDDING_DENOISE_ENABLED",
            bool(payload.get("enabled", True)),
        ),
        model_code_attribute_retry_only=bool(
            payload.get("model_code_attribute_retry_only", True)
        ),
        min_hits=int(payload.get("min_hits", 8)),
        candidate_limit=int(payload.get("candidate_limit", 160)),
        max_rerank_hits=int(payload.get("max_rerank_hits", 160)),
        keyword_boost=float(payload.get("keyword_boost", 1.5)),
        title_keyword_boost=float(payload.get("title_keyword_boost", 2.0)),
        score_field=str(payload.get("score_field") or "embedding_denoise_score"),
        attribute_evidence_gate_enabled=bool(
            payload.get("attribute_evidence_gate_enabled", True)
        ),
        attribute_term_min_score=float(payload.get("attribute_term_min_score", 0.62)),
        attribute_cjk_term_min_score=float(
            payload.get("attribute_cjk_term_min_score", 0.72)
        ),
        max_attribute_candidate_terms=int(
            payload.get("max_attribute_candidate_terms", 96)
        ),
        max_attribute_evidence_terms=int(
            payload.get("max_attribute_evidence_terms", 6)
        ),
        missing_attribute_penalty=float(payload.get("missing_attribute_penalty", 0.25)),
        attribute_evidence_boost=float(payload.get("attribute_evidence_boost", 3.0)),
    )


__all__ = ["SearchEmbeddingDenoisePolicy", "get_search_embedding_denoise_policy"]
