from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path


_ASSET_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "search_recall_policy.json"
)


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class SearchRecallPolicy:
    exact_relax_retry_enabled: bool
    low_recall_total_hits: int
    min_keyword_count: int
    max_keyword_count: int
    model_code_attribute_retry: bool


@lru_cache(maxsize=1)
def get_search_recall_policy() -> SearchRecallPolicy:
    payload = json.loads(_ASSET_PATH.read_text(encoding="utf-8"))
    return SearchRecallPolicy(
        exact_relax_retry_enabled=_truthy_env(
            "BILI_SEARCH_EXACT_RELAX_RETRY_ENABLED",
            bool(payload.get("exact_relax_retry_enabled", True)),
        ),
        low_recall_total_hits=int(payload.get("low_recall_total_hits", 3)),
        min_keyword_count=int(payload.get("min_keyword_count", 2)),
        max_keyword_count=int(payload.get("max_keyword_count", 5)),
        model_code_attribute_retry=bool(payload.get("model_code_attribute_retry", True)),
    )


__all__ = ["SearchRecallPolicy", "get_search_recall_policy"]
