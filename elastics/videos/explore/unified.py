from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tclogger import logger

from dsl.fields.word import override_auto_require_short_han_exact


@dataclass(slots=True)
class UnifiedExploreFinalizeConfig:
    query: str
    qmod: list[str] | str | None
    owner_intent_info: dict | None
    constraint_filter: dict | None = None
    auto_constraint: bool = True
    extra_filters: list[dict] = field(default_factory=list)
    suggest_info: dict = field(default_factory=dict)
    verbose: bool = False
    most_relevant_limit: int = 0
    rank_method: Any = None
    rank_top_k: int = 0
    group_owner_limit: int = 0
    prefer: Any = None
    knn_field: str = ""
    knn_k: int = 0
    knn_num_candidates: int = 0
    allow_short_han_retry: bool = True


def get_unified_explore_total_hits(result: dict) -> int:
    output = ((result or {}).get("data") or [{}])[0].get("output") or {}
    total_hits = output.get("total_hits")
    if total_hits is not None:
        return int(total_hits)
    hits = output.get("hits") or []
    return len(hits)


def annotate_unified_explore_result(
    result: dict,
    qmod: list[str] | str | None,
    owner_intent_info: dict | None,
) -> dict:
    if result.get("data"):
        output = result["data"][0].setdefault("output", {})
        output["qmod"] = qmod
    result["intent_info"] = owner_intent_info
    return result


def retry_unified_explore_without_short_han_exact(
    searcher,
    result: dict,
    *,
    config: UnifiedExploreFinalizeConfig,
) -> dict | None:
    if not searcher._should_retry_without_short_han_exact(
        config.query,
        total_hits=get_unified_explore_total_hits(result),
        allow_retry=config.allow_short_han_retry,
    ):
        return None

    logger.warn(
        "> Retry unified explore without auto-exact short Han segments",
        verbose=config.verbose,
    )
    with override_auto_require_short_han_exact("first"):
        retry_res = searcher.unified_explore(
            query=config.query,
            qmod=config.qmod,
            extra_filters=config.extra_filters,
            constraint_filter=config.constraint_filter,
            auto_constraint=config.auto_constraint,
            suggest_info=config.suggest_info,
            verbose=config.verbose,
            most_relevant_limit=config.most_relevant_limit,
            rank_method=config.rank_method,
            rank_top_k=config.rank_top_k,
            group_owner_limit=config.group_owner_limit,
            prefer=config.prefer,
            knn_field=config.knn_field,
            knn_k=config.knn_k,
            knn_num_candidates=config.knn_num_candidates,
            _allow_short_han_retry=False,
        )
    retry_info = dict(retry_res.get("retry_info") or {})
    retry_info["relaxed_short_han_exact"] = True
    retry_res["retry_info"] = retry_info
    return retry_res


def finalize_unified_explore_result(
    searcher,
    result: dict,
    *,
    config: UnifiedExploreFinalizeConfig,
) -> dict:
    result = annotate_unified_explore_result(
        result,
        qmod=config.qmod,
        owner_intent_info=config.owner_intent_info,
    )
    retry_res = retry_unified_explore_without_short_han_exact(
        searcher,
        result,
        config=config,
    )
    return retry_res or result
