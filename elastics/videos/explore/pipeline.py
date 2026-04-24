from __future__ import annotations

import time

from dataclasses import dataclass, field
from typing import Any

from tclogger import logger

from .steps import StepBuilder


@dataclass(slots=True)
class ExplorePipelineConfig:
    query: str
    recall_mode: str
    step_name: str
    extra_filters: list[dict] = field(default_factory=list)
    constraint_filter: dict | None = None
    suggest_info: dict = field(default_factory=dict)
    verbose: bool = False
    rank_method: Any = None
    rank_top_k: int = 0
    group_owner_limit: int = 0
    prefer: Any = None
    enable_rerank: bool = False
    rerank_max_hits: int = 0
    rerank_keyword_boost: float = 0.0
    rerank_title_keyword_boost: float = 0.0
    knn_field: str = ""
    recall_source_fields: list[str] | None = None
    recall_timeout: float = 0.0
    owner_intent_info: dict | None = None


def _tag_filter_only_result(result: dict, recall_mode: str) -> dict:
    if recall_mode == "word" or not result.get("data"):
        return result
    qmod_map = {
        "vector": ["vector"],
        "hybrid": ["word", "vector"],
    }
    result["data"][0]["output"]["qmod"] = qmod_map.get(recall_mode, [])
    result["data"][0]["output"]["filter_only"] = True
    return result


def _build_recall_kwargs(searcher, config: ExplorePipelineConfig) -> dict:
    recall_kwargs = {
        "searcher": searcher,
        "query": config.query,
        "mode": config.recall_mode,
        "extra_filters": config.extra_filters,
        "timeout": config.recall_timeout,
        "verbose": config.verbose,
    }
    if config.constraint_filter:
        recall_kwargs["constraint_filter"] = config.constraint_filter
    if config.recall_source_fields:
        recall_kwargs["source_fields"] = config.recall_source_fields
    if config.suggest_info:
        recall_kwargs["suggest_info"] = config.suggest_info
    if config.recall_mode in ("vector", "hybrid"):
        recall_kwargs["knn_field"] = config.knn_field
    return recall_kwargs


def run_explore_pipeline(searcher, config: ExplorePipelineConfig) -> dict:
    logger.enter_quiet(not config.verbose)
    perf = {"total_ms": 0}
    explore_start = time.perf_counter()
    steps = StepBuilder()

    if not searcher.has_search_keywords(config.query):
        logger.exit_quiet(not config.verbose)
        result = searcher._filter_only_explore(
            query=config.query,
            extra_filters=config.extra_filters,
            suggest_info=config.suggest_info,
            verbose=config.verbose,
            rank_top_k=config.rank_top_k,
            group_owner_limit=config.group_owner_limit,
        )
        return _tag_filter_only_result(result, config.recall_mode)

    step = steps.add_step(
        config.step_name,
        status="running",
        input_data={"query": config.query, "recall_mode": config.recall_mode},
    )
    logger.hint(f"> [step 0] {config.recall_mode} recall ...", verbose=config.verbose)
    recall_start = time.perf_counter()

    recall_pool = searcher.recall_manager.recall(
        **_build_recall_kwargs(searcher, config)
    )
    recall_pool = searcher._supplement_with_owner_intent_hits(
        pool=recall_pool,
        query=config.query,
        owner_intent_info=config.owner_intent_info,
        source_fields=config.recall_source_fields,
        extra_filters=config.extra_filters,
        timeout=config.recall_timeout,
        rank_top_k=config.rank_top_k,
        verbose=config.verbose,
    )
    perf["recall_ms"] = round((time.perf_counter() - recall_start) * 1000, 2)

    if not recall_pool.hits:
        steps.update_step(
            step,
            {"hits": [], "total_hits": 0, "recall_info": recall_pool.lanes_info},
        )
        steps.add_step(
            "group_hits_by_owner",
            output={"authors": []},
            comment="无搜索结果",
        )
        logger.exit_quiet(not config.verbose)
        return steps.finalize(config.query, perf=perf)

    logger.hint(
        f"> [step 1] Fetch & rank {len(recall_pool.hits)} candidates ...",
        verbose=config.verbose,
    )
    search_res, rerank_info = searcher._fetch_and_rank(
        recall_hits=recall_pool.hits,
        query=config.query,
        rank_method=config.rank_method,
        rank_top_k=config.rank_top_k,
        prefer=config.prefer,
        enable_rerank=config.enable_rerank,
        rerank_max_hits=config.rerank_max_hits,
        rerank_keyword_boost=config.rerank_keyword_boost,
        rerank_title_keyword_boost=config.rerank_title_keyword_boost,
        extra_filters=config.extra_filters,
        verbose=config.verbose,
        pool_hints=recall_pool.pool_hints,
    )
    search_res = searcher._blend_owner_intent_hits(
        search_res,
        config.owner_intent_info,
    )
    search_res["total_hits"] = recall_pool.total_hits
    search_res["recall_info"] = recall_pool.lanes_info
    perf["fetch_ms"] = search_res.get("fetch_ms", 0)
    perf["highlight_ms"] = search_res.get("highlight_ms", 0)
    perf["recall_candidates"] = len(recall_pool.hits)
    steps.update_step(step, search_res)

    if rerank_info:
        perf["rerank_ms"] = rerank_info.get("rerank_ms", 0)
        perf["reranked_count"] = rerank_info.get("reranked_count", 0)
        steps.add_step(
            "rerank",
            output=rerank_info,
            comment=f"重排了 {rerank_info.get('reranked_count', 0)} 个结果",
        )

    logger.hint(
        "> [step 2] Group by owner ...",
        verbose=config.verbose,
    )
    group_res = searcher._build_group_step(
        search_res,
        config.group_owner_limit,
        owner_intent_info=config.owner_intent_info,
    )
    steps.add_step(
        "group_hits_by_owner",
        output={"authors": group_res},
        input_data={"limit": config.group_owner_limit},
    )

    perf["total_ms"] = round((time.perf_counter() - explore_start) * 1000, 2)
    logger.exit_quiet(not config.verbose)
    return steps.finalize(config.query, perf=perf)
