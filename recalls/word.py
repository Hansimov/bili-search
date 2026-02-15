"""
Multi-Lane Word Recall

Runs parallel ES searches optimized for different dimensions:
- relevance: BM25 match, sorted by _score (default)
- popularity: BM25 match, sorted by stat.view desc
- recency: BM25 match, sorted by pubdate desc
- quality: BM25 match, sorted by stat_score desc

This ensures that relevant, popular, recent, AND high-quality documents
are all present in the candidate pool before ranking.

Performance advantage: Instead of scanning 10000 docs in one query,
runs 4 focused queries for ~200 docs each, in parallel.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from tclogger import logger

from recalls.base import RecallResult, RecallPool, NoiseFilter
from ranks.constants import MIN_BM25_SCORE

# Lane configurations: (lane_name, sort_spec, limit)
# Increased limits for better recall coverage:
#   - More candidates per lane → better chance of finding good docs
#   - Parallel execution means wall-clock time ≈ slowest single lane
WORD_RECALL_LANES = {
    "relevance": {
        "sort": None,  # Default _score sorting
        "limit": 500,
        "desc": "BM25 keyword relevance",
    },
    "popularity": {
        "sort": [{"stat.view": "desc"}],
        "limit": 200,
        "desc": "Most viewed matching docs",
    },
    "recency": {
        "sort": [{"pubdate": "desc"}],
        "limit": 200,
        "desc": "Most recent matching docs",
    },
    "quality": {
        "sort": [{"stat_score": "desc"}],
        "limit": 200,
        "desc": "Highest quality matching docs",
    },
    "engagement": {
        "sort": [{"stat.coin": "desc"}],
        "limit": 100,
        "desc": "Highest engagement (coins) matching docs",
    },
}

# Default lanes to run in parallel
DEFAULT_LANES = ["relevance", "popularity", "recency", "quality", "engagement"]


class MultiLaneWordRecall:
    """Multi-lane word-based recall strategy.

    Runs multiple ES BM25 searches in parallel, each sorted by a
    different signal, then merges into a single deduplicated pool.

    This guarantees that the recall pool contains candidates from
    all four dimensions (relevant, popular, recent, quality), which
    the downstream ranker can then draw from for diversified ranking.

    Example:
        >>> recall = MultiLaneWordRecall()
        >>> pool = recall.recall(
        ...     searcher=video_searcher,
        ...     query="黑神话",
        ...     lanes=["relevance", "popularity", "recency", "quality"],
        ... )
        >>> len(pool.hits)  # ~400 after dedup
        387
        >>> pool.lanes_info
        {'relevance': {'hit_count': 200, ...}, 'popularity': {...}, ...}
    """

    def __init__(
        self,
        lanes_config: dict = None,
        max_workers: int = 5,
    ):
        """Initialize multi-lane recall.

        Args:
            lanes_config: Override lane configurations. Defaults to WORD_RECALL_LANES.
            max_workers: Max parallel threads for ES queries.
        """
        self.lanes_config = lanes_config or WORD_RECALL_LANES
        self.max_workers = max_workers

    def _run_lane(
        self,
        searcher,
        query: str,
        lane_name: str,
        lane_config: dict,
        source_fields: list[str],
        extra_filters: list[dict],
        suggest_info: dict,
        timeout: float,
    ) -> RecallResult:
        """Run a single recall lane.

        Args:
            searcher: VideoSearcherV2 instance.
            query: Search query string.
            lane_name: Name of this recall lane.
            lane_config: Configuration for this lane (sort, limit).
            source_fields: Fields to retrieve from ES.
            extra_filters: Additional filter clauses.
            suggest_info: Suggestion info for query rewriting.
            timeout: ES search timeout in seconds.

        Returns:
            RecallResult for this lane.
        """
        start = time.perf_counter()
        sort_spec = lane_config.get("sort")
        limit = lane_config.get("limit", 200)

        try:
            if sort_spec is not None:
                # For non-relevance lanes, use custom sort via raw ES query
                res = self._search_with_sort(
                    searcher=searcher,
                    query=query,
                    source_fields=source_fields,
                    extra_filters=extra_filters,
                    suggest_info=suggest_info,
                    sort_spec=sort_spec,
                    limit=limit,
                    timeout=timeout,
                )
            else:
                # Relevance lane: use standard BM25 search
                res = searcher.search(
                    query=query,
                    source_fields=source_fields,
                    extra_filters=extra_filters,
                    suggest_info=suggest_info,
                    parse_hits=True,
                    add_region_info=False,
                    add_highlights_info=False,
                    is_highlight=False,
                    boost=True,
                    rank_method="heads",
                    limit=limit,
                    rank_top_k=limit,
                    timeout=timeout,
                    verbose=False,
                )

            took_ms = round((time.perf_counter() - start) * 1000, 2)
            hits = res.get("hits", [])

            # Apply content quality penalty first (reduces inflated BM25 scores
            # for short texts and very low-engagement docs)
            NoiseFilter.apply_content_quality_penalty(hits)

            # Apply score-ratio noise filtering within this lane
            # This removes docs that barely match the query (BM25 rare-keyword noise)
            original_count = len(hits)
            hits = NoiseFilter.filter_by_score_ratio(hits)

            return RecallResult(
                hits=hits,
                lane=lane_name,
                total_hits=res.get("total_hits", 0),
                took_ms=took_ms,
                timed_out=res.get("timed_out", False),
            )
        except Exception as e:
            took_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.warn(f"× Recall lane '{lane_name}' failed: {e}")
            return RecallResult(
                hits=[],
                lane=lane_name,
                total_hits=0,
                took_ms=took_ms,
                timed_out=False,
            )

    def _search_with_sort(
        self,
        searcher,
        query: str,
        source_fields: list[str],
        extra_filters: list[dict],
        suggest_info: dict,
        sort_spec: list[dict],
        limit: int,
        timeout: float,
    ) -> dict:
        """Execute a search with custom sort order.

        Builds a search body with the BM25 query but overrides the sort
        to use a custom field (e.g., stat.view, pubdate, stat_score).
        Uses min_score to ensure documents still match the query.

        Args:
            searcher: VideoSearcherV2 instance.
            query: Search query.
            source_fields: Fields to retrieve.
            extra_filters: Filter clauses.
            suggest_info: Suggestion info.
            sort_spec: ES sort specification (e.g., [{"stat.view": "desc"}]).
            limit: Max results.
            timeout: Timeout in seconds.

        Returns:
            Parsed search result dict.
        """
        from elastics.videos.constants import (
            SEARCH_MATCH_FIELDS,
            EXPLORE_BOOSTED_FIELDS,
        )
        from elastics.videos.constants import SEARCH_MATCH_TYPE, TERMINATE_AFTER
        from elastics.structure import construct_boosted_fields, set_timeout

        # Build the same query DSL as standard search
        boosted_match_fields, boosted_date_fields = construct_boosted_fields(
            match_fields=SEARCH_MATCH_FIELDS,
            boost=True,
            boosted_fields=EXPLORE_BOOSTED_FIELDS,
        )
        _, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
            query=query,
            suggest_info=suggest_info,
            boosted_match_fields=boosted_match_fields,
            boosted_date_fields=boosted_date_fields,
            match_type=SEARCH_MATCH_TYPE,
            extra_filters=extra_filters,
        )

        # Build search body with custom sort
        search_body = {
            "query": query_dsl_dict,
            "_source": source_fields,
            "sort": sort_spec + [{"_score": "desc"}],  # Tiebreak by score
            "track_total_hits": True,
            "size": limit,
            "terminate_after": min(TERMINATE_AFTER, 500000),  # Smaller scan for speed
            "min_score": MIN_BM25_SCORE,  # Filter noise: docs must reasonably match
        }
        search_body = set_timeout(search_body, timeout=timeout)

        # Submit to ES
        es_res_dict = searcher.submit_to_es(search_body, context=f"recall_{sort_spec}")

        # Parse with minimal processing
        query_info = searcher.query_rewriter.get_query_info(query)
        parse_res = searcher.hit_parser.parse(
            query_info,
            match_fields=SEARCH_MATCH_FIELDS,
            res_dict=es_res_dict,
            request_type="search",
            drop_no_highlights=False,
            add_region_info=False,
            add_highlights_info=False,
            match_type=SEARCH_MATCH_TYPE,
            limit=limit,
            verbose=False,
        )
        return parse_res

    def recall(
        self,
        searcher,
        query: str,
        source_fields: list[str] = None,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        lanes: list[str] = None,
        timeout: float = 5.0,
        verbose: bool = False,
    ) -> RecallPool:
        """Run multi-lane recall and merge results.

        Args:
            searcher: VideoSearcherV2 instance for ES access.
            query: Search query string.
            source_fields: Fields to retrieve. Defaults to minimal fields.
            extra_filters: Additional filter clauses.
            suggest_info: Suggestion info for query rewriting.
            lanes: Which lanes to run. Defaults to all four.
            timeout: Timeout per lane in seconds.
            verbose: Enable verbose logging.

        Returns:
            RecallPool with merged, deduplicated candidates.
        """
        if source_fields is None:
            source_fields = [
                "bvid",
                "title",
                "desc",
                "stat",
                "pubdate",
                "duration",
                "stat_score",
            ]
        if lanes is None:
            lanes = DEFAULT_LANES

        start = time.perf_counter()

        # Run lanes in parallel
        results: list[RecallResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for lane_name in lanes:
                if lane_name not in self.lanes_config:
                    continue
                lane_config = self.lanes_config[lane_name]
                future = executor.submit(
                    self._run_lane,
                    searcher=searcher,
                    query=query,
                    lane_name=lane_name,
                    lane_config=lane_config,
                    source_fields=source_fields,
                    extra_filters=extra_filters,
                    suggest_info=suggest_info,
                    timeout=timeout,
                )
                futures[future] = lane_name

            for future in as_completed(futures):
                lane_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if verbose:
                        logger.mesg(
                            f"  Lane '{lane_name}': {len(result.hits)} hits "
                            f"({result.took_ms:.0f}ms)"
                        )
                except Exception as e:
                    logger.warn(f"× Lane '{lane_name}' failed: {e}")

        # Merge all lane results
        pool = RecallPool.merge(*results)
        pool.took_ms = round((time.perf_counter() - start) * 1000, 2)

        if verbose:
            logger.mesg(
                f"  Multi-lane recall: {len(pool.hits)} unique candidates "
                f"from {len(results)} lanes ({pool.took_ms:.0f}ms)"
            )

        return pool
