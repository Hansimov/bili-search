"""
Vector Recall

KNN vector-based recall with optional supplemental word recall.
Handles both broad queries (KNN search) and narrow filter queries
(dual-sort filter approach).
"""

import time
from concurrent.futures import ThreadPoolExecutor
from tclogger import logger

from recalls.base import RecallResult, RecallPool, NoiseFilter

# Default limits
VECTOR_KNN_K = 1000
VECTOR_NUM_CANDIDATES = 10000
VECTOR_WORD_LIMIT = 1000
VECTOR_WORD_TIMEOUT = 3.0


class VectorRecall:
    """KNN vector-based recall with supplemental word recall.

    For broad queries:
    1. Converts query to LSH embedding (bit vector)
    2. Runs KNN search for semantic similarity recall
    3. Runs parallel word recall for keyword-matching candidates
    4. Merges both pools for comprehensive coverage

    For narrow filter queries (user/bvid filters):
    1. Uses dual-sort filter search (recent + popular)
    2. No KNN needed (filter limits result set)

    Example:
        >>> recall = VectorRecall()
        >>> pool = recall.recall(
        ...     searcher=video_searcher,
        ...     query="deepseek v3",
        ... )
    """

    def __init__(
        self,
        knn_k: int = VECTOR_KNN_K,
        num_candidates: int = VECTOR_NUM_CANDIDATES,
        word_recall_limit: int = VECTOR_WORD_LIMIT,
        word_recall_timeout: float = VECTOR_WORD_TIMEOUT,
    ):
        self.knn_k = knn_k
        self.num_candidates = num_candidates
        self.word_recall_limit = word_recall_limit
        self.word_recall_timeout = word_recall_timeout

    def _word_recall(
        self,
        searcher,
        query: str,
        source_fields: list[str],
        extra_filters: list[dict],
        limit: int = None,
    ) -> RecallResult:
        """Run supplemental word recall in parallel with KNN."""
        effective_limit = limit if limit is not None else self.word_recall_limit
        start = time.perf_counter()
        try:
            res = searcher.search(
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                parse_hits=True,
                add_region_info=False,
                add_highlights_info=False,
                is_highlight=False,
                boost=True,
                rank_method="heads",
                limit=effective_limit,
                rank_top_k=effective_limit,
                timeout=self.word_recall_timeout,
                verbose=False,
            )
            took_ms = round((time.perf_counter() - start) * 1000, 2)
            return RecallResult(
                hits=res.get("hits", []),
                lane="word_supplement",
                total_hits=res.get("total_hits", 0),
                took_ms=took_ms,
                timed_out=res.get("timed_out", False),
            )
        except Exception as e:
            took_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.warn(f"× Supplemental word recall failed: {e}")
            return RecallResult(hits=[], lane="word_supplement", took_ms=took_ms)

    def _knn_recall(
        self,
        searcher,
        query: str,
        source_fields: list[str],
        extra_filters: list[dict],
        constraint_filter: dict = None,
        knn_field: str = "text_emb",
        timeout: float = 8.0,
    ) -> RecallResult:
        """Run KNN vector search."""
        start = time.perf_counter()
        try:
            res = searcher.knn_search(
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                constraint_filter=constraint_filter,
                knn_field=knn_field,
                k=self.knn_k,
                num_candidates=self.num_candidates,
                parse_hits=True,
                add_region_info=False,
                rank_method="heads",
                limit=self.knn_k,
                rank_top_k=self.knn_k,
                skip_ranking=True,
                timeout=timeout,
                verbose=False,
            )
            took_ms = round((time.perf_counter() - start) * 1000, 2)
            hits = res.get("hits", [])

            # Filter KNN noise: LSH bit vector hamming distances have narrow ranges
            # where many irrelevant docs cluster near relevant ones.
            # A stricter ratio removes clearly irrelevant docs.
            hits = NoiseFilter.filter_knn_by_score_ratio(hits)

            return RecallResult(
                hits=hits,
                lane="knn",
                total_hits=res.get("total_hits", 0),
                took_ms=took_ms,
                timed_out=res.get("timed_out", False),
            )
        except Exception as e:
            took_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.warn(f"× KNN recall failed: {e}")
            return RecallResult(hits=[], lane="knn", took_ms=took_ms)

    def _filter_recall(
        self,
        searcher,
        query: str,
        source_fields: list[str],
        extra_filters: list[dict],
        timeout: float,
    ) -> RecallResult:
        """Run dual-sort filter recall for narrow filter queries."""
        start = time.perf_counter()
        try:
            res = searcher.dual_sort_filter_search(
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                parse_hits=True,
                add_region_info=False,
                add_highlights_info=False,
                per_owner_recent_limit=2000,
                per_owner_popular_limit=2000,
                timeout=timeout,
                verbose=False,
            )
            took_ms = round((time.perf_counter() - start) * 1000, 2)
            return RecallResult(
                hits=res.get("hits", []),
                lane="filter",
                total_hits=res.get("total_hits", 0),
                took_ms=took_ms,
                timed_out=res.get("timed_out", False),
            )
        except Exception as e:
            took_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.warn(f"× Filter recall failed: {e}")
            return RecallResult(hits=[], lane="filter", took_ms=took_ms)

    def recall(
        self,
        searcher,
        query: str,
        source_fields: list[str] = None,
        extra_filters: list[dict] = [],
        constraint_filter: dict = None,
        knn_field: str = "text_emb",
        enable_word_supplement: bool = True,
        word_recall_limit: int = None,
        timeout: float = 8.0,
        verbose: bool = False,
    ) -> RecallPool:
        """Run vector recall with optional word supplement.

        Args:
            searcher: VideoSearcherV2 instance.
            query: Search query.
            source_fields: Fields to retrieve.
            extra_filters: Additional filter clauses.
            constraint_filter: Optional es_tok_constraints query dict for
                token-level filtering via the es-tok plugin.
            knn_field: Dense vector field name.
            enable_word_supplement: Whether to run word recall in parallel.
            word_recall_limit: Override word supplement limit. If None, uses
                self.word_recall_limit.
            timeout: Timeout for KNN search.
            verbose: Enable verbose logging.

        Returns:
            RecallPool with vector + word candidates.
        """
        if source_fields is None:
            source_fields = [
                "bvid",
                "title",
                "tags",
                "desc",
                "owner",
                "stat",
                "pubdate",
                "duration",
            ]

        # Check for narrow filters
        query_info, filter_clauses = searcher.get_filters_from_query(
            query=query, extra_filters=extra_filters
        )
        has_narrow = searcher.has_narrow_filters(filter_clauses)

        results = []

        if has_narrow:
            # For narrow filters, use filter-first approach
            result = self._filter_recall(
                searcher=searcher,
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                timeout=timeout,
            )
            results.append(result)
            if verbose:
                logger.mesg(
                    f"  Filter recall: {len(result.hits)} hits ({result.took_ms:.0f}ms)"
                )
        else:
            # Run KNN + word supplement in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                knn_future = executor.submit(
                    self._knn_recall,
                    searcher=searcher,
                    query=query,
                    source_fields=source_fields,
                    extra_filters=extra_filters,
                    constraint_filter=constraint_filter,
                    knn_field=knn_field,
                    timeout=timeout,
                )

                word_future = None
                if enable_word_supplement:
                    word_future = executor.submit(
                        self._word_recall,
                        searcher=searcher,
                        query=query,
                        source_fields=source_fields,
                        extra_filters=extra_filters,
                        limit=word_recall_limit,
                    )

                # Collect results
                knn_result = knn_future.result()
                results.append(knn_result)
                if verbose:
                    logger.mesg(
                        f"  KNN recall: {len(knn_result.hits)} hits "
                        f"({knn_result.took_ms:.0f}ms)"
                    )

                if word_future:
                    word_result = word_future.result()
                    results.append(word_result)
                    if verbose:
                        logger.mesg(
                            f"  Word supplement: {len(word_result.hits)} hits "
                            f"({word_result.took_ms:.0f}ms)"
                        )

        pool = RecallPool.merge(*results)
        if verbose:
            logger.mesg(
                f"  Vector recall total: {len(pool.hits)} unique candidates "
                f"({pool.took_ms:.0f}ms)"
            )

        return pool
