"""
Recall Manager

Orchestrates recall strategy selection based on query mode (qmod).
Provides a unified interface for all recall operations.
"""

import time
from tclogger import logger

from recalls.base import RecallPool
from recalls.word import MultiLaneWordRecall
from recalls.vector import VectorRecall


class RecallManager:
    """Manages recall strategy selection and execution.

    Selects the appropriate recall strategy based on query mode:
    - word: Multi-lane word recall (relevance + popularity + recency + quality)
    - vector: KNN vector recall with word supplement
    - hybrid: Both word and vector recall, merged

    Example:
        >>> manager = RecallManager()
        >>> pool = manager.recall(
        ...     searcher=video_searcher,
        ...     query="黑神话",
        ...     mode="word",
        ... )
        >>> len(pool.hits)
        387
    """

    def __init__(self):
        self.word_recall = MultiLaneWordRecall()
        self.vector_recall = VectorRecall()

    def recall(
        self,
        searcher,
        query: str,
        mode: str = "word",
        source_fields: list[str] = None,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        knn_field: str = "text_emb",
        timeout: float = 5.0,
        verbose: bool = False,
    ) -> RecallPool:
        """Execute recall based on mode.

        Args:
            searcher: VideoSearcherV2 instance.
            query: Search query string.
            mode: Recall mode - "word", "vector", or "hybrid".
            source_fields: Fields to retrieve from ES.
            extra_filters: Additional filter clauses.
            suggest_info: Suggestion info for query rewriting.
            knn_field: Dense vector field for KNN.
            timeout: Timeout per search in seconds.
            verbose: Enable verbose logging.

        Returns:
            RecallPool with candidates from selected strategy.
        """
        start = time.perf_counter()

        if mode == "word":
            pool = self.word_recall.recall(
                searcher=searcher,
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                suggest_info=suggest_info,
                timeout=timeout,
                verbose=verbose,
            )
        elif mode == "vector":
            pool = self.vector_recall.recall(
                searcher=searcher,
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                knn_field=knn_field,
                enable_word_supplement=True,
                timeout=timeout,
                verbose=verbose,
            )
        elif mode == "hybrid":
            # Run both word and vector recall, merge pools
            word_pool = self.word_recall.recall(
                searcher=searcher,
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                suggest_info=suggest_info,
                lanes=["relevance"],  # Only relevance lane for word
                timeout=timeout,
                verbose=verbose,
            )
            vector_pool = self.vector_recall.recall(
                searcher=searcher,
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                knn_field=knn_field,
                enable_word_supplement=False,  # Word recall already done above
                timeout=timeout,
                verbose=verbose,
            )
            # Merge both pools
            from recalls.base import RecallResult

            word_result = RecallResult(
                hits=word_pool.hits,
                lane="word",
                total_hits=word_pool.total_hits,
                took_ms=word_pool.took_ms,
                timed_out=word_pool.timed_out,
            )
            vector_result = RecallResult(
                hits=vector_pool.hits,
                lane="vector",
                total_hits=vector_pool.total_hits,
                took_ms=vector_pool.took_ms,
                timed_out=vector_pool.timed_out,
            )
            pool = RecallPool.merge(word_result, vector_result)
        else:
            raise ValueError(f"Unknown recall mode: {mode}")

        # Apply pool-level noise filtering to remove low-confidence candidates
        pre_filter_count = len(pool.hits)
        pool = pool.filter_noise()

        pool.took_ms = round((time.perf_counter() - start) * 1000, 2)

        if verbose:
            noise_removed = pre_filter_count - len(pool.hits)
            filter_msg = f" (filtered {noise_removed} noise)" if noise_removed else ""
            logger.mesg(
                f"  RecallManager ({mode}): {len(pool.hits)} candidates "
                f"in {pool.took_ms:.0f}ms{filter_msg}"
            )

        return pool
