"""
Recall Manager

Orchestrates recall strategy selection based on query mode (qmod).
Provides a unified interface for all recall operations with multi-round
recall support for improved coverage and relevance.

Multi-round recall:
    Round 1: Standard recall (word/vector/hybrid)
    Round 2 (if needed): Supplementary recall to fill candidate deficit
        - Relaxed scoring to capture more borderline candidates
        - Owner-focused recall when owner intent is detected
        - Broader keyword variations
"""

import re
import time
from tclogger import logger

from recalls.base import RecallPool, RecallResult
from recalls.word import MultiLaneWordRecall
from recalls.vector import VectorRecall
from ranks.constants import EXPLORE_RANK_TOP_K


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
        constraint_filter: dict = None,
        suggest_info: dict = {},
        knn_field: str = "text_emb",
        timeout: float = 5.0,
        target_count: int = EXPLORE_RANK_TOP_K,
        verbose: bool = False,
    ) -> RecallPool:
        """Execute multi-round recall based on mode.

        Round 1: Standard recall using the specified mode.
        Round 2 (if pool < target_count): Supplementary recall strategies:
            - Owner-focused recall when owner intent is detected
            - Relaxed-threshold recall to fill the deficit

        Args:
            searcher: VideoSearcherV2 instance.
            query: Search query string.
            mode: Recall mode - "word", "vector", or "hybrid".
            source_fields: Fields to retrieve from ES.
            extra_filters: Additional filter clauses.
            constraint_filter: Optional es_tok_constraints query dict.
            suggest_info: Suggestion info for query rewriting.
            knn_field: Dense vector field for KNN.
            timeout: Timeout per search in seconds.
            target_count: Target number of candidates for the ranking phase.
            verbose: Enable verbose logging.

        Returns:
            RecallPool with candidates from selected strategy.
        """
        start = time.perf_counter()

        # =====================================================================
        # Round 1: Standard recall
        # =====================================================================
        pool = self._round1_recall(
            searcher=searcher,
            query=query,
            mode=mode,
            source_fields=source_fields,
            extra_filters=extra_filters,
            constraint_filter=constraint_filter,
            suggest_info=suggest_info,
            knn_field=knn_field,
            timeout=timeout,
            verbose=verbose,
        )

        # Apply pool-level noise filtering with target_count guarantee
        pre_filter_count = len(pool.hits)
        pool = pool.filter_noise(target_count=target_count)

        if verbose:
            noise_removed = pre_filter_count - len(pool.hits)
            filter_msg = f" (filtered {noise_removed} noise)" if noise_removed else ""
            logger.mesg(
                f"  Round 1 ({mode}): {len(pool.hits)} candidates "
                f"in {pool.took_ms:.0f}ms{filter_msg}"
            )

        # =====================================================================
        # Round 2: Supplementary recall if pool is below target
        # =====================================================================
        if len(pool.hits) < target_count and mode in ("word", "hybrid"):
            pool = self._round2_supplementary(
                pool=pool,
                searcher=searcher,
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                suggest_info=suggest_info,
                timeout=timeout,
                target_count=target_count,
                verbose=verbose,
            )

        # =====================================================================
        # Round 3: Owner-focused recall (for all modes)
        # =====================================================================
        owner_names = self._detect_owner_intent(query, pool.hits)
        if owner_names:
            pool = self._owner_focused_recall(
                pool=pool,
                searcher=searcher,
                query=query,
                owner_names=owner_names,
                source_fields=source_fields,
                extra_filters=extra_filters,
                timeout=timeout,
                verbose=verbose,
            )

        pool.took_ms = round((time.perf_counter() - start) * 1000, 2)

        if verbose:
            logger.mesg(
                f"  RecallManager final: {len(pool.hits)} candidates "
                f"in {pool.took_ms:.0f}ms"
            )

        return pool

    def _round1_recall(
        self,
        searcher,
        query: str,
        mode: str,
        source_fields: list[str],
        extra_filters: list[dict],
        constraint_filter: dict,
        suggest_info: dict,
        knn_field: str,
        timeout: float,
        verbose: bool,
    ) -> RecallPool:
        """Execute standard round-1 recall."""
        if mode == "word":
            return self.word_recall.recall(
                searcher=searcher,
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                suggest_info=suggest_info,
                timeout=timeout,
                verbose=verbose,
            )
        elif mode == "vector":
            word_limit = None
            if constraint_filter:
                word_limit = max(200, self.vector_recall.word_recall_limit // 2)

            return self.vector_recall.recall(
                searcher=searcher,
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                constraint_filter=constraint_filter,
                knn_field=knn_field,
                enable_word_supplement=True,
                word_recall_limit=word_limit,
                timeout=timeout,
                verbose=verbose,
            )
        elif mode == "hybrid":
            word_pool = self.word_recall.recall(
                searcher=searcher,
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                suggest_info=suggest_info,
                lanes=["relevance"],
                timeout=timeout,
                verbose=verbose,
            )
            vector_pool = self.vector_recall.recall(
                searcher=searcher,
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                constraint_filter=constraint_filter,
                knn_field=knn_field,
                enable_word_supplement=False,
                timeout=timeout,
                verbose=verbose,
            )
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
            return RecallPool.merge(word_result, vector_result)
        else:
            raise ValueError(f"Unknown recall mode: {mode}")

    def _round2_supplementary(
        self,
        pool: RecallPool,
        searcher,
        query: str,
        source_fields: list[str],
        extra_filters: list[dict],
        suggest_info: dict,
        timeout: float,
        target_count: int,
        verbose: bool,
    ) -> RecallPool:
        """Run supplementary recall to fill deficit in the candidate pool.

        Strategies:
        1. Broader recall with relaxed min_score on popularity/recency/quality lanes
        2. Increase limits for the relevance lane

        Args:
            pool: Current recall pool (below target_count).
            searcher: VideoSearcherV2 instance.
            query: Search query.
            source_fields: Fields to retrieve.
            extra_filters: Filter clauses.
            suggest_info: Suggestion info.
            timeout: Timeout per lane.
            target_count: Desired minimum pool size.
            verbose: Verbose logging.

        Returns:
            Enlarged RecallPool.
        """
        deficit = target_count - len(pool.hits)
        if deficit <= 0:
            return pool

        if verbose:
            logger.mesg(
                f"  Round 2: supplementary recall for {deficit} more candidates..."
            )

        # Run broader recall with MUCH larger limits to go deeper in the index.
        # Original lanes use limits up to 600. Supplementary must exceed these
        # to find genuinely new documents. The extra candidates are filtered by
        # noise removal, so overshooting is safe.
        supplementary_lanes_config = {
            "relevance_broad": {
                "sort": None,
                "limit": max(1500, deficit * 10),
                "desc": "Deep BM25 recall to fill deficit",
            },
            "popularity_broad": {
                "sort": [{"stat.view": "desc"}],
                "limit": max(1000, deficit * 5),
                "desc": "Deep popular content recall",
            },
            "quality_broad": {
                "sort": [{"stat_score": "desc"}],
                "limit": max(1000, deficit * 5),
                "desc": "Deep quality content recall",
            },
        }

        # Create a temporary word recall with broader config
        from recalls.word import MultiLaneWordRecall

        broad_recall = MultiLaneWordRecall(lanes_config=supplementary_lanes_config)
        supplement_pool = broad_recall.recall(
            searcher=searcher,
            query=query,
            source_fields=source_fields,
            extra_filters=extra_filters,
            suggest_info=suggest_info,
            lanes=list(supplementary_lanes_config.keys()),
            timeout=timeout,
            verbose=verbose,
        )

        if not supplement_pool.hits:
            return pool

        # Merge: add only new bvids not already in pool, capped at deficit
        existing_bvids = {h.get("bvid") for h in pool.hits if h.get("bvid")}
        new_hits = [
            h
            for h in supplement_pool.hits
            if h.get("bvid") and h.get("bvid") not in existing_bvids
        ]

        # Cap supplementary additions: we only need enough to fill the deficit
        # plus a margin (2x) for the ranker to select from.
        max_supplement = max(deficit * 2, 200)
        if len(new_hits) > max_supplement:
            new_hits = new_hits[:max_supplement]

        if new_hits:
            # Create merged pool
            combined_hits = pool.hits + new_hits
            combined_tags = dict(pool.lane_tags)
            for hit in new_hits:
                bvid = hit.get("bvid")
                if bvid:
                    combined_tags[bvid] = combined_tags.get(bvid, set()) | {
                        "supplement"
                    }

            new_pool = RecallPool(
                hits=combined_hits,
                lanes_info={
                    **pool.lanes_info,
                    "supplement": {
                        "new_hits": len(new_hits),
                        "total_supplement": len(supplement_pool.hits),
                    },
                },
                total_hits=max(pool.total_hits, supplement_pool.total_hits),
                took_ms=pool.took_ms,
                timed_out=pool.timed_out or supplement_pool.timed_out,
                lane_tags=combined_tags,
            )

            if verbose:
                logger.mesg(
                    f"  Round 2: added {len(new_hits)} new candidates "
                    f"(pool: {len(pool.hits)} → {len(new_pool.hits)})"
                )

            return new_pool

        return pool

    @staticmethod
    def _detect_owner_intent(query: str, hits: list[dict]) -> list[str]:
        """Detect if query partially matches any owner names in the pool.

        Checks if all meaningful tokens in the query appear in an owner name.
        For example, '红警08' matches owner '红警HBK08' because both '红警'
        and '08' appear in the owner name tokens.

        Args:
            query: Search query.
            hits: Recall pool hits (must have 'owner' field).

        Returns:
            List of matching owner names, sorted by frequency descending.
        """
        if not query or not hits:
            return []

        # Tokenize query into meaningful chunks (CJK, alpha, numeric)
        query_tokens = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", query.lower()))
        if not query_tokens or len(query_tokens) < 1:
            return []

        # Count docs per owner
        owner_counts: dict[str, int] = {}
        for hit in hits:
            owner = hit.get("owner")
            if isinstance(owner, dict):
                name = owner.get("name", "")
            else:
                name = ""
            if name:
                owner_counts[name] = owner_counts.get(name, 0) + 1

        # Find owners whose token set contains ALL query tokens
        matches = []
        for name, count in owner_counts.items():
            name_tokens = set(
                re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", name.lower())
            )
            if query_tokens.issubset(name_tokens):
                matches.append((name, count))

        # Sort by frequency descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return [name for name, count in matches]

    def _owner_focused_recall(
        self,
        pool: RecallPool,
        searcher,
        query: str,
        owner_names: list[str],
        source_fields: list[str],
        extra_filters: list[dict],
        timeout: float,
        verbose: bool,
    ) -> RecallPool:
        """Run focused recall for detected owner names.

        When the query matches a specific creator (e.g., '红警08' → '红警HBK08'),
        fetch more content from that creator to ensure they're well-represented
        in the candidate pool.

        Args:
            pool: Current recall pool.
            searcher: VideoSearcherV2 instance.
            query: Search query.
            owner_names: Detected matching owner names.
            source_fields: Fields to retrieve.
            extra_filters: Filter clauses.
            timeout: Timeout.
            verbose: Verbose logging.

        Returns:
            Enlarged RecallPool with owner-specific docs.
        """
        if not owner_names:
            return pool

        if verbose:
            logger.mesg(f"  Owner intent detected: {owner_names[:3]}")

        existing_bvids = {h.get("bvid") for h in pool.hits if h.get("bvid")}
        all_new_hits = []

        for owner_name in owner_names[:2]:  # Top 2 matching owners
            # Build owner filter
            owner_filter = {"match_phrase": {"owner.name": owner_name}}
            combined_filters = list(extra_filters) + [owner_filter]

            try:
                # Search for this owner's content matching the query
                res = searcher.search(
                    query=query,
                    source_fields=source_fields
                    or [
                        "bvid",
                        "title",
                        "tags",
                        "desc",
                        "owner",
                        "stat",
                        "pubdate",
                        "duration",
                        "stat_score",
                    ],
                    extra_filters=combined_filters,
                    parse_hits=True,
                    add_region_info=False,
                    add_highlights_info=False,
                    is_highlight=False,
                    boost=False,
                    rank_method="heads",
                    limit=200,
                    rank_top_k=200,
                    timeout=timeout,
                    verbose=False,
                )
                owner_hits = res.get("hits", [])

                # Tag and add only new docs
                new_count = 0
                for hit in owner_hits:
                    bvid = hit.get("bvid")
                    if bvid and bvid not in existing_bvids:
                        hit["_owner_matched"] = True
                        hit["_matched_owner_name"] = owner_name
                        all_new_hits.append(hit)
                        existing_bvids.add(bvid)
                        new_count += 1

                if verbose:
                    logger.mesg(
                        f"  Owner recall '{owner_name}': "
                        f"{len(owner_hits)} total, {new_count} new"
                    )
            except Exception as e:
                logger.warn(f"  Owner recall for '{owner_name}' failed: {e}")

        if all_new_hits:
            combined_hits = pool.hits + all_new_hits
            combined_tags = dict(pool.lane_tags)
            for hit in all_new_hits:
                bvid = hit.get("bvid")
                if bvid:
                    combined_tags[bvid] = combined_tags.get(bvid, set()) | {"owner"}

            new_pool = RecallPool(
                hits=combined_hits,
                lanes_info={
                    **pool.lanes_info,
                    "owner_recall": {
                        "owners": owner_names[:2],
                        "new_hits": len(all_new_hits),
                    },
                },
                total_hits=pool.total_hits,
                took_ms=pool.took_ms,
                timed_out=pool.timed_out,
                lane_tags=combined_tags,
            )

            if verbose:
                logger.mesg(
                    f"  Owner recall: added {len(all_new_hits)} "
                    f"(pool: {len(pool.hits)} → {len(new_pool.hits)})"
                )

            return new_pool

        return pool
