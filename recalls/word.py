"""
Multi-Lane Word Recall

Runs parallel ES searches optimized for different dimensions:
- relevance: BM25 match, sorted by _score (text match quality)
- title_match: BM25 match on title+tags (catches entity/exact queries)
- popularity: BM25 match, sorted by stat.view desc (raw reach)
- recency: BM25 match, sorted by pubdate desc (freshness)
- quality: BM25 match, sorted by stat_score desc (content quality)

Design rationale:
- Each lane captures a distinct signal dimension to maximize diversity.
- The title_match lane is critical for entity queries like '通义实验室',
  '飓风营救', '红警08' where the query should match in the title or tags.
  Without it, popular-but-irrelevant docs dominate via other lanes.
  Tags are included because they often contain entity names, movie titles,
  and topic keywords that are as strong a signal as the title itself.
- Relevance lane gets a larger limit (500) because BM25 is the primary
  signal and we want broad keyword coverage in the candidate pool.

Performance: 5 parallel queries of ~200-500 docs each vs scanning 10000 docs
in one query. Wall-clock time ≈ slowest single lane.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from tclogger import logger

from recalls.base import RecallResult, RecallPool, NoiseFilter
from ranks.constants import MIN_BM25_SCORE

# Lane configurations: (lane_name, sort_spec, limit)
#
# Each lane captures a distinct signal dimension:
#   - relevance: text match quality (BM25 _score) — broad recall
#   - title_match: title+tags BM25 — high precision for entity queries
#   - popularity: raw reach/awareness (view count)
#   - recency: freshness (publish date)
#   - quality: content quality (stat_score from DocScorer)
#
# The title_match lane was added to fix entity query problems:
#   For queries like '通义实验室', '飓风营救', '红警08', the standard
#   relevance lane searches all fields (title, tags, desc, owner.name)
#   which dilutes title matches with partial desc/tag matches. The
#   title_match lane searches title + tags fields with higher weight,
#   ensuring docs whose title/tags match the query are always recalled.
#   Tags are included because they often contain entity names, movie
#   titles, and topic keywords that are as strong a signal as the title.
#
# Performance: parallel execution means wall-clock ≈ slowest single lane
WORD_RECALL_LANES = {
    "relevance": {
        "sort": None,  # Default _score sorting
        "limit": 600,
        "desc": "BM25 keyword relevance — text match quality",
    },
    "title_match": {
        "sort": None,  # _score sorting on title+tags fields
        "limit": 400,
        "title_tags": True,  # Special flag: search title + tags fields only
        "desc": "Title+tags BM25 — high precision entity/name matching",
    },
    "owner_name": {
        "sort": None,  # _score sorting on owner.name + title fields
        "limit": 200,
        "owner_name": True,  # Special flag: search owner.name + title
        "desc": "Owner name BM25 — UP主名称精确匹配",
    },
    "popularity": {
        "sort": [{"stat.view": "desc"}],
        "limit": 300,
        "desc": "Most viewed — raw reach and awareness",
    },
    "recency": {
        "sort": [{"pubdate": "desc"}],
        "limit": 250,
        "desc": "Most recent — freshness and timeliness",
    },
    "quality": {
        "sort": [{"stat_score": "desc"}],
        "limit": 300,
        "desc": "Highest quality — composite DocScorer stat quality",
    },
}

# Default lanes to run in parallel
DEFAULT_LANES = [
    "relevance",
    "title_match",
    "owner_name",
    "popularity",
    "recency",
    "quality",
]


class MultiLaneWordRecall:
    """Multi-lane word-based recall strategy.

    Runs multiple ES BM25 searches in parallel, each sorted by a
    different signal, then merges into a single deduplicated pool.

    This guarantees that the recall pool contains candidates from
    all five dimensions (relevant, title-matched, popular, recent, quality),
    which the downstream ranker can then draw from for diversified ranking.

    The five lanes are intentionally distinct:
    - relevance: text match quality (BM25 _score) across all fields
    - title_match: title+tags BM25 for entity/name precision
    - popularity: raw reach/awareness (view count)
    - recency: freshness (publish date)
    - quality: content quality (stat_score combining engagement ratios,
      anomaly detection, and balanced stat signals from DocScorer)

    Title-match tagging:
    After recall, all hits are tagged with `_title_matched` based on
    whether the query terms appear in the document title or tags. This
    signal is used by the downstream ranker to boost relevance scoring.

    Example:
        >>> recall = MultiLaneWordRecall()
        >>> pool = recall.recall(
        ...     searcher=video_searcher,
        ...     query="黑神话",
        ...     lanes=["relevance", "title_match", "popularity", "recency", "quality"],
        ... )
        >>> len(pool.hits)  # ~500-800 after dedup
        650
    """

    def __init__(
        self,
        lanes_config: dict = None,
        max_workers: int = 6,
    ):
        """Initialize multi-lane recall.

        Args:
            lanes_config: Override lane configurations. Defaults to WORD_RECALL_LANES.
            max_workers: Max parallel threads for ES queries (5 lanes).
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
            lane_config: Configuration for this lane (sort, limit, title_only).
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
        title_tags = lane_config.get("title_tags", False)
        owner_name = lane_config.get("owner_name", False)

        try:
            if title_tags:
                # Title+tags lane: search title and tags for high precision
                res = self._search_title_tags(
                    searcher=searcher,
                    query=query,
                    source_fields=source_fields,
                    extra_filters=extra_filters,
                    suggest_info=suggest_info,
                    limit=limit,
                    timeout=timeout,
                )
            elif owner_name:
                # Owner name lane: search owner.name + title for UP主 matching
                res = self._search_owner_name(
                    searcher=searcher,
                    query=query,
                    source_fields=source_fields,
                    extra_filters=extra_filters,
                    suggest_info=suggest_info,
                    limit=limit,
                    timeout=timeout,
                )
            elif sort_spec is not None:
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

    def _search_title_tags(
        self,
        searcher,
        query: str,
        source_fields: list[str],
        extra_filters: list[dict],
        suggest_info: dict,
        limit: int,
        timeout: float,
    ) -> dict:
        """Execute a BM25 search using title + tags fields only.

        This lane is designed for entity/name queries like '通义实验室',
        '飓风营救', '红警08' where matching in the title or tags is a strong
        signal of relevance. By restricting to title+tags fields, we avoid
        dilution from partial matches in desc/owner.name while capturing
        entity names and topic keywords from both title and tags.

        Uses title.words + tags.words with es_tok_query_string for best precision.

        Args:
            searcher: VideoSearcherV2 instance.
            query: Search query.
            source_fields: Fields to retrieve.
            extra_filters: Filter clauses.
            suggest_info: Suggestion info.
            limit: Max results.
            timeout: Timeout in seconds.

        Returns:
            Parsed search result dict.
        """
        from elastics.videos.constants import SEARCH_MATCH_TYPE, TERMINATE_AFTER
        from elastics.structure import construct_boosted_fields, set_timeout

        # Title + tags fields with appropriate boosts
        # Title gets highest boost (most specific signal for entity queries)
        # Tags get moderate boost (often contain entity names and topic keywords)
        title_tags_match_fields = ["title.words", "tags.words"]
        title_tags_boosted_fields = {
            "title.words": 5.0,  # Title is strongest signal
            "tags.words": 3.0,  # Tags capture entity names and topics
            "title.pinyin": 0.5,  # Pinyin fallback for romanized input
        }

        boosted_match_fields, boosted_date_fields = construct_boosted_fields(
            match_fields=title_tags_match_fields,
            boost=True,
            boosted_fields=title_tags_boosted_fields,
        )
        _, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
            query=query,
            suggest_info=suggest_info,
            boosted_match_fields=boosted_match_fields,
            boosted_date_fields=boosted_date_fields,
            match_type=SEARCH_MATCH_TYPE,
            extra_filters=extra_filters,
        )

        search_body = {
            "query": query_dsl_dict,
            "_source": source_fields,
            "track_total_hits": True,
            "size": limit,
            "terminate_after": min(TERMINATE_AFTER, 500000),
        }
        search_body = set_timeout(search_body, timeout=timeout)

        es_res_dict = searcher.submit_to_es(search_body, context="recall_title_tags")

        query_info = searcher.query_rewriter.get_query_info(query)
        parse_res = searcher.hit_parser.parse(
            query_info,
            match_fields=title_tags_match_fields,
            res_dict=es_res_dict,
            request_type="search",
            drop_no_highlights=False,
            add_region_info=False,
            add_highlights_info=False,
            match_type=SEARCH_MATCH_TYPE,
            limit=limit,
            verbose=False,
        )

        # Tag all hits from this lane as title-matched
        for hit in parse_res.get("hits", []):
            hit["_title_matched"] = True

        return parse_res

    def _search_owner_name(
        self,
        searcher,
        query: str,
        source_fields: list[str],
        extra_filters: list[dict],
        suggest_info: dict,
        limit: int,
        timeout: float,
    ) -> dict:
        """Execute a BM25 search emphasizing owner.name + title fields.

        This lane is designed for queries that target specific creators/UP主,
        like '红警08' (owner '红警HBK08'), '通义实验室' (owner '通义大模型').
        By boosting owner.name heavily, docs from matching creators are
        strongly preferred even if title matches are partial.

        Args:
            searcher: VideoSearcherV2 instance.
            query: Search query.
            source_fields: Fields to retrieve.
            extra_filters: Filter clauses.
            suggest_info: Suggestion info.
            limit: Max results.
            timeout: Timeout in seconds.

        Returns:
            Parsed search result dict.
        """
        from elastics.videos.constants import SEARCH_MATCH_TYPE, TERMINATE_AFTER
        from elastics.structure import construct_boosted_fields, set_timeout

        # Owner.name gets highest boost, title secondary
        owner_match_fields = ["owner.name.words", "title.words", "tags.words"]
        owner_boosted_fields = {
            "owner.name.words": 8.0,  # Owner name is the primary signal
            "owner.name.pinyin": 2.0,  # Pinyin fallback for romanized input
            "title.words": 3.0,  # Title for context
            "tags.words": 2.0,  # Tags often contain owner names
        }

        boosted_match_fields, boosted_date_fields = construct_boosted_fields(
            match_fields=owner_match_fields,
            boost=True,
            boosted_fields=owner_boosted_fields,
        )
        _, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
            query=query,
            suggest_info=suggest_info,
            boosted_match_fields=boosted_match_fields,
            boosted_date_fields=boosted_date_fields,
            match_type=SEARCH_MATCH_TYPE,
            extra_filters=extra_filters,
        )

        search_body = {
            "query": query_dsl_dict,
            "_source": source_fields,
            "track_total_hits": True,
            "size": limit,
            "terminate_after": min(TERMINATE_AFTER, 500000),
        }
        search_body = set_timeout(search_body, timeout=timeout)

        es_res_dict = searcher.submit_to_es(search_body, context="recall_owner_name")

        query_info = searcher.query_rewriter.get_query_info(query)
        parse_res = searcher.hit_parser.parse(
            query_info,
            match_fields=owner_match_fields,
            res_dict=es_res_dict,
            request_type="search",
            drop_no_highlights=False,
            add_region_info=False,
            add_highlights_info=False,
            match_type=SEARCH_MATCH_TYPE,
            limit=limit,
            verbose=False,
        )

        # Tag all hits from this lane as owner-name matched
        for hit in parse_res.get("hits", []):
            hit["_owner_lane"] = True

        return parse_res

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
            lanes: Which lanes to run. Defaults to all five.
            timeout: Timeout per lane in seconds.
            verbose: Enable verbose logging.

        Returns:
            RecallPool with merged, deduplicated candidates.
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

        # Tag title matches for all hits in the merged pool
        self._tag_title_matches(pool.hits, query)

        # Tag owner matches for all hits in the merged pool
        self._tag_owner_matches(pool.hits, query)

        pool.took_ms = round((time.perf_counter() - start) * 1000, 2)

        if verbose:
            title_matched = sum(1 for h in pool.hits if h.get("_title_matched"))
            logger.mesg(
                f"  Multi-lane recall: {len(pool.hits)} unique candidates "
                f"({title_matched} title-matched) "
                f"from {len(results)} lanes ({pool.took_ms:.0f}ms)"
            )

        return pool

    @staticmethod
    def _tag_title_matches(hits: list[dict], query: str) -> None:
        """Tag hits with _title_matched if query terms appear in title or tags.

        This is a lightweight character-level check (not tokenization).
        A hit is tagged if ALL significant query terms (length >= 2) appear
        in the title OR tags. Single-character query terms are checked
        individually.

        Tags are included because they often contain entity names, movie
        titles, and topic keywords that are as strong a signal as the title.

        The _title_matched tag is used by the downstream ranker to apply
        a relevance bonus via TITLE_MATCH_BONUS.

        Args:
            hits: List of hit dicts to tag (modified in-place).
            query: Original query string.
        """
        if not query or not hits:
            return

        # Extract meaningful terms from query (skip single chars for multi-term queries)
        # Simple approach: split by spaces and common delimiters
        import re

        terms = re.split(r"[\s\-_,，。、/\\|]+", query.strip())
        terms = [t.lower() for t in terms if t]

        if not terms:
            return

        # For very short queries (1-2 chars total), do simple substring match
        query_clean = query.strip().lower()

        for hit in hits:
            # Skip if already tagged by title_match lane
            if hit.get("_title_matched"):
                continue

            title = (hit.get("title") or "").lower()
            tags = (hit.get("tags") or "").lower()
            # Combine title and tags for matching
            title_tags = title + " " + tags if tags else title

            if not title_tags.strip():
                continue

            if len(query_clean) <= 2:
                # Short query: simple substring match in title or tags
                hit["_title_matched"] = query_clean in title_tags
            else:
                # Multi-term: check if query substring appears in title or tags
                hit["_title_matched"] = query_clean in title_tags or all(
                    t in title_tags for t in terms if len(t) >= 2
                )

    @staticmethod
    def _tag_owner_matches(hits: list[dict], query: str) -> None:
        """Tag hits with _owner_matched if query tokens match the owner name.

        Detects "owner intent" — when the user searches for a specific creator
        by checking if all meaningful tokens in the query appear in owner.name.
        For example, '红警08' matches owner '红警HBK08' because both '红警'
        and '08' appear in the owner name.

        Also stores the matched owner names in each hit for downstream use
        by the diversified ranker.

        Args:
            hits: List of hit dicts to tag (modified in-place).
            query: Original query string.
        """
        if not query or not hits:
            return

        import re

        # Tokenize query into meaningful chunks (CJK, alpha, numeric)
        query_tokens = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", query.lower()))
        if not query_tokens:
            return

        # Collect all unique owner names and their token sets
        owner_token_map: dict[str, set] = {}
        for hit in hits:
            owner = hit.get("owner")
            if isinstance(owner, dict):
                name = owner.get("name", "")
            else:
                name = ""
            if name and name not in owner_token_map:
                owner_token_map[name] = set(
                    re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", name.lower())
                )

        # Find owners whose token set contains ALL query tokens
        matching_owners = set()
        for name, name_tokens in owner_token_map.items():
            if query_tokens and query_tokens.issubset(name_tokens):
                matching_owners.add(name)

        if not matching_owners:
            return

        # Tag hits from matching owners
        for hit in hits:
            owner = hit.get("owner")
            if isinstance(owner, dict):
                name = owner.get("name", "")
            else:
                name = ""
            if name in matching_owners:
                hit["_owner_matched"] = True
                hit["_matched_owner_name"] = name
