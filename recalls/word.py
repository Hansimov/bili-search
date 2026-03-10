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
- Tags are included because they often contain entity names, movie titles,
  and topic keywords that are as strong a signal as the title itself.
- Relevance lane gets a larger limit because BM25 is the primary signal and
  we want broad keyword coverage in the candidate pool.

Performance: 5 parallel queries of ~200-600 docs each vs scanning 10000 docs
in one query. Wall-clock time is approximately the slowest single lane.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tclogger import logger

from recalls.base import NoiseFilter, RecallPool, RecallResult
from ranks.constants import MIN_BM25_SCORE


WORD_RECALL_LANES = {
    "relevance": {
        "sort": None,
        "limit": 600,
        "desc": "BM25 keyword relevance — text match quality",
    },
    "title_match": {
        "sort": None,
        "limit": 400,
        "title_tags": True,
        "desc": "Title+tags BM25 — high precision entity/name matching",
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

DEFAULT_LANES = [
    "relevance",
    "title_match",
    "popularity",
    "recency",
    "quality",
]


class MultiLaneWordRecall:
    """Multi-lane word-based recall strategy."""

    def __init__(self, lanes_config: dict = None, max_workers: int = 5):
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
        """Run a single recall lane."""
        start = time.perf_counter()
        sort_spec = lane_config.get("sort")
        limit = lane_config.get("limit", 200)
        title_tags = lane_config.get("title_tags", False)

        try:
            if title_tags:
                res = self._search_title_tags(
                    searcher=searcher,
                    query=query,
                    source_fields=source_fields,
                    extra_filters=extra_filters,
                    suggest_info=suggest_info,
                    limit=limit,
                    timeout=timeout,
                )
            elif sort_spec is not None:
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

            NoiseFilter.apply_content_quality_penalty(hits)
            hits = NoiseFilter.filter_by_score_ratio(
                hits,
                lane_name=lane_name,
                lane_tags=None,
            )

            return RecallResult(
                hits=hits,
                lane=lane_name,
                total_hits=res.get("total_hits", len(hits)),
                took_ms=took_ms,
                timed_out=res.get("timed_out", False),
            )
        except Exception as e:
            logger.warn(f"× Recall lane '{lane_name}' failed: {e}")
            return RecallResult(hits=[], lane=lane_name, took_ms=0, timed_out=True)

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
        """Execute a BM25 search emphasizing title and tags fields."""
        from elastics.structure import construct_boosted_fields, set_timeout
        from elastics.videos.constants import SEARCH_MATCH_TYPE, TERMINATE_AFTER

        title_tags_match_fields = ["title.words", "tags.words"]
        title_tags_boosted_fields = {
            "title.words": 5.0,
            "tags.words": 3.0,
            "title.pinyin": 0.5,
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

        for hit in parse_res.get("hits", []):
            hit["_title_matched"] = True

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
        """Execute a search with custom sort order."""
        from elastics.structure import construct_boosted_fields, set_timeout
        from elastics.videos.constants import (
            EXPLORE_BOOSTED_FIELDS,
            SEARCH_MATCH_FIELDS,
            SEARCH_MATCH_TYPE,
            TERMINATE_AFTER,
        )

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

        search_body = {
            "query": query_dsl_dict,
            "_source": source_fields,
            "sort": sort_spec + [{"_score": "desc"}],
            "track_total_hits": True,
            "size": limit,
            "terminate_after": min(TERMINATE_AFTER, 500000),
            "min_score": MIN_BM25_SCORE,
        }
        search_body = set_timeout(search_body, timeout=timeout)

        es_res_dict = searcher.submit_to_es(search_body, context=f"recall_{sort_spec}")

        query_info = searcher.query_rewriter.get_query_info(query)
        return searcher.hit_parser.parse(
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
        """Run multi-lane recall and merge results."""
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

        pool = RecallPool.merge(*results)
        self._tag_title_matches(pool.hits, query)
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
        """Tag hits with _title_matched if query terms appear in title or tags."""
        if not query or not hits:
            return

        import re

        raw_terms = re.split(r"[\s\-_,，。、/\\|]+", query.strip())
        terms = []
        for term in raw_terms:
            if not term:
                continue
            clean_term = term.lstrip("+!")
            if clean_term:
                terms.append(clean_term.lower())

        query_clean = " ".join(terms)

        for hit in hits:
            if hit.get("_title_matched"):
                continue

            title = (hit.get("title") or "").lower()
            tags = (hit.get("tags") or "").lower()
            title_tags = title + " " + tags if tags else title

            if not title_tags.strip():
                continue

            if len(query_clean) <= 2:
                hit["_title_matched"] = query_clean in title_tags
            else:
                hit["_title_matched"] = query_clean in title_tags or all(
                    term in title_tags for term in terms if len(term) >= 2
                )
