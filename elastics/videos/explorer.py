import time

from tclogger import logger
from typing import Union

from converters.highlight.char_match import get_char_highlighter
from elastics.structure import build_auto_constraint_filter
from elastics.videos.constants import CONSTRAINT_FIELDS_DEFAULT, EXPLORE_TIMEOUT
from elastics.videos.constants import KNN_K, KNN_NUM_CANDIDATES, KNN_TIMEOUT
from elastics.videos.constants import KNN_TEXT_EMB_FIELD
from elastics.videos.constants import QMOD
from elastics.videos.searcher_v2 import VideoSearcherV2
from dsl.fields.qmod import (
    is_hybrid_qmod,
    has_rerank_qmod,
)

from ranks.constants import (
    RANK_METHOD_TYPE,
    RANK_METHOD,
    RANK_PREFER_TYPE,
    RANK_PREFER,
    AUTHOR_SORT_FIELD_TYPE,
    AUTHOR_SORT_FIELD,
    EXPLORE_RANK_TOP_K,
    EXPLORE_GROUP_OWNER_LIMIT,
    RERANK_MAX_HITS as KNN_RERANK_MAX_HITS,
    RERANK_KEYWORD_BOOST as KNN_RERANK_KEYWORD_BOOST,
    RERANK_TITLE_KEYWORD_BOOST as KNN_RERANK_TITLE_KEYWORD_BOOST,
)
from ranks.reranker import get_reranker
from ranks.grouper import AuthorGrouper
from recalls.manager import RecallManager

STEP_ZH_NAMES = {
    "most_relevant_search": {
        "name_zh": "搜索相关",
        "output_type": "hits",
    },
    "knn_search": {
        "name_zh": "向量搜索",
        "output_type": "hits",
    },
    "rerank": {
        "name_zh": "精排重排",
        "output_type": "info",
    },
    "hybrid_search": {
        "name_zh": "混合搜索",
        "output_type": "hits",
    },
    "group_hits_by_owner": {
        "name_zh": "UP主聚合",
        "output_type": "info",
    },
}


class StepBuilder:
    """Helper to build step result dicts with consistent format.

    Eliminates boilerplate in explore methods.

    Usage:
        steps = StepBuilder()
        with steps.step("most_relevant_search", input={...}) as s:
            result = self.search(...)
            s.set_output(result)
        return steps.finalize(query)
    """

    def __init__(self):
        self.steps: list[dict] = []
        self.step_idx: int = -1
        self.final_status: str = "finished"

    def add_step(
        self,
        name: str,
        status: str = "finished",
        input_data: dict = None,
        output: dict = None,
        comment: str = "",
    ) -> dict:
        """Add a step result."""
        self.step_idx += 1
        step = {
            "step": self.step_idx,
            "name": name,
            "name_zh": STEP_ZH_NAMES.get(name, {}).get("name_zh", name),
            "status": status,
            "input": input_data or {},
            "output_type": STEP_ZH_NAMES.get(name, {}).get("output_type", "info"),
            "output": output or {},
            "comment": comment,
        }
        self.steps.append(step)
        return step

    def update_step(self, step: dict, output: dict, status: str = "finished"):
        """Update the output and status of an existing step."""
        step["output"] = output
        if isinstance(output, dict) and output.get("timed_out"):
            step["status"] = "timedout"
            self.final_status = "timedout"
        else:
            step["status"] = status

    def finalize(self, query: str, **extra) -> dict:
        """Build final response dict."""
        result = {
            "query": query,
            "status": self.final_status,
            "data": self.steps,
        }
        result.update(extra)
        return result


class VideoExplorer(VideoSearcherV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recall_manager = RecallManager()

    def get_user_docs(self, mids: list[str]) -> dict:
        user_docs_list = self.mongo.get_docs(
            "users",
            ids=mids,
            id_field="mid",
            include_fields=["mid", "name", "face"],
        )
        user_docs_dict = {user_doc["mid"]: user_doc for user_doc in user_docs_list}
        return user_docs_dict

    def group_hits_by_owner(
        self,
        search_res: dict,
        sort_field: AUTHOR_SORT_FIELD_TYPE = AUTHOR_SORT_FIELD,
        limit: int = 25,
    ) -> list[dict]:
        """Group hits by owner (UP主) and sort by specified field.

        Uses the AuthorGrouper class from ranks module.

        IMPORTANT: Returns a LIST (not dict) to ensure order preservation
        across JSON serialization/deserialization (network transport to frontend).

        Args:
            search_res: Search result containing hits list.
            sort_field: Field to sort grouped authors by:
                - "first_appear_order": Order by when author first appears in hits (default)
                  This ensures author order matches the video list order.
                - "top_rank_score": Max rank_score among author's videos
                - "sum_rank_score": Sum of rank_scores
                - Other options: sum_count, sum_view, sum_sort_score
            limit: Max number of author groups to return.

        Returns:
            List of author groups, sorted by sort_field.
            Order is guaranteed to be preserved in JSON transport.
        """
        # Use AuthorGrouper from ranks module - return as list for JSON order preservation
        grouper = AuthorGrouper()
        authors_list = grouper.group_from_search_result_as_list(
            search_res=search_res,
            sort_field=sort_field,
            limit=limit,
        )

        # Add user faces from MongoDB
        mids = [author.get("mid") for author in authors_list]
        user_docs = self.get_user_docs(mids)
        grouper.add_user_faces_to_list(authors_list, user_docs)

        return authors_list

    def merge_scores_into_hits(
        self,
        full_hits: list[dict],
        score_hits: list[dict],
        score_fields: list[str] = None,
    ) -> list[dict]:
        """Merge score fields from score_hits into full_hits by bvid.

        This is used to preserve original scores (from KNN/hybrid search)
        when fetching full documents by bvid.

        Args:
            full_hits: Full document hits (from bvid-based search).
            score_hits: Hits with score information (from KNN/hybrid search).
            score_fields: List of score field names to merge.
                         Defaults to ["score", "hybrid_score", "word_rank", "knn_rank"].

        Returns:
            full_hits with score fields merged in.
        """
        if not score_hits:
            return full_hits

        if score_fields is None:
            score_fields = [
                "score",
                "hybrid_score",
                "word_rank",
                "knn_rank",
                "rank_score",
                "sort_score",
                # Recall metadata fields — critical for downstream ranking
                "_title_matched",  # Title-match bonus in diversified ranker
                "_owner_matched",  # Owner-match bonus in diversified ranker
                "_matched_owner_name",  # For debugging/analysis
                "_owner_lane",  # From owner_name recall lane
                "_recall_lanes",  # Lane membership for multi-lane analysis
            ]

        # Build bvid -> score data mapping
        score_map = {}
        for hit in score_hits:
            bvid = hit.get("bvid")
            if bvid:
                score_data = {}
                for field in score_fields:
                    if field in hit:
                        score_data[field] = hit[field]
                if score_data:
                    score_map[bvid] = score_data

        # Merge scores into full_hits
        for hit in full_hits:
            bvid = hit.get("bvid")
            if bvid and bvid in score_map:
                hit.update(score_map[bvid])

        return full_hits

    # =========================================================================
    # Filter-Only Explore (no search keywords)
    # =========================================================================

    def _filter_only_explore(
        self,
        query: str,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        verbose: bool = False,
        rank_top_k: int = EXPLORE_RANK_TOP_K,
        group_owner_limit: int = EXPLORE_GROUP_OWNER_LIMIT,
    ) -> dict:
        """Handle queries with no search keywords (filter-only).

        For queries like "user:123" or "date>2024" with no text keywords,
        there's no relevance signal. Results are ranked by stats + recency.

        This replaces the legacy explore() method for the no-keywords case.

        Args:
            query: Query string (contains only filter expressions).
            extra_filters: Additional filter clauses.
            suggest_info: Suggestion info for query rewriting.
            verbose: Enable verbose logging.
            rank_top_k: Max results to return.
            group_owner_limit: Max author groups.

        Returns:
            dict: {"query": str, "status": str, "data": list[dict]}
        """
        logger.enter_quiet(not verbose)
        perf = {"total_ms": 0}
        explore_start = time.perf_counter()
        steps = StepBuilder()

        # Step 0: Search with filter-only (no keyword matching)
        step = steps.add_step(
            "most_relevant_search",
            status="running",
            input_data={"query": query, "filter_only": True},
        )
        logger.hint("> [step 0] Filter-only search ...", verbose=verbose)

        search_res = self.search(
            query=query,
            extra_filters=extra_filters,
            suggest_info=suggest_info,
            parse_hits=True,
            add_region_info=True,
            add_highlights_info=False,
            is_highlight=False,
            boost=True,
            rank_method="heads",
            limit=rank_top_k * 2,
            rank_top_k=rank_top_k,
            timeout=EXPLORE_TIMEOUT,
            verbose=verbose,
        )

        # Apply filter-only ranking (stats + recency, no relevance)
        search_res = self.hit_ranker.filter_only_rank(search_res, top_k=rank_top_k)
        search_res["filter_only"] = True
        steps.update_step(step, search_res)

        if not search_res.get("hits"):
            steps.add_step(
                "group_hits_by_owner",
                output={"authors": []},
                comment="无搜索结果",
            )
            perf["total_ms"] = round((time.perf_counter() - explore_start) * 1000, 2)
            logger.exit_quiet(not verbose)
            return steps.finalize(query, perf=perf)

        # Step 1: Group by owner
        logger.hint("> [step 1] Group by owner ...", verbose=verbose)
        group_res = self._build_group_step(search_res, group_owner_limit)
        steps.add_step(
            "group_hits_by_owner",
            output={"authors": group_res},
            input_data={"limit": group_owner_limit},
        )

        perf["total_ms"] = round((time.perf_counter() - explore_start) * 1000, 2)
        logger.exit_quiet(not verbose)
        return steps.finalize(query, perf=perf)

    # =========================================================================
    # V2 Explore Methods (multi-lane recall + diversified ranking)
    # =========================================================================

    def _fetch_and_rank(
        self,
        recall_hits: list[dict],
        query: str,
        rank_method: RANK_METHOD_TYPE,
        rank_top_k: int,
        prefer: RANK_PREFER_TYPE,
        enable_rerank: bool = False,
        rerank_max_hits: int = KNN_RERANK_MAX_HITS,
        rerank_keyword_boost: float = KNN_RERANK_KEYWORD_BOOST,
        rerank_title_keyword_boost: float = KNN_RERANK_TITLE_KEYWORD_BOOST,
        extra_filters: list[dict] = [],
        verbose: bool = False,
        pool_hints: object = None,
    ) -> tuple[dict, dict]:
        """Fetch full docs for recall hits, optionally rerank, then rank.

        Shared logic for all explore variants. Returns (search_res, rerank_info).

        Args:
            recall_hits: Hit dicts with at least 'bvid' from recall.
            query: Original query string.
            rank_method: Ranking method to apply.
            rank_top_k: Max results after ranking.
            prefer: Ranking preference mode.
            enable_rerank: Whether to apply embedding reranking.
            rerank_max_hits: Max hits to rerank.
            extra_filters: Extra filter clauses.
            verbose: Verbose logging.

        Returns:
            Tuple of (search_res, rerank_info).
        """
        bvids = [h.get("bvid") for h in recall_hits if h.get("bvid")]

        # Fetch full documents
        start = time.perf_counter()
        full_doc_res = self.fetch_docs_by_bvids(
            bvids=bvids,
            add_region_info=True,
            limit=len(bvids),
            timeout=EXPLORE_TIMEOUT,
            verbose=verbose,
        )
        fetch_ms = round((time.perf_counter() - start) * 1000, 2)

        full_hits = full_doc_res.get("hits", [])
        # Merge recall scores into full docs
        self.merge_scores_into_hits(full_hits, recall_hits)

        # Parse query info (needed for both reranking and highlighting)
        query_info = self.query_rewriter.get_query_info(query)
        keywords_body = query_info.get("keywords_body", [])
        constraint_texts = query_info.get("constraint_texts", [])
        # Include constraint texts in embed_text for semantic context,
        # but fall back to raw query if no keywords at all.
        all_embed_parts = keywords_body + constraint_texts
        embed_text = " ".join(all_embed_parts) if all_embed_parts else query

        # Optional reranking
        rerank_info = {}
        if enable_rerank:
            reranker = get_reranker()
            if reranker.is_available():
                rerank_expr_tree = query_info.get("query_expr_tree")
                start = time.perf_counter()
                reranked_hits, rerank_perf = reranker.rerank(
                    query=embed_text,
                    hits=full_hits[:rerank_max_hits],
                    keywords=keywords_body,
                    expr_tree=rerank_expr_tree,
                    keyword_boost=rerank_keyword_boost,
                    title_keyword_boost=rerank_title_keyword_boost,
                    max_rerank=rerank_max_hits,
                    score_field="rerank_score",
                    verbose=verbose,
                )
                rerank_ms = round((time.perf_counter() - start) * 1000, 2)
                if len(full_hits) > rerank_max_hits:
                    reranked_hits.extend(full_hits[rerank_max_hits:])
                full_doc_res["hits"] = reranked_hits
                full_hits = reranked_hits
                rerank_info = {
                    "reranked_count": min(len(full_hits), rerank_max_hits),
                    "rerank_ms": rerank_ms,
                    "perf": rerank_perf,
                }

        # Apply ranking (trims to rank_top_k)
        full_doc_res["hits"] = full_hits
        full_doc_res = self.hit_ranker.rank(
            full_doc_res,
            method=rank_method,
            top_k=rank_top_k,
            prefer=prefer,
            query=query,
            pool_hints=pool_hints,
        )

        # Add char-level highlighting AFTER ranking — only highlights the
        # final ranked results instead of all recall candidates. This is safe
        # because neither the reranker nor the ranker reads highlight fields.
        highlight_start = time.perf_counter()
        ranked_hits = full_doc_res.get("hits", [])
        char_highlighter = get_char_highlighter()
        char_highlighter.add_highlights_to_hits(
            hits=ranked_hits,
            keywords=embed_text,
            fields=["title", "tags", "desc", "owner.name"],
            tag="hit",
        )
        highlight_ms = round((time.perf_counter() - highlight_start) * 1000, 2)

        full_doc_res["fetch_ms"] = fetch_ms
        full_doc_res["highlight_ms"] = highlight_ms

        return full_doc_res, rerank_info

    def _build_group_step(
        self,
        search_res: dict,
        group_owner_limit: int,
        group_sort_field: str = "first_appear_order",
    ) -> tuple[list[dict], dict]:
        """Build author group step result.

        Args:
            search_res: Search result to group.
            group_owner_limit: Max author groups.
            group_sort_field: Sort field for author groups.

        Returns:
            Tuple of (authors_list, group_step_output).
        """
        group_res = self.group_hits_by_owner(
            search_res=search_res,
            sort_field=group_sort_field,
            limit=group_owner_limit,
        )
        return group_res

    def _run_explore_pipeline(
        self,
        query: str,
        recall_mode: str,
        step_name: str,
        extra_filters: list[dict] = [],
        constraint_filter: dict = None,
        suggest_info: dict = {},
        verbose: bool = False,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD,
        rank_top_k: int = EXPLORE_RANK_TOP_K,
        group_owner_limit: int = EXPLORE_GROUP_OWNER_LIMIT,
        prefer: RANK_PREFER_TYPE = RANK_PREFER,
        enable_rerank: bool = False,
        rerank_max_hits: int = KNN_RERANK_MAX_HITS,
        rerank_keyword_boost: float = KNN_RERANK_KEYWORD_BOOST,
        rerank_title_keyword_boost: float = KNN_RERANK_TITLE_KEYWORD_BOOST,
        knn_field: str = KNN_TEXT_EMB_FIELD,
        recall_source_fields: list[str] = None,
        recall_timeout: float = EXPLORE_TIMEOUT,
    ) -> dict:
        """Shared pipeline for all explore variants.

        All explore methods follow the same structure:
        1. Check for keywords → filter_only_explore if none
        2. Run recall (word/vector/hybrid)
        3. Fetch full docs + optional rerank + ranking
        4. Group by author

        This method unifies the common logic to reduce code duplication.

        Args:
            query: Search query string.
            recall_mode: "word", "vector", or "hybrid".
            step_name: Step name for the main search step.
            extra_filters: Additional filter clauses.
            suggest_info: Suggestion info.
            verbose: Verbose logging.
            rank_method: Ranking method.
            rank_top_k: Max results after ranking.
            group_owner_limit: Max author groups.
            prefer: Ranking preference mode.
            enable_rerank: Whether to apply reranking.
            rerank_max_hits: Max hits to rerank.
            rerank_keyword_boost: Keyword boost factor.
            rerank_title_keyword_boost: Title keyword boost factor.
            knn_field: Dense vector field for KNN.
            recall_source_fields: Fields to retrieve in recall.
            recall_timeout: Timeout for recall operations.

        Returns:
            dict: {"query": str, "status": str, "data": list[dict]}
        """
        logger.enter_quiet(not verbose)
        perf = {"total_ms": 0}
        explore_start = time.perf_counter()
        steps = StepBuilder()

        # Check for keywords → fallback to filter-only
        if not self.has_search_keywords(query):
            logger.exit_quiet(not verbose)
            result = self._filter_only_explore(
                query=query,
                extra_filters=extra_filters,
                suggest_info=suggest_info,
                verbose=verbose,
                rank_top_k=rank_top_k,
                group_owner_limit=group_owner_limit,
            )
            # Tag result with recall mode if needed
            if recall_mode != "word" and result.get("data") and len(result["data"]) > 0:
                qmod_map = {
                    "vector": ["vector"],
                    "hybrid": ["word", "vector"],
                }
                result["data"][0]["output"]["qmod"] = qmod_map.get(recall_mode, [])
                result["data"][0]["output"]["filter_only"] = True
            return result

        # Step 0: Recall
        step = steps.add_step(
            step_name,
            status="running",
            input_data={"query": query, "recall_mode": recall_mode},
        )
        logger.hint(f"> [step 0] {recall_mode} recall ...", verbose=verbose)
        recall_start = time.perf_counter()

        # Build recall kwargs
        recall_kwargs = dict(
            searcher=self,
            query=query,
            mode=recall_mode,
            extra_filters=extra_filters,
            timeout=recall_timeout,
            verbose=verbose,
        )
        if constraint_filter:
            recall_kwargs["constraint_filter"] = constraint_filter
        if recall_source_fields:
            recall_kwargs["source_fields"] = recall_source_fields
        if suggest_info:
            recall_kwargs["suggest_info"] = suggest_info
        if recall_mode in ("vector", "hybrid"):
            recall_kwargs["knn_field"] = knn_field

        recall_pool = self.recall_manager.recall(**recall_kwargs)
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
            logger.exit_quiet(not verbose)
            return steps.finalize(query, perf=perf)

        # Step 1: Fetch full docs + rank
        logger.hint(
            f"> [step 1] Fetch & rank {len(recall_pool.hits)} candidates ...",
            verbose=verbose,
        )
        search_res, rerank_info = self._fetch_and_rank(
            recall_hits=recall_pool.hits,
            query=query,
            rank_method=rank_method,
            rank_top_k=rank_top_k,
            prefer=prefer,
            enable_rerank=enable_rerank,
            rerank_max_hits=rerank_max_hits,
            rerank_keyword_boost=rerank_keyword_boost,
            rerank_title_keyword_boost=rerank_title_keyword_boost,
            extra_filters=extra_filters,
            verbose=verbose,
            pool_hints=recall_pool.pool_hints,
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

        # Step 2: Group by owner
        logger.hint("> [step 2] Group by owner ...", verbose=verbose)
        group_res = self._build_group_step(search_res, group_owner_limit)
        steps.add_step(
            "group_hits_by_owner",
            output={"authors": group_res},
            input_data={"limit": group_owner_limit},
        )

        perf["total_ms"] = round((time.perf_counter() - explore_start) * 1000, 2)
        logger.exit_quiet(not verbose)
        return steps.finalize(query, perf=perf)

    def explore_v2(
        self,
        query: str,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        verbose: bool = False,
        # Recall & ranking params
        rank_method: RANK_METHOD_TYPE = RANK_METHOD,
        rank_top_k: int = EXPLORE_RANK_TOP_K,
        group_owner_limit: int = EXPLORE_GROUP_OWNER_LIMIT,
        prefer: RANK_PREFER_TYPE = RANK_PREFER,
        # Rerank params
        enable_rerank: bool = False,
        rerank_max_hits: int = KNN_RERANK_MAX_HITS,
        rerank_keyword_boost: float = KNN_RERANK_KEYWORD_BOOST,
        rerank_title_keyword_boost: float = KNN_RERANK_TITLE_KEYWORD_BOOST,
    ) -> dict:
        """V2 Word explore using multi-lane recall + diversified ranking.

        Uses 5 parallel recall lanes (relevance, title_match, popularity,
        recency, quality) for comprehensive candidate coverage, then
        three-phase diversified ranking (headline → slots → fused) for
        quality results.

        Returns:
            dict: {"query": str, "status": str, "data": list[dict]}
        """
        return self._run_explore_pipeline(
            query=query,
            recall_mode="word",
            step_name="most_relevant_search",
            extra_filters=extra_filters,
            suggest_info=suggest_info,
            verbose=verbose,
            rank_method=rank_method,
            rank_top_k=rank_top_k,
            group_owner_limit=group_owner_limit,
            prefer=prefer,
            enable_rerank=enable_rerank,
            rerank_max_hits=rerank_max_hits,
            rerank_keyword_boost=rerank_keyword_boost,
            rerank_title_keyword_boost=rerank_title_keyword_boost,
            recall_source_fields=[
                "bvid",
                "title",
                "tags",
                "desc",
                "stat",
                "pubdate",
                "duration",
                "stat_score",
            ],
        )

    def knn_explore_v2(
        self,
        query: str,
        extra_filters: list[dict] = [],
        constraint_filter: dict = None,
        verbose: bool = False,
        # KNN params
        knn_field: str = KNN_TEXT_EMB_FIELD,
        # Rerank params
        enable_rerank: bool = True,
        rerank_max_hits: int = KNN_RERANK_MAX_HITS,
        rerank_keyword_boost: float = KNN_RERANK_KEYWORD_BOOST,
        rerank_title_keyword_boost: float = KNN_RERANK_TITLE_KEYWORD_BOOST,
        # Explore params
        rank_method: RANK_METHOD_TYPE = "diversified",
        rank_top_k: int = EXPLORE_RANK_TOP_K,
        group_owner_limit: int = EXPLORE_GROUP_OWNER_LIMIT,
        prefer: RANK_PREFER_TYPE = RANK_PREFER,
    ) -> dict:
        """V2 KNN explore using vector recall + diversified ranking."""
        return self._run_explore_pipeline(
            query=query,
            recall_mode="vector",
            step_name="knn_search",
            extra_filters=extra_filters,
            constraint_filter=constraint_filter,
            verbose=verbose,
            rank_method=rank_method,
            rank_top_k=rank_top_k,
            group_owner_limit=group_owner_limit,
            prefer=prefer,
            enable_rerank=enable_rerank,
            rerank_max_hits=rerank_max_hits,
            rerank_keyword_boost=rerank_keyword_boost,
            rerank_title_keyword_boost=rerank_title_keyword_boost,
            knn_field=knn_field,
            recall_timeout=KNN_TIMEOUT,
        )

    def hybrid_explore_v2(
        self,
        query: str,
        extra_filters: list[dict] = [],
        constraint_filter: dict = None,
        suggest_info: dict = {},
        verbose: bool = False,
        # KNN params
        knn_field: str = KNN_TEXT_EMB_FIELD,
        # Rerank params
        enable_rerank: bool = False,
        rerank_max_hits: int = KNN_RERANK_MAX_HITS,
        rerank_keyword_boost: float = KNN_RERANK_KEYWORD_BOOST,
        rerank_title_keyword_boost: float = KNN_RERANK_TITLE_KEYWORD_BOOST,
        # Explore params
        rank_method: RANK_METHOD_TYPE = "diversified",
        rank_top_k: int = EXPLORE_RANK_TOP_K,
        group_owner_limit: int = EXPLORE_GROUP_OWNER_LIMIT,
        prefer: RANK_PREFER_TYPE = RANK_PREFER,
    ) -> dict:
        """V2 Hybrid explore using combined word+vector recall + diversified ranking."""
        return self._run_explore_pipeline(
            query=query,
            recall_mode="hybrid",
            step_name="hybrid_search",
            extra_filters=extra_filters,
            constraint_filter=constraint_filter,
            suggest_info=suggest_info,
            verbose=verbose,
            rank_method=rank_method,
            rank_top_k=rank_top_k,
            group_owner_limit=group_owner_limit,
            prefer=prefer,
            enable_rerank=enable_rerank,
            rerank_max_hits=rerank_max_hits,
            rerank_keyword_boost=rerank_keyword_boost,
            rerank_title_keyword_boost=rerank_title_keyword_boost,
            knn_field=knn_field,
            recall_source_fields=[
                "bvid",
                "stat",
                "pubdate",
                "duration",
                "stat_score",
                "title",
                "tags",
                "desc",
                "owner",
            ],
        )

    def unified_explore(
        self,
        query: str,
        qmod: Union[str, list[str]] = None,
        extra_filters: list[dict] = [],
        constraint_filter: dict = None,
        auto_constraint: bool = True,
        suggest_info: dict = {},
        verbose: bool = False,
        # Common explore params
        most_relevant_limit: int = 10000,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD,
        rank_top_k: int = EXPLORE_RANK_TOP_K,
        group_owner_limit: int = EXPLORE_GROUP_OWNER_LIMIT,
        # Ranking preference
        prefer: RANK_PREFER_TYPE = RANK_PREFER,
        # KNN/Hybrid specific params
        knn_field: str = KNN_TEXT_EMB_FIELD,
        knn_k: int = KNN_K,
        knn_num_candidates: int = KNN_NUM_CANDIDATES,
    ) -> dict:
        """Unified explore that automatically selects search method based on query mode.

        Now uses V2 methods with multi-lane recall and diversified ranking
        for better performance and result quality.

        The query mode can be specified via:
        1. The qmod parameter (str or list[str])
        2. DSL expression in query (e.g., "黑神话 q=v" or "q=wv" or "q=wvr")

        Query modes:
        - w: word-based retrieval (ES text search)
        - v: vector-based KNN retrieval (embedding similarity)
        - r: rerank with float embeddings for precise similarity

        Valid combinations (must have at least w or v):
        - q=w: word only (multi-lane recall + diversified ranking)
        - q=v: vector only (KNN + word supplement + rerank)
        - q=wv: hybrid word+vector
        - q=wr: word + rerank
        - q=vr: vector + rerank
        - q=wvr: hybrid + rerank

        Args:
            query: Query string.
            qmod: Override mode(s).
            extra_filters: Additional filter clauses.
            constraint_filter: Explicit constraint filter. If provided, used
                as-is. If None and auto_constraint is True, one is built
                automatically from the query's tokenization.
            auto_constraint: When True and constraint_filter is None, auto-
                build a constraint filter from query tokens using the index
                tokenizer. This dramatically improves precision for compound
                queries by ensuring KNN results contain the key terms.
            suggest_info: Suggestion info.
            verbose: Enable verbose logging.
            most_relevant_limit: Max docs for searches.
            rank_method: Ranking method (default: "diversified").
            rank_top_k: Top-k for ranking.
            group_owner_limit: Max owner groups.
            prefer: Ranking preference.
            knn_field: Dense vector field.
            knn_k: KNN neighbors count.
            knn_num_candidates: KNN candidates per shard.

        Returns:
            Explore results dict (qmod is in first step_result's output).
        """
        from dsl.fields.qmod import normalize_qmod

        # Extract query mode from query if not provided
        if qmod is None:
            qmod = self.get_qmod_from_query(query)
        else:
            qmod = normalize_qmod(qmod)

        logger.hint(f"> Query mode (qmod): {qmod}", verbose=verbose)

        # Check mode flags
        is_hybrid = is_hybrid_qmod(qmod)
        has_word = "word" in qmod
        has_vector = "vector" in qmod
        enable_rerank = has_rerank_qmod(qmod)

        # Graceful fallback: if vector/hybrid mode but embed client is
        # unavailable, degrade to word-only to avoid hanging
        if has_vector and not self.embed_client.is_available():
            logger.warn(
                f"> Embed client unavailable, falling back from "
                f"qmod={qmod} to word-only mode"
            )
            qmod = ["word"]
            is_hybrid = False
            has_word = True
            has_vector = False

        # Auto-build constraint filter for vector/hybrid modes
        if constraint_filter is None and auto_constraint and has_vector:
            try:
                # Extract raw keywords (without DSL modifiers like q=v)
                query_info = self.query_rewriter.get_query_info(query)

                # Use DSL-specified constraint_filter (+/-tokens) if available
                dsl_constraint = query_info.get("constraint_filter", {})
                if dsl_constraint:
                    constraint_filter = dsl_constraint
                    logger.hint(
                        f"> DSL constraint: {constraint_filter}",
                        verbose=verbose,
                    )
                else:
                    # Fall back to auto-building from keywords
                    keywords_body = query_info.get("keywords_body", [])
                    raw_query = " ".join(keywords_body) if keywords_body else ""
                    if raw_query:
                        constraint_filter = build_auto_constraint_filter(
                            es_client=self.es.client,
                            index_name=self.index_name,
                            query=raw_query,
                            fields=CONSTRAINT_FIELDS_DEFAULT,
                        )
                        if constraint_filter:
                            logger.hint(
                                f"> Auto-constraint: {constraint_filter}",
                                verbose=verbose,
                            )
            except Exception as e:
                logger.warn(f"Auto-constraint build failed: {e}")

        if is_hybrid:
            # Hybrid mode: combined word+vector recall + diversified ranking
            result = self.hybrid_explore_v2(
                query=query,
                extra_filters=extra_filters,
                constraint_filter=constraint_filter,
                suggest_info=suggest_info,
                verbose=verbose,
                knn_field=knn_field,
                rank_method=rank_method,
                rank_top_k=rank_top_k,
                group_owner_limit=group_owner_limit,
                enable_rerank=enable_rerank,
                prefer=prefer,
            )
            if result.get("data") and len(result["data"]) > 0:
                result["data"][0]["output"]["qmod"] = qmod
            return result

        elif has_vector:
            # Vector-only mode: KNN recall + rerank + diversified ranking
            result = self.knn_explore_v2(
                query=query,
                extra_filters=extra_filters,
                constraint_filter=constraint_filter,
                verbose=verbose,
                knn_field=knn_field,
                rank_method=rank_method,
                rank_top_k=rank_top_k,
                group_owner_limit=group_owner_limit,
                enable_rerank=True,  # Always rerank for vector search
                prefer=prefer,
            )
            if result.get("data") and len(result["data"]) > 0:
                result["data"][0]["output"]["qmod"] = qmod
            return result

        else:
            # Word-only mode: multi-lane recall + diversified ranking
            result = self.explore_v2(
                query=query,
                extra_filters=extra_filters,
                suggest_info=suggest_info,
                verbose=verbose,
                rank_method=rank_method,
                rank_top_k=rank_top_k,
                group_owner_limit=group_owner_limit,
                enable_rerank=enable_rerank,
                prefer=prefer,
            )
            if result.get("data") and len(result["data"]) > 0:
                result["data"][0]["output"]["qmod"] = qmod
            return result
