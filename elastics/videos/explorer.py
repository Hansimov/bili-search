import json

from copy import deepcopy
from tclogger import dict_get, get_by_threshold
from tclogger import logstr, logger, brk
from typing import Union, Literal

from converters.dsl.fields.bvid import bvids_to_filter
from elastics.videos.constants import SEARCH_MATCH_FIELDS, EXPLORE_BOOSTED_FIELDS
from elastics.videos.constants import SEARCH_MATCH_TYPE
from elastics.videos.constants import RANK_METHOD_TYPE, RANK_METHOD_DEFAULT
from elastics.videos.constants import AGG_TIMEOUT, EXPLORE_TIMEOUT
from elastics.videos.constants import TERMINATE_AFTER
from elastics.videos.constants import KNN_K, KNN_NUM_CANDIDATES, KNN_TIMEOUT
from elastics.videos.constants import KNN_TEXT_EMB_FIELD
from elastics.videos.constants import QMOD_SINGLE_TYPE, QMOD_DEFAULT
from elastics.videos.constants import HYBRID_RRF_K
from elastics.structure import construct_boosted_fields
from elastics.videos.searcher_v2 import VideoSearcherV2
from converters.dsl.fields.qmod import extract_qmod_from_expr_tree, is_hybrid_qmod

STEP_ZH_NAMES = {
    "init": {
        "name_zh": "初始化",
    },
    "construct_query_dsl_dict": {
        "name_zh": "解析查询",
        "output_type": "info",
    },
    "construct_knn_query": {
        "name_zh": "构建向量查询",
        "output_type": "info",
    },
    "aggregation": {
        "name_zh": "聚合",
        "output_type": "info",
    },
    "most_relevant_search": {
        "name_zh": "搜索相关",
        "output_type": "hits",
    },
    "knn_search": {
        "name_zh": "向量搜索",
        "output_type": "hits",
    },
    "hybrid_search": {
        "name_zh": "混合搜索",
        "output_type": "hits",
    },
    "most_popular_search": {
        "name_zh": "搜索热门",
        "output_type": "hits",
    },
    "group_hits_by_owner": {
        "name_zh": "UP主聚合",
        "output_type": "info",
    },
}


class VideoExplorer(VideoSearcherV2):
    def get_total_hits(self, agg_result: dict):
        return dict_get(agg_result, "hits.total.value", None)

    def get_stat_filter_by_threshold(
        self,
        agg_result: dict,
        field: Literal["view", "like", "coin", "favorite"],
        threshold: float = None,
        max_doc_count: int = None,
        res_format: Literal["dict", "tuple"] = "dict",
    ) -> Union[dict, tuple]:
        total_hits = self.get_total_hits(agg_result)
        if total_hits is None:
            logger.warn(f"× Not found total_hits")
            return None
        if max_doc_count is None and threshold is None:
            return {}
        if threshold is None:
            threshold = 0
        if max_doc_count is not None:
            if total_hits <= max_doc_count:
                return {}
            threshold = max((1 - max_doc_count / total_hits) * 100, threshold)
        stat_agg_dict = dict_get(agg_result, f"aggregations.{field}_ps.values", None)
        if stat_agg_dict is None:
            logger.warn(f"× Not found aggregation: {field}_ps")
            return None
        stat_key, stat_value = get_by_threshold(
            stat_agg_dict,
            threshold=threshold,
            direction="upper_bound",
            target="key",
        )
        if stat_key is None and stat_value is None:
            logger.warn(f"× Not found threshold: {threshold}")
            return None
        if res_format == "tuple":
            return stat_key, stat_value
        else:
            stat_filter = {"range": {f"stat.{field}": {"gte": int(stat_value)}}}
            return stat_filter

    def get_score_threshold_by_ratio(
        self,
        agg_result: dict,
        ratio: float = None,
        max_doc_count: int = None,
    ) -> float:
        """ratio: 0.0 - 1.0, means should be greater than ratio * max_score
        for example, if ratio=0.75, means should be greater than 75% of max_score
        """
        field = "score"
        total_hits = self.get_total_hits(agg_result)
        if total_hits is None:
            logger.warn(f"× Not found total_hits")
            return None
        if max_doc_count is None and ratio is None:
            return 0
        score_agg_dict = dict_get(agg_result, f"aggregations.{field}_ps.values", None)
        if score_agg_dict is None:
            logger.warn(f"× Not found aggregation: {field}_ps")
            return None
        try:
            max_score = max(score_agg_dict.values())
            min_score = min(score_agg_dict.values())
        except:
            return None
        if ratio is None:
            ratio = 0
        if max_doc_count is not None and max_doc_count < total_hits:
            doc_count_percentile = max_doc_count / total_hits * 100
            percent_by_count, _ = get_by_threshold(
                score_agg_dict,
                threshold=doc_count_percentile,
                direction="upper_bound",
                target="key",
            )
            ratio_by_count = percent_by_count / 100
            # ratio could be min/max of the constraints by max_doc_count and ratio,
            # - if use min, could ensure there are enough-but-not-too-much candidate docs
            # - if use max, would get less candidates, especially useful when the first recall docs are too many
            ratio = max(ratio, ratio_by_count)
        score_threshold = round(max(max_score * ratio, min_score), 4)
        return score_threshold

    def set_total_hits(self, agg_res: dict, search_res: dict) -> None:
        agg_total_hits = self.get_total_hits(agg_res)
        search_total_hits = search_res.get("total_hits", 0)
        if agg_total_hits is not None and search_total_hits is not None:
            if search_total_hits == TERMINATE_AFTER:
                search_res["total_hits"] = agg_total_hits
        return search_res

    def format_result(
        self, res: dict, res_format: Literal["json", "str"] = "json"
    ) -> Union[dict, str]:
        if res_format == "str":
            # used for EventSourceResponse
            return json.dumps(res)
        else:
            # used for normal response
            return res

    def update_step_output(
        self, step_yield: dict, step_output: dict = None, field: str = None
    ) -> dict:
        if isinstance(step_output, dict) and step_output.get("timed_out", None) is True:
            step_yield["status"] = "timedout"
        else:
            step_yield["status"] = "finished"
        if field is not None:
            step_yield["output"][field] = step_output
        else:
            step_yield["output"] = step_output
        return step_yield

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
        sort_field: Literal[
            "sum_count",
            "sum_view",
            "sum_sort_score",
            "sum_rank_score",
            "top_rank_score",
            "first_appear_order",
        ] = "first_appear_order",
        limit: int = 25,
    ) -> dict:
        """Group hits by owner (UP主) and sort by specified field.

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
            Dict of author groups keyed by mid.
        """
        group_res = {}
        first_appear_idx = {}  # Track first appearance index for each author

        for idx, hit in enumerate(search_res.get("hits", [])):
            name = dict_get(hit, "owner.name", None)
            mid = dict_get(hit, "owner.mid", None)
            pubdate = dict_get(hit, "pubdate") or 0
            view = dict_get(hit, "stat.view") or 0
            sort_score = dict_get(hit, "sort_score") or 0
            rank_score = dict_get(hit, "rank_score") or 0
            if mid is None or name is None:
                continue

            # Track first appearance index for this author
            if mid not in first_appear_idx:
                first_appear_idx[mid] = idx

            item = group_res.get(mid, None)
            if item is None:
                group_res[mid] = {
                    "mid": mid,
                    "name": name,
                    "latest_pubdate": pubdate,
                    "sum_view": view,
                    "sum_sort_score": sort_score,
                    "sum_rank_score": rank_score,
                    "top_rank_score": rank_score,
                    "first_appear_order": idx,  # Store first appearance index
                    "sum_count": 0,
                    "hits": [],
                }
            else:
                latest_pubdate = group_res[mid]["latest_pubdate"]
                if pubdate > latest_pubdate:
                    group_res[mid]["latest_pubdate"] = pubdate
                    group_res[mid]["name"] = name
                sum_view = group_res[mid]["sum_view"] or 0
                sum_sort_score = group_res[mid]["sum_sort_score"] or 0
                sum_rank_score = group_res[mid]["sum_rank_score"] or 0
                top_rank_score = group_res[mid]["top_rank_score"] or 0
                group_res[mid]["sum_view"] = sum_view + view
                group_res[mid]["sum_sort_score"] = sum_sort_score + sort_score
                group_res[mid]["sum_rank_score"] = sum_rank_score + rank_score
                group_res[mid]["top_rank_score"] = max(top_rank_score, rank_score)
            group_res[mid]["hits"].append(hit)
            group_res[mid]["sum_count"] += len(group_res[mid]["hits"])

        # sort by sort_field, and limit to top N
        # For first_appear_order, lower index = earlier appearance = higher priority
        if sort_field == "first_appear_order":
            sorted_items = sorted(
                group_res.items(), key=lambda item: item[1][sort_field], reverse=False
            )[:limit]
        else:
            sorted_items = sorted(
                group_res.items(), key=lambda item: item[1][sort_field], reverse=True
            )[:limit]
        group_res = dict(sorted_items)

        # add user faces
        mids = list(group_res.keys())
        user_docs = self.get_user_docs(mids)
        for mid, user_doc in user_docs.items():
            group_res[mid]["face"] = user_doc.get("face", "")
        return group_res

    def is_status_timedout(self, result: dict) -> bool:
        return result.get("status", None) == "timedout"

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

    def explore(
        self,
        # `query_dsl_dict` related params
        query: str,
        query_dsl_dict: dict = None,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        match_type: str = SEARCH_MATCH_TYPE,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        boost: bool = True,
        boosted_fields: dict = EXPLORE_BOOSTED_FIELDS,
        verbose: bool = False,
        # `explore` related params
        most_relevant_limit: int = 10000,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD_DEFAULT,
        rank_top_k: int = 400,
        group_owner_limit: int = 20,
    ) -> dict:
        """Explore and return all step results in a single response.

        Returns:
            dict: {
                "query": str,
                "status": "finished" | "timedout",
                "data": list[dict]  # list of step results
            }
        """
        logger.enter_quiet(not verbose)

        step_results = []  # Collect all step results
        final_status = "finished"

        # Step 0: Construct query_dsl_dict
        step_idx = 0
        step_name = "construct_query_dsl_dict"
        logger.hint("> [step 1] Query constructing ...")
        if query_dsl_dict is None:
            boosted_fields_params = {
                "match_fields": match_fields,
                "boost": boost,
                "boosted_fields": boosted_fields,
            }
            boosted_match_fields, boosted_date_fields = construct_boosted_fields(
                **boosted_fields_params
            )
            query_rewrite_dsl_params = {
                "query": query,
                "suggest_info": suggest_info,
                "boosted_match_fields": boosted_match_fields,
                "boosted_date_fields": boosted_date_fields,
                "match_type": match_type,
                "extra_filters": extra_filters,
            }
            _, _, query_dsl_dict = self.get_info_of_query_rewrite_dsl(
                **query_rewrite_dsl_params
            )
        query_dsl_dict_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "finished",
            "input": query_rewrite_dsl_params,
            "output_type": STEP_ZH_NAMES[step_name]["output_type"],
            "output": query_dsl_dict,
            "comment": "",
        }
        step_results.append(query_dsl_dict_result)

        # Step 1: Most-relevant docs
        step_idx += 1
        step_name = "most_relevant_search"
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(f"> {step_str} Top relevant docs")
        relevant_search_params = {
            "query": query,
            "suggest_info": suggest_info,
            "source_fields": ["bvid", "stat", "pubdate", "duration"],  # reduce io
            "extra_filters": extra_filters,
            "use_script_score": False,  # speed up
            "rank_method": rank_method,  # better ranking
            "add_region_info": False,  # speed up
            "add_highlights_info": False,  # speed up
            "is_profile": False,
            "is_highlight": False,  # speed up
            "limit": most_relevant_limit,
            "rank_top_k": rank_top_k,  # final returned docs with partial fields
            "timeout": EXPLORE_TIMEOUT,
            "verbose": verbose,
        }
        relevant_search_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "running",
            "input": relevant_search_params,
            "output": {},
            "output_type": "hits",
            "comment": "",
        }
        relevant_search_res = self.search(**relevant_search_params)
        self.update_step_output(relevant_search_result, step_output=relevant_search_res)
        if self.is_status_timedout(relevant_search_result):
            final_status = "timedout"
            step_results.append(relevant_search_result)
            logger.exit_quiet(not verbose)
            return {"query": query, "status": final_status, "data": step_results}

        # Step 3: Fetch full-docs by return ranked ids
        full_doc_search_params = deepcopy(relevant_search_params)
        bvids = [hit.get("bvid", None) for hit in relevant_search_res.get("hits", [])]
        bvid_filter = bvids_to_filter(bvids)
        full_doc_search_params.pop("source_fields", None)
        full_doc_search_params.update(
            {
                "rank_method": rank_method,  # same with relevant search
                "is_highlight": True,
                "add_region_info": True,
                "add_highlights_info": True,
                "extra_filters": extra_filters + [bvid_filter],
                "limit": len(bvids),
            }
        )
        full_doc_search_res = self.search(**full_doc_search_params)
        full_doc_search_res["total_hits"] = relevant_search_res.get("total_hits", 0)
        self.update_step_output(relevant_search_result, step_output=full_doc_search_res)
        if self.is_status_timedout(relevant_search_result):
            final_status = "timedout"
            step_results.append(relevant_search_result)
            logger.exit_quiet(not verbose)
            return {"query": query, "status": final_status, "data": step_results}
        step_results.append(relevant_search_result)

        # Step 4: Group hits by owner
        step_idx += 1
        step_name = "group_hits_by_owner"
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(f"> {step_str} Group hits by owner, sort by sum of view")
        group_hits_by_owner_params = {
            "search_res": full_doc_search_res,
            "limit": group_owner_limit,
        }
        group_hits_by_owner_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "running",
            "input": {"limit": group_owner_limit},
            "output": {},
            "output_type": STEP_ZH_NAMES[step_name]["output_type"],
            "comment": "",
        }
        group_res = self.group_hits_by_owner(**group_hits_by_owner_params)
        self.update_step_output(
            group_hits_by_owner_result, step_output=group_res, field="authors"
        )
        step_results.append(group_hits_by_owner_result)

        logger.exit_quiet(not verbose)
        return {"query": query, "status": final_status, "data": step_results}

    def knn_explore(
        self,
        # Query and filter params
        query: str,
        extra_filters: list[dict] = [],
        verbose: bool = False,
        # KNN-specific params
        knn_field: str = KNN_TEXT_EMB_FIELD,
        knn_k: int = KNN_K,
        knn_num_candidates: int = KNN_NUM_CANDIDATES,
        similarity: float = None,
        # Explore params
        most_relevant_limit: int = 10000,
        rank_method: RANK_METHOD_TYPE = "relevance",  # Default to pure relevance ranking
        rank_top_k: int = 400,
        group_owner_limit: int = 20,
        group_sort_field: Literal[
            "sum_count",
            "sum_view",
            "sum_sort_score",
            "sum_rank_score",
            "top_rank_score",
            "first_appear_order",
        ] = "first_appear_order",  # Default: order by first appearance in ranked hits
    ) -> dict:
        """KNN-based explore using text embeddings instead of keyword matching.

        This method performs vector similarity search using the text_emb field,
        while still supporting all DSL filter expressions (date, stat, user, etc.).

        IMPORTANT: For vector search, relevance is the ONLY metric that matters.
        Results are ranked purely by vector similarity score - no stats/pubdate weighting.
        This ensures the most semantically relevant results appear first.

        The workflow is:
        1. Extract filters from DSL query (non-word expressions)
        2. Convert query words to embedding vector via TEI
        3. Perform KNN search with filters
        4. Fetch full documents for top results
        5. Group results by owner (UP主) - ordered by first appearance in hits

        Args:
            query: Query string (can include DSL filter expressions).
            extra_filters: Additional filter clauses.
            verbose: Enable verbose logging.
            knn_field: Dense vector field for KNN search.
            knn_k: Number of nearest neighbors.
            knn_num_candidates: Candidates per shard.
            similarity: Minimum similarity threshold.
            most_relevant_limit: Max docs for initial KNN search.
            rank_method: Ranking method for results.
            rank_top_k: Top-k for final ranking.
            group_owner_limit: Max groups for owner aggregation.

        Returns:
            dict: {
                "query": str,
                "status": "finished" | "timedout" | "error",
                "data": list[dict]  # list of step results
            }
        """
        logger.enter_quiet(not verbose)

        step_results = []
        final_status = "finished"

        # Step 0: Construct KNN query info
        step_idx = 0
        step_name = "construct_knn_query"
        logger.hint("> [step 0] KNN query constructing ...")

        # Extract filters and query info
        query_info, filter_clauses = self.get_filters_from_query(
            query=query,
            extra_filters=extra_filters,
        )

        # Get query words for embedding
        words_expr = query_info.get("words_expr", "")
        keywords_body = query_info.get("keywords_body", [])

        if keywords_body:
            embed_text = " ".join(keywords_body)
        else:
            embed_text = words_expr or query

        # Check embed client availability
        if not self.embed_client.is_available():
            logger.warn("× Embed client not available, KNN explore cannot proceed")
            error_result = {
                "step": step_idx,
                "name": step_name,
                "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
                "status": "error",
                "input": {"query": query},
                "output_type": STEP_ZH_NAMES[step_name]["output_type"],
                "output": {"error": "Embed client not available"},
                "comment": "TEI服务不可用",
            }
            step_results.append(error_result)
            logger.exit_quiet(not verbose)
            return {"query": query, "status": "error", "data": step_results}

        # Convert query to embedding
        query_hex = self.embed_client.text_to_hex(embed_text)
        if not query_hex:
            logger.warn("× Failed to get embedding for query")
            error_result = {
                "step": step_idx,
                "name": step_name,
                "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
                "status": "error",
                "input": {"query": query, "embed_text": embed_text},
                "output_type": STEP_ZH_NAMES[step_name]["output_type"],
                "output": {"error": "Failed to compute embedding"},
                "comment": "无法计算查询向量",
            }
            step_results.append(error_result)
            logger.exit_quiet(not verbose)
            return {"query": query, "status": "error", "data": step_results}

        query_vector = self.embed_client.hex_to_byte_array(query_hex)

        knn_query_info = {
            "embed_text": embed_text,
            "query_vector_len": len(query_vector),
            "filter_count": len(filter_clauses),
            "filters": filter_clauses[:3] if filter_clauses else [],  # Show first 3
            "qmod": ["vector"],  # vector-only mode
        }

        knn_query_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "finished",
            "input": {"query": query, "extra_filters": extra_filters},
            "output_type": STEP_ZH_NAMES[step_name]["output_type"],
            "output": knn_query_info,
            "comment": "",
        }
        step_results.append(knn_query_result)

        # Step 1: KNN search for most relevant docs
        step_idx += 1
        step_name = "knn_search"
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(f"> {step_str} KNN search for relevant docs")

        knn_search_params = {
            "query": query,
            "source_fields": [
                "bvid",
                "owner",
                "stat",
                "pubdate",
                "duration",
            ],  # Include owner for author grouping
            "extra_filters": extra_filters,
            "knn_field": knn_field,
            "k": knn_k,  # Use knn_k parameter for KNN search
            "num_candidates": knn_num_candidates,
            "similarity": similarity,
            "add_region_info": False,
            "is_explain": False,
            "rank_method": rank_method,
            "limit": knn_k,  # Limit to k results
            "rank_top_k": rank_top_k,
            "timeout": KNN_TIMEOUT,
            "verbose": verbose,
        }

        knn_search_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "running",
            "input": knn_search_params,
            "output": {},
            "output_type": "hits",
            "comment": "",
        }

        knn_search_res = self.knn_search(**knn_search_params)
        self.update_step_output(knn_search_result, step_output=knn_search_res)

        # Track timeout status but continue processing if we have any hits
        if self.is_status_timedout(knn_search_result):
            final_status = "timedout"

        # Step 2: Fetch full docs for ranked results
        bvids = [hit.get("bvid", None) for hit in knn_search_res.get("hits", [])]
        if not bvids:
            # No results - still need to add empty group_hits_by_owner step for frontend
            logger.warn("× No results from KNN search")
            step_results.append(knn_search_result)
            # Add empty group_hits_by_owner result with appropriate comment
            step_idx += 1
            # Distinguish between timeout and normal no-results
            if final_status == "timedout":
                comment = "搜索超时，请稍后重试"
            else:
                comment = "无搜索结果"
            empty_group_result = {
                "step": step_idx,
                "name": "group_hits_by_owner",
                "name_zh": STEP_ZH_NAMES["group_hits_by_owner"]["name_zh"],
                "status": "finished",
                "input": {"limit": group_owner_limit, "sort_field": group_sort_field},
                "output": {"authors": {}},
                "output_type": STEP_ZH_NAMES["group_hits_by_owner"]["output_type"],
                "comment": comment,
            }
            step_results.append(empty_group_result)
            logger.exit_quiet(not verbose)
            return {"query": query, "status": final_status, "data": step_results}

        # Use fetch_docs_by_bvids to get full docs without word matching
        full_doc_search_res = self.fetch_docs_by_bvids(
            bvids=bvids,
            add_region_info=True,
            limit=len(bvids),
            timeout=EXPLORE_TIMEOUT,
            verbose=verbose,
        )

        # Merge KNN scores from original search into full doc hits
        knn_hits = knn_search_res.get("hits", [])
        full_hits = full_doc_search_res.get("hits", [])
        self.merge_scores_into_hits(full_hits, knn_hits)

        # Re-apply ranking after merging scores
        # For KNN explore, "relevance" is the preferred method - pure vector similarity ranking
        if rank_method == "relevance":
            full_doc_search_res = self.hit_ranker.relevance_rank(
                full_doc_search_res, top_k=rank_top_k
            )
        elif rank_method == "rrf":
            full_doc_search_res = self.hit_ranker.rrf_rank(
                full_doc_search_res, top_k=rank_top_k
            )
        elif rank_method == "stats":
            full_doc_search_res = self.hit_ranker.stats_rank(
                full_doc_search_res, top_k=rank_top_k
            )
        else:  # "heads"
            full_doc_search_res = self.hit_ranker.heads(
                full_doc_search_res, top_k=rank_top_k
            )

        full_doc_search_res["total_hits"] = knn_search_res.get("total_hits", 0)
        self.update_step_output(knn_search_result, step_output=full_doc_search_res)

        # Track if any step timed out, but continue to group_hits_by_owner if we have hits
        if self.is_status_timedout(knn_search_result):
            final_status = "timedout"

        step_results.append(knn_search_result)

        # Step 3: Group hits by owner
        # IMPORTANT: Always execute this step even if previous steps timed out,
        # as long as we have some hits to group. This ensures "相关作者" is always populated.
        # For KNN explore, use top_rank_score to find authors with highest relevance hits
        step_idx += 1
        step_name = "group_hits_by_owner"
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(f"> {step_str} Group hits by owner")

        group_hits_by_owner_params = {
            "search_res": full_doc_search_res,
            "sort_field": group_sort_field,
            "limit": group_owner_limit,
        }

        group_hits_by_owner_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "running",
            "input": {"limit": group_owner_limit, "sort_field": group_sort_field},
            "output": {},
            "output_type": STEP_ZH_NAMES[step_name]["output_type"],
            "comment": "",
        }

        group_res = self.group_hits_by_owner(**group_hits_by_owner_params)
        self.update_step_output(
            group_hits_by_owner_result, step_output=group_res, field="authors"
        )
        step_results.append(group_hits_by_owner_result)

        logger.exit_quiet(not verbose)
        return {"query": query, "status": final_status, "data": step_results}

    def hybrid_explore(
        self,
        # Query and filter params
        query: str,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        verbose: bool = False,
        # KNN params
        knn_field: str = KNN_TEXT_EMB_FIELD,
        knn_k: int = KNN_K,
        knn_num_candidates: int = KNN_NUM_CANDIDATES,
        # Hybrid params
        rrf_k: int = HYBRID_RRF_K,
        fusion_method: Literal["rrf", "weighted"] = "rrf",
        # Explore params
        most_relevant_limit: int = 10000,
        rank_method: RANK_METHOD_TYPE = "relevance",  # Default to pure relevance ranking
        rank_top_k: int = 400,
        group_owner_limit: int = 20,
        group_sort_field: Literal[
            "sum_count",
            "sum_view",
            "sum_sort_score",
            "sum_rank_score",
            "top_rank_score",
            "first_appear_order",
        ] = "first_appear_order",  # Default: order by first appearance in ranked hits
    ) -> dict:
        """Hybrid explore combining word-based and vector-based retrieval.

        This method prioritizes RELEVANCE over popularity/recency.
        Results are ranked by vector similarity first, ensuring the most
        semantically relevant content appears at the top.

        Workflow:
        1. Performs word-based search (for keyword matching)
        2. Performs KNN vector search (for semantic similarity)
        3. Fuses results using RRF, with vector scores weighted higher
        4. Ranks by pure relevance score (no stats/pubdate weighting)
        5. Groups results by owner (UP主) using top relevance score

        Args:
            query: Query string (can include DSL filter expressions).
            extra_filters: Additional filter clauses.
            suggest_info: Suggestion info.
            verbose: Enable verbose logging.
            knn_field: Dense vector field for KNN search.
            knn_k: Number of nearest neighbors.
            knn_num_candidates: Candidates per shard.
            rrf_k: K parameter for RRF fusion.
            fusion_method: "rrf" or "weighted".
            most_relevant_limit: Max docs for searches.
            rank_method: Ranking method ("relevance" recommended for vector search).
            rank_top_k: Top-k for final ranking.
            group_owner_limit: Max groups for owner aggregation.
            group_sort_field: Field to sort author groups by.

        Returns:
            dict: {
                "query": str,
                "status": "finished" | "timedout" | "error",
                "data": list[dict]  # list of step results, qmod in first step's output
            }
        """
        logger.enter_quiet(not verbose)

        step_results = []
        final_status = "finished"

        # Step 0: Parse query and extract info
        step_idx = 0
        step_name = "construct_query_dsl_dict"
        logger.hint("> [step 0] Query constructing ...")

        query_info = self.query_rewriter.get_query_info(query)
        query_dsl_dict_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "finished",
            "input": {"query": query},
            "output_type": STEP_ZH_NAMES[step_name]["output_type"],
            "output": {
                "words_expr": query_info.get("words_expr", ""),
                "keywords_body": query_info.get("keywords_body", []),
                "qmod": ["word", "vector"],  # hybrid mode
            },
            "comment": "",
        }
        step_results.append(query_dsl_dict_result)

        # Step 1: Hybrid search
        step_idx += 1
        step_name = "hybrid_search"
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(f"> {step_str} Hybrid search (word + KNN)")

        hybrid_search_params = {
            "query": query,
            "source_fields": [
                "bvid",
                "owner",
                "stat",
                "pubdate",
                "duration",
            ],  # Include owner for author grouping
            "extra_filters": extra_filters,
            "suggest_info": suggest_info,
            "knn_field": knn_field,
            "knn_k": knn_k,  # Use knn_k parameter for KNN portion
            "knn_num_candidates": knn_num_candidates,
            "rrf_k": rrf_k,
            "fusion_method": fusion_method,
            "add_region_info": False,
            "add_highlights_info": False,
            "is_highlight": False,
            "rank_method": rank_method,
            "limit": min(most_relevant_limit, 2000),  # Cap limit for faster search
            "rank_top_k": rank_top_k,
            "timeout": EXPLORE_TIMEOUT,
            "verbose": verbose,
        }

        hybrid_search_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "running",
            "input": hybrid_search_params,
            "output": {},
            "output_type": "hits",
            "comment": "",
        }

        hybrid_search_res = self.hybrid_search(**hybrid_search_params)
        self.update_step_output(hybrid_search_result, step_output=hybrid_search_res)

        # Track timeout status but continue processing
        if self.is_status_timedout(hybrid_search_result):
            final_status = "timedout"

        # Step 2: Fetch full docs for ranked results
        bvids = [hit.get("bvid", None) for hit in hybrid_search_res.get("hits", [])]
        if not bvids:
            # No results - still need to add empty group_hits_by_owner step for frontend
            logger.warn("× No results from hybrid search")
            step_results.append(hybrid_search_result)
            # Add empty group_hits_by_owner result with appropriate comment
            step_idx += 1
            # Distinguish between timeout and normal no-results
            if final_status == "timedout":
                comment = "搜索超时，请稍后重试"
            else:
                comment = "无搜索结果"
            empty_group_result = {
                "step": step_idx,
                "name": "group_hits_by_owner",
                "name_zh": STEP_ZH_NAMES["group_hits_by_owner"]["name_zh"],
                "status": "finished",
                "input": {"limit": group_owner_limit, "sort_field": group_sort_field},
                "output": {"authors": {}},
                "output_type": STEP_ZH_NAMES["group_hits_by_owner"]["output_type"],
                "comment": comment,
            }
            step_results.append(empty_group_result)
            logger.exit_quiet(not verbose)
            return {
                "query": query,
                "status": final_status,
                "data": step_results,
            }

        bvid_filter = bvids_to_filter(bvids)

        full_doc_search_params = {
            "query": query,
            "extra_filters": extra_filters + [bvid_filter],
            "rank_method": "heads",  # Use heads to preserve order, we'll merge scores
            "is_highlight": True,
            "add_region_info": True,
            "add_highlights_info": True,
            "limit": len(bvids),
            "timeout": EXPLORE_TIMEOUT,
            "verbose": verbose,
        }

        full_doc_search_res = self.search(**full_doc_search_params)

        # Merge hybrid scores from original search into full doc hits
        hybrid_hits = hybrid_search_res.get("hits", [])
        full_hits = full_doc_search_res.get("hits", [])
        self.merge_scores_into_hits(full_hits, hybrid_hits)

        # Re-apply ranking after merging scores
        # For hybrid explore, RRF fusion already produced well-ranked results
        # Use heads to preserve RRF order, set rank_score = hybrid_score for consistency
        if rank_method == "relevance":
            # Preserve RRF fusion order, just set rank_score for downstream use
            for hit in full_hits:
                hit["rank_score"] = hit.get("hybrid_score", 0) or 0
            full_hits.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
            full_doc_search_res["hits"] = full_hits[:rank_top_k]
            full_doc_search_res["return_hits"] = len(full_doc_search_res["hits"])
            full_doc_search_res["rank_method"] = "relevance"
        elif rank_method == "rrf":
            full_doc_search_res = self.hit_ranker.rrf_rank(
                full_doc_search_res, top_k=rank_top_k
            )
        elif rank_method == "stats":
            full_doc_search_res = self.hit_ranker.stats_rank(
                full_doc_search_res, top_k=rank_top_k
            )
        else:  # "heads"
            full_doc_search_res = self.hit_ranker.heads(
                full_doc_search_res, top_k=rank_top_k
            )

        full_doc_search_res["total_hits"] = hybrid_search_res.get("total_hits", 0)
        full_doc_search_res["fusion_method"] = hybrid_search_res.get(
            "fusion_method", fusion_method
        )
        full_doc_search_res["word_hits_count"] = hybrid_search_res.get(
            "word_hits_count", 0
        )
        full_doc_search_res["knn_hits_count"] = hybrid_search_res.get(
            "knn_hits_count", 0
        )

        self.update_step_output(hybrid_search_result, step_output=full_doc_search_res)

        # Track if any step timed out, but continue to group_hits_by_owner if we have hits
        if self.is_status_timedout(hybrid_search_result):
            final_status = "timedout"

        step_results.append(hybrid_search_result)

        # Step 3: Group hits by owner
        # IMPORTANT: Always execute this step even if previous steps timed out,
        # as long as we have some hits to group. This ensures "相关作者" is always populated.

        # Step 3: Group hits by owner
        # For hybrid explore, use top_rank_score to find authors with highest relevance hits
        step_idx += 1
        step_name = "group_hits_by_owner"
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(f"> {step_str} Group hits by owner")

        group_hits_by_owner_params = {
            "search_res": full_doc_search_res,
            "sort_field": group_sort_field,
            "limit": group_owner_limit,
        }

        group_hits_by_owner_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "running",
            "input": {"limit": group_owner_limit, "sort_field": group_sort_field},
            "output": {},
            "output_type": STEP_ZH_NAMES[step_name]["output_type"],
            "comment": "",
        }

        group_res = self.group_hits_by_owner(**group_hits_by_owner_params)
        self.update_step_output(
            group_hits_by_owner_result, step_output=group_res, field="authors"
        )
        step_results.append(group_hits_by_owner_result)

        logger.exit_quiet(not verbose)
        return {
            "query": query,
            "status": final_status,
            "data": step_results,
        }

    def unified_explore(
        self,
        query: str,
        qmod: Union[str, list[str]] = None,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        verbose: bool = False,
        # Common explore params
        most_relevant_limit: int = 10000,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD_DEFAULT,
        rank_top_k: int = 400,
        group_owner_limit: int = 25,
        # KNN/Hybrid specific params
        knn_field: str = KNN_TEXT_EMB_FIELD,
        knn_k: int = KNN_K,
        knn_num_candidates: int = KNN_NUM_CANDIDATES,
    ) -> dict:
        """Unified explore that automatically selects search method based on query mode.

        The query mode can be specified via:
        1. The qmod parameter (str or list[str])
        2. DSL expression in query (e.g., "黑神话 q=v" or "q=wv")

        Args:
            query: Query string.
            qmod: Override mode(s). Can be:
                - str: "w", "v", "wv" (shorthand) or "word", "vector"
                - list[str]: ["word"], ["vector"], ["word", "vector"]
                If None, extracted from query.
            extra_filters: Additional filter clauses.
            suggest_info: Suggestion info.
            verbose: Enable verbose logging.
            most_relevant_limit: Max docs for searches.
            rank_method: Ranking method.
            rank_top_k: Top-k for ranking.
            group_owner_limit: Max owner groups.
            knn_field: Dense vector field.
            knn_k: KNN neighbors count.
            knn_num_candidates: KNN candidates per shard.

        Returns:
            Explore results dict (qmod is in first step_result's output).
        """
        from converters.dsl.fields.qmod import normalize_qmod

        # Extract query mode from query if not provided
        if qmod is None:
            qmod = self.get_qmod_from_query(query)
        else:
            qmod = normalize_qmod(qmod)

        logger.hint(f"> Query mode (qmod): {qmod}", verbose=verbose)

        is_hybrid = is_hybrid_qmod(qmod)
        has_word = "word" in qmod
        has_vector = "vector" in qmod

        if is_hybrid:
            # Hybrid mode (both word and vector)
            # For hybrid mode, force relevance ranking to prioritize semantic relevance
            hybrid_rank_method = "relevance"
            result = self.hybrid_explore(
                query=query,
                extra_filters=extra_filters,
                suggest_info=suggest_info,
                verbose=verbose,
                knn_field=knn_field,
                knn_k=knn_k,
                knn_num_candidates=knn_num_candidates,
                most_relevant_limit=most_relevant_limit,
                rank_method=hybrid_rank_method,
                rank_top_k=rank_top_k,
                group_owner_limit=group_owner_limit,
            )
            return result

        elif has_vector:
            # Vector-only mode
            # IMPORTANT: For vector search, ALWAYS use relevance ranking
            # This ensures "综合排序" and "最高相关" produce identical results
            # because relevance IS the only meaningful metric for vector search
            vector_rank_method = "relevance"
            result = self.knn_explore(
                query=query,
                extra_filters=extra_filters,
                verbose=verbose,
                knn_field=knn_field,
                knn_k=knn_k,
                knn_num_candidates=knn_num_candidates,
                most_relevant_limit=most_relevant_limit,
                rank_method=vector_rank_method,
                rank_top_k=rank_top_k,
                group_owner_limit=group_owner_limit,
            )
            return result

        else:
            # Word-only mode
            result = self.explore(
                query=query,
                extra_filters=extra_filters,
                suggest_info=suggest_info,
                verbose=verbose,
                most_relevant_limit=most_relevant_limit,
                rank_method=rank_method,
                rank_top_k=rank_top_k,
                group_owner_limit=group_owner_limit,
            )
            # Add qmod to first step's output for word-only mode
            if result.get("data") and len(result["data"]) > 0:
                result["data"][0]["output"]["qmod"] = ["word"]
            return result
