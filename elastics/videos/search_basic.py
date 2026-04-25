from copy import deepcopy
from tclogger import logger, dict_to_str, get_now, tcdatetime
from typing import Literal, Union

from converters.query.dsl import ScriptScoreQueryDSLConstructor
from elastics.structure import get_highlight_settings, construct_boosted_fields
from elastics.structure import set_min_score, set_terminate_after
from elastics.structure import set_timeout, set_profile
from elastics.videos.constants import (
    AGG_PERCENTS,
    AGG_TIMEOUT,
    AGG_TOP_K,
    DOC_EXCLUDED_SOURCE_FIELDS,
    IS_HIGHLIGHT,
    MATCH_BOOL,
    MATCH_OPERATOR,
    MATCH_TYPE,
    NO_HIGHLIGHT_REDUNDANCE_RATIO,
    SEARCH_BOOSTED_FIELDS,
    SEARCH_LIMIT,
    SEARCH_MATCH_BOOL,
    SEARCH_MATCH_FIELDS,
    SEARCH_MATCH_OPERATOR,
    SEARCH_MATCH_TYPE,
    SEARCH_TIMEOUT,
    SOURCE_FIELDS,
    SUGGEST_BOOSTED_FIELDS,
    SUGGEST_DETAIL_LEVELS,
    SUGGEST_LIMIT,
    SUGGEST_MATCH_BOOL,
    SUGGEST_MATCH_FIELDS,
    SUGGEST_MATCH_OPERATOR,
    SUGGEST_MATCH_TYPE,
    SUGGEST_TIMEOUT,
    TERMINATE_AFTER,
    TRACK_TOTAL_HITS,
    USE_SCRIPT_SCORE,
)


class VideoSearchBasicMixin:
    def construct_search_body(
        self,
        query_dsl_dict: dict,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        drop_no_highlights: bool = False,
        is_explain: bool = False,
        is_profile: bool = False,
        is_highlight: bool = IS_HIGHLIGHT,
        use_script_score: bool = USE_SCRIPT_SCORE,
        score_threshold: float = None,
        limit: int = SEARCH_LIMIT,
        terminate_after: int = TERMINATE_AFTER,
        timeout: Union[int, float, str] = SEARCH_TIMEOUT,
    ) -> dict:
        """construct script_score from query_dsl_dict, and return search_body"""
        common_params = {
            "_source": source_fields,
            "explain": is_explain,
            "track_total_hits": TRACK_TOTAL_HITS,
        }
        if is_highlight:
            common_params["highlight"] = get_highlight_settings(match_fields)
        script_score_constructor = ScriptScoreQueryDSLConstructor()
        if use_script_score:
            script_query_dsl_dict = script_score_constructor.construct(
                query_dsl_dict, score_threshold=score_threshold, combine_type="sort"
            )
            search_body = {
                **script_query_dsl_dict,
                **common_params,
            }
        else:
            search_body = {
                "query": query_dsl_dict,
                **common_params,
            }
        search_body = set_timeout(search_body, timeout=timeout)
        search_body = set_min_score(search_body, min_score=score_threshold)
        search_body = set_terminate_after(search_body, terminate_after=terminate_after)
        search_body = set_profile(search_body, profile=is_profile)
        if limit and limit > 0:
            if drop_no_highlights:
                search_body["size"] = int(limit * NO_HIGHLIGHT_REDUNDANCE_RATIO)
            else:
                search_body["size"] = limit
        return search_body

    def suggest(
        self,
        query: str,
        match_fields: list[str] = SUGGEST_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        match_type: MATCH_TYPE = SUGGEST_MATCH_TYPE,
        match_bool: MATCH_BOOL = SUGGEST_MATCH_BOOL,
        match_operator: MATCH_OPERATOR = SUGGEST_MATCH_OPERATOR,
        extra_filters: list[dict] = [],
        parse_hits: bool = True,
        drop_no_highlights: bool = False,
        is_explain: bool = False,
        is_profile: bool = False,
        is_highlight: bool = IS_HIGHLIGHT,
        boost: bool = True,
        boosted_fields: dict = SUGGEST_BOOSTED_FIELDS,
        combined_fields_list: list[list[str]] = [],
        use_script_score: bool = USE_SCRIPT_SCORE,
        use_pinyin: bool = True,
        detail_level: int = -1,
        detail_levels: dict = SUGGEST_DETAIL_LEVELS,
        limit: int = SUGGEST_LIMIT,
        timeout: Union[int, float, str] = SUGGEST_TIMEOUT,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        return self.search(
            query=query,
            match_fields=match_fields,
            source_fields=source_fields,
            match_type=match_type,
            match_bool=match_bool,
            match_operator=match_operator,
            extra_filters=extra_filters,
            request_type="suggest",
            parse_hits=parse_hits,
            drop_no_highlights=drop_no_highlights,
            is_explain=is_explain,
            is_profile=is_profile,
            is_highlight=is_highlight,
            boost=boost,
            boosted_fields=boosted_fields,
            combined_fields_list=combined_fields_list,
            use_script_score=use_script_score,
            use_pinyin=use_pinyin,
            detail_level=detail_level,
            detail_levels=detail_levels,
            limit=limit,
            timeout=timeout,
            verbose=verbose,
        )

    def random(
        self,
        seed: Union[int, str] = None,
        seed_update_seconds: int = None,
        source_fields: list[str] = SOURCE_FIELDS,
        parse_hits: bool = True,
        is_explain: bool = False,
        filters: list[dict] = None,
        limit: int = 1,
        verbose: bool = False,
    ):
        logger.enter_quiet(not verbose)
        now = get_now()
        now_ts = int(now.timestamp())
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        today = tcdatetime(year=now.year, month=now.month, day=now.day)
        today_ts = int(today.timestamp())

        if seed is None:
            if seed_update_seconds is None:
                seed = now_ts
            else:
                seed_update_seconds = max(int(abs(seed_update_seconds)), 1)
                seed = now_ts // seed_update_seconds
        else:
            seed = int(seed)

        if filters is None:
            past_month_day_ts = today_ts - 3600 * 24 * 30
            filters = [
                {"range": {"stat.coin": {"gte": 100}}},
                {"range": {"stat.danmaku": {"gte": 100}}},
                {"range": {"pubdate": {"gte": past_month_day_ts}}},
            ]

        search_body = {
            "query": {
                "function_score": {
                    "query": {"bool": {"filter": filters}},
                    "random_score": {"seed": seed, "field": "_seq_no"},
                    "boost_mode": "replace",
                }
            },
            "_source": source_fields,
            "explain": is_explain,
        }
        if limit and limit > 0:
            search_body["size"] = limit
        logger.note(f"> Get random docs with seed:", end=" ")
        logger.mesg(f"[{seed}] ({now_str})")
        res = self.es.client.search(index=self.index_name, body=search_body)
        res_dict = res.body
        if parse_hits:
            return_res = self.hit_parser.parse({}, [], res_dict, request_type="random")
        else:
            return_res = res_dict
        logger.exit_quiet(not verbose)
        return return_res

    def latest(
        self,
        source_fields: list[str] = SOURCE_FIELDS,
        parse_hits: bool = True,
        is_explain: bool = False,
        limit: int = SUGGEST_LIMIT,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        logger.enter_quiet(not verbose)
        search_body = {
            "query": {"match_all": {}},
            "sort": [{"pubdate": {"order": "desc"}}],
            "_source": source_fields,
            "explain": is_explain,
        }
        if limit and limit > 0:
            search_body["size"] = limit
        logger.note(f"> Get latest {limit} docs:")
        res = self.es.client.search(index=self.index_name, body=search_body)
        res_dict = res.body
        if parse_hits:
            return_res = self.hit_parser.parse({}, [], res_dict, request_type="latest")
        else:
            return_res = res_dict
        logger.exit_quiet(not verbose)
        return return_res

    def doc(
        self,
        bvid: str,
        included_source_fields: list[str] = [],
        excluded_source_fields: list[str] = DOC_EXCLUDED_SOURCE_FIELDS,
        verbose: bool = False,
    ) -> dict:
        logger.enter_quiet(not verbose)
        logger.note(f"> Get video details:", end=" ")
        logger.mesg(f"[{bvid}]")
        res = self.es.client.get(
            index=self.index_name,
            id=bvid,
            source_excludes=excluded_source_fields or None,
            source_includes=included_source_fields or None,
        )
        res_dict = res.body["_source"]

        pubdate_t = tcdatetime.fromtimestamp(res_dict.get("pubdate", 0))
        pubdate_str = pubdate_t.strftime("%Y-%m-%d %H:%M:%S")
        res_dict["pubdate_str"] = pubdate_str

        reduced_dict = {
            k: v
            for k, v in res_dict.items()
            if k in ["title", "bvid", "pubdate_str", "desc"]
        }
        logger.success(dict_to_str(reduced_dict), indent=4)
        logger.exit_quiet(not verbose)
        return res_dict

    def fetch_docs_by_bvids(
        self,
        bvids: list[str],
        source_fields: list[str] = SOURCE_FIELDS,
        add_region_info: bool = True,
        limit: int = None,
        timeout: Union[int, float, str] = SEARCH_TIMEOUT,
        verbose: bool = False,
    ) -> dict:
        """Fetch documents by bvid list without word matching.

        This method retrieves full documents using only bvid filter,
        without any query matching. Useful for fetching KNN/hybrid results.

        Args:
            bvids: List of bvids to fetch.
            source_fields: Fields to include in results.
            add_region_info: Whether to add region info.
            limit: Maximum results (defaults to len(bvids)).
            timeout: Search timeout.
            verbose: Enable verbose logging.

        Returns:
            Search results dict with hits.
        """
        logger.enter_quiet(not verbose)

        if not bvids:
            logger.exit_quiet(not verbose)
            return {
                "timed_out": False,
                "total_hits": 0,
                "return_hits": 0,
                "hits": [],
            }

        if limit is None:
            limit = len(bvids)

        # Build search body with only bvid filter (no query matching)
        search_body = {
            "query": {"bool": {"filter": [{"terms": {"bvid.keyword": bvids}}]}},
            "_source": source_fields,
            "size": limit,
            "track_total_hits": True,
        }
        search_body = set_timeout(search_body, timeout=timeout)

        logger.mesg(f"> Fetching {len(bvids)} docs by bvid", verbose=verbose)

        # Submit to ES
        res_dict = self.submit_to_es(search_body, context="fetch_docs_by_bvids")

        # Parse results
        parse_res = self.hit_parser.parse(
            query_info={},
            match_fields=[],
            res_dict=res_dict,
            request_type="search",
            drop_no_highlights=False,
            add_region_info=add_region_info,
            add_highlights_info=False,
            match_type="cross_fields",
            match_operator="or",
            detail_level=-1,
            limit=limit,
            verbose=verbose,
        )

        logger.exit_quiet(not verbose)
        return parse_res

    def post_process_return_res(self, return_res: dict) -> dict:
        """Remove some non-jsonable items from return_res."""
        return_res["query_info"].pop("query_expr_tree", None)
        return_res["rewrite_info"].pop("rewrited_expr_trees", None)
        return return_res

    def sanitize_search_body_for_client(self, search_body: dict) -> dict:
        """Remove large data from search_body before returning to client.

        This reduces network payload by:
        1. Removing bvid terms filter arrays which can contain hundreds of ids
        2. Removing query_vector arrays from KNN queries

        Args:
            search_body: The search body dict to sanitize.

        Returns:
            A sanitized copy with large arrays replaced by placeholders.
        """
        if not search_body:
            return search_body

        sanitized = deepcopy(search_body)

        def sanitize_large_arrays(obj: dict, max_terms: int = 10) -> None:
            """Recursively find and truncate large arrays."""
            if not isinstance(obj, dict):
                return
            for key, value in list(obj.items()):
                if key == "terms" and isinstance(value, dict):
                    # Found a terms filter, check each field
                    for field, terms_list in value.items():
                        if isinstance(terms_list, list) and len(terms_list) > max_terms:
                            # Replace with count indicator
                            value[field] = f"[{len(terms_list)} items omitted]"
                elif key == "query_vector" and isinstance(value, list):
                    # Remove query_vector from KNN queries - it's large and not needed by client
                    obj[key] = f"[{len(value)} dims vector omitted]"
                elif isinstance(value, dict):
                    sanitize_large_arrays(value, max_terms)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            sanitize_large_arrays(item, max_terms)

        sanitize_large_arrays(sanitized)
        return sanitized

    def construct_agg_body_of_percentile(
        self,
        query_dsl_dict: dict,
        timeout: Union[int, float, str] = AGG_TIMEOUT,
        # sort_field: str = AGG_SORT_FIELD,
        # sort_order: str = AGG_SORT_ORDER,
    ) -> dict:
        """construct aggregations body for percentile"""
        common_params = {
            "size": 0,
            "track_total_hits": TRACK_TOTAL_HITS,
            "_source": False,
        }
        # sort_dict = [{sort_field: {"order": sort_order}}]
        aggs_dict = {
            "score_ps": {
                "percentiles": {
                    "script": {"source": "_score"},
                    "percents": AGG_PERCENTS,
                }
            },
            "view_ps": {
                "percentiles": {
                    "field": "stat.view",
                    "percents": AGG_PERCENTS,
                }
            },
            "pubdate_ps": {
                "percentiles": {
                    "field": "pubdate",
                    "percents": AGG_PERCENTS,
                }
            },
        }
        agg_body = {
            **common_params,
            "query": query_dsl_dict,
            # "sort": sort_dict,
            "aggs": aggs_dict,
        }
        agg_body = set_timeout(agg_body, timeout=timeout)
        return agg_body

    def construct_agg_body_of_top_hits(
        self,
        query_dsl_dict: dict,
        timeout: Union[int, float, str] = AGG_TIMEOUT,
        top_k: int = AGG_TOP_K,
    ) -> dict:
        """construct aggregations body for top_hits"""
        common_params = {
            "size": 0,
            "track_total_hits": TRACK_TOTAL_HITS,
            "_source": False,
        }
        query_dsl_dict_without_sort = deepcopy(query_dsl_dict)
        query_dsl_dict_without_sort["sort"] = ["_doc"]
        aggs_dict = {
            "pubdate_tops": {
                "top_hits": {
                    "side": top_k,
                    "sort": [{"pubdate": {"order": "desc"}}],
                    "_source": False,
                }
            },
            "favorite_tops": {
                "top_hits": {
                    "side": top_k,
                    "sort": [{"stat.favorite": {"order": "desc"}}],
                    "_source": False,
                }
            },
        }
        agg_body = {
            **common_params,
            "query": query_dsl_dict_without_sort,
            "aggs": aggs_dict,
        }
        agg_body = set_timeout(agg_body, timeout=timeout)
        return agg_body

    def agg(
        self,
        query: str = None,
        query_dsl_dict: dict = None,
        agg_type: Literal["percentile", "top_hits"] = "top_hits",
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        boost: bool = True,
        boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
        timeout: Union[int, float, str] = AGG_TIMEOUT,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        """If query_dsl_dict is provided, use it directly, then only `timeout` and `verbose` is taking effect."""
        if not query_dsl_dict:
            # construct boosted fields
            boosted_match_fields, boosted_date_fields = construct_boosted_fields(
                match_fields=match_fields,
                boost=boost,
                boosted_fields=boosted_fields,
            )
            # construct query_dsl_dict
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
        # construct agg_body
        if agg_type == "percentile":
            agg_body = self.construct_agg_body_of_percentile(
                query_dsl_dict=query_dsl_dict, timeout=timeout
            )
        else:
            agg_body = self.construct_agg_body_of_top_hits(
                query_dsl_dict=query_dsl_dict, timeout=timeout
            )
        if verbose:
            logger.hint("> agg_body:")
            logger.mesg(dict_to_str(agg_body, add_quotes=True, align_list=False))
        # submit agg_body to es client
        es_res_dict = self.submit_to_es(agg_body, context="agg")
        return es_res_dict
