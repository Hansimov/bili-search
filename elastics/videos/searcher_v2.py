from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from sedb import MongoOperator, ElasticOperator
from tclogger import logger, dict_to_str, get_now, tcdatetime
from typing import Union, Literal

from configs.envs import MONGO_ENVS, SECRETS, ELASTIC_PRO_ENVS
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.dsl.rewrite import DslExprRewriter
from converters.dsl.elastic import DslExprToElasticConverter
from converters.dsl.filter import QueryDslDictFilterMerger
from elastics.structure import get_highlight_settings, construct_boosted_fields
from elastics.structure import set_min_score, set_terminate_after
from elastics.structure import set_timeout, set_profile
from elastics.structure import construct_knn_query, construct_knn_search_body
from elastics.videos.constants import ELASTIC_VIDEOS_PRO_INDEX
from elastics.videos.constants import SEARCH_REQUEST_TYPE, SEARCH_REQUEST_TYPE_DEFAULT
from elastics.videos.constants import SOURCE_FIELDS, DOC_EXCLUDED_SOURCE_FIELDS
from elastics.videos.constants import SEARCH_MATCH_FIELDS, SEARCH_BOOSTED_FIELDS
from elastics.videos.constants import SUGGEST_MATCH_FIELDS, SUGGEST_BOOSTED_FIELDS
from elastics.videos.constants import DATE_MATCH_FIELDS
from elastics.videos.constants import MATCH_TYPE, MATCH_BOOL, MATCH_OPERATOR
from elastics.videos.constants import SEARCH_MATCH_TYPE, SUGGEST_MATCH_TYPE
from elastics.videos.constants import SEARCH_MATCH_BOOL, SEARCH_MATCH_OPERATOR
from elastics.videos.constants import SUGGEST_MATCH_BOOL, SUGGEST_MATCH_OPERATOR
from elastics.videos.constants import SEARCH_DETAIL_LEVELS, SUGGEST_DETAIL_LEVELS
from elastics.videos.constants import SEARCH_LIMIT, SUGGEST_LIMIT, RANK_TOP_K, AGG_TOP_K
from elastics.videos.constants import SEARCH_TIMEOUT, SUGGEST_TIMEOUT
from elastics.videos.constants import NO_HIGHLIGHT_REDUNDANCE_RATIO
from elastics.videos.constants import USE_SCRIPT_SCORE_DEFAULT
from elastics.videos.constants import RANK_METHOD_TYPE, RANK_METHOD_DEFAULT
from elastics.videos.constants import TRACK_TOTAL_HITS, IS_HIGHLIGHT
from elastics.videos.constants import AGG_TIMEOUT, AGG_PERCENTS
from elastics.videos.constants import AGG_SORT_FIELD, AGG_SORT_ORDER
from elastics.videos.constants import TERMINATE_AFTER
from elastics.videos.constants import KNN_TEXT_EMB_FIELD, KNN_K, KNN_NUM_CANDIDATES
from elastics.videos.constants import KNN_TIMEOUT
from elastics.videos.constants import QMOD_SINGLE_TYPE, QMOD_DEFAULT
from elastics.videos.constants import HYBRID_WORD_WEIGHT, HYBRID_VECTOR_WEIGHT
from elastics.videos.constants import HYBRID_RRF_K
from elastics.videos.hits import VideoHitsParser, SuggestInfoParser
from elastics.videos.ranker import VideoHitsRanker
from converters.embed import TextEmbedSearchClient
from converters.dsl.fields.qmod import extract_qmod_from_expr_tree


class VideoSearcherV2:
    def __init__(
        self,
        index_name: str = ELASTIC_VIDEOS_PRO_INDEX,
        elastic_env_name: str = None,
        mongo_env_name: str = None,
    ):
        """
        - index_name:
            name of elastic index for videos
            - example: "bili_videos_dev4"
        - elastic_env_name:
            name of elastic envs in secrets.json
            - example: "elastic", "elastic_dev"
        - mongo_env_name:
            name of mongo envs in secrets.json
            - example: "mongo"
        """
        self.index_name = index_name
        self.elastic_env_name = elastic_env_name
        self.mongo_env_name = mongo_env_name
        self.init_processors()

    def init_processors(self):
        if self.elastic_env_name:
            elastic_envs = SECRETS[self.elastic_env_name]
        else:
            elastic_envs = ELASTIC_PRO_ENVS
        if self.mongo_env_name:
            mongo_envs = SECRETS[self.mongo_env_name]
        else:
            mongo_envs = MONGO_ENVS
        self.es = ElasticOperator(elastic_envs, connect_cls=self.__class__)
        self.mongo = MongoOperator(mongo_envs, connect_cls=self.__class__)
        self.hit_parser = VideoHitsParser()
        self.hit_ranker = VideoHitsRanker()
        self.query_rewriter = DslExprRewriter()
        self.elastic_converter = DslExprToElasticConverter()
        self.filter_merger = QueryDslDictFilterMerger()
        self.suggest_parser = SuggestInfoParser("v2")
        # Lazy-initialized embed client for KNN search
        self._embed_client = None

    @property
    def embed_client(self) -> TextEmbedSearchClient:
        """Lazy-initialized embed client for KNN search."""
        if self._embed_client is None:
            self._embed_client = TextEmbedSearchClient(lazy_init=True)
        return self._embed_client

    def submit_to_es(self, body: dict) -> dict:
        try:
            res = self.es.client.search(index=self.index_name, body=body)
            res_dict = res.body
        except Exception as e:
            logger.warn(f"× Error: {e}")
            res_dict = {}
        return res_dict

    def construct_search_body(
        self,
        query_dsl_dict: dict,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        drop_no_highlights: bool = False,
        is_explain: bool = False,
        is_profile: bool = False,
        is_highlight: bool = IS_HIGHLIGHT,
        use_script_score: bool = USE_SCRIPT_SCORE_DEFAULT,
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
        use_script_score: bool = USE_SCRIPT_SCORE_DEFAULT,
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
        res_dict = self.submit_to_es(search_body)

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

    def rewrite_by_suggest(
        self,
        query_info: dict,
        suggest_info: dict = {},
        rewrite_info: dict = {},
        request_type: SEARCH_REQUEST_TYPE = SEARCH_REQUEST_TYPE_DEFAULT,
        return_res: dict = {},
    ) -> dict:
        """Example output:
        ```python
        {
            ... (input `return_res` key-vals),
            "suggest_info": dict,
            "rewrite_info": dict,
        }
        ```
        #ANCHOR[id=return_res]
        """
        if request_type == "suggest":
            suggest_info = self.suggest_parser.parse(
                qwords=query_info["keywords_body"],
                hits=return_res["hits"],
            )
            rewrite_info = self.query_rewriter.rewrite_query_info_by_suggest_info(
                query_info, suggest_info
            )
        else:
            suggest_info = suggest_info or {}
            rewrite_info = rewrite_info or {}
        return_res["suggest_info"] = suggest_info
        return_res["rewrite_info"] = rewrite_info
        return return_res

    def get_info_of_query_rewrite_dsl(
        self,
        query: str,
        suggest_info: dict = {},
        boosted_match_fields: list[str] = SEARCH_MATCH_FIELDS,
        boosted_date_fields: list[str] = DATE_MATCH_FIELDS,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        extra_filters: list[dict] = [],
    ) -> tuple[dict, dict, dict]:
        """return `(query_info, rewrite_info, query_dsl_dict)`"""
        # get query_info, and get rewrite_info with suggest_info
        query_info = self.query_rewriter.get_query_info(query)
        rewrite_info = self.query_rewriter.rewrite_query_info_by_suggest_info(
            query_info, suggest_info
        )
        # get expr_tree, and construct query_dsl_dict from expr_tree
        expr_tree = rewrite_info.get("rewrited_expr_tree", None)
        expr_tree = expr_tree or query_info.get("query_expr_tree", None)
        self.elastic_converter.word_converter.switch_mode(
            match_fields=boosted_match_fields,
            date_match_fields=boosted_date_fields,
            match_type=match_type,
        )
        query_dsl_dict = self.elastic_converter.expr_tree_to_dict(expr_tree)
        # merge extra_filters to query_dsl_dict
        query_dsl_dict = self.filter_merger.merge(query_dsl_dict, extra_filters)
        # logger.hint("> query_dsl_dict:")
        # logger.mesg(dict_to_str(query_dsl_dict, add_quotes=True, align_list=False))
        # logger.hint("> expr_tree:")
        # logger.mesg(expr_tree.yaml())
        return query_info, rewrite_info, query_dsl_dict

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
        es_res_dict = self.submit_to_es(agg_body)
        return es_res_dict

    def search(
        self,
        query: str,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        match_bool: MATCH_BOOL = SEARCH_MATCH_BOOL,
        match_operator: MATCH_OPERATOR = SEARCH_MATCH_OPERATOR,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        request_type: SEARCH_REQUEST_TYPE = SEARCH_REQUEST_TYPE_DEFAULT,
        parse_hits: bool = True,
        drop_no_highlights: bool = False,
        add_region_info: bool = True,
        add_highlights_info: bool = True,
        is_explain: bool = False,
        is_profile: bool = False,
        is_highlight: bool = IS_HIGHLIGHT,
        boost: bool = True,
        boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
        combined_fields_list: list[list[str]] = [],
        use_script_score: bool = USE_SCRIPT_SCORE_DEFAULT,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD_DEFAULT,
        score_threshold: float = None,
        use_pinyin: bool = False,
        detail_level: int = -1,
        detail_levels: dict = SEARCH_DETAIL_LEVELS,
        limit: int = SEARCH_LIMIT,
        rank_top_k: int = RANK_TOP_K,
        terminate_after: int = TERMINATE_AFTER,
        timeout: Union[int, float, str] = SEARCH_TIMEOUT,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        logger.enter_quiet(not verbose)
        # init params by detail_level
        if detail_level in detail_levels:
            match_detail = detail_levels[detail_level]
            match_type = match_detail["match_type"]
            match_bool = match_detail["bool"]
            match_operator = match_detail.get("operator", "or")
            use_pinyin = match_detail.get("pinyin", use_pinyin)
            extra_filters = match_detail.get("filters", extra_filters)
            timeout = match_detail.get("timeout", timeout)
        # construct boosted fields
        boosted_match_fields, boosted_date_fields = construct_boosted_fields(
            match_fields=match_fields,
            boost=boost,
            boosted_fields=boosted_fields,
            use_pinyin=use_pinyin,
        )
        query_rewrite_dsl_params = {
            "query": query,
            "suggest_info": suggest_info,
            "boosted_match_fields": boosted_match_fields,
            "boosted_date_fields": boosted_date_fields,
            "match_type": match_type,
            "extra_filters": extra_filters,
        }
        query_info, rewrite_info, query_dsl_dict = self.get_info_of_query_rewrite_dsl(
            **query_rewrite_dsl_params
        )
        # construct search_body
        search_body_params = {
            "query_dsl_dict": query_dsl_dict,
            "match_fields": boosted_match_fields,
            "source_fields": source_fields,
            "drop_no_highlights": drop_no_highlights,
            "is_explain": is_explain,
            "is_profile": is_profile,
            "is_highlight": is_highlight,
            "use_script_score": use_script_score,
            "score_threshold": score_threshold,
            "limit": limit,
            "terminate_after": terminate_after,
            "timeout": timeout,
        }
        search_body = self.construct_search_body(**search_body_params)
        logger.mesg(
            dict_to_str(search_body, add_quotes=True), indent=2, verbose=verbose
        )
        # submit search_body to es client
        es_res_dict = self.submit_to_es(search_body)
        # parse results
        if parse_hits:
            parse_res = self.hit_parser.parse(
                query_info,
                match_fields=match_fields,
                res_dict=es_res_dict,
                request_type=request_type,
                drop_no_highlights=drop_no_highlights,
                add_region_info=add_region_info,
                add_highlights_info=add_highlights_info,
                match_type=match_type,
                match_operator=match_operator,
                detail_level=detail_level,
                limit=limit,
                verbose=verbose,
            )
            if rank_method == "rrf":
                parse_res = self.hit_ranker.rrf_rank(parse_res, top_k=rank_top_k)
            elif rank_method == "stats":
                parse_res = self.hit_ranker.stats_rank(parse_res, top_k=rank_top_k)
            else:  # "heads"
                parse_res = self.hit_ranker.heads(parse_res, top_k=rank_top_k)
        else:
            parse_res = es_res_dict
        # rewrite_by_suggest, only apply for "suggest" request_type
        return_res = self.rewrite_by_suggest(
            query_info,
            suggest_info=suggest_info,
            rewrite_info=rewrite_info,
            request_type=request_type,
            return_res=parse_res,
        )
        return_res = self.post_process_return_res(parse_res)
        # Sanitize search_body to reduce network payload (removes large terms arrays)
        return_res["search_body"] = self.sanitize_search_body_for_client(search_body)
        logger.exit_quiet(not verbose)
        return return_res

    def get_filters_from_query(
        self,
        query: str,
        extra_filters: list[dict] = [],
    ) -> tuple[dict, list[dict]]:
        """Extract filter clauses from DSL query expression.

        This extracts non-word filters (date, stat, user, etc.) from the query
        and combines them with extra_filters for use in KNN search.

        Args:
            query: The query string (may contain DSL expressions like "d>2024-01-01").
            extra_filters: Additional filter clauses to include.

        Returns:
            Tuple of (query_info, filter_clauses).
        """
        query_info = self.query_rewriter.get_query_info(query)
        expr_tree = query_info.get("query_expr_tree", None)

        if expr_tree is None:
            return query_info, extra_filters

        # Extract filter atoms (non-word expressions) from expr_tree
        filter_expr_tree = expr_tree.filter_atoms_by_keys(exclude_keys=["word_expr"])

        # Convert filter expr_tree to elastic filter dict
        if filter_expr_tree and filter_expr_tree.children:
            self.elastic_converter.word_converter.switch_mode(
                match_fields=[],
                date_match_fields=[],
                match_type="cross_fields",
            )
            filter_dict = self.elastic_converter.expr_tree_to_dict(filter_expr_tree)

            # Extract filter clauses from the dict
            filter_clauses = self.filter_merger.get_query_filters_from_query_dsl_dict(
                filter_dict
            )
        else:
            filter_clauses = []

        # Combine with extra_filters
        all_filters = filter_clauses + list(extra_filters)
        return query_info, all_filters

    def construct_knn_search_body(
        self,
        query_vector: list[int],
        filter_clauses: list[dict] = None,
        source_fields: list[str] = SOURCE_FIELDS,
        knn_field: str = KNN_TEXT_EMB_FIELD,
        k: int = KNN_K,
        num_candidates: int = KNN_NUM_CANDIDATES,
        similarity: float = None,
        limit: int = SEARCH_LIMIT,
        timeout: Union[int, float, str] = KNN_TIMEOUT,
        is_explain: bool = False,
    ) -> dict:
        """Construct KNN search body.

        Args:
            query_vector: Query vector as byte array (signed int8 list).
            filter_clauses: Filter clauses to apply during KNN search.
            source_fields: Fields to include in _source.
            knn_field: The dense_vector field to search.
            k: Number of nearest neighbors to return.
            num_candidates: Number of candidates to consider per shard.
            similarity: Minimum similarity threshold.
            limit: Maximum number of results.
            timeout: Search timeout.
            is_explain: Whether to include explanation.

        Returns:
            Complete search body dict.
        """
        knn_query = construct_knn_query(
            query_vector=query_vector,
            field=knn_field,
            k=k,
            num_candidates=num_candidates,
            filter_clauses=filter_clauses,
            similarity=similarity,
        )

        search_body = construct_knn_search_body(
            knn_query=knn_query,
            source_fields=source_fields,
            size=limit,
            timeout=timeout,
            track_total_hits=TRACK_TOTAL_HITS,
            is_explain=is_explain,
        )

        return search_body

    def knn_search(
        self,
        query: str,
        source_fields: list[str] = SOURCE_FIELDS,
        extra_filters: list[dict] = [],
        knn_field: str = KNN_TEXT_EMB_FIELD,
        k: int = KNN_K,
        num_candidates: int = KNN_NUM_CANDIDATES,
        similarity: float = None,
        parse_hits: bool = True,
        add_region_info: bool = True,
        is_explain: bool = False,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD_DEFAULT,
        limit: int = SEARCH_LIMIT,
        rank_top_k: int = RANK_TOP_K,
        timeout: Union[int, float, str] = KNN_TIMEOUT,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        """Perform KNN search using text embeddings.

        This method:
        1. Extracts filter clauses from DSL query (date, stat, user filters)
        2. Extracts query words and converts them to embedding vector
        3. Performs KNN search on text_emb field with filters
        4. Parses and ranks the results

        Args:
            query: Query string (can include DSL expressions for filtering).
            source_fields: Fields to include in results.
            extra_filters: Additional filter clauses.
            knn_field: The dense_vector field to search.
            k: Number of nearest neighbors.
            num_candidates: Candidates per shard.
            similarity: Minimum similarity threshold.
            parse_hits: Whether to parse hits into structured format.
            add_region_info: Whether to add region info to results.
            is_explain: Whether to include ES explanation.
            rank_method: Ranking method ("heads", "rrf", "stats").
            limit: Maximum results to return.
            rank_top_k: Top-k for ranking.
            timeout: Search timeout.
            verbose: Enable verbose logging.

        Returns:
            Search results dict with hits and metadata.
        """
        logger.enter_quiet(not verbose)

        # Extract query info and filter clauses from DSL query
        query_info, filter_clauses = self.get_filters_from_query(
            query=query,
            extra_filters=extra_filters,
        )

        # Get query words for embedding
        words_expr = query_info.get("words_expr", "")
        keywords_body = query_info.get("keywords_body", [])

        # Build text for embedding from query words
        if keywords_body:
            embed_text = " ".join(keywords_body)
        else:
            embed_text = words_expr or query

        # Convert query text to embedding vector
        if not self.embed_client.is_available():
            logger.warn("× Embed client not available, KNN search cannot proceed")
            logger.exit_quiet(not verbose)
            return {
                "query": query,
                "request_type": "knn_search",
                "timed_out": True,
                "total_hits": 0,
                "return_hits": 0,
                "hits": [],
                "query_info": query_info,
                "error": "Embed client not available",
            }

        query_hex = self.embed_client.text_to_hex(embed_text)
        if not query_hex:
            logger.warn("× Failed to get embedding for query")
            logger.exit_quiet(not verbose)
            return {
                "query": query,
                "request_type": "knn_search",
                "timed_out": True,
                "total_hits": 0,
                "return_hits": 0,
                "hits": [],
                "query_info": query_info,
                "error": "Failed to compute embedding",
            }

        # Convert hex string to byte array for ES
        query_vector = self.embed_client.hex_to_byte_array(query_hex)

        # Construct KNN search body
        search_body = self.construct_knn_search_body(
            query_vector=query_vector,
            filter_clauses=filter_clauses if filter_clauses else None,
            source_fields=source_fields,
            knn_field=knn_field,
            k=k,
            num_candidates=num_candidates,
            similarity=similarity,
            limit=limit,
            timeout=timeout,
            is_explain=is_explain,
        )

        logger.mesg(
            dict_to_str(search_body, add_quotes=True), indent=2, verbose=verbose
        )

        # Submit to ES
        es_res_dict = self.submit_to_es(search_body)

        # Parse results
        if parse_hits:
            parse_res = self.hit_parser.parse(
                query_info,
                match_fields=[],  # No highlight for KNN search
                res_dict=es_res_dict,
                request_type="knn_search",
                drop_no_highlights=False,
                add_region_info=add_region_info,
                add_highlights_info=False,  # KNN search doesn't use keyword highlights
                match_type="cross_fields",
                match_operator="or",
                detail_level=-1,
                limit=limit,
                verbose=verbose,
            )
            # Apply ranking - for KNN search, "relevance" is the default and preferred method
            if rank_method == "relevance":
                parse_res = self.hit_ranker.relevance_rank(parse_res, top_k=rank_top_k)
            elif rank_method == "rrf":
                parse_res = self.hit_ranker.rrf_rank(parse_res, top_k=rank_top_k)
            elif rank_method == "stats":
                parse_res = self.hit_ranker.stats_rank(parse_res, top_k=rank_top_k)
            else:  # "heads"
                parse_res = self.hit_ranker.heads(parse_res, top_k=rank_top_k)
        else:
            parse_res = es_res_dict

        # Build return result
        return_res = {
            **(parse_res if isinstance(parse_res, dict) else {}),
            "query_info": query_info,
            "suggest_info": {},
            "rewrite_info": {},
            # Sanitize search_body to reduce network payload
            "search_body": self.sanitize_search_body_for_client(search_body),
        }

        # Clean up non-jsonable items
        return_res["query_info"].pop("query_expr_tree", None)

        logger.exit_quiet(not verbose)
        return return_res

    def get_qmod_from_query(self, query: str) -> list[str]:
        """Extract query mode (qmod) from query string.

        Parses the query for q=w/v/wv expression and returns the mode list.

        Args:
            query: Query string that may contain q=w/v/wv.

        Returns:
            List of query modes, e.g. ["word"], ["vector"], ["word", "vector"].
        """
        try:
            expr_tree = self.elastic_converter.construct_expr_tree(query)
            return extract_qmod_from_expr_tree(expr_tree)
        except Exception as e:
            logger.warn(f"× Failed to parse qmod: {e}")
            return (
                QMOD_DEFAULT.copy()
                if isinstance(QMOD_DEFAULT, list)
                else [QMOD_DEFAULT]
            )

    def hybrid_search(
        self,
        query: str,
        source_fields: list[str] = SOURCE_FIELDS,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        # Word search params
        boost: bool = True,
        boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
        use_script_score: bool = USE_SCRIPT_SCORE_DEFAULT,
        # KNN params
        knn_field: str = KNN_TEXT_EMB_FIELD,
        knn_k: int = KNN_K,
        knn_num_candidates: int = KNN_NUM_CANDIDATES,
        # Hybrid fusion params
        word_weight: float = HYBRID_WORD_WEIGHT,
        vector_weight: float = HYBRID_VECTOR_WEIGHT,
        rrf_k: int = HYBRID_RRF_K,
        fusion_method: Literal["rrf", "weighted"] = "rrf",
        # Common params
        parse_hits: bool = True,
        add_region_info: bool = True,
        add_highlights_info: bool = True,
        is_highlight: bool = IS_HIGHLIGHT,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD_DEFAULT,
        limit: int = SEARCH_LIMIT,
        rank_top_k: int = RANK_TOP_K,
        timeout: Union[int, float, str] = SEARCH_TIMEOUT,
        verbose: bool = False,
    ) -> dict:
        """Perform hybrid search combining word-based and vector-based retrieval.

        This method:
        1. Performs word-based search using traditional ES query
        2. Performs KNN vector search using text embeddings
        3. Fuses results using RRF or weighted combination

        Args:
            query: Query string (can include DSL expressions).
            source_fields: Fields to include in results.
            match_fields: Fields for word matching.
            match_type: Type of matching for word search.
            extra_filters: Additional filter clauses.
            suggest_info: Suggestion info from previous searches.
            boost: Whether to boost fields.
            boosted_fields: Field boost weights.
            use_script_score: Whether to use script scoring.
            knn_field: Dense vector field for KNN.
            knn_k: Number of KNN neighbors.
            knn_num_candidates: KNN candidates per shard.
            word_weight: Weight for word-based scores (weighted fusion).
            vector_weight: Weight for vector-based scores (weighted fusion).
            rrf_k: K parameter for RRF fusion.
            fusion_method: "rrf" or "weighted".
            parse_hits: Whether to parse hits.
            add_region_info: Whether to add region info.
            add_highlights_info: Whether to add highlight info.
            is_highlight: Whether to highlight matches.
            rank_method: Final ranking method.
            limit: Maximum results.
            rank_top_k: Top-k for ranking.
            timeout: Search timeout.
            verbose: Enable verbose logging.

        Returns:
            Search results with fused hits from both methods.
        """
        logger.enter_quiet(not verbose)

        # Get query info
        query_info = self.query_rewriter.get_query_info(query)

        # Execute word search and KNN search in parallel
        logger.hint("> Hybrid: Running word search and KNN search in parallel ...")

        # Cap word search limit to avoid excessive results (max 2000 for word search)
        word_limit = min(limit, 2000)
        word_search_params = {
            "query": query,
            "source_fields": source_fields,
            "match_fields": match_fields,
            "match_type": match_type,
            "extra_filters": extra_filters,
            "suggest_info": suggest_info,
            "boost": boost,
            "boosted_fields": boosted_fields,
            "use_script_score": use_script_score,
            "parse_hits": True,
            "add_region_info": add_region_info,
            "add_highlights_info": add_highlights_info,
            "is_highlight": is_highlight,
            "rank_method": "heads",  # Use heads, we'll re-rank after fusion
            "limit": word_limit,
            "rank_top_k": word_limit,
            "timeout": timeout,
            "verbose": verbose,
        }

        # KNN k must not exceed num_candidates, and we cap limit to avoid excessive results
        knn_limit = min(limit, knn_num_candidates)
        knn_search_params = {
            "query": query,
            "source_fields": source_fields,
            "extra_filters": extra_filters,
            "knn_field": knn_field,
            "k": knn_limit,  # k must be <= num_candidates
            "num_candidates": knn_num_candidates,
            "parse_hits": True,
            "add_region_info": add_region_info,
            "rank_method": "heads",  # Use heads, we'll re-rank after fusion
            "limit": knn_limit,
            "rank_top_k": knn_limit,
            "timeout": timeout,
            "verbose": verbose,
        }

        # Use ThreadPoolExecutor to run both searches in parallel
        word_search_res = {}
        knn_search_res = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both search tasks
            word_future = executor.submit(self.search, **word_search_params)
            knn_future = executor.submit(self.knn_search, **knn_search_params)

            # Wait for both to complete
            word_search_res = word_future.result()
            knn_search_res = knn_future.result()

        logger.hint("> Hybrid: Both searches completed")

        # Fuse results
        logger.hint("> Hybrid: Fusing results ...")
        word_hits = word_search_res.get("hits", [])
        knn_hits = knn_search_res.get("hits", [])

        # Use capped limit for fusion to avoid excessive results
        fusion_limit = min(limit, max(len(word_hits), len(knn_hits), 1000))

        if fusion_method == "rrf":
            fused_hits = self._rrf_fusion(
                word_hits, knn_hits, k=rrf_k, limit=fusion_limit
            )
        else:
            fused_hits = self._weighted_fusion(
                word_hits,
                knn_hits,
                word_weight=word_weight,
                vector_weight=vector_weight,
                limit=fusion_limit,
            )

        # Build result dict
        parse_res = {
            "timed_out": word_search_res.get("timed_out", False)
            or knn_search_res.get("timed_out", False),
            "total_hits": max(
                word_search_res.get("total_hits", 0),
                knn_search_res.get("total_hits", 0),
            ),
            "return_hits": len(fused_hits),
            "hits": fused_hits,
            "word_hits_count": len(word_hits),
            "knn_hits_count": len(knn_hits),
            "fusion_method": fusion_method,
        }

        # Apply final ranking
        # For hybrid search, RRF fusion already produces a well-ranked list
        # So we use "heads" to preserve the RRF order, not "relevance" which would filter incorrectly
        if rank_method == "relevance":
            # For hybrid search with relevance ranking, just take top-k by hybrid_score
            # Don't apply min_score filtering because hybrid_score has different scale
            parse_res = self.hit_ranker.heads(parse_res, top_k=rank_top_k)
            parse_res["rank_method"] = "relevance"
        elif rank_method == "rrf":
            parse_res = self.hit_ranker.rrf_rank(parse_res, top_k=rank_top_k)
        elif rank_method == "stats":
            parse_res = self.hit_ranker.stats_rank(parse_res, top_k=rank_top_k)
        else:  # "heads"
            parse_res = self.hit_ranker.heads(parse_res, top_k=rank_top_k)

        # Build return result
        return_res = {
            **parse_res,
            "query_info": query_info,
            "suggest_info": word_search_res.get("suggest_info", {}),
            "rewrite_info": word_search_res.get("rewrite_info", {}),
        }

        # Clean up non-jsonable items
        return_res["query_info"].pop("query_expr_tree", None)

        logger.exit_quiet(not verbose)
        return return_res

    def _rrf_fusion(
        self,
        word_hits: list[dict],
        knn_hits: list[dict],
        k: int = HYBRID_RRF_K,
        limit: int = SEARCH_LIMIT,
    ) -> list[dict]:
        """Fuse results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) for each result list.

        Args:
            word_hits: Hits from word-based search.
            knn_hits: Hits from KNN search.
            k: RRF k parameter (default 60).
            limit: Maximum results to return.

        Returns:
            Fused and sorted list of hits.
        """
        # Build bvid -> hit mapping and calculate RRF scores
        hit_map = {}  # bvid -> hit dict
        rrf_scores = {}  # bvid -> rrf score

        # Process word hits
        for rank, hit in enumerate(word_hits, start=1):
            bvid = hit.get("bvid")
            if not bvid:
                continue
            hit_map[bvid] = hit
            rrf_scores[bvid] = rrf_scores.get(bvid, 0) + 1 / (k + rank)
            hit_map[bvid]["word_rank"] = rank

        # Process KNN hits
        for rank, hit in enumerate(knn_hits, start=1):
            bvid = hit.get("bvid")
            if not bvid:
                continue
            if bvid not in hit_map:
                hit_map[bvid] = hit
            rrf_scores[bvid] = rrf_scores.get(bvid, 0) + 1 / (k + rank)
            hit_map[bvid]["knn_rank"] = rank

        # Sort by RRF score
        sorted_bvids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        # Build result list
        fused_hits = []
        for bvid in sorted_bvids[:limit]:
            hit = hit_map[bvid]
            hit["hybrid_score"] = rrf_scores[bvid]
            hit["fusion_method"] = "rrf"
            fused_hits.append(hit)

        return fused_hits

    def _weighted_fusion(
        self,
        word_hits: list[dict],
        knn_hits: list[dict],
        word_weight: float = HYBRID_WORD_WEIGHT,
        vector_weight: float = HYBRID_VECTOR_WEIGHT,
        limit: int = SEARCH_LIMIT,
    ) -> list[dict]:
        """Fuse results using weighted score combination.

        Args:
            word_hits: Hits from word-based search.
            knn_hits: Hits from KNN search.
            word_weight: Weight for word scores.
            vector_weight: Weight for vector scores.
            limit: Maximum results.

        Returns:
            Fused and sorted list of hits.
        """
        # Normalize weights
        total_weight = word_weight + vector_weight
        word_weight = word_weight / total_weight
        vector_weight = vector_weight / total_weight

        # Get max scores for normalization
        word_max_score = (
            max((h.get("score", 0) or 0 for h in word_hits), default=1) or 1
        )
        knn_max_score = max((h.get("score", 0) or 0 for h in knn_hits), default=1) or 1

        # Build mappings
        hit_map = {}  # bvid -> hit
        word_scores = {}  # bvid -> normalized word score
        knn_scores = {}  # bvid -> normalized knn score

        for hit in word_hits:
            bvid = hit.get("bvid")
            if not bvid:
                continue
            hit_map[bvid] = hit
            word_scores[bvid] = (hit.get("score", 0) or 0) / word_max_score

        for hit in knn_hits:
            bvid = hit.get("bvid")
            if not bvid:
                continue
            if bvid not in hit_map:
                hit_map[bvid] = hit
            knn_scores[bvid] = (hit.get("score", 0) or 0) / knn_max_score

        # Calculate weighted scores
        weighted_scores = {}
        for bvid in hit_map:
            w_score = word_scores.get(bvid, 0)
            k_score = knn_scores.get(bvid, 0)
            weighted_scores[bvid] = word_weight * w_score + vector_weight * k_score

        # Sort by weighted score
        sorted_bvids = sorted(
            weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True
        )

        # Build result list
        fused_hits = []
        for bvid in sorted_bvids[:limit]:
            hit = hit_map[bvid]
            hit["hybrid_score"] = weighted_scores[bvid]
            hit["word_score_norm"] = word_scores.get(bvid, 0)
            hit["knn_score_norm"] = knn_scores.get(bvid, 0)
            hit["fusion_method"] = "weighted"
            fused_hits.append(hit)

        return fused_hits
