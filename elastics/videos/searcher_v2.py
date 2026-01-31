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
from elastics.videos.hits import VideoHitsParser, SuggestInfoParser
from elastics.videos.ranker import VideoHitsRanker


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
        return_res["search_body"] = search_body
        logger.exit_quiet(not verbose)
        return return_res
