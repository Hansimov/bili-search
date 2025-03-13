from copy import deepcopy
from pprint import pformat
from sedb import ElasticOperator
from tclogger import logger, logstr, brk, dict_to_str, get_now, tcdatetime
from typing import Union, Literal

from configs.envs import ELASTIC_ENVS
from converters.query.filter import QueryFilterExtractor
from converters.query.dsl import MultiMatchQueryDSLConstructor
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.query.rewrite import QueryRewriter
from converters.query.field import is_pinyin_field, deboost_field
from converters.query.field import remove_suffixes_from_fields
from elastics.videos.constants import VIDEOS_INDEX_DEFAULT
from elastics.videos.constants import SEARCH_REQUEST_TYPE, SEARCH_REQUEST_TYPE_DEFAULT
from elastics.videos.constants import SOURCE_FIELDS, DOC_EXCLUDED_SOURCE_FIELDS
from elastics.videos.constants import SEARCH_MATCH_FIELDS, SEARCH_BOOSTED_FIELDS
from elastics.videos.constants import SUGGEST_MATCH_FIELDS, SUGGEST_BOOSTED_FIELDS
from elastics.videos.constants import DATE_MATCH_FIELDS, DATE_BOOSTED_FIELDS
from elastics.videos.constants import MATCH_TYPE, MATCH_BOOL, MATCH_OPERATOR
from elastics.videos.constants import SEARCH_MATCH_TYPE, SUGGEST_MATCH_TYPE
from elastics.videos.constants import SEARCH_MATCH_BOOL, SEARCH_MATCH_OPERATOR
from elastics.videos.constants import SUGGEST_MATCH_BOOL, SUGGEST_MATCH_OPERATOR
from elastics.videos.constants import SEARCH_DETAIL_LEVELS, MAX_SEARCH_DETAIL_LEVEL
from elastics.videos.constants import SUGGEST_DETAIL_LEVELS, MAX_SUGGEST_DETAIL_LEVEL
from elastics.videos.constants import SEARCH_LIMIT, SUGGEST_LIMIT
from elastics.videos.constants import SEARCH_TIMEOUT, SUGGEST_TIMEOUT
from elastics.videos.constants import NO_HIGHLIGHT_REDUNDANCE_RATIO
from elastics.videos.hits import VideoHitsParser, SuggestInfoParser


class VideoSearcherV1:
    def __init__(self, index_name: str = VIDEOS_INDEX_DEFAULT):
        self.index_name = index_name
        self.es = ElasticOperator(
            ELASTIC_ENVS,
            connect_msg=f"{logstr.mesg(self.__class__.__name__)} -> {logstr.mesg(brk('elastic'))}",
        )
        self.init_processors()

    def init_processors(self):
        self.hit_parser = VideoHitsParser()
        self.suggest_parser = SuggestInfoParser()
        self.query_rewriter = QueryRewriter()

    def get_highlight_settings(
        self,
        match_fields: list[str],
        removable_suffixes: list[str] = [".words"],
        tag: str = "hit",
    ):
        highlight_fields = [
            deboost_field(field) for field in match_fields if not is_pinyin_field(field)
        ]
        if removable_suffixes:
            highlight_fields.extend(
                remove_suffixes_from_fields(
                    highlight_fields, suffixes=removable_suffixes
                )
            )

        highlight_fields = sorted(list(set(highlight_fields)))
        highlight_fields_dict = {field: {} for field in highlight_fields}

        highlight_settings = {
            "pre_tags": [f"<{tag}>"],
            "post_tags": [f"</{tag}>"],
            "fields": highlight_fields_dict,
        }
        return highlight_settings

    def boost_fields(self, match_fields: list, boosted_fields: dict):
        boosted_match_fields = deepcopy(match_fields)
        for key in boosted_fields:
            if key in boosted_match_fields:
                key_index = boosted_match_fields.index(key)
                boosted_match_fields[key_index] += f"^{boosted_fields[key]}"
        return boosted_match_fields

    def construct_boosted_fields(
        self,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        boost: bool = True,
        boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
        use_pinyin: bool = False,
    ) -> tuple[list[str], list[str]]:
        if not use_pinyin:
            match_fields = [
                field for field in match_fields if not field.endswith(".pinyin")
            ]
        date_fields = [
            field
            for field in match_fields
            if not field.endswith(".pinyin")
            and any(field.startswith(date_field) for date_field in DATE_MATCH_FIELDS)
        ]
        if boost:
            boosted_match_fields = self.boost_fields(match_fields, boosted_fields)
            boosted_date_fields = self.boost_fields(date_fields, DATE_BOOSTED_FIELDS)
        else:
            boosted_match_fields = match_fields
            boosted_date_fields = date_fields
        return boosted_match_fields, boosted_date_fields

    def submit_and_parse(
        self,
        query_info: dict,
        search_body: dict,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        parse_hits: bool = True,
        request_type: SEARCH_REQUEST_TYPE = SEARCH_REQUEST_TYPE_DEFAULT,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        match_operator: MATCH_OPERATOR = SEARCH_MATCH_OPERATOR,
        detail_level: int = -1,
        limit: int = SEARCH_LIMIT,
        timeout: Union[int, float, str] = SEARCH_TIMEOUT,
        verbose: bool = False,
    ) -> dict:
        if timeout:
            if isinstance(timeout, str):
                search_body["timeout"] = timeout
            elif isinstance(timeout, int) or isinstance(timeout, float):
                timeout_str = round(timeout * 1000)
                search_body["timeout"] = f"{timeout_str}ms"
            else:
                logger.warn(f"× Invalid type of `timeout`: {type(timeout)}")
        logger.note(dict_to_str(search_body, add_quotes=True, align_list=False))
        if limit and limit > 0:
            search_body["size"] = int(limit * NO_HIGHLIGHT_REDUNDANCE_RATIO)

        logger.note(f"> Get search results by query:", end=" ")
        logger.mesg(f"[{query_info['query']}]")
        try:
            res = self.es.client.search(index=self.index_name, body=search_body)
            res_dict = res.body
        except Exception as e:
            logger.warn(f"× Error: {e}")
            res_dict = {}

        if parse_hits:
            return_res = self.hit_parser.parse(
                query_info,
                match_fields,
                res_dict,
                request_type=request_type,
                drop_no_highlights=True,
                match_type=match_type,
                match_operator=match_operator,
                detail_level=detail_level,
                limit=limit,
                verbose=verbose,
            )
        else:
            logger.mesg(dict_to_str(res_dict))
            return_res = res_dict
        return return_res

    def rewrite_with_suggest(
        self, query_info: dict, suggest_info: dict
    ) -> tuple[dict, str]:
        """if suggest_info is provided, get rewrite_info and keywords_rewrited, which is used in construct query_dsl_dict;
        and if suggest_info is not provided, return empty rewrite_info and original keywords in query_info.
        """
        if suggest_info:
            rewrite_info = self.query_rewriter.rewrite(query_info, suggest_info)
            rewrite_list = rewrite_info.get("list", [])
            if rewrite_list:
                keywords_rewrited = rewrite_list[0]
            else:
                keywords_rewrited = " ".join(query_info["keywords"])
        else:
            rewrite_info = {}
            keywords_rewrited = " ".join(query_info["keywords"])
        return rewrite_info, keywords_rewrited

    def suggest_and_rewrite(
        self,
        query_info: dict,
        suggest_info: dict = {},
        rewrite_info: dict = {},
        request_type: SEARCH_REQUEST_TYPE = SEARCH_REQUEST_TYPE_DEFAULT,
        return_res: dict = {},
    ) -> dict:
        """if request_type is "suggest", parse suggest_info, and get rewrite_info:
        - as in most cases, when request_type is "suggest", suggest_info is not provided,
        - so in order to provide suggest_info for next search, we need to add this info
        And if reqeust_type is "search" and the rewrite_info is provided, reuse it.
        Then add suggest_info and rewrite_info to return_res.
        """
        if request_type == "suggest":
            qwords = query_info["keywords_body"]
            suggest_info = self.suggest_parser.parse(
                qwords=qwords, hits=return_res["hits"]
            )
            rewrite_info = self.query_rewriter.rewrite(
                query_info=query_info, suggest_info=suggest_info
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
        match_bool: MATCH_BOOL = SEARCH_MATCH_BOOL,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        match_operator: MATCH_OPERATOR = SEARCH_MATCH_OPERATOR,
        extra_filters: list[dict] = [],
        combined_fields_list: list[list[str]] = [],
        **kwargs,  # for compatibility with different input params
    ) -> tuple[dict, dict, dict]:
        """This is version v1, which uses regex to extract query_info, convert rewrite_info, and construct query_dsl_dict."""
        # get query_info and rewrite_info
        query_info_extractor = QueryFilterExtractor()
        query_info = query_info_extractor.split_keyword_and_filter_expr(query)
        rewrite_info, keywords_rewrited = self.rewrite_with_suggest(
            query_info=query_info, suggest_info=suggest_info
        )
        # construct query_dsl_dict from keywords_rewrited
        query_constructor = MultiMatchQueryDSLConstructor()
        query_dsl_dict = query_constructor.construct(
            keywords_rewrited,
            match_fields=boosted_match_fields,
            date_match_fields=boosted_date_fields,
            match_bool=match_bool,
            match_type=match_type,
            match_operator=match_operator,
            combined_fields_list=combined_fields_list,
        )
        # add filter_dicts and extra_filters to query_dsl_dict
        _, filter_dicts = query_info_extractor.construct(query)
        if filter_dicts or extra_filters:
            query_dsl_dict["bool"]["filter"] = filter_dicts + extra_filters
        return query_info, rewrite_info, query_dsl_dict

    def construct_search_body(
        self,
        query_dsl_dict: dict,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        is_explain: bool = False,
        use_script_score: bool = True,
    ) -> dict:
        """construct script_score or rrf dict from query_dsl_dict, and return search_body"""
        script_score_constructor = ScriptScoreQueryDSLConstructor()
        common_params = {
            "_source": source_fields,
            "explain": is_explain,
            "track_total_hits": True,
        }
        if use_script_score:
            scripted_query_dsl_dict = script_score_constructor.construct(query_dsl_dict)
            search_body = {
                "query": scripted_query_dsl_dict,
                "highlight": self.get_highlight_settings(match_fields),
                **common_params,
            }
        else:
            rrf_dsl_dict = script_score_constructor.construct_rrf(query_dsl_dict)
            search_body = {**rrf_dsl_dict, **common_params}
        return search_body

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
        is_explain: bool = False,
        boost: bool = True,
        boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
        combined_fields_list: list[list[str]] = [],
        use_script_score: bool = True,
        use_pinyin: bool = False,
        detail_level: int = -1,
        detail_levels: dict = SEARCH_DETAIL_LEVELS,
        limit: int = SEARCH_LIMIT,
        timeout: Union[int, float, str] = SEARCH_TIMEOUT,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        """This is 1st version of `search`, which uses regex to parse query dsl and construct elastic dict.
        This version would be deprecated in the future. Use latest `search` instead.

        The main difference between `search` and `suggest` is that:
        - `search` has are more fuzzy (loose) and compositive match rules than `suggest`,
        and has more match fields.

        I have compared the results of different match types, and conclude that:
        - `phrase_prefix`: for precise and complete match
        - `most_fields`: for loose and composite match

        The main difference between "multi_match" and "multi matches in must" of `search` is that:
        - `multi matches in must` requires all keywords to be matched
        - `multi_match` only requires any keyword to be matched in any field.
        """
        # enter quiet
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
        boosted_match_fields, boosted_date_fields = self.construct_boosted_fields(
            match_fields=match_fields,
            boost=boost,
            boosted_fields=boosted_fields,
            use_pinyin=use_pinyin,
        )
        # this part is implemented with different versions:
        # - developer could customize `get_info_of_query_rewrite_dsl`
        # - v1 use regex, v2 use lark
        query_rewrite_dsl_params = {
            "query": query,
            "suggest_info": suggest_info,
            "boosted_match_fields": boosted_match_fields,
            "boosted_date_fields": boosted_date_fields,
            "match_bool": match_bool,
            "match_type": match_type,
            "match_operator": match_operator,
            "extra_filters": extra_filters,
            "combined_fields_list": combined_fields_list,
        }
        query_info, rewrite_info, query_dsl_dict = self.get_info_of_query_rewrite_dsl(
            **query_rewrite_dsl_params
        )
        # construct search_body
        search_body_params = {
            "query_dsl_dict": query_dsl_dict,
            "match_fields": boosted_match_fields,
            "source_fields": source_fields,
            "is_explain": is_explain,
            "use_script_score": use_script_score,
        }
        search_body = self.construct_search_body(**search_body_params)
        # submit search_body, parse results, and suggest and rewrite
        submit_and_parse_params = {
            "query_info": query_info,
            "search_body": search_body,
            "match_fields": match_fields,
            "match_type": match_type,
            "match_operator": match_operator,
            "request_type": request_type,
            "parse_hits": parse_hits,
            "detail_level": detail_level,
            "limit": limit,
            "timeout": timeout,
            "verbose": verbose,
        }
        return_res = self.submit_and_parse(**submit_and_parse_params)
        return_res = self.suggest_and_rewrite(
            query_info,
            suggest_info=suggest_info,
            rewrite_info=rewrite_info,
            request_type=request_type,
            return_res=return_res,
        )
        # exit quiet
        logger.exit_quiet(not verbose)
        return return_res

    def multi_level_search(
        self,
        query: str,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        match_bool: MATCH_BOOL = SEARCH_MATCH_BOOL,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        request_type: SEARCH_REQUEST_TYPE = SEARCH_REQUEST_TYPE_DEFAULT,
        parse_hits: bool = True,
        is_explain: bool = False,
        boost: bool = True,
        boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
        combined_fields_list: list[list[str]] = [],
        use_script_score: bool = True,
        use_pinyin: bool = False,
        detail_level: int = -1,
        detail_levels: dict = SEARCH_DETAIL_LEVELS,
        max_detail_level: int = MAX_SEARCH_DETAIL_LEVEL,
        limit: int = SEARCH_LIMIT,
        timeout: Union[int, float, str] = SEARCH_TIMEOUT,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        return_res = {
            "total_hits": 0,
            "return_hits": 0,
            "hits": [],
        }
        max_detail_level = min(max_detail_level, max(detail_levels.keys()))

        if detail_level < 1:
            detail_level = min(detail_levels.keys())
        elif detail_level > max_detail_level:
            return return_res
        else:
            detail_level = int(detail_level)

        while return_res["total_hits"] == 0 and detail_level <= max_detail_level:
            return_res = self.search(
                query=query,
                match_fields=match_fields,
                source_fields=source_fields,
                match_type=match_type,
                match_bool=match_bool,
                extra_filters=extra_filters,
                suggest_info=suggest_info,
                request_type=request_type,
                parse_hits=parse_hits,
                is_explain=is_explain,
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
            detail_level += 1
        return return_res

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
        is_explain: bool = False,
        boost: bool = True,
        boosted_fields: dict = SUGGEST_BOOSTED_FIELDS,
        combined_fields_list: list[list[str]] = [],
        use_script_score: bool = True,
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
            is_explain=is_explain,
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

    def multi_level_suggest(
        self,
        query: str,
        match_fields: list[str] = SUGGEST_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        match_type: MATCH_TYPE = SUGGEST_MATCH_TYPE,
        match_bool: MATCH_BOOL = SEARCH_MATCH_BOOL,
        parse_hits: bool = True,
        is_explain: bool = False,
        boost: bool = True,
        boosted_fields: dict = SUGGEST_BOOSTED_FIELDS,
        combined_fields_list: list[list[str]] = [],
        use_script_score: bool = True,
        use_pinyin: bool = True,
        detail_level: int = -1,
        detail_levels: dict = SUGGEST_DETAIL_LEVELS,
        max_detail_level: int = MAX_SUGGEST_DETAIL_LEVEL,
        limit: int = SUGGEST_LIMIT,
        timeout: Union[int, float, str] = SUGGEST_TIMEOUT,
        verbose: bool = False,
    ):
        return self.multi_level_search(
            query=query,
            match_fields=match_fields,
            source_fields=source_fields,
            match_type=match_type,
            match_bool=match_bool,
            request_type="suggest",
            parse_hits=parse_hits,
            is_explain=is_explain,
            boost=boost,
            boosted_fields=boosted_fields,
            combined_fields_list=combined_fields_list,
            use_script_score=use_script_score,
            use_pinyin=use_pinyin,
            detail_level=detail_level,
            detail_levels=detail_levels,
            max_detail_level=max_detail_level,
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
            return_res = self.hit_parser.parse("", [], res_dict, request_type="random")
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
            return_res = self.hit_parser.parse("", [], res_dict, request_type="latest")
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

        pudate_str = tcdatetime.fromtimestamp(res_dict.get("pubdate", 0)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        res_dict["pubdate_str"] = pudate_str

        reduced_dict = {
            k: v
            for k, v in res_dict.items()
            if k in ["title", "bvid", "pubdate_str", "desc"]
        }
        logger.success(pformat(reduced_dict, indent=4, sort_dicts=False))
        logger.exit_quiet(not verbose)
        return res_dict
VideoSearcher = VideoSearcherV1
