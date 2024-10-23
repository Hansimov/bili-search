from copy import deepcopy
from datetime import datetime
from pprint import pformat
from tclogger import logger, logstr, dict_to_str
from typing import Union, Literal

from converters.query.filter import QueryFilterExtractor
from converters.query.dsl import MultiMatchQueryDSLConstructor
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.query.rewrite import QueryRewriter
from converters.query.field import is_pinyin_field, deboost_field
from converters.query.field import remove_suffixes_from_fields
from elastics.client import ElasticSearchClient
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
from elastics.videos.hits import VideoHitsParser


class VideoSearcher:
    def __init__(
        self, index_name: str = "bili_videos_dev3", elastic_verbose: bool = True
    ):
        self.index_name = index_name
        self.es = ElasticSearchClient(verbose=elastic_verbose)
        self.es.connect()
        self.hit_parser = VideoHitsParser()
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
        request_type: Literal["search", "suggest"] = "search",
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
        """
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
        logger.enter_quiet(not verbose)

        if detail_level in detail_levels:
            match_detail = detail_levels[detail_level]
            match_type = match_detail["match_type"]
            match_bool = match_detail["bool"]
            match_operator = match_detail.get("operator", "or")
            use_pinyin = match_detail.get("pinyin", use_pinyin)
            extra_filters = match_detail.get("filters", extra_filters)
            timeout = match_detail.get("timeout", timeout)

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
            boosted_fields = self.boost_fields(match_fields, boosted_fields)
            date_boosted_fields = self.boost_fields(date_fields, DATE_BOOSTED_FIELDS)
        else:
            boosted_fields = match_fields
            date_boosted_fields = date_fields

        filter_extractor = QueryFilterExtractor()
        query_keywords, filters = filter_extractor.construct(query)
        if suggest_info:
            query_keywords = self.query_rewriter.rewrite(query_keywords, suggest_info)
        query_without_filters = " ".join(query_keywords)

        query_constructor = MultiMatchQueryDSLConstructor()
        query_dsl_dict = query_constructor.construct(
            query_without_filters,
            match_fields=boosted_fields,
            date_match_fields=date_boosted_fields,
            match_bool=match_bool,
            match_type=match_type,
            match_operator=match_operator,
            combined_fields_list=combined_fields_list,
        )

        if filters or extra_filters:
            query_dsl_dict["bool"]["filter"] = filters + extra_filters

        script_score_constructor = ScriptScoreQueryDSLConstructor()

        if use_script_score:
            query_dsl_dict = script_score_constructor.construct(query_dsl_dict)
            search_body = {
                "query": query_dsl_dict,
                "_source": source_fields,
                "explain": is_explain,
                "highlight": self.get_highlight_settings(match_fields),
                "track_total_hits": True,
            }
        else:
            rrf_dsl_dict = script_score_constructor.construct_rrf(query_dsl_dict)
            search_body = {
                **rrf_dsl_dict,
                "_source": source_fields,
                "explain": is_explain,
                "track_total_hits": True,
            }

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
        logger.mesg(f"[{query}]")

        try:
            res = self.es.client.search(index=self.index_name, body=search_body)
            res_dict = res.body
        except Exception as e:
            logger.warn(f"× Error: {e}")
            res_dict = {}

        if parse_hits:
            return_res = self.hit_parser.parse(
                query,
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
        # logger.success(pformat(return_res, sort_dicts=False, indent=4))
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
        request_type: Literal["search", "suggest"] = "search",
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
        now = datetime.now()
        now_ts = int(now.timestamp())
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        today = datetime(year=now.year, month=now.month, day=now.day)
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

        pudate_str = datetime.fromtimestamp(res_dict.get("pubdate", 0)).strftime(
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
