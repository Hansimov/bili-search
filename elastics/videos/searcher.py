from copy import deepcopy
from datetime import datetime
from pprint import pformat
from tclogger import logger, logstr, dict_to_str
from typing import Union

from converters.query.filter import QueryFilterExtractor
from converters.query.dsl import MultiMatchQueryDSLConstructor
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from elastics.client import ElasticSearchClient
from elastics.videos.constants import SOURCE_FIELDS, DOC_EXCLUDED_SOURCE_FIELDS
from elastics.videos.constants import SEARCH_MATCH_FIELDS, SEARCH_BOOSTED_FIELDS
from elastics.videos.constants import SUGGEST_MATCH_FIELDS, SUGGEST_BOOSTED_FIELDS
from elastics.videos.constants import DATE_BOOSTED_FIELDS
from elastics.videos.constants import SEARCH_COMBINED_FIELDS_LIST
from elastics.videos.constants import SUGGEST_COMBINED_FIELDS_LIST
from elastics.videos.constants import MATCH_TYPE, MATCH_BOOL, MATCH_OPERATOR
from elastics.videos.constants import SEARCH_MATCH_TYPE, SUGGEST_MATCH_TYPE
from elastics.videos.constants import SEARCH_MATCH_BOOL, SEARCH_MATCH_OPERATOR
from elastics.videos.constants import SEARCH_DETAIL_LEVELS, MAX_SEARCH_DETAIL_LEVEL
from elastics.videos.constants import SUGGEST_LIMIT, SEARCH_LIMIT
from elastics.videos.constants import NO_HIGHLIGHT_REDUNDANCE_RATIO
from elastics.videos.hits import VideoHitsParser


class VideoSearcher:
    def __init__(self, index_name: str = "bili_videos_dev2"):
        self.index_name = index_name
        self.es = ElasticSearchClient()
        self.es.connect()
        self.hit_parser = VideoHitsParser()

    def get_highlight_settings(self, match_fields: list[str], tag: str = "hit"):
        highlight_fields = [
            field.split("^", 1)[0]
            for field in match_fields
            if not field.endswith(".pinyin")
        ]
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
        parse_hits: bool = True,
        is_explain: bool = False,
        boost: bool = True,
        boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
        combined_fields_list: list[list[str]] = [],
        use_script_score: bool = True,
        use_pinyin: bool = False,
        detail_level: int = -1,
        limit: int = SEARCH_LIMIT,
        timeout: Union[int, float, str] = 2,
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

        if detail_level in SEARCH_DETAIL_LEVELS:
            match_detail = SEARCH_DETAIL_LEVELS[detail_level]
            match_type = match_detail["match_type"]
            match_bool = match_detail["bool"]
            match_operator = match_detail.get("operator", "or")
            use_pinyin = match_detail.get("pinyin", use_pinyin)

        if not use_pinyin:
            match_fields = [
                field for field in match_fields if not field.endswith(".pinyin")
            ]

        if boost:
            boosted_fields = self.boost_fields(match_fields, boosted_fields)
            date_fields = [
                field for field in match_fields if not field.endswith(".pinyin")
            ]
            date_boosted_fields = self.boost_fields(date_fields, DATE_BOOSTED_FIELDS)
        else:
            boosted_fields = match_fields
            date_boosted_fields = match_fields

        filter_extractor = QueryFilterExtractor()
        query_keywords, filters = filter_extractor.construct(query)
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

        if filters:
            query_dsl_dict["bool"]["filter"] = filters

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

        logger.note(
            dict_to_str(search_body, add_quotes=True, is_colored=True, align_list=False)
        )
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
                request_type="search",
                drop_no_highlights=True,
                match_type=match_type,
                match_operator=match_operator,
                detail_level=detail_level,
                limit=limit,
                verbose=verbose,
            )
        else:
            return_res = res_dict
        # logger.success(pformat(return_res, sort_dicts=False, indent=4))
        logger.exit_quiet(not verbose)
        return return_res

    def detailed_search(
        self,
        query: str,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        match_bool: MATCH_BOOL = SEARCH_MATCH_BOOL,
        parse_hits: bool = True,
        is_explain: bool = False,
        boost: bool = True,
        boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
        combined_fields_list: list[list[str]] = [],
        detail_level: int = -1,
        max_detail_level: int = MAX_SEARCH_DETAIL_LEVEL,
        limit: int = SEARCH_LIMIT,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        return_res = {
            "total_hits": 0,
            "return_hits": 0,
            "hits": [],
        }
        max_detail_level = min(max_detail_level, max(SEARCH_DETAIL_LEVELS.keys()))

        if detail_level < 1:
            detail_level = min(SEARCH_DETAIL_LEVELS.keys())
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
                parse_hits=parse_hits,
                is_explain=is_explain,
                boost=boost,
                boosted_fields=boosted_fields,
                combined_fields_list=combined_fields_list,
                detail_level=detail_level,
                limit=limit,
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
        match_bool: MATCH_BOOL = SEARCH_MATCH_BOOL,
        match_operator: MATCH_OPERATOR = SEARCH_MATCH_OPERATOR,
        parse_hits: bool = True,
        is_explain: bool = False,
        boost: bool = True,
        boosted_fields: dict = SUGGEST_BOOSTED_FIELDS,
        combined_fields_list: list[list[str]] = [],
        use_script_score: bool = True,
        use_pinyin: bool = True,
        limit: int = SUGGEST_LIMIT,
        timeout: Union[int, float, str] = 2,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        return self.search(
            query=query,
            match_fields=match_fields,
            source_fields=source_fields,
            match_type=match_type,
            match_bool=match_bool,
            match_operator=match_operator,
            parse_hits=parse_hits,
            is_explain=is_explain,
            boost=boost,
            boosted_fields=boosted_fields,
            combined_fields_list=combined_fields_list,
            use_script_score=use_script_score,
            use_pinyin=use_pinyin,
            detail_level=-1,
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
        if seed is None:
            if seed_update_seconds is None:
                seed = now_ts
            else:
                seed_update_seconds = max(int(abs(seed_update_seconds)), 1)
                seed = now_ts // seed_update_seconds
        else:
            seed = int(seed)

        if filters is None:
            past_month_ts = now_ts = now_ts - 3600 * 24 * 30
            filters = [
                {"range": {"stat.coin": {"gte": 1000}}},
                {"range": {"pubdate": {"gte": past_month_ts}}},
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
