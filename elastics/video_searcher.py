import math

from copy import deepcopy
from datetime import datetime
from pprint import pformat
from tclogger import logger, logstr, get_now_ts, dict_to_str
from typing import Literal, Union

from elastics.client import ElasticSearchClient
from elastics.highlighter import PinyinHighlighter, HighlightMerger
from elastics.structure import get_es_source_val
from converters.times import DateFormatChecker
from converters.query_filter_extractor import QueryFilterExtractor


class MultiMatchQueryDSLConstructor:
    def remove_boost_from_fields(self, fields: list[str]) -> list[str]:
        return [field.split("^", 1)[0] for field in fields]

    def remove_pinyin_fields(self, fields: list[str]) -> list[str]:
        return [field for field in fields if not field.endswith(".pinyin")]

    def remove_field_from_fields(
        self, field_to_remove: str, fields: list[str]
    ) -> list[str]:
        return [field for field in fields if not field.startswith(field_to_remove)]

    def is_field_in_fields(self, field_to_check: str, fields: list[str]) -> bool:
        for field in fields:
            if field.startswith(field_to_check):
                return True
        return False

    def construct(
        self,
        query: str,
        match_fields: list[str] = ["title", "owner.name", "desc", "pubdate_str"],
        date_match_fields: list[str] = ["title", "owner.name", "desc", "pubdate_str"],
        match_bool: str = "must",
        match_type: str = "phrase_prefix",
        match_operator: str = "or",
    ) -> dict:
        query_keywords = query.split()
        fields_without_pubdate = self.remove_field_from_fields(
            "pubdate_str", match_fields
        )
        date_match_fields_without_pubdate = self.remove_field_from_fields(
            "pubdate_str", date_match_fields
        )
        if self.is_field_in_fields("pubdate_str", match_fields):
            date_format_checker = DateFormatChecker()
            match_bool_clause = []
            splitted_fields_groups_by_pubdate = [
                {
                    "fields": [
                        field
                        for field in date_match_fields
                        if field.startswith("pubdate_str")
                    ],
                    "type": "bool_prefix",
                },
            ]
            for keyword in query_keywords:
                date_format_checker.init_year_month_day()
                is_keyword_date_format = date_format_checker.is_in_date_range(
                    keyword, start="2009-09-09", end=datetime.now(), verbose=False
                )
                if is_keyword_date_format:
                    if date_format_checker.matched_format == "%Y":
                        splitted_fields_groups_by_pubdate.append(
                            {
                                "fields": date_match_fields_without_pubdate,
                                "type": match_type,
                            }
                        )
                    date_keyword = date_format_checker.rewrite(
                        keyword, sep="-", check_format=False, use_current_year=True
                    )
                    should_clause = []
                    for fields_group in splitted_fields_groups_by_pubdate:
                        field_keyword = keyword
                        for field in fields_group["fields"]:
                            if field.startswith("pubdate_str"):
                                field_keyword = date_keyword
                                break
                        multi_match_clause = {
                            "multi_match": {
                                "query": field_keyword,
                                "type": fields_group["type"],
                                "fields": fields_group["fields"],
                                "operator": match_operator,
                            },
                        }
                        should_clause.append(multi_match_clause)
                    bool_should_clause = {
                        "bool": {
                            "should": should_clause,
                            "minimum_should_match": 1,
                        }
                    }
                else:
                    bool_should_clause = {
                        "multi_match": {
                            "query": keyword,
                            "type": match_type,
                            "fields": fields_without_pubdate,
                            "operator": match_operator,
                        }
                    }
                match_bool_clause.append(bool_should_clause)
            query_dsl_dict = {
                "bool": {match_bool: match_bool_clause},
            }
        else:
            multi_match_clauses = []
            for keyword in query_keywords:
                multi_match_clause = {
                    "multi_match": {
                        "query": keyword,
                        "type": match_type,
                        "fields": match_fields,
                        "operator": match_operator,
                    }
                }
                multi_match_clauses.append(multi_match_clause)
            query_dsl_dict = {
                "bool": {match_bool: multi_match_clauses},
            }
        return query_dsl_dict


class ScriptScoreQueryDSLConstructor:
    def field_to_var(self, field: str):
        return field.replace(".", "_")

    def assign_var(
        self, field: str, get_value_func: str = None, default_value: float = 0
    ):
        field_var = self.field_to_var(field)
        if not get_value_func:
            get_value_func = f"doc['{field}'].value"
        if field == "pubdate":
            get_value_func = "doc['pubdate'].value"
        return f"double {field_var} = (doc['{field}'].size() > 0) ? {get_value_func} : {default_value};"

    def log_func(self, field: str, min_value: float = 2) -> str:
        func_str = f"Math.log10(Math.max({field}, {min_value}))"
        return func_str

    def pow_func(
        self,
        field: str,
        power: float = 1,
        min_value: float = 1,
        power_precision: int = 4,
    ) -> str:
        func_str = (
            f"Math.pow(Math.max({field}, {min_value}), {power:.{power_precision}f})"
        )
        return func_str

    def pubdate_decay_func(
        self,
        field: str = "pubdate",
        now_ts_field: str = "params.now_ts",
        half_life_days: int = 7,
        power: float = 1.5,
        min_value: float = 0.2,
    ) -> str:
        passed_seconds_str = f"({now_ts_field} - {field})"
        seconds_per_day = 86400
        scaled_pass_days = f"{passed_seconds_str}/{seconds_per_day}/{half_life_days}"
        power_str = self.pow_func(scaled_pass_days, power=power, min_value=min_value)
        func_str = f"1 / (1 + {power_str}) + {min_value}"
        return func_str

    def get_script_source(self):
        assign_vars = []
        stat_powers = {
            "stat.view": 0.1,
            "stat.like": 0.1,
            "stat.coin": 0.25,
            "stat.favorite": 0.15,
            "stat.reply": 0.2,
            "stat.danmaku": 0.15,
            "stat.share": 0.2,
        }
        stat_pow_ratio = (
            1 / math.sqrt(len(stat_powers.keys())) / sum(stat_powers.values())
        )
        for field in list(stat_powers.keys()) + ["pubdate"]:
            assign_vars.append(self.assign_var(field))
        assign_vars_str = "\n".join(assign_vars)
        stat_func_str = " * ".join(
            self.pow_func(self.field_to_var(field), field_power * stat_pow_ratio, 1)
            for field, field_power in stat_powers.items()
        )
        score_str = self.pow_func("_score", 1, 1)
        func_str = (
            f"return ({stat_func_str}) * ({self.pubdate_decay_func()}) * {score_str};"
        )
        script_source = f"{assign_vars_str}\n{func_str}"
        return script_source

    def construct(self, query_dsl_dict: dict) -> dict:
        script_score_dsl_dict = {
            "script_score": {
                "query": query_dsl_dict,
                "script": {
                    "source": self.get_script_source(),
                    "params": {
                        "now_ts": get_now_ts(),
                    },
                },
            }
        }
        return script_score_dsl_dict


class VideoSearcher:
    SOURCE_FIELDS = [
        "title",
        "bvid",
        "owner",
        "pic",
        "duration",
        "desc",
        "stat",
        "tname",
        "tags",
        "pubdate_str",
        "insert_at_str",
    ]
    SEARCH_MATCH_FIELDS = [
        "title",
        "title.pinyin",
        "tags",
        "tags.pinyin",
        "owner.name",
        "owner.name.pinyin",
        "desc",
        "desc.pinyin",
        "pubdate_str",
    ]
    SUGGEST_MATCH_FIELDS = [
        "title",
        "title.pinyin",
        "tags",
        "tags.pinyin",
        "owner.name",
        "owner.name.pinyin",
        "pubdate_str",
    ]
    DATE_MATCH_FIELDS = [
        "title",
        "desc",
        "owner.name",
        "pubdate_str",
    ]
    SEARCH_BOOSTED_FIELDS = {
        "title": 2.5,
        "title.pinyin": 0.25,
        "tags": 2,
        "tags.pinyin": 0.2,
        "owner.name": 2,
        "owner.name.pinyin": 0.2,
        "desc": 0.1,
        "desc.pinyin": 0.01,
        "pubdate_str": 2.5,
    }
    SUGGEST_BOOSTED_FIELDS = {
        "title": 2.5,
        "title.pinyin": 0.5,
        "tags": 2,
        "tags.pinyin": 0.4,
        "owner.name": 2,
        "owner.name.pinyin": 0.4,
        "pubdate_str": 2.5,
    }
    DATE_BOOSTED_FIELDS = {
        "title": 0.1,
        "owner.name": 0.1,
        "desc": 0.05,
        "pubdate_str": 2.5,
    }
    DOC_EXCLUDED_SOURCE_FIELDS = []

    MATCH_TYPE = Literal[
        "best_fields",
        "most_fields",
        "cross_fields",
        "phrase",
        "phrase_prefix",
        "bool_prefix",
    ]
    MATCH_BOOL = Literal["must", "should", "must_not", "filter"]
    MATCH_OPERATOR = Literal["or", "and"]

    SEARCH_MATCH_TYPE = "phrase_prefix"
    SUGGEST_MATCH_TYPE = "phrase_prefix"
    SEARCH_MATCH_BOOL = "must"
    SEARCH_MATCH_OPERATOR = "or"

    SEARCH_DETAIL_LEVELS = {
        1: {"match_type": "phrase_prefix", "bool": "must", "pinyin": False},
        2: {"match_type": "cross_fields", "bool": "must", "operator": "and"},
        3: {"match_type": "cross_fields", "bool": "must", "pinyin": True},
        4: {"match_type": "most_fields", "bool": "must", "pinyin": True},
        5: {"match_type": "most_fields", "bool": "should"},
    }
    MAX_SEARCH_DETAIL_LEVEL = 4

    SUGGEST_LIMIT = 10
    SEARCH_LIMIT = 50
    # This constant is to contain more hits for redundance,
    # as drop_no_highlights would drop some hits
    NO_HIGHLIGHT_REDUNDANCE_RATIO = 2

    def __init__(self, index_name: str = "bili_videos"):
        self.index_name = index_name
        self.es = ElasticSearchClient()
        self.es.connect()

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

        if detail_level in self.SEARCH_DETAIL_LEVELS:
            match_detail = self.SEARCH_DETAIL_LEVELS[detail_level]
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
            date_boosted_fields = self.boost_fields(
                date_fields, self.DATE_BOOSTED_FIELDS
            )
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
        )

        if filters:
            query_dsl_dict["bool"]["filter"] = filters

        if use_script_score:
            script_score_constructor = ScriptScoreQueryDSLConstructor()
            query_dsl_dict = script_score_constructor.construct(query_dsl_dict)

        search_body = {
            "query": query_dsl_dict,
            "_source": source_fields,
            "explain": is_explain,
            "highlight": self.get_highlight_settings(match_fields),
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

        if detail_level > 2 and match_bool == "should":
            search_body["query"]["bool"]["minimum_should_match"] = (
                len(query_keywords) - 1
            )
        logger.note(dict_to_str(search_body, is_colored=True, align_list=False))
        if limit and limit > 0:
            search_body["size"] = int(limit * self.NO_HIGHLIGHT_REDUNDANCE_RATIO)
        logger.note(f"> Get search results by query:", end=" ")
        logger.mesg(f"[{query}]")

        try:
            res = self.es.client.search(index=self.index_name, body=search_body)
            res_dict = res.body
        except Exception as e:
            logger.warn(f"× Error: {e}")
            res_dict = {}

        return_res = self.parse_hits(
            query,
            match_fields,
            res_dict,
            request_type="search",
            is_parse_hits=parse_hits,
            drop_no_highlights=True,
            match_type=match_type,
            match_operator=match_operator,
            detail_level=detail_level,
            limit=limit,
            verbose=verbose,
        )
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
        detail_level: int = -1,
        max_detail_level: int = MAX_SEARCH_DETAIL_LEVEL,
        limit: int = SEARCH_LIMIT,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        detail_level_upper_bound = int(
            min(max_detail_level, max(self.SEARCH_DETAIL_LEVELS.keys()))
        )

        if detail_level < 1:
            detail_level = 1
        elif detail_level > detail_level_upper_bound:
            detail_level = detail_level_upper_bound
        else:
            detail_level = int(detail_level)

        return_res = {
            "total_hits": 0,
            "return_hits": 0,
            "hits": [],
        }

        while (
            return_res["total_hits"] == 0 and detail_level <= detail_level_upper_bound
        ):
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
        return_res = self.parse_hits(
            "", [], res_dict, request_type="random", is_parse_hits=parse_hits
        )
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
        return_res = self.parse_hits(
            "", [], res_dict, request_type="latest", is_parse_hits=parse_hits
        )
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

    def get_pinyin_highlights(
        self, query: str, match_fields: list[str], _source: dict
    ) -> dict:
        pinyin_highlights = {}
        if query:
            pinyin_highlighter = PinyinHighlighter()
            for field in match_fields:
                if field.endswith(".pinyin"):
                    no_pinyin_field = field.replace(".pinyin", "")
                    text = get_es_source_val(_source, no_pinyin_field)
                    if text is None:
                        continue
                    highlighted_text = pinyin_highlighter.highlight(
                        query, text, tag="hit"
                    )
                    if highlighted_text:
                        pinyin_highlights[field] = [highlighted_text]
        return pinyin_highlights

    def parse_hits(
        self,
        query: str,
        match_fields: list[str],
        res_dict: dict,
        request_type: Literal[
            "suggest", "search", "random", "latest", "doc"
        ] = "search",
        is_parse_hits: bool = True,
        drop_no_highlights: bool = False,
        match_type: MATCH_TYPE = "most_fields",
        match_operator: MATCH_OPERATOR = "or",
        detail_level: int = -1,
        limit: int = -1,
        verbose: bool = False,
    ) -> list[dict]:
        if not is_parse_hits:
            return res_dict
        if not res_dict:
            hits_info = {
                "request_type": request_type,
                "detail_level": detail_level,
                "took": -1,
                "timed_out": True,
                "total_hits": 0,
                "return_hits": 0,
                "hits": [],
            }
            return hits_info
        hits = []
        for hit in res_dict["hits"]["hits"]:
            _source = hit["_source"]
            score = hit["_score"]
            common_highlights = hit.get("highlight", {})
            pinyin_highlights = self.get_pinyin_highlights(query, match_fields, _source)

            merged_highlights = {}
            merger = HighlightMerger()

            pinyin_fields = [
                field.replace(".pinyin", "") for field in pinyin_highlights.keys()
            ]
            merged_fields = list(common_highlights.keys()) + pinyin_fields

            for field in merged_fields:
                common_highlight = common_highlights.get(field, [])
                pinyin_highlight = pinyin_highlights.get(field + ".pinyin", [])
                merged_highlight = merger.merge(
                    get_es_source_val(_source, field),
                    common_highlight + pinyin_highlight,
                    tag="hit",
                )
                if merged_highlight:
                    merged_highlights[field] = [merged_highlight]

            if (
                drop_no_highlights
                and (not common_highlights)
                and (not pinyin_highlights)
                and match_type == "most_fields"
                and match_operator == "or"
            ):
                continue
            hit_info = {
                **_source,
                "score": score,
                "common_highlights": common_highlights,
                "pinyin_highlights": pinyin_highlights,
                "merged_highlights": merged_highlights,
            }
            hits.append(hit_info)
        if limit > 0:
            hits = hits[:limit]
        hits_info = {
            "request_type": request_type,
            "detail_level": detail_level,
            "took": res_dict["took"],
            "timed_out": res_dict["timed_out"],
            "total_hits": res_dict["hits"]["total"]["value"],
            "return_hits": len(hits),
            "hits": hits,
        }
        logger.enter_quiet(not verbose)
        # logger.success(pformat(hits_info, indent=4, sort_dicts=False))
        logger.note(f"Request type: [{request_type}]")
        logger.mesg(f"  * detail level: {detail_level}")
        logger.mesg(f"  * return hits count: {hits_info['return_hits']}")
        logger.mesg(f"  * total hits count: {hits_info['total_hits']}")
        logger.mesg(f"  * took: {hits_info['took']}ms")
        logger.mesg(f"  * timed_out: {hits_info['timed_out']}")
        logger.exit_quiet(not verbose)
        return hits_info


if __name__ == "__main__":
    video_searcher = VideoSearcher("bili_videos_dev2")
    # logger.note("> Getting random results ...")
    # res = video_searcher.random()
    # logger.mesg(res)

    # query = "田文镜"
    # logger.note("> Searching results:", end=" ")
    # logger.file(f"[{query}]")
    # res = video_searcher.search(query, limit=5, verbose=True)
    # hits = res.pop("hits")
    # logger.success(dict_to_str(res, is_colored=False))
    # for idx, hit in enumerate(hits):
    #     logger.note(f"* Hit {idx}:")
    #     logger.file(dict_to_str(hit, align_list=False), indent=4)

    # query = "twj"
    # logger.note(f"> Suggest results: " + logstr.file(f"[{query}]"))
    # res = video_searcher.suggest(query, limit=5, verbose=True)
    # hits = res.pop("hits")
    # logger.success(dict_to_str(res, is_colored=False))
    # for idx, hit in enumerate(hits):
    #     logger.note(f"* Hit {idx}:")
    #     logger.file(dict_to_str(hit, align_list=False), indent=4)

    # query = "Hansimov 2018"
    # query = "黑神话 2024 :coin>1000 :view<100000"
    # match_fields = ["title^2.5", "owner.name^2", "desc", "pubdate_str^2.5"]
    # date_match_fields = ["title^0.5", "owner.name^0.25", "desc^0.2", "pubdate_str^2.5"]

    # filter_extractor = QueryFilterExtractor()
    # query_keywords, filters = filter_extractor.construct(query)
    # query_without_filters = " ".join(query_keywords)

    # query_constructor = MultiMatchQueryDSLConstructor()
    # query_dsl_dict = query_constructor.construct(
    #     query=query_without_filters,
    #     match_fields=match_fields,
    #     date_match_fields=date_match_fields,
    # )

    # if filters:
    #     query_dsl_dict["bool"]["filter"] = filters

    # logger.note(f"> Construct DSL for query:", end=" ")
    # logger.mesg(f"[{query}]")
    # logger.success(pformat(query_dsl_dict, sort_dicts=False, indent=2, compact=True))
    # script_query_dsl_dict = ScriptScoreQueryDSLConstructor().construct(query_dsl_dict)
    # logger.note(pformat(script_query_dsl_dict, sort_dicts=False, compact=True))
    # logger.mesg(ScriptScoreQueryDSLConstructor().get_script_source())

    # searcher = VideoSearcher("bili_videos_dev")
    # search_res = searcher.search(
    #     query,
    #     source_fields=["title", "owner.name", "desc", "pubdate_str", "stat"],
    #     boost=True,
    #     use_script_score=True,
    #     detail_level=1,
    #     limit=3,
    #     timeout=1,
    #     verbose=True,
    # )
    # if search_res["took"] < 0:
    #     logger.warn(pformat(search_res, sort_dicts=False, indent=4))
    # searcher.suggest("yingshi", limit=3, verbose=True, timeout=1)

    # python -m elastics.video_searcher
