from copy import deepcopy
from datetime import datetime
from pprint import pformat
from tclogger import logger, get_now_ts
from typing import Literal, Union

from elastics.client import ElasticSearchClient
from elastics.highlighter import PinyinHighlighter, HighlightMerger
from elastics.structure import get_es_source_val
from converters.times import DateFormatChecker


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
        match_bool: str = "must",
        match_type: str = "phrase_prefix",
        match_operator: str = "or",
    ) -> dict:
        query_keywords = query.split()
        fields_without_pubdate_str = self.remove_field_from_fields(
            "pubdate_str", match_fields
        )
        if self.is_field_in_fields("pubdate_str", match_fields):
            date_format_checker = DateFormatChecker()
            splitted_fields_groups = [
                {
                    "fields": fields_without_pubdate_str,
                    "type": match_type,
                },
                {
                    "fields": ["pubdate_str"],
                    "type": "bool_prefix",
                },
            ]
            match_bool_clause = []
            for keyword in query_keywords:
                if date_format_checker.is_in_date_range(
                    keyword, start="2009-09-09", end=datetime.now(), verbose=False
                ):
                    date_keyword = date_format_checker.rewrite(
                        keyword, sep="-", check_format=False, use_current_year=True
                    )
                    date_format_checker.init_year_month_day()
                    should_clause = []
                    for fields_group in splitted_fields_groups:
                        if "pubdate_str" in fields_group["fields"]:
                            field_keyword = date_keyword
                        else:
                            field_keyword = keyword
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
                            "fields": fields_without_pubdate_str,
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
    def assign_var(
        self, field: str, get_value_func: str = None, default_value: float = 0
    ):
        field_var = field.replace(".", "_")
        if not get_value_func:
            get_value_func = f"doc['{field}'].value"
        if field == "pubdate":
            get_value_func = "doc['pubdate'].value.getMillis()/1000"
        return f"double {field_var} = (doc['{field}'].size() > 0) ? {get_value_func} : {default_value};"

    def log_func(self, field: str) -> str:
        field_var = field.replace(".", "_")
        func_str = f"Math.log10({field_var}+2)"
        return func_str

    def pubdate_decay_func(
        self,
        field: str = "pubdate",
        half_life_days: int = 7,
        power: float = 1.5,
        lower_bound: float = 0.001,
    ) -> str:
        field_var = field.replace(".", "_")
        passed_seconds_str = f"(params.now_ts - {field_var})"
        seconds_per_day = 86400
        func_str = f"1/(1+Math.pow({passed_seconds_str}/{seconds_per_day}/{half_life_days}, {power})) + {lower_bound}"
        return func_str

    def get_script_source(self):
        # score = log(stat.view+1) * log(stat.like+1) * log(stat.coin+1) * (1/(1+Math.pow(passed_days/7, 1.5)) + 0.001)
        assign_vars = []
        stat_fields = ["stat.view", "stat.like", "stat.coin"]
        for field in stat_fields + ["pubdate"]:
            assign_vars.append(self.assign_var(field))
        assign_vars_str = "\n".join(assign_vars)
        stat_func_str = " * ".join(self.log_func(field) for field in stat_fields)
        func_str = f"return {stat_func_str} * ({self.pubdate_decay_func()});"
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
        "pubdate_str",
        "insert_at_str",
    ]
    SUGGEST_MATCH_FIELDS = ["title", "title.pinyin"]
    SEARCH_MATCH_FIELDS = [
        "title",
        "title.pinyin",
        "owner.name",
        "owner.name.pinyin",
        "desc",
        "desc.pinyin",
        "pubdate_str",
    ]
    BOOSTED_FIELDS = {
        "title": 2.5,
        "title.pinyin": 0.25,
        "owner.name": 2,
        "owner.name.pinyin": 0.2,
        "desc": 1,
        "desc.pinyin": 0.1,
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

    SUGGEST_MATCH_TYPE = "phrase_prefix"
    SEARCH_MATCH_TYPE = "phrase_prefix"
    SEARCH_MATCH_BOOL = "must"
    SEARCH_MATCH_OPERATOR = "or"

    SEARCH_DETAIL_LEVELS = {
        1: {"match_type": "phrase_prefix", "bool": "must"},
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

    def suggest(
        self,
        query: str,
        match_fields: list[str] = SUGGEST_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        match_type: MATCH_TYPE = SUGGEST_MATCH_TYPE,
        parse_hits: bool = True,
        is_explain: bool = False,
        limit: int = SUGGEST_LIMIT,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        """
        Multi-match query:
            - https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-multi-match-query.html#multi-match-types

        I have compared the suggestion results of:
            - `suggest`with "completion"
            - `multi_search` with "phrase_prefix"

        And the conclusion is that `multi_search` is better and more flexible.
        """
        logger.enter_quiet(not verbose)
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "type": match_type,
                    "fields": match_fields,
                }
            },
            "_source": source_fields,
            "explain": is_explain,
            "highlight": self.get_highlight_settings(match_fields),
        }
        if limit and limit > 0:
            search_body["size"] = int(limit * self.NO_HIGHLIGHT_REDUNDANCE_RATIO)

        logger.note(f"> Get suggestions by query:", end=" ")
        logger.mesg(f"[{query}]")
        res = self.es.client.search(index=self.index_name, body=search_body)
        res_dict = res.body
        return_res = self.parse_hits(
            query,
            match_fields,
            res_dict,
            request_type="suggest",
            is_parse_hits=parse_hits,
            drop_no_highlights=True,
            limit=limit,
        )
        logger.exit_quiet(not verbose)
        return return_res

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
        boosted_fields: dict = BOOSTED_FIELDS,
        use_script_score: bool = True,
        detail_level: int = -1,
        limit: int = SEARCH_LIMIT,
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
            use_pinyin_fields = match_detail.get("pinyin", False)

        if not use_pinyin_fields:
            match_fields = [
                field for field in match_fields if not field.endswith(".pinyin")
            ]

        if boost:
            boosted_fields = self.boost_fields(match_fields, boosted_fields)
        else:
            boosted_fields = match_fields

        query_constructor = MultiMatchQueryDSLConstructor()
        query_dsl_dict = query_constructor.construct(
            query,
            match_fields=boosted_fields,
            match_bool=match_bool,
            match_type=match_type,
            match_operator=match_operator,
        )
        if use_script_score:
            script_score_constructor = ScriptScoreQueryDSLConstructor()
            query_dsl_dict = script_score_constructor.construct(query_dsl_dict)

        search_body = {
            "query": query_dsl_dict,
            "_source": source_fields,
            "explain": is_explain,
            "highlight": self.get_highlight_settings(match_fields),
        }

        query_keywords = query.split()
        if detail_level > 2 and match_bool == "should":
            search_body["query"]["bool"]["minimum_should_match"] = (
                len(query_keywords) - 1
            )

        logger.note(pformat(search_body, sort_dicts=False))
        if limit and limit > 0:
            search_body["size"] = int(limit * self.NO_HIGHLIGHT_REDUNDANCE_RATIO)
        logger.note(f"> Get search results by query:", end=" ")
        logger.mesg(f"[{query}]")
        res = self.es.client.search(index=self.index_name, body=search_body)
        res_dict = res.body
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
        )
        return_res["detail_level"] = detail_level
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
        boosted_fields: dict = BOOSTED_FIELDS,
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

    def random(
        self,
        seed: Union[int, str] = None,
        seed_update_seconds: int = None,
        source_fields: list[str] = SOURCE_FIELDS,
        parse_hits: bool = True,
        is_explain: bool = False,
        limit: int = 1,
        verbose: bool = False,
    ):
        logger.enter_quiet(not verbose)
        now = datetime.now()
        ts = round(now.timestamp())
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if seed is None:
            if seed_update_seconds is None:
                seed = ts
            else:
                seed_update_seconds = max(int(abs(seed_update_seconds)), 1)
                seed = ts // seed_update_seconds
        else:
            seed = int(seed)
        search_body = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "filter": [
                                {"range": {"stat.view": {"gte": 1000000}}},
                                {"range": {"pubdate": {"gte": "now-30d/d"}}},
                            ]
                        }
                    },
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
        logger.enter_quiet(not verbose)
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
                    text = get_es_source_val(_source, field.replace(".pinyin", ""))
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
    ) -> list[dict]:
        if not is_parse_hits:
            return res_dict
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
            "total_hits": res_dict["hits"]["total"]["value"],
            "return_hits": len(hits),
            "hits": hits,
        }
        logger.success(pformat(hits_info, indent=4, sort_dicts=False))
        logger.note(f"Request type: [{request_type}]")
        logger.mesg(f"  * detail level: {detail_level}")
        logger.mesg(f"  * return hits count: {hits_info['return_hits']}")
        logger.mesg(f"  * total hits count: {hits_info['total_hits']}")
        return hits_info


if __name__ == "__main__":
    # query = "Hansimov 2018"
    query = "影视飓风 2024-07"
    match_fields = ["title^2.5", "owner.name^2", "desc", "pubdate_str^2.5"]
    constructor = MultiMatchQueryDSLConstructor()
    query_dsl_dict = constructor.construct(query=query, match_fields=match_fields)
    logger.note(f"> Construct DSL for query:", end=" ")
    logger.mesg(f"[{query}]")
    logger.success(pformat(query_dsl_dict, sort_dicts=False, indent=2))

    logger.note(ScriptScoreQueryDSLConstructor().get_script_source())

    searcher = VideoSearcher("bili_videos_dev")
    searcher.search(
        query,
        source_fields=["title", "owner.name", "desc", "pubdate_str", "stat"],
        boost=True,
        use_script_score=True,
        detail_level=1,
        limit=3,
        verbose=True,
    )

    # python -m elastics.video_searcher
