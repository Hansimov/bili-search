from copy import deepcopy
from datetime import datetime
from pprint import pformat
from tclogger import logger, logstr, get_now_ts, dict_to_str
from typing import Literal, Union

from elastics.client import ElasticSearchClient
from elastics.highlighter import PinyinHighlighter, HighlightMerger
from elastics.structure import get_es_source_val
from converters.query.filter import QueryFilterExtractor
from converters.query.dsl import (
    MultiMatchQueryDSLConstructor,
    ScriptScoreQueryDSLConstructor,
)
from elastics.videos.constants import SOURCE_FIELDS, DOC_EXCLUDED_SOURCE_FIELDS
from elastics.videos.constants import SEARCH_MATCH_FIELDS, SEARCH_BOOSTED_FIELDS
from elastics.videos.constants import SUGGEST_MATCH_FIELDS, SUGGEST_BOOSTED_FIELDS
from elastics.videos.constants import DATE_MATCH_FIELDS, DATE_BOOSTED_FIELDS
from elastics.videos.constants import MATCH_TYPE, MATCH_BOOL, MATCH_OPERATOR
from elastics.videos.constants import (
    SEARCH_MATCH_TYPE,
    SEARCH_MATCH_BOOL,
    SEARCH_MATCH_OPERATOR,
)
from elastics.videos.constants import SUGGEST_MATCH_TYPE
from elastics.videos.constants import SEARCH_DETAIL_LEVELS, MAX_SEARCH_DETAIL_LEVEL
from elastics.videos.constants import (
    SUGGEST_LIMIT,
    SEARCH_LIMIT,
    NO_HIGHLIGHT_REDUNDANCE_RATIO,
)


class VideoSearcher:
    def __init__(self, index_name: str = "bili_videos_dev2"):
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

        if detail_level > 2 and match_bool == "should":
            search_body["query"]["bool"]["minimum_should_match"] = (
                len(query_keywords) - 1
            )
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
            min(max_detail_level, max(SEARCH_DETAIL_LEVELS.keys()))
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

            for hit_field in merged_fields:
                for suffix in [".pinyin", ".words"]:
                    field = hit_field.removesuffix(suffix)
                common_highlight = common_highlights.get(hit_field, [])
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

    # query = "影视飓风"
    # logger.note("> Searching results:", end=" ")
    # logger.file(f"[{query}]")
    # res = video_searcher.search(query, limit=3, use_script_score=True, verbose=True)
    # hits = res.pop("hits")
    # logger.success(dict_to_str(res, is_colored=False))
    # for idx, hit in enumerate(hits):
    #     logger.note(f"* Hit {idx}:")
    #     logger.file(dict_to_str(hit, align_list=False), indent=4)

    query = "影视飓feng 2024"
    logger.note(f"> Suggest results: " + logstr.file(f"[{query}]"))
    res = video_searcher.suggest(query, limit=5, verbose=True)
    hits = res.pop("hits")
    logger.success(dict_to_str(res, is_colored=False))
    for idx, hit in enumerate(hits):
        logger.note(f"* Hit {idx}:")
        logger.file(dict_to_str(hit, align_list=False), indent=4)

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

    # python -m elastics.videos.searcher
