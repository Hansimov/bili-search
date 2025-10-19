from tclogger import logger, dict_to_str, dict_get
from typing import Union, Literal

from elastics.structure import get_es_source_val
from converters.field.region_infos import REGION_INFOS_BY_ID
from converters.highlight.merge import HighlightMerger
from converters.highlight.count import HighlightsCounter
from converters.highlight.pinyin import PinyinHighlighter
from elastics.videos.constants import MATCH_TYPE, MATCH_OPERATOR
from elastics.videos.constants import SEARCH_MATCH_TYPE, SEARCH_MATCH_OPERATOR
from elastics.videos.constants import SEARCH_REQUEST_TYPE, SEARCH_REQUEST_TYPE_DEFAULT


class VideoHitsParser:
    def __init__(self):
        self.pinyin_highlighter = PinyinHighlighter()
        self.highlight_merger = HighlightMerger()

    def get_pinyin_highlights(
        self, keywords: Union[str, list[str]], match_fields: list[str], source: dict
    ) -> dict:
        if not keywords:
            return {}
        pinyin_highlights = {}
        for field in match_fields:
            if field.endswith(".pinyin"):
                no_pinyin_field = field.replace(".pinyin", "")
                text = get_es_source_val(source, no_pinyin_field)
                if text is None:
                    continue
                highlighted_text = self.pinyin_highlighter.highlight(
                    keywords, text, tag="hit"
                )
                if highlighted_text:
                    pinyin_highlights[field] = [highlighted_text]
        return pinyin_highlights

    def get_merged_fields(
        self, common_highlights: dict, pinyin_highlights: dict
    ) -> set:
        pinyin_highlight_fields = list(pinyin_highlights.keys())
        common_highlight_fields = list(common_highlights.keys())
        all_fields = set(pinyin_highlight_fields + common_highlight_fields)
        pinyin_suffix_fields = [
            field.replace(".pinyin", "") for field in pinyin_highlight_fields
        ]
        words_suffix_fields = [
            field.replace(".words", "") for field in common_highlight_fields
        ]
        other_fields = [
            field
            for field in all_fields
            if not any(field.endswith(suffix) for suffix in [".pinyin", ".words"])
        ]
        merged_fields = set(other_fields + pinyin_suffix_fields + words_suffix_fields)
        return merged_fields

    def get_merged_and_segged_highlights(
        self,
        source: dict,
        common_highlights: dict,
        pinyin_highlights: dict,
    ) -> tuple[dict, dict]:
        merged_highlights = {}
        segged_highlights = {}
        merged_fields = self.get_merged_fields(common_highlights, pinyin_highlights)
        for field in merged_fields:
            common_highlight = common_highlights.get(field, [])
            words_highlight = common_highlights.get(field + ".words", [])
            pinyin_highlight = pinyin_highlights.get(field + ".pinyin", [])
            highlight_to_merge = common_highlight + words_highlight + pinyin_highlight
            merged_res = self.highlight_merger.extract_and_merge(
                get_es_source_val(source, field), highlight_to_merge, tag="hit"
            )
            merged_highlight = merged_res["merged"]
            segged_highlight = merged_res["segged"]
            if merged_highlight:
                merged_highlights[field] = [merged_highlight]
            if segged_highlight:
                segged_highlights[field] = segged_highlight
        return merged_highlights, segged_highlights

    def add_region_info_to_source(self, source: dict) -> dict:
        tid = source.get("tid", None)
        region_name = REGION_INFOS_BY_ID.get(tid, {}).get("region_name", "")
        region_parent_name = REGION_INFOS_BY_ID.get(tid, {}).get("parent_name", "")
        source["region_name"] = region_name
        source["region_parent_name"] = region_parent_name
        return source

    def parse(
        self,
        query_info: dict,
        match_fields: list[str],
        res_dict: dict,
        request_type: SEARCH_REQUEST_TYPE = SEARCH_REQUEST_TYPE_DEFAULT,
        drop_no_highlights: bool = False,
        add_region_info: bool = True,
        add_highlights_info: bool = True,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        match_operator: MATCH_OPERATOR = SEARCH_MATCH_OPERATOR,
        detail_level: int = -1,
        limit: int = -1,
        verbose: bool = False,
    ) -> dict:
        """Example of output:
        ```json
        {
            "query": "红警08",
            "request_type": "search",
            "detail_level": -1,
            "took": 750,
            "timed_out": false,
            "total_hits": 1234,
            "return_hits": 10,
            "hits": [
                {
                    "bvid": "BV19LqeYnEyv",
                    "tid": 17,
                    "title": "红警08！深夜高歌一曲！",
                    "tname": "单机游戏",
                    "ptid": 4,
                    "pubdate": "1734027686",
                    "rtags": "游戏, 单机游戏",
                    "tags": "红色警戒, 唱歌, 演唱, 08",
                    "score": 2644.0063,
                    "stat":{
                        "view": 11111,
                        "danmaku": 222,
                        "reply": 33,
                        "favorite": 444,
                        "coin": 55,
                        "share": 6
                    },
                    "highlights": {
                        "common": {
                            "owner.name.words: ["红警HBK<hit>08</hit>"],
                            "title.words": ["红警<hit>08</hit>！深夜高歌一曲！"],
                            "tags.words": ["红色警戒, 唱歌, 演唱, <hit>08</hit>"]
                        },
                        "pinyin": {
                            "title.pinyin": ["<hit>红警</hit>08！深夜高歌一曲！"],
                            "owner.name.pinyin": ["<hit>红警</hit>HBK08"]
                        },
                        "merged": {
                            "title": ["<hit>红警08</hit>！深夜高歌一曲！"],
                            "owner.name": ["<hit>红警</hit>HBK<hit>08</hit>"],
                            "tags": ["红色警戒, 唱歌, 演唱, <hit>08</hit>"]
                        },
                        "segged": {
                            "title": ["08", "红警"],
                            "owner.name": ["08", "红警"],
                            "tags": ["08"]
                        }
                    }
                },
                ...
            ],
            "query_info": <query_info>
        }
        ```
        """
        qwords = query_info.get("keywords_body", None)
        query = query_info.get("query", None)
        if not res_dict:
            hits_info = {
                "query": query,
                "request_type": request_type,
                "detail_level": detail_level,
                "took": -1,
                "timed_out": True,
                "total_hits": 0,
                "return_hits": 0,
                "hits": [],
                "suggest_info": {},
                "query_info": query_info,
            }
            return hits_info
        hits = []
        for hit in res_dict["hits"]["hits"]:
            source = hit["_source"]
            score = hit["_score"]
            sort_score = hit.get("sort", [None])[0]
            if add_region_info:
                self.add_region_info_to_source(source)
            hit_info = {
                **source,
                "score": score,
                "sort_score": sort_score,
            }
            if add_highlights_info:
                common_highlights = hit.get("highlight", {})
                pinyin_highlights = self.get_pinyin_highlights(
                    keywords=qwords, match_fields=match_fields, source=source
                )
                is_hit_ignored = (
                    drop_no_highlights
                    # and (match_type == "most_fields" and match_operator == "or")
                    and ((not common_highlights) and (not pinyin_highlights))
                )
                if is_hit_ignored:
                    continue
                merged_highlights, segged_highlights = (
                    self.get_merged_and_segged_highlights(
                        source=source,
                        common_highlights=common_highlights,
                        pinyin_highlights=pinyin_highlights,
                    )
                )
                highlights_info = {
                    "highlights": {
                        "common": common_highlights,
                        "pinyin": pinyin_highlights,
                        "merged": merged_highlights,
                        "segged": segged_highlights,
                    },
                }
                hit_info.update(highlights_info)
            hits.append(hit_info)
        if limit > 0:
            hits = hits[:limit]
        hits_info = {
            "query": query,
            "request_type": request_type,
            "detail_level": detail_level,
            "took": res_dict["took"],
            "timed_out": res_dict["timed_out"],
            "total_hits": dict_get(res_dict, "hits.total.value", -1),
            "return_hits": len(hits),
            "hits": hits,
            "query_info": query_info,
        }
        if verbose:
            logger.note(f"Request type: [{request_type}]")
            log_info = {
                "detail_level": detail_level,
                "return_hits": hits_info["return_hits"],
                "total_hits": hits_info["total_hits"],
                "took": hits_info["took"],
                "timed_out": hits_info["timed_out"],
            }
            logger.okay(dict_to_str(log_info), indent=2)
        return hits_info


class SuggestInfoParser:
    def __init__(self, version: Literal["v1", "v2"] = "v1"):
        self.highlights_counter = HighlightsCounter()
        self.version = version
        self.init_keywords_params()

    def init_keywords_params(self):
        if self.version == "v1":
            self.keywords_params = {"is_calc_hwords_str_count": True}
        else:
            self.keywords_params = {"is_calc_hwords_str_count": False}

    def parse(self, qwords: list[str], hits: list[dict]) -> dict:
        """Example output:
        ```python
        {
            "qword_hword_count":    dict[str, dict[str, int]]
            "hword_count_qword":    dict[str, tuple[int, str]]
            "group_replaces_count": dict[tuple[str], int]
            "related_authors":      dict[str, dict]
        }
        ```
        #ANCHOR[id=suggest_info]
        """
        keywords_info = self.highlights_counter.count_keywords(
            qwords, hits, **self.keywords_params
        )
        related_authors = self.highlights_counter.count_authors(hits)
        suggest_info = {
            **keywords_info,
            "related_authors": related_authors,
        }
        return suggest_info
