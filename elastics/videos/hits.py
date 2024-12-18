from typing import Literal
from tclogger import logger

from elastics.structure import get_es_source_val
from converters.highlight.merge import HighlightMerger
from converters.highlight.count import HighlightsCounter
from converters.highlight.pinyin import PinyinHighlighter
from converters.query.filter import QueryFilterExtractor
from elastics.videos.constants import MATCH_TYPE, MATCH_OPERATOR


class VideoHitsParser:
    def __init__(self):
        self.pinyin_highlighter = PinyinHighlighter()
        self.highlights_counter = HighlightsCounter()
        self.split_query = QueryFilterExtractor().split_keyword_and_filter_expr

    def get_pinyin_highlights(
        self, query: str, match_fields: list[str], _source: dict
    ) -> dict:
        if not query:
            return {}
        pinyin_highlights = {}
        for field in match_fields:
            if field.endswith(".pinyin"):
                no_pinyin_field = field.replace(".pinyin", "")
                text = get_es_source_val(_source, no_pinyin_field)
                if text is None:
                    continue
                highlighted_text = self.pinyin_highlighter.highlight(
                    query, text, tag="hit"
                )
                if highlighted_text:
                    pinyin_highlights[field] = [highlighted_text]
        return pinyin_highlights

    def parse(
        self,
        query: str,
        match_fields: list[str],
        res_dict: dict,
        request_type: Literal[
            "suggest", "search", "random", "latest", "doc"
        ] = "search",
        drop_no_highlights: bool = False,
        match_type: MATCH_TYPE = "most_fields",
        match_operator: MATCH_OPERATOR = "or",
        detail_level: int = -1,
        limit: int = -1,
        verbose: bool = False,
    ) -> dict:
        query_info = self.split_query(query)
        qwords = query_info["keywords_body"]
        qwords_str = " ".join(qwords)
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
            _source = hit["_source"]
            score = hit["_score"]
            common_highlights = hit.get("highlight", {})
            pinyin_highlights = self.get_pinyin_highlights(
                qwords_str, match_fields, _source
            )
            all_fields = {**common_highlights, **pinyin_highlights}.keys()

            merger = HighlightMerger()
            segged_highlights = {}
            merged_highlights = {}

            pinyin_suffix_fields = [
                field.replace(".pinyin", "") for field in pinyin_highlights.keys()
            ]
            words_suffix_fields = [
                field.replace(".words", "") for field in common_highlights.keys()
            ]
            other_fields = [
                field
                for field in all_fields
                if not any(field.endswith(suffix) for suffix in [".pinyin", ".words"])
            ]
            merged_fields = set(
                other_fields + pinyin_suffix_fields + words_suffix_fields
            )

            for field in merged_fields:
                common_highlight = common_highlights.get(field, [])
                words_highlight = common_highlights.get(field + ".words", [])
                pinyin_highlight = pinyin_highlights.get(field + ".pinyin", [])
                highlight_to_merge = (
                    common_highlight + words_highlight + pinyin_highlight
                )
                merged_res = merger.extract_and_merge(
                    get_es_source_val(_source, field), highlight_to_merge, tag="hit"
                )
                merged_highlight = merged_res["merged"]
                segged_highlight = merged_res["segged"]
                if merged_highlight:
                    merged_highlights[field] = [merged_highlight]
                if segged_highlight:
                    segged_highlights[field] = segged_highlight

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
                "highlights": {
                    "common": common_highlights,
                    "pinyin": pinyin_highlights,
                    "merged": merged_highlights,
                    "segged": segged_highlights,
                },
            }
            hits.append(hit_info)
        if limit > 0:
            hits = hits[:limit]
        count_res = self.highlights_counter.count_keywords(qwords, hits)
        hits_info = {
            "query": query,
            "request_type": request_type,
            "detail_level": detail_level,
            "took": res_dict["took"],
            "timed_out": res_dict["timed_out"],
            "total_hits": res_dict["hits"]["total"]["value"],
            "return_hits": len(hits),
            "hits": hits,
            "suggest_info": {
                "qword_hword_count": count_res["qword_hword_count"],
                "hwords_str_count": count_res["hwords_str_count"],
                "related_authors": self.highlights_counter.count_authors(hits),
            },
            "query_info": query_info,
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
