from typing import Literal
from tclogger import logger

from elastics.structure import get_es_source_val
from elastics.highlighter import PinyinHighlighter, HighlightMerger
from elastics.videos.constants import MATCH_TYPE, MATCH_OPERATOR


class VideoHitsParser:
    def __init__(self):
        self.pinyin_highlighter = PinyinHighlighter()

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
    ) -> list[dict]:
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
