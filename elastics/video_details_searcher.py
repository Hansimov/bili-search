from datetime import datetime
from pprint import pformat
from tclogger import logger
from typing import Literal, Union

from elastics.client import ElasticSearchClient
from elastics.highlighter import PinyinHighlighter
from elastics.structure import get_es_source_val


class VideoDetailsSearcher:
    SCRIPT_FIELDS = {
        "pubdate.datetime": {
            "script": {
                "source": "doc['pubdate'].value.format(DateTimeFormatter.ofPattern('yyyy-MM-dd HH:mm:ss').withZone(ZoneId.of('UTC+8')))"
            }
        }
    }

    SOURCE_FIELDS = ["title", "bvid", "owner", "pic", "duration", "desc", "stat"]
    SUGGEST_MATCH_FIELDS = ["title", "title.pinyin"]
    SEARCH_MATCH_FIELDS = [
        "title",
        "title.pinyin",
        "owner.name",
        "owner.name.pinyin",
        "desc",
        "desc.pinyin",
    ]
    DOC_EXCLUDED_SOURCE_FIELDS = ["rights", "argue_info"]

    MATCH_TYPE = Literal[
        "best_fields",
        "most_fields",
        "cross_fields",
        "phrase",
        "phrase_prefix",
        "bool_prefix",
    ]

    SUGGEST_LIMIT = 10
    SEARCH_LIMIT = 50

    def __init__(self, index_name: str = "bili_video_details"):
        self.index_name = index_name
        self.es = ElasticSearchClient()
        self.es.connect()

    def get_highlight_settings(self, match_fields: list[str], tag: str = "hit"):
        highlight_fields = [
            field for field in match_fields if not field.endswith(".pinyin")
        ]
        highlight_fields_dict = {field: {} for field in highlight_fields}

        highlight_settings = {
            "pre_tags": [f"<{tag}>"],
            "post_tags": [f"</{tag}>"],
            "fields": highlight_fields_dict,
        }
        return highlight_settings

    def suggest(
        self,
        query: str,
        match_fields: list[str] = SUGGEST_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        match_type: MATCH_TYPE = "phrase_prefix",
        parse_hits: bool = True,
        is_explain: bool = False,
        limit: int = 10,
    ) -> Union[dict, list[dict]]:
        """
        Multi-match query:
            - https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-multi-match-query.html#multi-match-types

        I have compared the suggestion results of:
            - `suggest`with "completion"
            - `multi_search` with "phrase_prefix"

        And the conclusion is that `multi_search` is better and more flexible.
        """
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "type": match_type,
                    "fields": match_fields,
                }
            },
            "_source": source_fields,
            "script_fields": self.SCRIPT_FIELDS,
            "explain": is_explain,
            "highlight": self.get_highlight_settings(match_fields),
        }
        if limit and limit > 0:
            search_body["size"] = limit

        logger.note(f"> Get suggestions by query:", end=" ")
        logger.mesg(f"[{query}]")
        res = self.es.client.search(index=self.index_name, body=search_body)
        res_dict = res.body
        hits_info = self.parse_hits(
            query, match_fields, res_dict, request_type="suggest"
        )

        if parse_hits:
            logger.success(pformat(hits_info, indent=4, sort_dicts=False))
            return_res = hits_info
        else:
            logger.success(pformat(res.body, indent=4, sort_dicts=False))
            return_res = res_dict

        logger.mesg(f"  * Hits count: {len(hits_info)}")
        return return_res

    def search(
        self,
        query: str,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        match_type: MATCH_TYPE = "cross_fields",
        parse_hits: bool = True,
        is_explain: bool = False,
        limit: int = 50,
    ) -> Union[dict, list[dict]]:
        """The main difference between `search` and `suggest` is that,
        `search` has are more fuzzy (loose) and compositive match rules than `suggest`,
        and has more match fields.
        """
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "type": match_type,
                    "fields": match_fields,
                }
            },
            "_source": source_fields,
            "script_fields": self.SCRIPT_FIELDS,
            "explain": is_explain,
            "highlight": self.get_highlight_settings(match_fields),
        }
        if limit and limit > 0:
            search_body["size"] = limit

        logger.note(f"> Get search results by query:", end=" ")
        logger.mesg(f"[{query}]")
        res = self.es.client.search(index=self.index_name, body=search_body)
        res_dict = res.body
        hits_info = self.parse_hits(
            query, match_fields, res_dict, request_type="search"
        )

        if parse_hits:
            logger.success(pformat(hits_info, indent=4, sort_dicts=False))
            return_res = hits_info
        else:
            logger.success(pformat(res.body, indent=4, sort_dicts=False))
            return_res = res_dict

        logger.mesg(f"  * Hits count: {len(hits_info)}")
        return return_res

    def random(
        self,
        seed: Union[int, str] = None,
        seed_update_seconds: int = None,
        source_fields: list[str] = SOURCE_FIELDS,
        parse_hits: bool = True,
        is_explain: bool = False,
        limit: int = 1,
    ):
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
                    "functions": [
                        {
                            "random_score": {
                                "seed": seed,
                                "field": "_seq_no",
                            }
                        }
                    ],
                    "score_mode": "sum",
                }
            },
            "_source": source_fields,
            "script_fields": self.SCRIPT_FIELDS,
            "explain": is_explain,
        }

        if limit and limit > 0:
            search_body["size"] = limit

        logger.note(f"> Get random docs with seed:", end=" ")
        logger.mesg(f"[{seed}] ({now_str})")
        res = self.es.client.search(index=self.index_name, body=search_body)
        res_dict = res.body
        hits_info = self.parse_hits("", [], res_dict, request_type="random")

        if parse_hits:
            logger.success(pformat(hits_info, indent=4, sort_dicts=False))
            return_res = hits_info
        else:
            logger.success(pformat(res.body, indent=4, sort_dicts=False))
            return_res = res_dict

        logger.mesg(f"  * Random count: {len(hits_info)}")
        return return_res

    def latest(
        self,
        source_fields: list[str] = SOURCE_FIELDS,
        parse_hits: bool = True,
        is_explain: bool = False,
        limit: int = 10,
    ) -> Union[dict, list[dict]]:
        search_body = {
            "query": {"match_all": {}},
            "sort": [{"pubdate": {"order": "desc"}}],
            "_source": source_fields,
            "script_fields": self.SCRIPT_FIELDS,
            "explain": is_explain,
        }

        if limit and limit > 0:
            search_body["size"] = limit

        logger.note(f"> Get latest {limit} docs:")
        res = self.es.client.search(index=self.index_name, body=search_body)
        res_dict = res.body
        hits_info = self.parse_hits("", [], res_dict, request_type="latest")

        if parse_hits:
            logger.success(pformat(hits_info, indent=4, sort_dicts=False))
            return_res = hits_info
        else:
            logger.success(pformat(res.body, indent=4, sort_dicts=False))
            return_res = res_dict

        logger.mesg(f"  * Latest count: {len(hits_info)}")
        return return_res

    def doc(
        self,
        bvid: str,
        included_source_fields: list[str] = [],
        excluded_source_fields: list[str] = DOC_EXCLUDED_SOURCE_FIELDS,
    ) -> dict:
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
        return res_dict

    def get_pinyin_highlights(
        self, query: str, match_fields: list[str], _source: dict
    ) -> dict:
        pinyin_highlights = {}
        if query:
            pinyin_highlighter = PinyinHighlighter()
            for field in match_fields:
                if field.endswith(".pinyin"):
                    highlighted_text = pinyin_highlighter.highlight(
                        query,
                        get_es_source_val(_source, field[: -len(".pinyin")]),
                        tag="hit",
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
    ) -> list[dict]:

        hits_info = {
            "count": res_dict["hits"]["total"]["value"],
            "hits": [],
            "request_type": request_type,
        }
        for hit in res_dict["hits"]["hits"]:
            _source = hit["_source"]
            score = hit["_score"]
            pubdate_str = hit["fields"]["pubdate.datetime"][0]
            common_highlights = hit.get("highlight", {})
            pinyin_highlights = self.get_pinyin_highlights(query, match_fields, _source)
            hit_info = {
                **_source,
                "score": score,
                "pubdate_str": pubdate_str,
                "common_highlights": common_highlights,
                "pinyin_highlights": pinyin_highlights,
            }
            hits_info["hits"].append(hit_info)
        return hits_info


if __name__ == "__main__":
    searcher = VideoDetailsSearcher()
    # searcher.suggest("teji")
    # searcher.random(seed_update_seconds=10, limit=3)
    # searcher.latest(limit=10)
    searcher.doc("BV1Qz421B7W9")

    # python -m elastics.video_details_searcher
