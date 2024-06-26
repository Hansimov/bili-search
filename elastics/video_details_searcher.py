from datetime import datetime
from pprint import pformat
from tclogger import logger
from typing import Literal, Union

from elastics.client import ElasticSearchClient


class VideoDetailsSearcher:
    script_fields = {
        "pubdate.datetime": {
            "script": {
                "source": "doc['pubdate'].value.format(DateTimeFormatter.ofPattern('yyyy-MM-dd HH:mm:ss').withZone(ZoneId.of('UTC+8')))"
            }
        }
    }

    def __init__(self, index_name: str = "bili_video_details"):
        self.index_name = index_name
        self.es = ElasticSearchClient()
        self.es.connect()

    def suggest(
        self,
        query: str,
        parse_hits: bool = True,
        match_fields: list[str] = ["title", "title.pinyin"],
        source_fields: list[str] = ["title", "bvid"],
        match_type: Literal[
            "best_fields",
            "most_fields",
            "cross_fields",
            "phrase",
            "phrase_prefix",
            "bool_prefix",
        ] = "phrase_prefix",
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
            "script_fields": self.script_fields,
            "explain": is_explain,
        }
        if limit and limit > 0:
            search_body["size"] = limit

        logger.note(f"> Suggest for query:", end=" ")
        logger.mesg(f"[{query}]")
        res = self.es.client.search(index=self.index_name, body=search_body)
        res_dict = res.body
        hits_info = self.parse_hits(res_dict)

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
        parse_hits: bool = True,
        source_fields: list[str] = ["title", "bvid"],
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
            "script_fields": self.script_fields,
            "explain": is_explain,
        }

        if limit and limit > 0:
            search_body["size"] = limit

        logger.note(f"> Random docs with seed:", end=" ")
        logger.mesg(f"[{seed}] ({now_str})")
        res = self.es.client.search(index=self.index_name, body=search_body)
        res_dict = res.body
        hits_info = self.parse_hits(res_dict)

        if parse_hits:
            logger.success(pformat(hits_info, indent=4, sort_dicts=False))
            return_res = hits_info
        else:
            logger.success(pformat(res.body, indent=4, sort_dicts=False))
            return_res = res_dict

        logger.mesg(f"  * Random count: {len(hits_info)}")
        return return_res

    def parse_hits(self, res_dict: dict) -> list[dict]:
        hits_info = []
        for hit in res_dict["hits"]["hits"]:
            hit_source = hit["_source"]
            bvid = hit_source["bvid"]
            title = hit_source["title"]
            score = hit["_score"]
            pubdate = hit["fields"]["pubdate.datetime"][0]

            hit_info = {
                "bvid": bvid,
                "title": title,
                "score": score,
                "pubdate": pubdate,
            }
            hits_info.append(hit_info)
        return hits_info


if __name__ == "__main__":
    searcher = VideoDetailsSearcher()
    # searcher.suggest("teji")
    searcher.random(seed_update_seconds=10, limit=3)

    # python -m elastics.video_details_searcher
