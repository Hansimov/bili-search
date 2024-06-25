from pprint import pformat
from tclogger import logger
from typing import Literal, Union

from elastics.client import ElasticSearchClient


class VideoDetailsSuggester:
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
            "script_fields": {
                "pubdate.datetime": {
                    "script": {
                        "source": "doc['pubdate'].value.format(DateTimeFormatter.ofPattern('yyyy-MM-dd HH:mm:ss').withZone(ZoneId.of('UTC+8')))"
                    }
                }
            },
            "explain": is_explain,
        }
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
    suggester = VideoDetailsSuggester()
    # query = "teji"
    query = "ali"
    suggester.suggest(query)

    # python -m elastics.video_details_suggest
