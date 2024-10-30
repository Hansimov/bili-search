from tclogger import logger, logstr, dict_to_str
from typing import Literal

from elastics.videos.searcher import VideoSearcher
from configs.envs import SEARCH_APP_ENVS
from converters.query.field import is_field_in_fields


class SearchTool:
    def __init__(
        self,
        mode: Literal["prod", "dev"] = "dev",
        source_fields: list[str] = [
            *["title", "desc", "tags"],
            *["bvid", "pic", "owner", "pubdate_str"],
            *["stat.view", "stat.coin", "stat.danmaku"],
        ],
        limit: int = 20,
    ):
        index_name = SEARCH_APP_ENVS["bili_videos_index"][mode]
        self.searcher = VideoSearcher(index_name, elastic_verbose=False)
        self.source_fields = source_fields
        self.limit = limit

    def shrink_hits_by_source_fields(self, hits: list[dict]) -> list[dict]:
        new_hits = [
            {k: v for k, v in hit.items() if is_field_in_fields(k, self.source_fields)}
            for hit in hits
        ]
        return new_hits

    def shrink_results(self, results: dict) -> dict:
        hits = results.get("hits", [])
        hits = self.shrink_hits_by_source_fields(hits)
        res = {"query": results.get("query", ""), "hits": hits}
        return res

    def add_links(self, results: dict) -> dict:
        hits = results.get("hits", [])
        for hit in hits:
            hit["link"] = f"https://www.bilibili.com/video/{hit['bvid']}"
        return results

    def search(
        self,
        query: str,
        source_fields: list[str] = [],
        limit: int = 20,
        is_shrink_results: bool = False,
    ) -> dict:
        source_fields = source_fields or self.source_fields
        limit = limit or self.limit
        res = self.searcher.multi_level_search(
            query, source_fields=source_fields, limit=limit
        )
        if is_shrink_results:
            res = self.shrink_results(res)
        res = self.add_links(res)

        return res
