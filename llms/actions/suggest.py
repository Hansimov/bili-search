from tclogger import logger, logstr, dict_to_str
from typing import Literal

from elastics.videos.searcher import VideoSearcher
from configs.envs import SEARCH_APP_ENVS


class SuggestTool:
    def __init__(
        self,
        mode: Literal["prod", "dev"] = "dev",
        source_fields: list[str] = ["title", "bvid", "owner", "pubdate_str"],
        limit: int = 10,
    ):
        index_name = SEARCH_APP_ENVS["bili_videos_index"][mode]
        self.searcher = VideoSearcher(index_name, elastic_verbose=False)
        self.source_fields = source_fields
        self.limit = limit

    def shrink_hits_by_source_fields(
        self, hits: list[dict], source_fields: list[str] = []
    ) -> list[dict]:
        new_hits = [
            {k: v for k, v in hit.items() if k in source_fields} for hit in hits
        ]
        return new_hits

    def shrink_results(self, results: dict) -> dict:
        hits = results.get("hits", [])
        hits = self.shrink_hits_by_source_fields(hits, source_fields=self.source_fields)
        res = {"query": results.get("query", ""), "hits": hits}
        return res

    def suggest(
        self,
        query: str,
        source_fields: list[str] = [],
        limit: int = 10,
        is_shrink_results: bool = False,
    ) -> dict:
        source_fields = source_fields or self.source_fields
        limit = limit or self.limit
        res = self.searcher.multi_level_suggest(
            query, source_fields=source_fields, limit=limit
        )
        if is_shrink_results:
            res = self.shrink_results(res)

        return res


if __name__ == "__main__":
    query = "红警 月亮3"
    logger.note(f"> Query: [{logstr.mesg(query)}]")
    suggester = SuggestTool()
    hits = suggester.suggest(query)
    logger.success(dict_to_str(hits, add_quotes=True))

    # python -m llms.actions.suggest
