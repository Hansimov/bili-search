from tclogger import logger, logstr, dict_to_str

from elastics.videos.explorer import VideoExplorer
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from llms.constants import SOURCE_FIELDS_FOR_LLM_EXPLORE
from llms.actions.utils import (
    shrink_hits_by_source_fields,
    add_links,
    add_pubdate_str,
    add_pub_to_now_str,
)


class ExploreTool:
    def __init__(
        self,
        index_name: str = ELASTIC_VIDEOS_DEV_INDEX,
        elastic_env_name: str = ELASTIC_DEV,
        source_fields: list[str] = SOURCE_FIELDS_FOR_LLM_EXPLORE,
        limit: int = 20,
    ):
        self.explorer = VideoExplorer(
            index_name=index_name, elastic_env_name=elastic_env_name
        )
        self.source_fields = source_fields
        self.limit = limit

    def extract_hits_from_explore_result(self, result: dict) -> list[dict]:
        """Extract hits from explore result data steps."""
        hits = []
        for step in result.get("data", []):
            if step.get("output_type") == "hits":
                output = step.get("output", {})
                step_hits = output.get("hits", [])
                if step_hits:
                    hits = step_hits
                    break
        return hits

    def shrink_results(self, result: dict) -> dict:
        hits = self.extract_hits_from_explore_result(result)
        hits = shrink_hits_by_source_fields(hits, self.source_fields)
        hits = add_links(hits)
        hits = add_pubdate_str(hits)
        hits = add_pub_to_now_str(hits)
        res = {
            "query": result.get("query", ""),
            "status": result.get("status", ""),
            "hits": hits,
        }
        return res

    def explore(
        self,
        query: str,
        limit: int = 0,
        is_shrink_results: bool = False,
    ) -> dict:
        limit = limit or self.limit
        res = self.explorer.explore(
            query=query,
            rank_top_k=limit,
            verbose=False,
        )
        if is_shrink_results:
            res = self.shrink_results(res)
        return res


if __name__ == "__main__":
    query = "影视飓风 罗永浩"
    logger.note(f"> Query: [{logstr.mesg(query)}]")
    explorer = ExploreTool()
    results = explorer.explore(query, is_shrink_results=True)
    logger.success(dict_to_str(results, add_quotes=True))

    # python -m llms.actions.explore
