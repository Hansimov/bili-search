from tclogger import logger, logstr, dict_to_str

from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from llms.constants import SOURCE_FIELDS_FOR_LLM_SEARCH
from llms.actions.utils import shrink_results


class SearchTool:
    def __init__(
        self,
        index_name: str = ELASTIC_VIDEOS_DEV_INDEX,
        elastic_env_name: str = ELASTIC_DEV,
        source_fields: list[str] = SOURCE_FIELDS_FOR_LLM_SEARCH,
        limit: int = 20,
    ):
        self.searcher = VideoSearcherV2(
            index_name=index_name, elastic_env_name=elastic_env_name
        )
        self.source_fields = source_fields
        self.limit = limit

    def search(
        self,
        query: str,
        source_fields: list[str] = [],
        limit: int = 20,
        is_shrink_results: bool = False,
    ) -> dict:
        source_fields = source_fields or self.source_fields
        limit = limit or self.limit
        res = self.searcher.search(
            query, source_fields=source_fields, limit=limit, verbose=False
        )
        if is_shrink_results:
            res = shrink_results(res, self.source_fields)

        return res


if __name__ == "__main__":
    query = "影视飓风 罗永浩"
    logger.note(f"> Query: [{logstr.mesg(query)}]")
    searcher = SearchTool()
    results = searcher.search(query, is_shrink_results=True)
    logger.success(dict_to_str(results, add_quotes=True))

    # python -m llms.actions.search
