from copy import deepcopy
from tclogger import logger, logstr, dict_to_str, brk

from llms.actions.suggest import SuggestTool


class EntityCatogorizer:
    def __init__(
        self, suggest_limit: int = 20, threshold: int = 2, verbose: bool = False
    ):
        self.suggestor = SuggestTool()
        self.suggest_limit = suggest_limit
        self.threshold = threshold
        self.verbose = verbose

    def merge_keywords(self, keywords: dict):
        res = {}
        for field, item in keywords.items():
            for key, count in item.items():
                if key in res:
                    res[key] += count
                else:
                    res[key] = count
        return res

    def count_to_ratio(self, authors: dict, total_count: int):
        total_count = total_count or self.suggest_limit
        res = deepcopy(authors)
        for name, item in res.items():
            res[name]["ratio"] = item["count"] / total_count
            res[name].pop("count")
        return res

    def categorize(self, query: str):
        suggestions = self.suggestor.suggest(query, limit=25)
        # logger.success(dict_to_str(suggestions))
        total_hits = len(suggestions.get("hits", []))
        suggest_info = suggestions.get("suggest_info", {})
        highlighted_keywords = self.merge_keywords(
            suggest_info.get("qword_hword_count", {})
        )
        related_authors = self.count_to_ratio(
            suggest_info.get("related_authors", {}), total_hits
        )
        res = {
            "query": query,
            "total_hits": total_hits,
            "highlighted_keywords": highlighted_keywords,
            "related_authors": related_authors,
        }
        if self.verbose:
            logger.success(dict_to_str(res, align_colon=False, add_quotes=True))
        return res


if __name__ == "__main__":
    queries = [
        *["08", "月亮3"],
        *["马鹿", "白鼠", "老e"],
        *["黑神话", "影视飓风", "lks"],
    ]
    agent = EntityCatogorizer(verbose=True)
    for query in queries:
        logger.note(f"> Query: {logstr.mesg(brk(query))}")
        res = agent.categorize(query)

    # python -m llms.actions.entity
