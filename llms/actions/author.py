from tclogger import logger, dict_to_str

from llms.client_by_model import LLMClientByModel, MODEL_CONFIG_TYPE
from llms.prompts.author import CHECK_AUTHOR_TOOL_DESC, CHECK_AUTHOR_TOOL_EXAMPLE
from llms.actions.parse import LLMActionsParser
from llms.actions.suggest import VideoSuggester


class AuthorChecker:
    def __init__(
        self,
        model_config: MODEL_CONFIG_TYPE = "qwen2-72b",
        suggest_limit: int = 20,
        min_count: int = 5,
    ):
        self.system_prompts = [CHECK_AUTHOR_TOOL_DESC, CHECK_AUTHOR_TOOL_EXAMPLE]
        self.client = LLMClientByModel(model_config, self.system_prompts).client
        self.parser = LLMActionsParser(verbose=True)
        self.suggestor = VideoSuggester()
        self.suggest_limit = suggest_limit
        self.min_count = min_count

    def de_backtick(self, input: str):
        return input.strip().strip("`").strip()

    def count_authors(self, suggestions: list[dict]) -> dict:
        res = {}
        for suggestion in suggestions:
            owner = suggestion.get("owner", {})
            name = owner.get("name", None)
            uid = owner.get("mid", None)
            if name in res.keys():
                res[name]["count"] += 1
            else:
                res[name] = {"uid": uid, "count": 1}
        res = dict(sorted(res.items(), key=lambda item: item[1]["count"], reverse=True))
        return res

    def filter(self, authors: dict) -> list:
        res = []
        total_count = sum([info["count"] for name, info in authors.items()])
        for name, info in authors.items():
            count = info["count"]
            if count >= self.min_count:
                res.append(
                    {
                        "name": name,
                        "uid": info["uid"],
                        "ratio": round(count / total_count, 2),
                    }
                )
        res = sorted(res, key=lambda item: item["ratio"], reverse=True)
        return res

    def check(self, query: str):
        query = self.de_backtick(query)
        suggestions = self.suggestor.suggest(query, limit=20)
        authors = self.count_authors(suggestions["hits"])
        filtered_authors = self.filter(authors)
        if filtered_authors:
            res = {
                "intension": "search_author",
                "authors": filtered_authors,
            }
        else:
            res = {
                "intension": "search_text",
            }
        return res


if __name__ == "__main__":
    queries = ["lks", "08", "月亮 3", "黑神话", "马鹿", "影视飓风", "白鼠", "老e"]
    agent = AuthorChecker()
    for query in queries:
        logger.note(f"> Query: {query}")
        authors = agent.check(query)
        logger.success(dict_to_str(authors, add_quotes=True))

    # python -m llms.actions.author
