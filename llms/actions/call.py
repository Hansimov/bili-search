import re

from tclogger import logger, logstr, dict_to_str, brk

from llms.actions.entity import EntityCatogorizer
from llms.actions.search import SearchTool
from llms.actions.explore import ExploreTool
from llms.actions.input import InputCleaner


class LLMActionsCaller:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.cleaner = InputCleaner()

    def call(self, actions: list[dict] = []):
        actions = [action for action in actions if action["action_type"] == "tool_call"]
        results = []
        for action in actions:
            tool_name = action.get("tool_name", "")
            tool_input = action.get("tool_input", "")
            tool_input = self.cleaner.clean(tool_input)

            if self.verbose:
                logger.note(f"> Calling tool: {(logstr.file(brk(tool_name)))}")

            if tool_name == "entity":
                categorizer = EntityCatogorizer()
                result = categorizer.categorize(tool_input)
            elif tool_name == "search":
                searcher = SearchTool()
                result = searcher.search(tool_input, is_shrink_results=True)
            elif tool_name == "explore":
                explorer = ExploreTool()
                result = explorer.explore(tool_input, is_shrink_results=True)
            else:
                logger.warn(f"× Unknown tool call: {tool_name}")
                continue

            results.append(
                {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_result": result,
                }
            )
        return results
