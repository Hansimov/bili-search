import re

from tclogger import logger, dict_to_str

from llms.actions.author import AuthorChecker


class LLMActionsCaller:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def call(self, actions: list[dict] = []):
        actions = [action for action in actions if action["type"] == "tool_call"]
        results = []
        for action in actions:
            call_name = action.get("call_name", "")
            call_input = action.get("input", "")
            if self.verbose:
                logger.note(f"> Calling tool: {call_name}")

            if call_name == "check_author":
                checker = AuthorChecker()
                result = checker.check(call_input)
            elif call_name == "query":
                pass
            else:
                logger.warn(f"× Unknown tool call: {call_name}")

            results.append(
                {
                    "call_name": call_name,
                    "input": call_input,
                    "result": result,
                }
            )
        return results
