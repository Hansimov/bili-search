import re

from tclogger import logger, dict_to_str
from typing import Union


class LLMActionsParser:
    RE_LB = r"\s*[\[\(]\s*"
    RE_RB = r"\s*[\)\]]\s*"
    RE_BQ = r"\s*`*\s*"
    RE_TEXT = r".*?"
    RE_THINK_TYPE = r"(think)"
    RE_TOOL_NAME = r"(check_author|search)"
    RE_RESULT_TYPE = r"(output|result)"
    RE_COLON = r"\s*:\s*"
    REP_THINK = (
        rf"(?P<llm_think>"
        rf"{RE_LB}(?P<think_type>{RE_THINK_TYPE}){RE_RB}"
        rf"(?P<think_content>{RE_TEXT})"
        rf"{RE_LB}/\s*(?P=think_type){RE_RB})"
    )
    REP_TOOL_CALL = (
        rf"(?P<tool_call>"
        rf"{RE_LB}(?P<tool_name>{RE_TOOL_NAME}){RE_RB}"
        rf"(?P<tool_input>{RE_TEXT})"
        rf"{RE_LB}/\s*(?P=tool_name){RE_RB})"
    )
    REP_TOOL_RESULT = (
        rf"(?P<tool_result>"
        rf"{RE_LB}(?:(?P<result_type>{RE_RESULT_TYPE})"
        rf"(?:{RE_COLON}(?P<from_tool>{RE_TOOL_NAME}))?){RE_RB}"
        rf"(?P<tool_output>{RE_TEXT})"
        rf"{RE_LB}/\s*(?P=result_type){RE_RB})"
    )
    REP_RESP = f"({REP_THINK}|{REP_TOOL_CALL}|{REP_TOOL_RESULT})"

    ACTION_TYPES = ["llm_think", "tool_call", "tool_result"]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def parse(self, content: str) -> tuple[list[dict], bool]:
        # logger.note(self.REP_RESP)
        if self.verbose:
            logger.note(f"> Parsing actions from content ...")
        matches = re.finditer(self.REP_RESP, content, re.MULTILINE | re.DOTALL)
        actions = []
        has_tool_call = False
        for match in matches:
            action = match.groupdict()
            if action.get("llm_think"):
                action["action_type"] = "llm_think"
            elif action.get("tool_call"):
                action["action_type"] = "tool_call"
                has_tool_call = True
            elif action.get("tool_result"):
                action["action_type"] = "tool_result"
            else:
                logger.warn(f"× Unknown action type: {action}")
            action = {
                k: v for k, v in action.items() if k not in self.ACTION_TYPES and v
            }
            actions.append(action)
        if self.verbose:
            if actions:
                logger.success(dict_to_str(actions))
            else:
                logger.warn("  × No actions found. Iteration Stopped.")
        return actions, has_tool_call

    def jsonize(self, result: Union[dict, list]):
        result_str = dict_to_str(
            result, align_colon=False, is_colored=False, add_quotes=True
        )
        return f"```json\n{result_str}\n```"


if __name__ == "__main__":
    content = """
    [think] 影视飓风可能是一个视频作者昵称或者视频系列，
    应当调用工具 `check_author` 来确认用户的意图是是搜索对应的文本，还是视频作者。 [/think]
    [check_author] `影视飓风` [/check_author]
    [output:check_author]
    ```json
    {"intension": "search_author", "name": "影视飓风"}
    ```
    [/output]
    [think] 可知用户想搜索是昵称为“影视飓风”的作者，因此需要使用昵称过滤器 [/think]
    [query] `:name=影视飓风` [/query]
    """
    parser = LLMActionsParser(verbose=True)
    actions = parser.parse(content)

    # python -m llms.actions.parse
