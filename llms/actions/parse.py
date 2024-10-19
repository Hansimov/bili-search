import re

from tclogger import logger, dict_to_str


class LLMActionsParser:
    RE_LB = r"\s*[\[\(]\s*"
    RE_RB = r"\s*[\)\]]\s*"
    RE_BQ = r"\s*`*\s*"
    RE_TEXT = r".*?"
    RE_THINK_NAME = r"(think)"
    RE_TOOL_NAME = r"(check_author|query)"
    RE_RESULT_NAME = r"(output|result)"
    RE_COLON = r"\s*:\s*"
    REP_THINK = (
        rf"(?P<llm_think>"
        rf"{RE_LB}(?P<think_name>{RE_THINK_NAME}){RE_RB}"
        rf"(?P<thoughts>{RE_TEXT})"
        rf"{RE_LB}/\s*(?P=think_name){RE_RB})"
    )
    REP_TOOL_CALL = (
        rf"(?P<tool_call>"
        rf"{RE_LB}(?P<call_name>{RE_TOOL_NAME}){RE_RB}"
        rf"(?P<input>{RE_TEXT})"
        rf"{RE_LB}/\s*(?P=call_name){RE_RB})"
    )
    REP_TOOL_RESULT = (
        rf"(?P<tool_result>"
        rf"{RE_LB}(?:(?P<result_name>{RE_RESULT_NAME})"
        rf"(?:{RE_COLON}(?P<from_tool>{RE_TOOL_NAME}))?){RE_RB}"
        rf"(?P<output>{RE_TEXT})"
        rf"{RE_LB}/\s*(?P=result_name){RE_RB})"
    )
    REP_RESP = f"({REP_THINK}|{REP_TOOL_CALL}|{REP_TOOL_RESULT})"

    ACTION_TYPES = ["llm_think", "tool_call", "tool_result"]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def parse(self, content: str) -> list[dict]:
        # logger.note(self.REP_RESP)
        matches = re.finditer(self.REP_RESP, content, re.MULTILINE | re.DOTALL)
        actions = []
        for match in matches:
            action = match.groupdict()
            if action.get("llm_think"):
                action["type"] = "llm_think"
            elif action.get("tool_call"):
                action["type"] = "tool_call"
            elif action.get("tool_result"):
                action["type"] = "tool_result"
            else:
                logger.warn(f"× Unknown action type: {action}")
            action = {
                k: v for k, v in action.items() if k not in self.ACTION_TYPES and v
            }
            actions.append(action)
        if self.verbose:
            logger.success(dict_to_str(actions))
        return actions


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
