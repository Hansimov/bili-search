from tclogger import logger, dict_to_str

from llms.client_by_model import LLMClientByModel, MODEL_CONFIG_TYPE

from llms.prompts.intro import COPILOT_INTRO, NOW_PROMPT
from llms.prompts.query import INSTRUCT_TO_QUERY_TOOL_DESC, QUERY_SYNTAX
from llms.prompts.query import INSTRUCT_TO_QUERY_TOOL_EXAMPLE
from llms.prompts.author import CHECK_AUTHOR_TOOL_DESC
from llms.actions.parse import LLMActionsParser
from llms.actions.call import LLMActionsCaller


class InstructToQueryAgent:
    def __init__(self, model_config: MODEL_CONFIG_TYPE = "qwen2-72b"):
        self.system_prompts = [
            *[COPILOT_INTRO, CHECK_AUTHOR_TOOL_DESC, QUERY_SYNTAX],
            *[INSTRUCT_TO_QUERY_TOOL_DESC, INSTRUCT_TO_QUERY_TOOL_EXAMPLE],
            NOW_PROMPT,
        ]
        self.client = LLMClientByModel(
            model_config, self.system_prompts, verbose_finish=False, verbose_usage=False
        ).client
        self.parser = LLMActionsParser()
        self.caller = LLMActionsCaller()

    def chat(self, messages: list):
        response_content = self.client.chat(messages)
        actions = self.parser.parse(response_content)
        results = self.caller.call(actions)
        logger.success(dict_to_str(results))


if __name__ == "__main__":
    agent = InstructToQueryAgent()
    user_prompt = "lks最近都有哪些高互动的视频？"
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    logger.note(user_prompt)
    agent.chat(messages)

    # python -m llms.agents.query
