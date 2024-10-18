from tclogger import logger

from llms.client_by_model import LLMClientByModel, MODEL_CONFIG_TYPE

from llms.prompts.intro import COPILOT_INTRO, NOW_PROMPT
from llms.prompts.query import INSTRUCT_TO_QUERY_TOOL_DESC, QUERY_SYNTAX
from llms.prompts.query import INSTRUCT_TO_QUERY_TOOL_EXAMPLE
from llms.prompts.author import CHECK_AUTHOR_TOOL_DESC


class InstructToQueryAgent:
    def __init__(self, model_config: MODEL_CONFIG_TYPE = "qwen2-72b"):
        self.system_prompts = [
            *[COPILOT_INTRO, CHECK_AUTHOR_TOOL_DESC, QUERY_SYNTAX],
            *[INSTRUCT_TO_QUERY_TOOL_DESC, INSTRUCT_TO_QUERY_TOOL_EXAMPLE],
            NOW_PROMPT,
        ]
        self.client = LLMClientByModel(model_config, self.system_prompts).client

    def chat(self, messages: list):
        self.client.chat(messages)
