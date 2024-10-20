from tclogger import logger, dict_to_str

from llms.client_by_model import LLMClientByModel, MODEL_CONFIG_TYPE

from llms.prompts.system import TODAY_PROMPT
from llms.prompts.syntax import SEARCH_SYNTAX
from llms.prompts.author import CHECK_AUTHOR_TOOL_DESC
from llms.prompts.copilot import COPILOT_DESC, COPILOT_EXAMPLE
from llms.actions.parse import LLMActionsParser
from llms.actions.call import LLMActionsCaller


class CopilotAgent:
    def __init__(
        self,
        model_config: MODEL_CONFIG_TYPE = "qwen2-72b",
        verbose_action: bool = False,
        verbose_chat: bool = True,
    ):
        self.system_prompts = [
            *[COPILOT_DESC, CHECK_AUTHOR_TOOL_DESC, SEARCH_SYNTAX],
            *[COPILOT_EXAMPLE, TODAY_PROMPT],
        ]
        self.parser = LLMActionsParser(verbose=verbose_action)
        self.caller = LLMActionsCaller(verbose=verbose_action)
        self.verbose_action = verbose_action
        self.verbose_chat = verbose_chat
        self.client = LLMClientByModel(
            model_config,
            system_prompts=self.system_prompts,
            verbose_user=verbose_chat,
            verbose_assistant=verbose_chat,
            verbose_content=verbose_chat,
            verbose_finish=False,
            verbose_usage=False,
        ).client

    def chat(self, messages: list):
        response_content = self.client.chat(messages)
        actions, has_tool_call = self.parser.parse(response_content)
        while actions and has_tool_call:
            results = self.caller.call(actions)
            if self.verbose_action:
                logger.success(dict_to_str(results))
            results_str = self.parser.jsonize(results)
            new_message = {"role": "user", "content": results_str}
            if self.verbose_action:
                logger.note(results_str)
            messages.append(new_message)
            response_content = self.client.chat(messages)
            actions, has_tool_call = self.parser.parse(response_content)


if __name__ == "__main__":
    agent = CopilotAgent("deepseek", verbose_action=False)
    user_prompt = "李沐在2021年发了什么作品"
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    agent.chat(messages)

    # python -m llms.agents.copilot
