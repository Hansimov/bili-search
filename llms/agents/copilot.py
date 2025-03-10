import asyncio

from tclogger import logger, dict_to_str

from llms.client_by_model import LLMClientByModel, MODEL_CONFIG_TYPE
from llms.prompts.system import TODAY_PROMPT
from llms.prompts.syntax import SEARCH_SYNTAX
from llms.prompts.entity import ENTITY_CATEGORIZER_TOOL_DESC
from llms.prompts.copilot import COPILOT_DESC, COPILOT_EXAMPLE
from llms.actions.parse import LLMActionsParser
from llms.actions.call import LLMActionsCaller


class CopilotAgent:
    def __init__(
        self,
        model_config: MODEL_CONFIG_TYPE = "volcengine",
        delta_func: callable = None,
        tool_func: callable = None,
        terminate_event: asyncio.Event = None,
        verbose_action: bool = False,
        verbose_chat: bool = True,
    ):
        self.system_prompts = [
            *[COPILOT_DESC, ENTITY_CATEGORIZER_TOOL_DESC, SEARCH_SYNTAX],
            *[COPILOT_EXAMPLE, TODAY_PROMPT],
        ]
        self.parser = LLMActionsParser(verbose=verbose_action)
        self.caller = LLMActionsCaller(verbose=verbose_action)
        self.verbose_action = verbose_action
        self.verbose_chat = verbose_chat
        self.client = LLMClientByModel(
            model_config,
            system_prompts=self.system_prompts,
            delta_func=delta_func,
            terminate_event=terminate_event,
            verbose_user=verbose_chat,
            verbose_assistant=verbose_chat,
            verbose_content=verbose_chat,
            verbose_finish=False,
            verbose_usage=False,
        ).client
        self.tool_func = tool_func

    def exec_tool_func(self, results: list[dict]):
        if self.tool_func:
            tool_func_args = {"info": results}
            if asyncio.iscoroutinefunction(self.tool_func):
                asyncio.run(self.tool_func(**tool_func_args))
            else:
                self.tool_func(**tool_func_args)

    def chat(self, messages: list):
        response_content = self.client.chat(messages)
        actions, has_tool_call = self.parser.parse(response_content)
        while actions and has_tool_call:
            results = self.caller.call(actions)
            self.exec_tool_func(results)
            logger.success(dict_to_str(results), verbose=self.verbose_action)
            results_str = self.parser.jsonize(results)
            new_message = {"role": "user", "content": results_str}
            logger.note(results_str, verbose=self.verbose_action)
            messages.append(new_message)
            response_content = self.client.chat(messages)
            actions, has_tool_call = self.parser.parse(response_content)


if __name__ == "__main__":
    agent = CopilotAgent("volcengine", verbose_action=False)
    user_prompt = "李沐2021年发了什么"
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    agent.chat(messages)

    # python -m llms.agents.copilot
