from typing import Literal

from configs.envs import LLMS_ENVS
from llms.client import LLMClient

MODEL_CONFIG_TYPE = Literal["deepseek", "qwen2-72b"]


class LLMClientByModel:
    def __init__(
        self,
        model_config: MODEL_CONFIG_TYPE = "qwen2-72b",
        system_prompts: list[str] = [],
    ):
        llm_envs = LLMS_ENVS[model_config]
        init_messages = [{"role": "system", "content": "\n\n".join(system_prompts)}]
        llm_client_args = {
            "endpoint": llm_envs["endpoint"],
            "api_key": llm_envs["api_key"],
            "model": llm_envs["model"],
            "api_format": llm_envs["api_format"],
            "stream": True,
            "init_messages": init_messages,
        }
        self.llm_envs = llm_envs
        self.init_messages = init_messages
        self.client = LLMClient(**llm_client_args)