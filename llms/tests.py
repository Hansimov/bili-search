from tclogger import logger

from configs.envs import LLMS_ENVS
from llms.client import LLMClient
from llms.prompts import COPILOT_INTRO_PROMPT, NOW_STR_PROMPT
from llms.prompts import DSL_SYNTAX_PROMPT, TOOL_INTENSION_TO_QUERY_PROMPT

# llm = LLMS_ENVS["deepseek"]
# llm = LLMS_ENVS["qwen2.5-coder-7b"]
llm = LLMS_ENVS["qwen2-72b"]
args_dict = {
    "endpoint": llm["endpoint"],
    "api_key": llm["api_key"],
    "model": llm["model"],
    "api_format": llm["api_format"],
    "stream": True,
}
client = LLMClient(**args_dict)


def test_intro():
    messages = [
        {"role": "system", "content": COPILOT_INTRO_PROMPT},
        {"role": "user", "content": "你是谁？"},
    ]
    client.chat(messages)


def test_intension_to_query():
    system_prompt = "\n\n".join(
        [
            COPILOT_INTRO_PROMPT,
            NOW_STR_PROMPT,
            DSL_SYNTAX_PROMPT,
            TOOL_INTENSION_TO_QUERY_PROMPT,
        ]
    )
    user_prompt = "影视飓风最近都有哪些高互动的视频？"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    logger.note(user_prompt)
    client.chat(messages)


if __name__ == "__main__":
    # test_intro()
    test_intension_to_query()

    # python -m llms.tests
