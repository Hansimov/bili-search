from tclogger import logger

from configs.envs import LLMS_ENVS
from llms.client import LLMClient
from llms.prompts.query import COPILOT_INTRO, NOW_STR_PROMPT
from llms.prompts.query import INSTRUCT_TO_QUERY_TOOL_DESC, QUERY_SYNTAX
from llms.prompts.query import INSTRUCT_TO_QUERY_TOOL_EXAMPLE
from llms.prompts.author import CHECK_AUTHOR_TOOL_DESC

# llm = LLMS_ENVS["deepseek"]
# llm = LLMS_ENVS["qwen2.5-coder-7b"]
llm = LLMS_ENVS["qwen2-72b"]

init_messages = [
    {
        "role": "system",
        "content": "\n\n".join(
            [
                COPILOT_INTRO,
                CHECK_AUTHOR_TOOL_DESC,
                INSTRUCT_TO_QUERY_TOOL_DESC,
                QUERY_SYNTAX,
                INSTRUCT_TO_QUERY_TOOL_EXAMPLE,
                NOW_STR_PROMPT,
            ]
        ),
    }
]

args_dict = {
    "endpoint": llm["endpoint"],
    "api_key": llm["api_key"],
    "model": llm["model"],
    "api_format": llm["api_format"],
    "stream": True,
    "init_messages": init_messages,
}
client = LLMClient(**args_dict)


def test_intro():
    messages = [
        {"role": "system", "content": COPILOT_INTRO},
        {"role": "user", "content": "你是谁？"},
    ]
    client.chat(messages)


def test_intension_to_query():
    user_prompt = "lks最近都有哪些高互动的视频？"
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    logger.note(user_prompt)
    client.chat(messages)


if __name__ == "__main__":
    # test_intro()
    test_intension_to_query()

    # python -m llms.tests
