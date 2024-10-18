from tclogger import logger

from llms.client_by_model import LLMClientByModel
from llms.agents.query import InstructToQueryAgent
from llms.prompts.intro import COPILOT_INTRO


def test_intro():
    agent = LLMClientByModel(system_prompts=[COPILOT_INTRO]).client
    user_prompt = "你是谁？"
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    logger.note(user_prompt)
    agent.chat(messages)


def test_instruct_to_query():
    agent = InstructToQueryAgent()
    user_prompt = "lks最近都有哪些高互动的视频？"
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    logger.note(user_prompt)
    agent.chat(messages)


if __name__ == "__main__":
    # test_intro()
    test_instruct_to_query()

    # python -m llms.tests
