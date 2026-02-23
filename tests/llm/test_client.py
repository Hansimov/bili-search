"""Tests for webu.llms.client (LLMClient / LLMClientByConfig) with DeepSeek API."""

from configs.envs import LLMS_ENVS
from webu.llms.client import LLMClient, LLMClientByConfig
from tclogger import logger


def get_deepseek_client_configs(
    stream: bool = True,
    verbose: bool = True,
) -> dict:
    """Build LLMClient kwargs from LLMS_ENVS['deepseek'], filtering extra keys."""
    ds = LLMS_ENVS["deepseek"]
    return {
        "endpoint": ds["endpoint"],
        "api_key": ds["api_key"],
        "model": ds["model"],
        "api_format": ds["api_format"],
        "stream": stream,
        "verbose": verbose,
    }


SIMPLE_MESSAGES = [
    {"role": "system", "content": "你是一个简洁的助手，用一两句话回答问题。"},
    {"role": "user", "content": "请用一句话介绍 Python 语言。"},
]

MULTI_TURN_MESSAGES = [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user", "content": "1+1等于几？"},
    {"role": "assistant", "content": "1+1等于2。"},
    {"role": "user", "content": "再加3等于几？"},
]


def test_non_stream_chat():
    """Test non-streaming chat with DeepSeek API."""
    logger.note("=" * 60)
    logger.note("[TEST] Non-streaming chat")
    logger.note("=" * 60)

    configs = get_deepseek_client_configs(stream=False, verbose=True)
    client = LLMClient(**configs)

    response = client.chat(messages=SIMPLE_MESSAGES, stream=False)

    assert response, "Response should not be empty"
    assert isinstance(response, str), "Response should be a string"
    logger.success(f"\n[PASS] Non-streaming chat returned: {len(response)} chars")
    return response


def test_stream_chat():
    """Test streaming chat with DeepSeek API."""
    logger.note("=" * 60)
    logger.note("[TEST] Streaming chat")
    logger.note("=" * 60)

    configs = get_deepseek_client_configs(stream=True, verbose=True)
    client = LLMClient(**configs)

    response = client.chat(messages=SIMPLE_MESSAGES, stream=True)

    assert response, "Response should not be empty"
    assert isinstance(response, str), "Response should be a string"
    logger.success(f"\n[PASS] Streaming chat returned: {len(response)} chars")
    return response


def test_multi_turn_chat():
    """Test multi-turn conversation with DeepSeek API."""
    logger.note("=" * 60)
    logger.note("[TEST] Multi-turn chat")
    logger.note("=" * 60)

    configs = get_deepseek_client_configs(stream=True, verbose=True)
    client = LLMClient(**configs)

    response = client.chat(messages=MULTI_TURN_MESSAGES, stream=True)

    assert response, "Response should not be empty"
    assert isinstance(response, str), "Response should be a string"
    logger.success(f"\n[PASS] Multi-turn chat returned: {len(response)} chars")
    return response


def test_client_by_config():
    """Test LLMClientByConfig with DeepSeek configs."""
    logger.note("=" * 60)
    logger.note("[TEST] LLMClientByConfig")
    logger.note("=" * 60)

    ds = LLMS_ENVS["deepseek"]
    # LLMClientByConfig expects LLMConfigsType (TypedDict),
    # filter out keys not in LLMClient.__init__ (e.g. 'tasks')
    configs = {
        "endpoint": ds["endpoint"],
        "api_key": ds["api_key"],
        "model": ds["model"],
        "api_format": ds["api_format"],
        "stream": True,
        "verbose": True,
    }
    client = LLMClientByConfig(configs)

    response = client.chat(messages=SIMPLE_MESSAGES, stream=True)

    assert response, "Response should not be empty"
    assert isinstance(response, str), "Response should be a string"
    logger.success(f"\n[PASS] LLMClientByConfig chat returned: {len(response)} chars")
    return response


def test_temperature_param():
    """Test chat with custom temperature parameter."""
    logger.note("=" * 60)
    logger.note("[TEST] Custom temperature")
    logger.note("=" * 60)

    configs = get_deepseek_client_configs(stream=False, verbose=True)
    client = LLMClient(**configs)

    response = client.chat(
        messages=SIMPLE_MESSAGES,
        stream=False,
        temperature=0.7,
    )

    assert response, "Response should not be empty"
    assert isinstance(response, str), "Response should be a string"
    logger.success(f"\n[PASS] Temperature chat returned: {len(response)} chars")
    return response


THINKING_MESSAGES = [
    {"role": "user", "content": "strawberry 这个单词中有几个字母 r？请仔细想想。"},
]


def test_thinking_stream():
    """Test streaming chat with DeepSeek thinking mode (deepseek-reasoner)."""
    logger.note("=" * 60)
    logger.note("[TEST] Thinking mode - streaming")
    logger.note("=" * 60)

    configs = get_deepseek_client_configs(stream=True, verbose=True)
    configs["verbose_think"] = True
    client = LLMClient(**configs)

    response = client.chat(
        messages=THINKING_MESSAGES,
        model="deepseek-reasoner",
        stream=True,
    )

    assert response, "Response should not be empty"
    assert isinstance(response, str), "Response should be a string"
    assert "<think>" in response, "Response should contain <think> tag"
    assert "</think>" in response, "Response should contain </think> tag"
    logger.success(f"\n[PASS] Thinking stream returned: {len(response)} chars")
    return response


def test_thinking_non_stream():
    """Test non-streaming chat with DeepSeek thinking mode (deepseek-reasoner)."""
    logger.note("=" * 60)
    logger.note("[TEST] Thinking mode - non-streaming")
    logger.note("=" * 60)

    configs = get_deepseek_client_configs(stream=False, verbose=True)
    configs["verbose_think"] = True
    client = LLMClient(**configs)

    response = client.chat(
        messages=THINKING_MESSAGES,
        model="deepseek-reasoner",
        stream=False,
    )

    assert response, "Response should not be empty"
    assert isinstance(response, str), "Response should be a string"
    assert "<think>" in response, "Response should contain <think> tag"
    assert "</think>" in response, "Response should contain </think> tag"
    logger.success(f"\n[PASS] Thinking non-stream returned: {len(response)} chars")
    return response


def test_thinking_via_param():
    """Test thinking mode via 'thinking' parameter (model=deepseek-chat + enable_thinking=True)."""
    logger.note("=" * 60)
    logger.note("[TEST] Thinking mode via enable_thinking param")
    logger.note("=" * 60)

    configs = get_deepseek_client_configs(stream=True, verbose=True)
    configs["verbose_think"] = True
    client = LLMClient(**configs)

    response = client.chat(
        messages=THINKING_MESSAGES,
        enable_thinking=True,
        stream=True,
    )

    assert response, "Response should not be empty"
    assert isinstance(response, str), "Response should be a string"
    assert "<think>" in response, "Response should contain <think> tag"
    assert "</think>" in response, "Response should contain </think> tag"
    logger.success(f"\n[PASS] Thinking via param returned: {len(response)} chars")
    return response


if __name__ == "__main__":
    tests = [
        ("non_stream_chat", test_non_stream_chat),
        ("stream_chat", test_stream_chat),
        ("multi_turn_chat", test_multi_turn_chat),
        ("client_by_config", test_client_by_config),
        ("temperature_param", test_temperature_param),
        ("thinking_stream", test_thinking_stream),
        ("thinking_non_stream", test_thinking_non_stream),
        ("thinking_via_param", test_thinking_via_param),
    ]

    results = {}
    for name, test_func in tests:
        try:
            test_func()
            results[name] = "PASS"
        except Exception as e:
            logger.warn(f"\n[FAIL] {name}: {e}")
            results[name] = f"FAIL: {e}"

    logger.note("\n" + "=" * 60)
    logger.note("[SUMMARY]")
    logger.note("=" * 60)
    for name, result in results.items():
        status = logger.success if result == "PASS" else logger.warn
        status(f"  {name}: {result}")
