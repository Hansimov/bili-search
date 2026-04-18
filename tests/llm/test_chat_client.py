"""Tests for llms.models.client — LLM API client with function calling.

Includes:
- Unit tests with mocked HTTP responses (no external deps)
- Integration tests that call real LLM API (marked slow)

Run:
    python -m tests.llm.test_chat_client
"""

import json

from unittest.mock import patch, MagicMock
from tclogger import logger

from configs.envs import LLM_CONFIG
from llms.models import LLMClient, ChatResponse, ToolCall, create_llm_client


# ============================================================
# Mock response builders
# ============================================================


def make_content_response(content: str, usage: dict = None) -> dict:
    """Build a mock API response with content."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage
        or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def make_tool_call_response(
    tool_name: str,
    arguments: dict,
    call_id: str = "call_test_123",
) -> dict:
    """Build a mock API response with a tool call."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


def make_error_response(message: str) -> dict:
    """Build a mock API error response."""
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
        }
    }


# ============================================================
# Unit tests (mocked HTTP)
# ============================================================


def test_parse_content_response():
    """Test parsing a content response."""
    logger.note("=" * 60)
    logger.note("[TEST] parse content response")

    client = LLMClient(
        endpoint="http://test/chat/completions",
        api_key="test-key",
        model="test-model",
    )

    data = make_content_response("Hello, world!")
    result = client._parse_response(data)

    assert isinstance(result, ChatResponse)
    assert result.content == "Hello, world!"
    assert not result.has_tool_calls
    assert result.finish_reason == "stop"
    assert result.usage["total_tokens"] == 15

    logger.success("[PASS] parse content response")


def test_parse_tool_call_response():
    """Test parsing a tool call response."""
    logger.note("=" * 60)
    logger.note("[TEST] parse tool call response")

    client = LLMClient(
        endpoint="http://test/chat/completions",
        api_key="test-key",
        model="test-model",
    )

    data = make_tool_call_response(
        "search_videos",
        {"query": "黑神话"},
        call_id="call_abc",
    )
    result = client._parse_response(data)

    assert isinstance(result, ChatResponse)
    assert result.has_tool_calls
    assert len(result.tool_calls) == 1

    tc = result.tool_calls[0]
    assert tc.name == "search_videos"
    assert tc.id == "call_abc"
    assert tc.parse_arguments() == {"query": "黑神话"}

    logger.success("[PASS] parse tool call response")


def test_parse_error_response():
    """Test parsing an API error response."""
    logger.note("=" * 60)
    logger.note("[TEST] parse error response")

    client = LLMClient(
        endpoint="http://test/chat/completions",
        api_key="test-key",
        model="test-model",
    )

    data = make_error_response("Rate limit exceeded")
    result = client._parse_response(data)

    assert "Error" in (result.content or "")
    assert result.finish_reason == "error"

    logger.success("[PASS] parse error response")


def test_tool_call_to_dict():
    """Test ToolCall serialization."""
    logger.note("=" * 60)
    logger.note("[TEST] ToolCall.to_dict")

    tc = ToolCall(id="call_1", name="search_videos", arguments='{"query": "test"}')
    d = tc.to_dict()

    assert d["id"] == "call_1"
    assert d["type"] == "function"
    assert d["function"]["name"] == "search_videos"
    assert d["function"]["arguments"] == '{"query": "test"}'

    logger.success("[PASS] ToolCall.to_dict")


def test_chat_response_to_message():
    """Test ChatResponse serialization to message dict."""
    logger.note("=" * 60)
    logger.note("[TEST] ChatResponse.to_message_dict")

    # Content response
    resp1 = ChatResponse(content="Hello")
    msg1 = resp1.to_message_dict()
    assert msg1["role"] == "assistant"
    assert msg1["content"] == "Hello"
    assert "tool_calls" not in msg1

    # Tool call response
    tc = ToolCall(id="call_1", name="search", arguments="{}")
    resp2 = ChatResponse(content=None, tool_calls=[tc])
    msg2 = resp2.to_message_dict()
    assert msg2["role"] == "assistant"
    assert len(msg2["tool_calls"]) == 1
    assert msg2["tool_calls"][0]["function"]["name"] == "search"

    logger.success("[PASS] ChatResponse.to_message_dict")


@patch("webu.llms.client.requests.post")
def test_chat_with_mock(mock_post):
    """Test full chat() call with mocked HTTP."""
    logger.note("=" * 60)
    logger.note("[TEST] chat() with mock")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = make_content_response("测试回答")
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    client = LLMClient(
        endpoint="http://test/chat/completions",
        api_key="test-key",
        model="test-model",
    )

    result = client.chat(
        messages=[{"role": "user", "content": "你好"}],
        temperature=0.7,
    )

    assert result.content == "测试回答"
    assert not result.has_tool_calls

    # Verify request was made correctly
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["model"] == "test-model"
    assert payload["messages"][0]["content"] == "你好"
    assert payload["temperature"] == 0.7
    assert payload["stream"] is False

    logger.success("[PASS] chat() with mock")


@patch("webu.llms.client.requests.post")
def test_chat_with_tools_mock(mock_post):
    """Test chat() with tool definitions."""
    logger.note("=" * 60)
    logger.note("[TEST] chat() with tools mock")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = make_tool_call_response(
        "search_google", {"query": "Gemini 2.5 official updates"}
    )
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    client = LLMClient(
        endpoint="http://test/chat/completions",
        api_key="test-key",
        model="test-model",
    )

    from llms.tools.defs import TOOL_DEFINITIONS

    result = client.chat(
        messages=[{"role": "user", "content": "影视飓风最近有什么视频"}],
        tools=TOOL_DEFINITIONS,
    )

    assert result.has_tool_calls
    assert result.tool_calls[0].name == "search_google"
    assert result.tool_calls[0].parse_arguments() == {
        "query": "Gemini 2.5 official updates"
    }

    # Verify tools were sent in payload
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "tools" in payload
    assert payload["tool_choice"] == "auto"

    logger.success("[PASS] chat() with tools mock")


@patch("webu.llms.client.requests.post")
def test_chat_timeout(mock_post):
    """Test chat() handles timeout gracefully."""
    logger.note("=" * 60)
    logger.note("[TEST] chat() timeout handling")

    import requests

    mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

    client = LLMClient(
        endpoint="http://test/chat/completions",
        api_key="test-key",
        model="test-model",
        timeout=5,
    )

    result = client.chat(messages=[{"role": "user", "content": "test"}])
    assert "Error" in (result.content or "")
    assert result.finish_reason == "error"

    logger.success("[PASS] chat() timeout handling")


def test_chat_stream_uses_low_buffer_iter_lines_for_sse_latency():
    """Test chat_stream uses a tiny iter_lines chunk size for lower SSE latency."""
    logger.note("=" * 60)
    logger.note("[TEST] chat_stream low-buffer iter_lines")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_lines.return_value = iter(
        [
            b'data: {"choices":[{"delta":{"content":"A"},"finish_reason":null}]}',
            b"data: [DONE]",
        ]
    )

    client = LLMClient(
        endpoint="http://test/chat/completions",
        api_key="test-key",
        model="test-model",
    )
    client.create_response = MagicMock(return_value=mock_response)

    chunks = list(client.chat_stream(messages=[{"role": "user", "content": "hi"}]))

    assert chunks[0]["choices"][0]["delta"]["content"] == "A"
    mock_response.iter_lines.assert_called_once_with(chunk_size=1)

    logger.success("[PASS] chat_stream low-buffer iter_lines")


def test_create_llm_client():
    """Test create_llm_client factory function."""
    logger.note("=" * 60)
    logger.note("[TEST] create_llm_client")

    client = create_llm_client(LLM_CONFIG, verbose=False)
    assert isinstance(client, LLMClient)
    assert client.endpoint, "endpoint should be set"
    assert client.model, "model should be set"

    logger.success(f"[PASS] create_llm_client (config={LLM_CONFIG})")


def test_create_llm_client_invalid():
    """Test create_llm_client with invalid config."""
    logger.note("=" * 60)
    logger.note("[TEST] create_llm_client invalid")

    try:
        create_llm_client("nonexistent_model")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent_model" in str(e)

    logger.success("[PASS] create_llm_client invalid")


def test_tool_call_parse_bad_json():
    """Test ToolCall handles malformed JSON arguments."""
    logger.note("=" * 60)
    logger.note("[TEST] ToolCall parse bad JSON")

    tc = ToolCall(id="call_1", name="test", arguments="not json{{{")
    result = tc.parse_arguments()
    assert result == {}

    logger.success("[PASS] ToolCall parse bad JSON")


# ============================================================
# Integration test: actual DeepSeek API call
# ============================================================


def test_llm_chat_integration():
    """Integration test: actual chat with LLM API.

    Requires LLM API key in configs/secrets.json.
    """
    logger.note("=" * 60)
    logger.note(f"[TEST] LLM API integration (live, config={LLM_CONFIG})")

    client = create_llm_client(LLM_CONFIG, verbose=True)

    result = client.chat(
        messages=[
            {"role": "system", "content": "你是一个简洁的助手，用一句话回答。"},
            {"role": "user", "content": "1+1等于几？"},
        ],
        temperature=0,
    )

    assert result.content is not None
    assert len(result.content) > 0
    assert result.finish_reason == "stop"
    logger.success(f"  Response: {result.content}")
    logger.success(f"[PASS] LLM API integration (config={LLM_CONFIG})")


def test_llm_function_calling_integration():
    """Integration test: function calling with LLM API."""
    logger.note("=" * 60)
    logger.note(f"[TEST] LLM function calling integration (live, config={LLM_CONFIG})")

    client = create_llm_client(LLM_CONFIG, verbose=True)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    result = client.chat(
        messages=[
            {"role": "user", "content": "北京今天天气怎么样？"},
        ],
        tools=tools,
        temperature=0,
    )

    assert result.has_tool_calls
    assert result.tool_calls[0].name == "get_weather"
    args = result.tool_calls[0].parse_arguments()
    assert "location" in args
    logger.success(f"  Tool call: {result.tool_calls[0].name}({args})")
    logger.success(f"[PASS] LLM function calling integration (config={LLM_CONFIG})")


if __name__ == "__main__":
    # Unit tests (always run)
    unit_tests = [
        ("parse_content_response", test_parse_content_response),
        ("parse_tool_call_response", test_parse_tool_call_response),
        ("parse_error_response", test_parse_error_response),
        ("tool_call_to_dict", test_tool_call_to_dict),
        ("chat_response_to_message", test_chat_response_to_message),
        ("chat_with_mock", test_chat_with_mock),
        ("chat_with_tools_mock", test_chat_with_tools_mock),
        ("chat_timeout", test_chat_timeout),
        ("create_llm_client", test_create_llm_client),
        ("create_llm_client_invalid", test_create_llm_client_invalid),
        ("tool_call_parse_bad_json", test_tool_call_parse_bad_json),
    ]

    # Integration tests (require live API)
    integration_tests = [
        ("llm_chat_integration", test_llm_chat_integration),
        ("llm_function_calling", test_llm_function_calling_integration),
    ]

    results = {}

    logger.note("\n[UNIT TESTS]")
    for name, test_func in unit_tests:
        try:
            test_func()
            results[name] = "PASS"
        except Exception as e:
            logger.warn(f"\n[FAIL] {name}: {e}")
            results[name] = f"FAIL: {e}"

    logger.note("\n[INTEGRATION TESTS]")
    for name, test_func in integration_tests:
        try:
            test_func()
            results[name] = "PASS"
        except Exception as e:
            logger.warn(f"\n[FAIL] {name}: {e}")
            results[name] = f"FAIL: {e}"

    logger.note("\n" + "=" * 60)
    logger.note("[SUMMARY]")
    logger.note("=" * 60)
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    for name, result in results.items():
        status = logger.success if result == "PASS" else logger.warn
        status(f"  {name}: {result}")
    logger.note(f"\n  {passed}/{total} tests passed")

    # python -m tests.llm.test_chat_client
