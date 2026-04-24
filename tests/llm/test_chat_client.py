"""Tests for llms.models.client under the XML-only orchestration contract."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

from configs.envs import LLM_CONFIG
from llms.models import ChatResponse
from llms.models import LLMClient
from llms.models import ToolCall
from llms.models import create_llm_client


def make_content_response(content: str, usage: dict | None = None) -> dict:
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


def make_ignored_tool_call_response() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_test_123",
                            "type": "function",
                            "function": {
                                "name": "search_videos",
                                "arguments": '{"query": "黑神话"}',
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
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
        }
    }


def make_reasoning_details_response(content: str, reasoning: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_details": [{"type": "text", "text": reasoning}],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
    }


def make_client() -> LLMClient:
    return LLMClient(
        endpoint="http://test/chat/completions",
        api_key="test-key",
        model="test-model",
    )


def test_parse_content_response():
    client = make_client()

    result = client._parse_response(make_content_response("Hello, world!"))

    assert isinstance(result, ChatResponse)
    assert result.content == "Hello, world!"
    assert result.finish_reason == "stop"
    assert result.usage["total_tokens"] == 15


def test_parse_response_ignores_provider_tool_calls():
    client = make_client()

    result = client._parse_response(make_ignored_tool_call_response())

    assert result.content is None
    assert result.finish_reason == "tool_calls"
    assert result.usage["total_tokens"] == 30


def test_parse_error_response():
    client = make_client()

    result = client._parse_response(make_error_response("Rate limit exceeded"))

    assert "Error" in (result.content or "")
    assert result.finish_reason == "error"


def test_tool_call_to_dict():
    tc = ToolCall(id="call_1", name="search_videos", arguments='{"query": "test"}')

    assert tc.to_dict() == {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "search_videos",
            "arguments": '{"query": "test"}',
        },
    }


def test_chat_response_to_message_keeps_content_only():
    resp = ChatResponse(content="Hello")

    assert resp.to_message_dict() == {"role": "assistant", "content": "Hello"}


@patch("webu.llms.client.requests.post")
def test_chat_with_mock(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = make_content_response("测试回答")
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    client = make_client()

    result = client.chat(
        messages=[{"role": "user", "content": "你好"}],
        temperature=0.7,
    )

    assert result.content == "测试回答"

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["model"] == "test-model"
    assert payload["messages"][0]["content"] == "你好"
    assert payload["temperature"] == 0.7
    assert payload["stream"] is False
    assert "tools" not in payload
    assert "tool_choice" not in payload


@patch("webu.llms.client.requests.post")
def test_chat_timeout(mock_post):
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


def test_chat_stream_uses_low_buffer_iter_lines_for_sse_latency():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_lines.return_value = iter(
        [
            b'data: {"choices":[{"delta":{"content":"A"},"finish_reason":null}]}',
            b"data: [DONE]",
        ]
    )

    client = make_client()
    client.create_response = MagicMock(return_value=mock_response)

    chunks = list(client.chat_stream(messages=[{"role": "user", "content": "hi"}]))

    assert chunks[0]["choices"][0]["delta"]["content"] == "A"
    mock_response.iter_lines.assert_called_once_with(chunk_size=1)


def test_parse_response_extracts_reasoning_details_from_minimax_payload():
    client = LLMClient(
        endpoint="https://api.minimaxi.com/v1/chat/completions",
        api_key="test-key",
        model="MiniMax-M2.7",
    )

    result = client._parse_response(
        make_reasoning_details_response("4", "先确认 2+2 的结果。")
    )

    assert result.content == "4"
    assert result.reasoning_content == "先确认 2+2 的结果。"


def test_parse_response_splits_inline_thinking_tags_from_content():
    client = make_client()

    result = client._parse_response(
        make_content_response("<think>先分析一下。</think>\n最终答案")
    )

    assert result.reasoning_content == "先分析一下。"
    assert result.content == "最终答案"


def test_chat_stream_normalizes_cumulative_minimax_deltas():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_lines.return_value = iter(
        [
            'data: {"choices":[{"delta":{"reasoning_details":[{"text":"先"}]},"finish_reason":null}]}'.encode(
                "utf-8"
            ),
            'data: {"choices":[{"delta":{"reasoning_details":[{"text":"先想"}]},"finish_reason":null}]}'.encode(
                "utf-8"
            ),
            'data: {"choices":[{"delta":{"content":"答"},"finish_reason":null}]}'.encode(
                "utf-8"
            ),
            'data: {"choices":[{"delta":{"content":"答案"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}'.encode(
                "utf-8"
            ),
            b"data: [DONE]",
        ]
    )

    client = LLMClient(
        endpoint="https://api.minimaxi.com/v1/chat/completions",
        api_key="test-key",
        model="MiniMax-M2.7",
    )
    client.create_response = MagicMock(return_value=mock_response)

    chunks = list(client.chat_stream(messages=[{"role": "user", "content": "hi"}]))

    assert chunks[0]["choices"][0]["delta"]["reasoning_content"] == "先"
    assert chunks[1]["choices"][0]["delta"]["reasoning_content"] == "想"
    assert chunks[2]["choices"][0]["delta"]["content"] == "答"
    assert chunks[3]["choices"][0]["delta"]["content"] == "案"


def test_create_llm_client():
    client = create_llm_client(LLM_CONFIG, verbose=False)
    assert isinstance(client, LLMClient)
    assert client.endpoint
    assert client.model


def test_create_llm_client_invalid():
    try:
        create_llm_client("nonexistent_model")
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "nonexistent_model" in str(exc)


def test_tool_call_parse_bad_json():
    tc = ToolCall(id="call_1", name="test", arguments="not json{{{")

    assert tc.parse_arguments() == {}
