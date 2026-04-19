from __future__ import annotations

import json

from unittest.mock import MagicMock, patch

from webu.llms.client import LLMClient


class FakeResponse:
    def __init__(
        self,
        *,
        payload: dict | None = None,
        lines: list[bytes] | None = None,
        status_code: int = 200,
        text: str = "",
    ):
        self._payload = payload
        self._lines = lines or []
        self.status_code = status_code
        self.text = text or (
            json.dumps(payload, ensure_ascii=False) if payload is not None else ""
        )

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("No payload available")
        return self._payload

    def iter_lines(self):
        return iter(self._lines)


def make_client() -> LLMClient:
    return LLMClient(
        endpoint="https://api.minimaxi.com/v1/chat/completions",
        api_key="test-key",
        model="MiniMax-M2.7",
        api_format="openai",
        verbose_user=False,
        verbose_assistant=False,
        verbose_content=False,
        verbose_think=False,
        verbose_usage=False,
        verbose_finish=False,
    )


def test_create_response_adds_reasoning_split_for_minimax():
    client = make_client()
    mock_response = MagicMock(status_code=200)

    with patch(
        "webu.llms.client.requests.post", return_value=mock_response
    ) as mock_post:
        client.create_response(
            messages=[{"role": "user", "content": "你好"}],
            stream=False,
            enable_thinking=True,
        )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["reasoning_split"] is True
    assert "enable_thinking" not in payload
    assert "thinking" not in payload


def test_parse_json_response_extracts_reasoning_details():
    client = make_client()
    response = FakeResponse(
        payload={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "4",
                        "reasoning_details": [{"type": "text", "text": "先确认 2+2。"}],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
    )

    content, usage = client.parse_json_response(response)

    assert content == "<think>先确认 2+2。</think>4"
    assert usage["total_tokens"] == 14


def test_parse_stream_response_normalizes_cumulative_minimax_chunks():
    client = make_client()
    response = FakeResponse(
        lines=[
            'data: {"choices":[{"delta":{"reasoning_details":[{"text":"先"}]},"finish_reason":null}]}'.encode(
                "utf-8"
            ),
            'data: {"choices":[{"delta":{"reasoning_details":[{"text":"先想"}]},"finish_reason":null}]}'.encode(
                "utf-8"
            ),
            'data: {"choices":[{"delta":{"content":"答"},"finish_reason":null}]}'.encode(
                "utf-8"
            ),
            'data: {"choices":[{"delta":{"content":"答案"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":6,"total_tokens":18}}'.encode(
                "utf-8"
            ),
            b"data: [DONE]",
        ]
    )

    content, usage = client.parse_stream_response(response)

    assert content == "<think>先想</think>答案"
    assert usage["total_tokens"] == 18
