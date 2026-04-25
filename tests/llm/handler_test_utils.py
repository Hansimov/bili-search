"""Shared helpers for ChatHandler orchestration tests."""

from __future__ import annotations

import json

from llms.models import ChatResponse, ToolCall


def make_content_response(content: str, usage: dict | None = None) -> ChatResponse:
    return ChatResponse(
        content=content,
        finish_reason="stop",
        usage=usage
        or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def make_tool_cmd_response(
    analysis: str,
    commands: str,
    usage: dict | None = None,
) -> ChatResponse:
    content = f"{analysis}\n{commands}" if analysis else commands
    return ChatResponse(
        content=content,
        finish_reason="stop",
        usage=usage
        or {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    )


def make_function_call_response(
    *tool_calls: ToolCall, usage: dict | None = None
) -> ChatResponse:
    xml_commands: list[str] = []
    for tool_call in tool_calls:
        arguments = tool_call.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        attrs = " ".join(
            f"{key}='{json.dumps(value, ensure_ascii=False)}'"
            for key, value in (arguments or {}).items()
        )
        xml_commands.append(
            f"<{tool_call.name} {attrs}/>" if attrs else f"<{tool_call.name}/>"
        )
    return ChatResponse(
        content="\n".join(xml_commands),
        finish_reason="stop",
        usage=usage
        or {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    )


def make_stream_chunk(
    *,
    delta: dict | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> dict:
    chunk = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def assistant_content(result: dict) -> str:
    return str(result["choices"][0]["message"].get("content") or "")


MOCK_EXPLORE_RESULT = {
    "query": "黑神话",
    "status": "finished",
    "data": [
        {
            "step": 0,
            "name": "most_relevant_search",
            "output": {
                "hits": [
                    {
                        "bvid": "BV1abc",
                        "title": "黑神话悟空全流程",
                        "owner": {"mid": 100, "name": "游戏UP主"},
                        "pubdate": 1708700000,
                        "stat": {"view": 500000},
                    },
                    {
                        "bvid": "BV1def",
                        "title": "黑神话评测",
                        "owner": {"mid": 200, "name": "测评达人"},
                        "pubdate": 1708600000,
                        "stat": {"view": 200000},
                    },
                ],
                "total_hits": 42,
            },
        }
    ],
}
