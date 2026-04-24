from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable

from tclogger import logger

from llms.models.types import ChatResponse


def build_error_stream_chunk(message: str) -> dict:
    return {
        "id": "stream-error",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "error",
            }
        ],
        "error": {"message": message},
    }


def accumulate_stream_chat_response(chunks: Iterable[dict]) -> ChatResponse:
    accumulated_content = ""
    accumulated_reasoning = ""
    finish_reason = None
    usage = {}

    for chunk in chunks:
        choices = chunk.get("choices", [])
        if not choices:
            if chunk.get("usage"):
                usage = chunk["usage"]
            continue

        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason") or finish_reason

        if delta.get("content"):
            accumulated_content += delta["content"]
        if delta.get("reasoning_content"):
            accumulated_reasoning += delta["reasoning_content"]
        if chunk.get("usage"):
            usage = chunk["usage"]

    return ChatResponse(
        content=accumulated_content or None,
        reasoning_content=accumulated_reasoning or None,
        finish_reason=finish_reason,
        usage=usage,
    )


def parse_chat_response_payload(
    data: dict,
    *,
    extract_message_parts: Callable[[dict], tuple[str | None, str | None]],
) -> ChatResponse:
    if "error" in data:
        error_msg = data["error"].get("message", str(data["error"]))
        logger.warn(f"× LLM API error: {error_msg}")
        return ChatResponse(content=f"[Error: {error_msg}]", finish_reason="error")

    choices = data.get("choices", [])
    if not choices:
        return ChatResponse(content="", finish_reason="error")

    choice = choices[0]
    message = choice.get("message", {})
    usage = data.get("usage", {})
    reasoning_content, content = extract_message_parts(message)
    if message.get("tool_calls"):
        logger.warn(
            "> Provider tool_calls ignored: bili-search runtime uses XML-only tool markup"
        )

    return ChatResponse(
        content=content or None,
        reasoning_content=reasoning_content or None,
        finish_reason=choice.get("finish_reason", ""),
        usage=usage,
    )
