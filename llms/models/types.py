from __future__ import annotations

import json

from tclogger import logger


class ToolCall:
    """Parsed tool call from LLM response."""

    def __init__(self, id: str, name: str, arguments: str | dict):
        self.id = id
        self.name = name
        self.arguments = arguments

    def parse_arguments(self) -> dict:
        if isinstance(self.arguments, dict):
            return self.arguments
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError:
            logger.warn(f"× Failed to parse tool arguments: {self.arguments}")
            return {}

    def to_dict(self) -> dict:
        arguments = self.arguments
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": arguments,
            },
        }


class ChatResponse:
    """Parsed chat completion response."""

    def __init__(
        self,
        content: str = None,
        reasoning_content: str = None,
        tool_calls: list[ToolCall] | None = None,
        finish_reason: str = None,
        usage: dict | None = None,
    ):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {}

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_message_dict(self) -> dict:
        msg = {"role": "assistant"}
        if self.tool_calls:
            msg["content"] = self.content
            msg["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        elif self.content is not None:
            msg["content"] = self.content
        return msg


__all__ = ["ChatResponse", "ToolCall"]
