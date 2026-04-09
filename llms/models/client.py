"""LLM API client interface built on top of webu.llms.client.

Keeps the structured ToolCall / ChatResponse interface used by bili-search
while reusing webu's request construction, timeout, and thinking-mode logic.
"""

from __future__ import annotations

import json
import os
import time

import requests

from tclogger import logger
from typing import Generator
from webu.llms.client import LLMClient as WebuLLMClient


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


class LLMClient(WebuLLMClient):
    """Structured chat client with optional OpenAI-style function calling."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        api_format: str = "openai",
        timeout: float = 120,
        verbose: bool = False,
        stream: bool | None = None,
        max_tokens: int | None = None,
        init_messages: list | None = None,
        enable_thinking: bool | None = None,
    ):
        super().__init__(
            endpoint=endpoint,
            api_key=api_key,
            api_format=api_format,
            model=model,
            stream=stream,
            max_tokens=max_tokens,
            timeout=timeout,
            init_messages=init_messages or [],
            enable_thinking=enable_thinking,
            verbose_user=False,
            verbose_assistant=False,
            verbose_content=False,
            verbose_think=False,
            verbose_usage=False,
            verbose_finish=False,
            verbose=verbose,
        )

    @staticmethod
    def _extra_body(tools: list[dict] | None = None) -> dict | None:
        if not tools:
            return None
        return {
            "tools": tools,
            "tool_choice": "auto",
        }

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
        enable_thinking: bool = None,
    ) -> ChatResponse:
        if self.verbose:
            logger.note(
                f"> LLM chat: model={model or self.model}, msgs={len(messages)}"
                + (f", tools={len(tools)}" if tools else "")
            )

        start = time.perf_counter()
        try:
            response = self.create_response(
                messages=messages,
                model=model,
                enable_thinking=enable_thinking,
                temperature=temperature,
                stream=False,
                max_tokens=max_tokens,
                extra_body=self._extra_body(tools),
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            logger.warn(f"× LLM request timed out after {self.timeout}s")
            return ChatResponse(
                content="[Error: Request timed out]",
                finish_reason="error",
            )
        except requests.exceptions.RequestException as exc:
            logger.warn(f"× LLM request error: {exc}")
            return ChatResponse(content=f"[Error: {exc}]", finish_reason="error")

        elapsed = round((time.perf_counter() - start) * 1000, 1)
        result = self._parse_response(data)

        if self.verbose:
            if result.has_tool_calls:
                names = [tool_call.name for tool_call in result.tool_calls]
                logger.success(f"  LLM → tool_calls: {names} ({elapsed}ms)")
            else:
                logger.success(
                    f"  LLM → content: {len(result.content or '')} chars ({elapsed}ms)"
                )
            if result.usage:
                logger.mesg(f"  usage: {result.usage}")

        return result

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
        enable_thinking: bool = None,
    ) -> Generator[dict, None, None]:
        if self.verbose:
            logger.note(
                f"> LLM stream: model={model or self.model}, msgs={len(messages)}"
                + (f", tools={len(tools)}" if tools else "")
            )

        try:
            response = self.create_response(
                messages=messages,
                model=model,
                enable_thinking=enable_thinking,
                temperature=temperature,
                stream=True,
                max_tokens=max_tokens,
                extra_body=self._extra_body(tools),
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warn(f"× LLM stream error: {exc}")
            return

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                continue

    def stream_to_response(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
        enable_thinking: bool = None,
    ) -> ChatResponse:
        accumulated_content = ""
        accumulated_reasoning = ""
        accumulated_tool_calls = {}
        finish_reason = None
        usage = {}

        for chunk in self.chat_stream(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            enable_thinking=enable_thinking,
        ):
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
            if delta.get("tool_calls"):
                for tool_call in delta["tool_calls"]:
                    index = tool_call.get("index", 0)
                    current = accumulated_tool_calls.setdefault(
                        index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    if tool_call.get("id"):
                        current["id"] = tool_call["id"]
                    func = tool_call.get("function", {})
                    if func.get("name"):
                        current["name"] = func["name"]
                    if func.get("arguments"):
                        current["arguments"] += func["arguments"]
            if chunk.get("usage"):
                usage = chunk["usage"]

        tool_calls = [
            ToolCall(
                id=entry["id"],
                name=entry["name"],
                arguments=entry["arguments"],
            )
            for _, entry in sorted(accumulated_tool_calls.items())
        ]
        return ChatResponse(
            content=accumulated_content or None,
            reasoning_content=accumulated_reasoning or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _parse_response(self, data: dict) -> ChatResponse:
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
        tool_calls = []
        for tool_call in message.get("tool_calls") or []:
            tool_calls.append(
                ToolCall(
                    id=tool_call.get("id", ""),
                    name=tool_call.get("function", {}).get("name", ""),
                    arguments=tool_call.get("function", {}).get("arguments", ""),
                )
            )

        return ChatResponse(
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", ""),
            usage=usage,
        )


def create_llm_client(
    model_config: str = None,
    timeout: float | None = None,
    verbose: bool = False,
) -> LLMClient:
    if model_config is None:
        from configs.envs import LLM_CONFIG

        model_config = LLM_CONFIG

    from configs.envs import LLMS_ENVS

    if model_config not in LLMS_ENVS:
        available = list(LLMS_ENVS.keys())
        raise ValueError(
            f"Unknown model config '{model_config}'. Available: {available}"
        )

    envs = LLMS_ENVS[model_config]
    resolved_timeout = timeout
    if resolved_timeout is None:
        env_timeout = os.getenv("BILI_LLM_TIMEOUT", "").strip()
        if env_timeout:
            resolved_timeout = float(env_timeout)
    if resolved_timeout is None:
        resolved_timeout = envs.get("timeout", 120)

    return LLMClient(
        endpoint=envs["endpoint"],
        api_key=envs.get("api_key", ""),
        model=envs["model"],
        api_format=envs.get("api_format", "openai"),
        timeout=resolved_timeout,
        stream=envs.get("stream"),
        max_tokens=envs.get("max_tokens"),
        init_messages=envs.get("init_messages") or [],
        enable_thinking=envs.get("enable_thinking"),
        verbose=verbose,
    )


__all__ = [
    "ChatResponse",
    "LLMClient",
    "ToolCall",
    "create_llm_client",
]
