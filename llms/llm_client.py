"""LLM API client with function calling support.

Communicates with OpenAI-compatible APIs (DeepSeek, Volcengine, etc.).
Supports native function calling for tool-use workflows.
"""

import json
import requests
import time

from tclogger import logger
from typing import Generator, Optional


class ToolCall:
    """Parsed tool call from LLM response."""

    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.name = name
        self.arguments = arguments

    def parse_arguments(self) -> dict:
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError:
            logger.warn(f"× Failed to parse tool arguments: {self.arguments}")
            return {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


class ChatResponse:
    """Parsed chat completion response."""

    def __init__(
        self,
        content: str = None,
        tool_calls: list[ToolCall] = None,
        finish_reason: str = None,
        usage: dict = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {}

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def to_message_dict(self) -> dict:
        """Convert to a message dict for appending to conversation history."""
        msg = {"role": "assistant"}
        if self.tool_calls:
            # Tool call message: content can be None per OpenAI spec
            msg["content"] = self.content
            msg["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        elif self.content is not None:
            msg["content"] = self.content
        return msg


class LLMClient:
    """LLM API client with function calling support.

    Compatible with OpenAI API format (DeepSeek, Volcengine, etc.).

    Usage:
        client = LLMClient(
            endpoint="https://api.deepseek.com/chat/completions",
            api_key="sk-...",
            model="deepseek-chat",
        )
        response = client.chat(messages=[...], tools=[...])
        if response.has_tool_calls:
            for tc in response.tool_calls:
                args = tc.parse_arguments()
                # execute tool...
        else:
            print(response.content)
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        timeout: float = 120,
        verbose: bool = False,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.verbose = verbose

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        stream: bool = False,
        temperature: float = None,
        max_tokens: int = None,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> ChatResponse:
        """Send non-streaming chat completion request.

        Args:
            messages: Conversation messages.
            tools: Tool definitions (OpenAI function calling format).
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            ChatResponse with content or tool_calls.
        """
        payload = self._build_payload(
            messages=messages,
            tools=tools,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        headers = self._build_headers()

        if self.verbose:
            logger.note(
                f"> LLM chat: model={self.model}, msgs={len(messages)}"
                + (f", tools={len(tools)}" if tools else "")
            )

        start = time.perf_counter()
        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            logger.warn(f"× LLM request timed out after {self.timeout}s")
            return ChatResponse(
                content="[Error: Request timed out]", finish_reason="error"
            )
        except requests.exceptions.RequestException as e:
            logger.warn(f"× LLM request error: {e}")
            return ChatResponse(content=f"[Error: {e}]", finish_reason="error")

        elapsed = round((time.perf_counter() - start) * 1000, 1)
        result = self._parse_response(data)

        if self.verbose:
            if result.has_tool_calls:
                names = [tc.name for tc in result.tool_calls]
                logger.success(f"  LLM → tool_calls: {names} ({elapsed}ms)")
            else:
                content_len = len(result.content or "")
                logger.success(f"  LLM → content: {content_len} chars ({elapsed}ms)")
            if result.usage:
                logger.mesg(f"  usage: {result.usage}")

        return result

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> Generator[dict, None, None]:
        """Send streaming chat completion request.

        Yields raw SSE data dicts as they arrive.
        """
        payload = self._build_payload(
            messages=messages,
            tools=tools,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        headers = self._build_headers()

        if self.verbose:
            logger.note(f"> LLM stream: model={self.model}, msgs={len(messages)}")

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warn(f"× LLM stream error: {e}")
            return

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                yield chunk
            except json.JSONDecodeError:
                continue

    def stream_to_response(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> ChatResponse:
        """Stream from LLM and accumulate into a ChatResponse.

        Used internally when we need to consume a potentially-streamed response
        and accumulate tool_calls or content.
        """
        accumulated_content = ""
        accumulated_tool_calls = {}
        finish_reason = None

        for chunk in self.chat_stream(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            finish_reason = choices[0].get("finish_reason") or finish_reason

            if "content" in delta and delta["content"]:
                accumulated_content += delta["content"]

            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    entry = accumulated_tool_calls[idx]
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    func = tc.get("function", {})
                    if func.get("name"):
                        entry["name"] = func["name"]
                    if func.get("arguments"):
                        entry["arguments"] += func["arguments"]

        tool_calls = []
        for idx in sorted(accumulated_tool_calls.keys()):
            tc = accumulated_tool_calls[idx]
            tool_calls.append(
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
            )

        return ChatResponse(
            content=accumulated_content or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    def _parse_response(self, data: dict) -> ChatResponse:
        """Parse a non-streaming API response."""
        # Handle API errors
        if "error" in data:
            error_msg = data["error"].get("message", str(data["error"]))
            logger.warn(f"× LLM API error: {error_msg}")
            return ChatResponse(content=f"[Error: {error_msg}]", finish_reason="error")

        choices = data.get("choices", [])
        if not choices:
            return ChatResponse(content="", finish_reason="error")

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "")
        usage = data.get("usage", {})

        content = message.get("content")
        tool_calls = []

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", ""),
                    )
                )

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )


def create_llm_client(
    model_config: str = "deepseek",
    verbose: bool = False,
) -> LLMClient:
    """Create an LLMClient from config name in secrets.json.

    Reuses the LLMS_ENVS config lookup from configs/envs.py.
    This is the main factory function for creating LLM clients.

    Args:
        model_config: Key in LLMS_ENVS (e.g. "deepseek", "volcengine").
        verbose: Enable verbose logging.

    Returns:
        Configured LLMClient instance.
    """
    from configs.envs import LLMS_ENVS

    if model_config not in LLMS_ENVS:
        available = list(LLMS_ENVS.keys())
        raise ValueError(
            f"Unknown model config '{model_config}'. Available: {available}"
        )

    envs = LLMS_ENVS[model_config]
    return LLMClient(
        endpoint=envs["endpoint"],
        api_key=envs.get("api_key", ""),
        model=envs["model"],
        verbose=verbose,
    )
