"""LLM API client interface built on top of webu.llms.client.

Keeps the structured ChatResponse interface used by bili-search while reusing
webu's request construction, timeout, and thinking-mode logic.

Design note:
    bili-search active orchestration is XML-only. This transport must not send
    provider `tools` payloads or parse provider `tool_calls` back into runtime
    planning state.
"""

from __future__ import annotations

import json
import os
import time

import requests

from tclogger import logger
from typing import Generator
from llms.models.responses import accumulate_stream_chat_response
from llms.models.responses import build_error_stream_chunk
from llms.models.responses import parse_chat_response_payload
from webu.llms.client import LLMClient as WebuLLMClient
from llms.models.types import ChatResponse, ToolCall


class LLMClient(WebuLLMClient):
    """Structured chat client used by the XML-only bili-search orchestrator."""

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

    def chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
        enable_thinking: bool = None,
    ) -> ChatResponse:
        if self.verbose:
            provider = self._resolve_provider(model=model)
            logger.note(
                f"> LLM chat: provider={provider}, model={model or self.model}, msgs={len(messages)}"
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
            logger.success(
                f"  LLM → content: {len(result.content or '')} chars ({elapsed}ms)"
            )
            if result.usage:
                logger.mesg(f"  usage: {result.usage}")

        return result

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
        enable_thinking: bool = None,
    ) -> Generator[dict, None, None]:
        if self.verbose:
            provider = self._resolve_provider(model=model)
            logger.note(
                f"> LLM stream: provider={provider}, model={model or self.model}, msgs={len(messages)}"
            )

        try:
            response = self.create_response(
                messages=messages,
                model=model,
                enable_thinking=enable_thinking,
                temperature=temperature,
                stream=True,
                max_tokens=max_tokens,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warn(f"× LLM stream error: {exc}")
            yield build_error_stream_chunk(str(exc))
            return

        # Use a tiny chunk size to reduce first-token latency for SSE consumers,
        # especially for delegated small-model tasks where users expect the
        # tool panel to start updating immediately.
        stream_state = {"reasoning": "", "content": ""}
        for line in response.iter_lines(chunk_size=1):
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
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta") or {}
                    if isinstance(delta, dict):
                        choices[0]["delta"] = self._normalize_stream_delta(
                            delta,
                            stream_state,
                        )
                yield chunk
            except json.JSONDecodeError:
                continue

    def stream_to_response(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
        enable_thinking: bool = None,
    ) -> ChatResponse:
        return accumulate_stream_chat_response(
            self.chat_stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                enable_thinking=enable_thinking,
            )
        )

    def _parse_response(self, data: dict) -> ChatResponse:
        return parse_chat_response_payload(
            data,
            extract_message_parts=self._extract_message_parts,
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
