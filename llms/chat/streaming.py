import json
import re
import threading
import time
import uuid

from typing import Generator, Optional

from llms.chat.content import _sanitize_content
from llms.prompts.copilot import build_system_prompt
from llms.runtime.usage import compute_perf_stats, normalize_usage


_THINKING_PROMPT = (
    "[思考模式] 你现在处于深度思考模式。请认真分析用户的问题，"
    "进行更深入、更全面的思考和搜索，给出更有深度和见解的回答。"
    "你可以进行多轮搜索来获取更全面的信息，"
    "并综合分析后给出详细、有条理的回答。\n\n"
)


class ChatStreamingMixin:
    def handle_stream(
        self,
        messages: list[dict],
        temperature: float = None,
        thinking: bool = False,
        max_iterations: int = None,
        cancelled: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """Handle a streaming chat completion request.

        Streams reasoning/tool/content events directly from the orchestrator
        so the frontend can render intermediate thinking and tool execution
        in real time.
        """
        start_time = time.perf_counter()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        yield self._format_stream_chunk(
            request_id=request_id,
            delta={"role": "assistant", "content": ""},
            thinking=thinking,
        )

        result, streamed_content = yield from self._relay_orchestration_stream(
            request_id=request_id,
            stream=self.orchestrator.run_stream(
                messages=messages,
                thinking=thinking,
                max_iterations=max_iterations,
                cancelled=cancelled,
            ),
        )

        final_content = self._ensure_response_context(
            messages,
            _sanitize_content(result.content or ""),
        )
        streamed_visible_content = _sanitize_content(streamed_content)
        if final_content:
            should_replay_content = (
                not streamed_visible_content
                or final_content.strip() != streamed_visible_content.strip()
            )
            if streamed_visible_content and should_replay_content:
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta={"retract_content": True},
                )
            if should_replay_content:
                for chunk in self._chunk_text_for_stream(final_content):
                    yield self._format_stream_chunk(
                        request_id=request_id,
                        delta={"content": chunk},
                    )

        normalized_usage = self._normalize_usage(result.usage)
        elapsed_seconds = time.perf_counter() - start_time
        perf_stats = self._compute_perf_stats(normalized_usage, elapsed_seconds)
        yield self._format_stream_chunk(
            request_id=request_id,
            delta={},
            finish_reason="stop",
            usage=normalized_usage,
            perf_stats=perf_stats,
            usage_trace=result.usage_trace,
            thinking=thinking,
        )
        yield "[DONE]"

    def _relay_orchestration_stream(
        self,
        *,
        request_id: str,
        stream,
    ) -> Generator[str, None, tuple[object, str]]:
        streamed_content = ""
        try:
            while True:
                event = next(stream)
                delta = event.get("delta") or {}
                if delta.get("retract_content"):
                    streamed_content = ""
                if delta.get("content"):
                    streamed_content += str(delta.get("content") or "")
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta=delta,
                    tool_events=event.get("tool_events"),
                )
        except StopIteration as stop:
            return stop.value, streamed_content

    @staticmethod
    def _chunk_text_for_stream(text: str, chunk_size: int = 96) -> list[str]:
        content = str(text or "")
        if not content:
            return []
        chunks = []
        buffer = ""
        for paragraph in re.split(r"(\n\n)", content):
            if not paragraph:
                continue
            if len(buffer) + len(paragraph) <= chunk_size:
                buffer += paragraph
                continue
            if buffer:
                chunks.append(buffer)
                buffer = ""
            if len(paragraph) <= chunk_size:
                buffer = paragraph
                continue
            for start in range(0, len(paragraph), chunk_size):
                chunks.append(paragraph[start : start + chunk_size])
        if buffer:
            chunks.append(buffer)
        return chunks

    def _build_messages(
        self, user_messages: list[dict], thinking: bool = False
    ) -> list[dict]:
        system_content = build_system_prompt(
            capabilities=self.search_capabilities,
            messages=user_messages,
        )
        if thinking:
            system_content = _THINKING_PROMPT + system_content
        return [{"role": "system", "content": system_content}] + list(user_messages)

    @staticmethod
    def _merge_final_usage(total_usage: dict, last_usage: dict) -> dict:
        """Merge accumulated and last-call usage for final reporting.

        prompt_tokens (and related cache fields) come from the LAST LLM call,
        reflecting the actual context size for the final answer.
        completion_tokens (and related output fields) come from the ACCUMULATED
        total, reflecting total output across all iterations.

        This gives users an intuitive view: "input" = what the LLM saw,
        "output" = total generated content.
        """
        result = dict(total_usage)

        # Replace prompt-related fields with last call's values
        prompt_keys = [
            "prompt_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
        ]
        for key in prompt_keys:
            if key in last_usage:
                result[key] = last_usage[key]
            elif key in result:
                del result[key]

        # Also replace nested prompt_tokens_details if present
        if "prompt_tokens_details" in last_usage:
            result["prompt_tokens_details"] = last_usage["prompt_tokens_details"]

        # Recompute total_tokens
        prompt = result.get("prompt_tokens", 0)
        completion = result.get("completion_tokens", 0)
        if prompt or completion:
            result["total_tokens"] = prompt + completion

        return result

    @staticmethod
    def _normalize_usage(usage: dict) -> dict:
        return normalize_usage(usage)

    @staticmethod
    def _compute_perf_stats(usage: dict, elapsed_seconds: float) -> dict:
        return compute_perf_stats(usage, elapsed_seconds)

    def _format_completion(
        self,
        request_id: str,
        content: str,
        usage: dict = None,
        perf_stats: dict = None,
        tool_events: list = None,
        usage_trace: dict = None,
        thinking: bool = False,
    ) -> dict:
        """Format response as OpenAI-compatible chat completion."""
        result = {
            "id": request_id,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage or {},
        }
        if perf_stats:
            result["perf_stats"] = perf_stats
        if tool_events:
            result["tool_events"] = tool_events
        if usage_trace:
            result["usage_trace"] = usage_trace
        if thinking:
            result["thinking"] = True
        return result

    def _format_stream_chunk(
        self,
        request_id: str,
        delta: dict,
        finish_reason: str = None,
        usage: dict = None,
        perf_stats: dict = None,
        tool_events: list = None,
        usage_trace: dict = None,
        thinking: bool = None,
    ) -> str:
        """Format a single SSE stream chunk as JSON string."""
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage:
            chunk["usage"] = usage
        if perf_stats:
            chunk["perf_stats"] = perf_stats
        if tool_events:
            chunk["tool_events"] = tool_events
        if usage_trace:
            chunk["usage_trace"] = usage_trace
        if thinking is not None:
            chunk["thinking"] = thinking
        return json.dumps(chunk, ensure_ascii=False)
