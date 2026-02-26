"""Chat handler with tool-calling loop.

Orchestrates the conversation between user, LLM, and search tools.
Implements the iterative tool-calling pattern:
  1. Send user message + system prompt + tool defs to LLM
  2. If LLM returns tool_calls → execute tools → feed results back → repeat
  3. If LLM returns content → return as final response

Token optimization strategy:
  - Each iteration re-sends the full conversation, so fewer iterations = fewer tokens
  - Compact DSL syntax is inline in the system prompt (no read_spec round-trip)
  - Tool results use compact JSON (no indent)
  - If max iterations exhausted, inject a nudge message to force content generation
"""

import json
import re
import time
import uuid

from tclogger import logger, dt_to_str
from typing import Generator

from llms.llm_client import LLMClient, ChatResponse, ToolCall, create_llm_client
from llms.tools.defs import TOOL_DEFINITIONS
from llms.tools.executor import ToolExecutor
from llms.prompts.copilot import build_system_prompt

# Maximum tool-calling iterations to prevent infinite loops.
# 5 allows multi-hop searches: parallel check_author+search → refine → follow-up → content.
MAX_TOOL_ITERATIONS = 5

# Maximum iterations for thinking mode — allows deeper exploration
MAX_TOOL_ITERATIONS_THINKING = 10

# Regex to strip leaked DeepSeek DSML function-calling markup from content
_DSML_PATTERN = re.compile(r"<｜.*?｜>")
_DSML_BLOCK_PATTERN = re.compile(
    r"<｜DSML｜function_calls>.*?</｜DSML｜function_calls>", re.DOTALL
)

# Nudge message injected before forcing content generation
_FORCE_CONTENT_NUDGE = (
    "你已经进行了充分的搜索。请根据以上搜索结果直接回答用户的问题。"
    "不要再调用任何工具。如果搜索结果不完全匹配，就根据已有信息给出最佳回答。"
)

# Thinking mode prompt: prepended to system prompt to encourage deeper reasoning
_THINKING_PROMPT = (
    "[思考模式] 你现在处于深度思考模式。请认真分析用户的问题，"
    "进行更深入、更全面的思考和搜索，给出更有深度和见解的回答。"
    "你可以进行多轮搜索来获取更全面的信息，"
    "并综合分析后给出详细、有条理的回答。\n\n"
)

# Characters per chunk when streaming content from non-streaming responses
_STREAM_CHUNK_SIZE = 4


def _sanitize_content(content: str) -> str:
    """Strip leaked DeepSeek DSML function-calling markup from content.

    DeepSeek may emit raw `<｜DSML｜...>` tags when tools=None is passed
    after a tool-calling conversation. This sanitizes the output.
    """
    # Remove full DSML blocks first
    content = _DSML_BLOCK_PATTERN.sub("", content)
    # Remove any remaining DSML tags
    content = _DSML_PATTERN.sub("", content)
    return content.strip()


class ChatHandler:
    """Main chat handler with tool-calling loop.

    Stateless per request - the client manages conversation history by
    sending the full message list with each request (OpenAI-compatible).

    Usage:
        from llms.tools.executor import SearchService
        handler = ChatHandler(
            llm_client=create_llm_client(),  # uses LLM_CONFIG from configs/envs.py
            search_client=SearchService(video_searcher, video_explorer),
        )

        # Non-streaming
        response = handler.handle(messages=[{"role": "user", "content": "..."}])

        # Streaming (yields SSE chunks)
        for chunk in handler.handle_stream(messages=[...]):
            print(chunk)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        search_client,
        max_iterations: int = MAX_TOOL_ITERATIONS,
        max_tool_results: int = 8,
        temperature: float = None,
        verbose: bool = False,
    ):
        self.llm_client = llm_client
        self.tool_executor = ToolExecutor(
            search_client=search_client,
            max_results=max_tool_results,
            verbose=verbose,
        )
        self.tool_defs = TOOL_DEFINITIONS
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.verbose = verbose

    @staticmethod
    def _accumulate_usage(total: dict, new: dict):
        """Accumulate all numeric usage fields dynamically.

        Picks up standard fields (prompt_tokens, completion_tokens, total_tokens)
        as well as provider-specific fields like DeepSeek's
        prompt_cache_hit_tokens and prompt_cache_miss_tokens.

        Also handles OpenAI/GPT nested usage structures:
        - prompt_tokens_details.cached_tokens
        - completion_tokens_details.reasoning_tokens
        These are flattened into top-level keys for uniform access.
        """
        for key, value in new.items():
            if isinstance(value, (int, float)):
                total[key] = total.get(key, 0) + value
            elif isinstance(value, dict):
                # Flatten nested dicts (e.g. prompt_tokens_details, completion_tokens_details)
                # into top-level keys like "prompt_tokens_details.cached_tokens"
                if key not in total:
                    total[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        total[key][sub_key] = total[key].get(sub_key, 0) + sub_value

    def _run_tool_loop(
        self,
        full_messages: list[dict],
        temperature: float = None,
        max_iterations: int = None,
    ) -> tuple[str, dict, list[dict]]:
        """Run the tool-calling loop and return (content, usage, tool_events).

        Shared between handle() and handle_stream() to avoid duplication.
        If the loop exhausts max_iterations without producing content,
        injects a nudge message and makes one final tools=None call.

        Args:
            full_messages: Full message list including system prompt.
            temperature: Override temperature.
            max_iterations: Override max iterations for this request.

        Returns:
            Tuple of (final_content, total_usage_dict, tool_events).
        """
        final_content = None
        total_usage = {}
        tool_events = []  # Track tool calls for status reporting
        iterations = max_iterations or self.max_iterations

        for iteration in range(iterations):
            if self.verbose:
                logger.hint(f"> Iteration {iteration + 1}/{iterations}")

            response = self.llm_client.chat(
                messages=full_messages,
                tools=self.tool_defs,
                temperature=temperature,
            )
            self._accumulate_usage(total_usage, response.usage)

            if response.has_tool_calls:
                if response.content:
                    final_content = response.content
                tool_names = [tc.name for tc in response.tool_calls]
                tool_events.append(
                    {
                        "iteration": iteration + 1,
                        "tools": tool_names,
                    }
                )
                self._process_tool_calls(full_messages, response)
                continue
            else:
                final_content = response.content or ""
                break
        else:
            # Max iterations exhausted — nudge LLM to produce content
            logger.warn(
                f"× Tool loop hit {iterations} iterations, "
                "forcing content generation"
            )
            # Inject a nudge message to guide the LLM
            full_messages.append(
                {
                    "role": "user",
                    "content": _FORCE_CONTENT_NUDGE,
                }
            )
            response = self.llm_client.chat(
                messages=full_messages,
                tools=None,  # No tools → LLM must generate content
                temperature=temperature,
            )
            self._accumulate_usage(total_usage, response.usage)
            final_content = (
                response.content or final_content or "[抱歉，处理超时，请重试]"
            )

        # Sanitize any leaked DSML markup
        final_content = _sanitize_content(final_content)

        return final_content, total_usage, tool_events

    def handle(
        self,
        messages: list[dict],
        temperature: float = None,
        thinking: bool = False,
        max_iterations: int = None,
    ) -> dict:
        """Handle a chat completion request (non-streaming).

        Runs the tool-calling loop synchronously and returns the final
        response in OpenAI-compatible format.

        Args:
            messages: User-provided conversation messages.
            temperature: Override temperature for this request.
            thinking: Enable thinking mode for deeper analysis.
            max_iterations: Override max iterations for this request.

        Returns:
            OpenAI-compatible chat completion response dict.
        """
        start_time = time.perf_counter()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        temp = temperature if temperature is not None else self.temperature

        # Resolve max_iterations: explicit override > thinking default > normal default
        resolved_iterations = max_iterations
        if resolved_iterations is None:
            resolved_iterations = (
                MAX_TOOL_ITERATIONS_THINKING if thinking else self.max_iterations
            )

        full_messages = self._build_messages(messages, thinking=thinking)
        final_content, total_usage, tool_events = self._run_tool_loop(
            full_messages, temp, max_iterations=resolved_iterations
        )

        elapsed_seconds = time.perf_counter() - start_time
        elapsed_ms = round(elapsed_seconds * 1000, 1)
        if self.verbose:
            logger.success(f"> Chat completed in {elapsed_ms}ms")

        # Normalize usage and compute performance stats
        normalized_usage = self._normalize_usage(total_usage)
        perf_stats = self._compute_perf_stats(normalized_usage, elapsed_seconds)

        return self._format_completion(
            request_id=request_id,
            content=final_content,
            usage=normalized_usage,
            perf_stats=perf_stats,
            tool_events=tool_events,
            thinking=thinking,
        )

    def handle_stream(
        self,
        messages: list[dict],
        temperature: float = None,
        thinking: bool = False,
        max_iterations: int = None,
    ) -> Generator[str, None, None]:
        """Handle a streaming chat completion request.

        Uses non-streaming chat() for the tool loop (reliable across all
        providers/proxies), then chunk-streams the final content to the client.
        Tool events are yielded in real-time between iterations.

        This hybrid approach avoids streaming API issues (hangs with certain
        proxies, unsupported stream_options) while still delivering:
        - Real-time tool event feedback during the tool-calling loop
        - Progressive content delivery for the final answer
        - Support for reasoning_content (DeepSeek chain-of-thought)

        Args:
            messages: User-provided conversation messages.
            temperature: Override temperature for this request.
            thinking: Enable thinking mode for deeper analysis.
            max_iterations: Override max iterations for this request.

        Yields:
            SSE data strings (JSON-encoded chunks or "[DONE]").
        """
        start_time = time.perf_counter()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        temp = temperature if temperature is not None else self.temperature

        # Resolve max_iterations: explicit override > thinking default > normal default
        resolved_iterations = max_iterations
        if resolved_iterations is None:
            resolved_iterations = (
                MAX_TOOL_ITERATIONS_THINKING if thinking else self.max_iterations
            )

        full_messages = self._build_messages(messages, thinking=thinking)
        total_usage = {}

        # First chunk: role + metadata
        yield self._format_stream_chunk(
            request_id=request_id,
            delta={"role": "assistant", "content": ""},
            thinking=thinking,
        )

        # --- Phase 1: Tool loop using non-streaming chat() ---
        # Non-streaming is more reliable than chat_stream() across providers
        final_content = None
        final_reasoning = None

        for iteration in range(resolved_iterations):
            if self.verbose:
                logger.hint(f"> Stream iteration {iteration + 1}/{resolved_iterations}")

            response = self.llm_client.chat(
                messages=full_messages,
                tools=self.tool_defs,
                temperature=temp,
            )
            self._accumulate_usage(total_usage, response.usage)

            if response.has_tool_calls:
                if response.content:
                    final_content = response.content
                tool_names = [tc.name for tc in response.tool_calls]
                tool_event = {"iteration": iteration + 1, "tools": tool_names}

                # Yield tool event in real-time
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta={},
                    tool_events=[tool_event],
                )

                self._process_tool_calls(full_messages, response)
                continue
            else:
                final_content = response.content or ""
                final_reasoning = response.reasoning_content
                break
        else:
            # Max iterations exhausted — force content generation
            if not final_content:
                logger.warn(
                    f"× Tool loop hit {resolved_iterations} iterations, "
                    "forcing content generation"
                )
                full_messages.append({"role": "user", "content": _FORCE_CONTENT_NUDGE})
                response = self.llm_client.chat(
                    messages=full_messages,
                    tools=None,
                    temperature=temp,
                )
                self._accumulate_usage(total_usage, response.usage)
                final_content = (
                    response.content or final_content or "[抱歉，处理超时，请重试]"
                )
                final_reasoning = response.reasoning_content

        # Sanitize content
        final_content = _sanitize_content(final_content)

        # --- Phase 2: Stream content in chunks ---
        # Stream reasoning_content first (e.g. DeepSeek chain-of-thought)
        if final_reasoning:
            for i in range(0, len(final_reasoning), _STREAM_CHUNK_SIZE):
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta={
                        "reasoning_content": final_reasoning[i : i + _STREAM_CHUNK_SIZE]
                    },
                )

        # Stream main content
        for i in range(0, len(final_content), _STREAM_CHUNK_SIZE):
            yield self._format_stream_chunk(
                request_id=request_id,
                delta={"content": final_content[i : i + _STREAM_CHUNK_SIZE]},
            )

        # Normalize usage and compute perf stats
        normalized_usage = self._normalize_usage(total_usage)
        elapsed_seconds = time.perf_counter() - start_time
        perf_stats = self._compute_perf_stats(normalized_usage, elapsed_seconds)

        if self.verbose:
            elapsed_ms = round(elapsed_seconds * 1000, 1)
            logger.success(f"> Stream completed in {elapsed_ms}ms")

        # Final chunk with stats
        yield self._format_stream_chunk(
            request_id=request_id,
            delta={},
            finish_reason="stop",
            usage=normalized_usage,
            perf_stats=perf_stats,
        )

        yield "[DONE]"

    def _build_messages(
        self, user_messages: list[dict], thinking: bool = False
    ) -> list[dict]:
        """Prepend system prompt to user messages.

        When thinking=True, prepends an additional thinking prompt
        to encourage deeper analysis.
        """
        system_content = build_system_prompt()
        if thinking:
            system_content = _THINKING_PROMPT + system_content

        system_message = {
            "role": "system",
            "content": system_content,
        }
        return [system_message] + list(user_messages)

    def _process_tool_calls(
        self,
        messages: list[dict],
        response: ChatResponse,
    ):
        """Execute tool calls and append results to the message list.

        Supports parallel tool calls — when the LLM returns multiple tool_calls
        in one response, all are executed and their results appended.

        Follows the OpenAI conversation format:
        1. Append the assistant message with tool_calls
        2. Append each tool result as a tool message
        """
        # Append assistant message with tool_calls
        messages.append(response.to_message_dict())

        # Execute each tool call and append results
        num_calls = len(response.tool_calls)
        for tool_call in response.tool_calls:
            result_message = self.tool_executor.execute(tool_call)
            messages.append(result_message)

            if self.verbose:
                logger.mesg(
                    f"  Tool '{tool_call.name}' → "
                    f"{len(result_message.get('content', ''))} chars"
                    + (f" ({num_calls} parallel)" if num_calls > 1 else "")
                )

    @staticmethod
    def _normalize_usage(usage: dict) -> dict:
        """Normalize usage dict: unify cache token formats across providers.

        Handles both DeepSeek flat format (prompt_cache_hit_tokens) and
        GPT nested format (prompt_tokens_details.cached_tokens), flattening
        the latter to the flat format for uniform access.

        Removes nested dict fields from the output to keep it clean.

        Returns:
            Normalized usage dict with flat keys only.
        """
        result = dict(usage)

        # GPT nested → flat
        prompt_details = result.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            gpt_cached = prompt_details.get("cached_tokens", 0)
            if gpt_cached and not result.get("prompt_cache_hit_tokens"):
                result["prompt_cache_hit_tokens"] = gpt_cached
                prompt_tokens = result.get("prompt_tokens", 0)
                result["prompt_cache_miss_tokens"] = max(0, prompt_tokens - gpt_cached)

        completion_details = result.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning = completion_details.get("reasoning_tokens", 0)
            if reasoning:
                result["reasoning_tokens"] = reasoning

        # Remove nested dicts (keep only flat numeric fields)
        for key in list(result.keys()):
            if isinstance(result[key], dict):
                del result[key]

        return result

    @staticmethod
    def _compute_perf_stats(usage: dict, elapsed_seconds: float) -> dict:
        """Compute performance statistics from usage and elapsed time.

        Only includes timing/rate metrics. Token counts stay in usage.

        Returns:
            Dict with tokens_per_second and elapsed timing.
        """
        completion_tokens = usage.get("completion_tokens", 0)

        tokens_per_second = (
            int(completion_tokens / elapsed_seconds)
            if elapsed_seconds > 0 and completion_tokens > 0
            else 0
        )

        elapsed_str = dt_to_str(elapsed_seconds, precision=1)

        return {
            "tokens_per_second": tokens_per_second,
            "total_elapsed": elapsed_str,
            "total_elapsed_ms": round(elapsed_seconds * 1000, 1),
        }

    def _format_completion(
        self,
        request_id: str,
        content: str,
        usage: dict = None,
        perf_stats: dict = None,
        tool_events: list = None,
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
        if thinking is not None:
            chunk["thinking"] = thinking
        return json.dumps(chunk, ensure_ascii=False)
