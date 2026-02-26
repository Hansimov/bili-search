"""Chat handler with prompt-based tool orchestration.

Orchestrates the conversation between user, LLM, and search tools
using inline XML commands in the LLM's response text (no function calling).

The flow:
  1. Send user message + system prompt → LLM responds with analysis + tool commands
  2. Parse tool commands from response → execute searches → inject results
  3. Send results back → LLM generates final answer
  4. Stream final answer to client

Token optimization strategy:
  - Each iteration re-sends the full conversation, so fewer iterations = fewer tokens
  - Compact DSL syntax is inline in the system prompt
  - Tool results use compact JSON (no indent)
  - If max iterations exhausted, inject a nudge message to force content generation
"""

import json
import re
import time
import uuid

from tclogger import logger, dt_to_str
from typing import Generator

from llms.llm_client import LLMClient, ChatResponse, create_llm_client
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

# Regex patterns for parsing inline tool commands from LLM responses
_SEARCH_CMD_RE = re.compile(
    r"""<search_videos\s+queries='(\[.*?\])'\s*/>""",
    re.DOTALL,
)
_CHECK_AUTHOR_CMD_RE = re.compile(
    r"""<check_author\s+name="([^"]+)"\s*/>""",
)
_TOOL_CMD_PATTERN = re.compile(
    r"""<(?:search_videos|check_author)\s[^>]*/>""",
)

# Nudge message injected before forcing content generation
_FORCE_CONTENT_NUDGE = (
    "你已经进行了充分的搜索。请根据以上搜索结果直接回答用户的问题。"
    "不要再输出任何搜索命令（如 <search_videos/> 或 <check_author/>）。"
    "如果搜索结果不完全匹配，就根据已有信息给出最佳回答。"
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
    # Remove any leaked tool commands
    content = _TOOL_CMD_PATTERN.sub("", content)
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

    @staticmethod
    def _parse_tool_commands(content: str) -> list[dict]:
        """Parse inline tool commands from LLM response text.

        Scans for <search_videos .../> and <check_author .../> XML tags.

        Returns:
            List of command dicts with 'type' and 'args' keys.
        """
        commands = []
        for match in _SEARCH_CMD_RE.finditer(content):
            queries_str = match.group(1)
            try:
                queries = json.loads(queries_str)
                if isinstance(queries, list):
                    commands.append(
                        {"type": "search_videos", "args": {"queries": queries}}
                    )
            except json.JSONDecodeError:
                pass
        for match in _CHECK_AUTHOR_CMD_RE.finditer(content):
            name = match.group(1)
            commands.append({"type": "check_author", "args": {"name": name}})
        return commands

    @staticmethod
    def _strip_tool_commands(content: str) -> str:
        """Remove inline tool commands from content, keeping analysis text."""
        return _TOOL_CMD_PATTERN.sub("", content).strip()

    def _execute_tool_commands(self, commands: list[dict]) -> list[dict]:
        """Execute parsed tool commands and return results.

        Dispatches to ToolExecutor's internal methods directly.

        Returns:
            List of result dicts with 'type', 'args', and 'result' keys.
        """
        results = []
        for cmd in commands:
            cmd_type = cmd["type"]
            args = cmd["args"]
            if cmd_type == "search_videos":
                result = self.tool_executor._search_videos(args)
            elif cmd_type == "check_author":
                result = self.tool_executor._check_author(args)
            else:
                continue
            results.append({"type": cmd_type, "args": args, "result": result})
            if self.verbose:
                result_str = json.dumps(result, ensure_ascii=False)
                logger.mesg(f"  {cmd_type} → {len(result_str)} chars")
        return results

    @staticmethod
    def _format_results_message(results: list[dict]) -> str:
        """Format tool results as a user message for conversation injection."""
        parts = ["[搜索结果]"]
        for r in results:
            cmd_type = r["type"]
            args = r["args"]
            result = r["result"]
            if cmd_type == "search_videos":
                queries = args.get("queries", [])
                parts.append(
                    f"\nsearch_videos(queries="
                    f"{json.dumps(queries, ensure_ascii=False)}):"
                )
            elif cmd_type == "check_author":
                name = args.get("name", "")
                parts.append(f'\ncheck_author(name="{name}"):')
            parts.append(json.dumps(result, ensure_ascii=False))
        return "\n".join(parts)

    def handle(
        self,
        messages: list[dict],
        temperature: float = None,
        thinking: bool = False,
        max_iterations: int = None,
    ) -> dict:
        """Handle a chat completion request (non-streaming).

        Uses prompt-based tool commands: the LLM outputs XML tool commands
        inline in its response text, which are parsed and executed.

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
        total_usage = {}
        tool_events = []
        final_content = None

        for iteration in range(resolved_iterations):
            if self.verbose:
                logger.hint(f"> Iteration {iteration + 1}/{resolved_iterations}")

            response = self.llm_client.chat(
                messages=full_messages,
                temperature=temp,
            )
            self._accumulate_usage(total_usage, response.usage)

            content = response.content or ""
            commands = self._parse_tool_commands(content)

            if commands:
                tool_names = [cmd["type"] for cmd in commands]
                tool_events.append({"iteration": iteration + 1, "tools": tool_names})
                results = self._execute_tool_commands(commands)
                full_messages.append({"role": "assistant", "content": content})
                full_messages.append(
                    {
                        "role": "user",
                        "content": self._format_results_message(results),
                    }
                )
                continue
            else:
                final_content = content
                break
        else:
            # Max iterations exhausted — nudge LLM to produce content
            logger.warn(
                f"× Tool loop hit {resolved_iterations} iterations, "
                "forcing content generation"
            )
            full_messages.append({"role": "user", "content": _FORCE_CONTENT_NUDGE})
            response = self.llm_client.chat(
                messages=full_messages,
                temperature=temp,
            )
            self._accumulate_usage(total_usage, response.usage)
            final_content = (
                response.content or final_content or "[抱歉，处理超时，请重试]"
            )

        final_content = _sanitize_content(final_content)

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

        Phase 1: Uses non-streaming chat() for the prompt-based tool loop
        (needs full response to parse XML tool commands). Between iterations,
        analysis text is yielded as reasoning_content and tool events in real-time.

        Phase 2: After tool execution, uses real streaming chat_stream() for
        the final content generation, yielding token deltas as they arrive
        from the LLM API. For simple Q&A (no tool calls), fake-streams the
        already-obtained response since it's typically fast.

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

        # --- Phase 1: Prompt-based tool loop using non-streaming chat() ---
        # Tool loop iterations MUST use non-streaming chat() because we need
        # the full response to parse XML tool commands.
        had_tools = False  # Track if any tool commands were executed

        for iteration in range(resolved_iterations):
            if self.verbose:
                logger.hint(f"> Stream iteration {iteration + 1}/{resolved_iterations}")

            response = self.llm_client.chat(
                messages=full_messages,
                temperature=temp,
            )
            self._accumulate_usage(total_usage, response.usage)

            content = response.content or ""
            commands = self._parse_tool_commands(content)

            if commands:
                had_tools = True

                # Yield thinking content (analysis portion, without commands)
                analysis = self._strip_tool_commands(content)
                if analysis:
                    for i in range(0, len(analysis), _STREAM_CHUNK_SIZE):
                        yield self._format_stream_chunk(
                            request_id=request_id,
                            delta={
                                "reasoning_content": analysis[
                                    i : i + _STREAM_CHUNK_SIZE
                                ]
                            },
                        )

                # Yield pending tool calls (before execution)
                tool_names = [cmd["type"] for cmd in commands]
                pending_calls = [
                    {
                        "type": cmd["type"],
                        "args": cmd["args"],
                        "status": "pending",
                    }
                    for cmd in commands
                ]
                tool_event_pending = {
                    "iteration": iteration + 1,
                    "tools": tool_names,
                    "calls": pending_calls,
                }
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta={},
                    tool_events=[tool_event_pending],
                )

                # Execute commands
                results = self._execute_tool_commands(commands)

                # Yield completed tool calls (after execution)
                completed_calls = [
                    {
                        "type": r["type"],
                        "args": r["args"],
                        "status": "completed",
                        "result": r["result"],
                    }
                    for r in results
                ]
                tool_event_completed = {
                    "iteration": iteration + 1,
                    "tools": tool_names,
                    "calls": completed_calls,
                }
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta={},
                    tool_events=[tool_event_completed],
                )

                # Inject assistant message + results into conversation
                full_messages.append({"role": "assistant", "content": content})
                full_messages.append(
                    {
                        "role": "user",
                        "content": self._format_results_message(results),
                    }
                )
                continue
            else:
                # No tool commands: this is the final content response.
                if had_tools:
                    # After tool iterations: discard this non-streaming response
                    # and use Phase 2 real streaming for better UX.
                    break
                else:
                    # Simple Q&A (no tools used): fake-stream the already-obtained
                    # response since it's typically fast and short.
                    final_content = _sanitize_content(content)
                    final_reasoning = response.reasoning_content

                    if final_reasoning:
                        for i in range(0, len(final_reasoning), _STREAM_CHUNK_SIZE):
                            yield self._format_stream_chunk(
                                request_id=request_id,
                                delta={
                                    "reasoning_content": final_reasoning[
                                        i : i + _STREAM_CHUNK_SIZE
                                    ]
                                },
                            )
                    for i in range(0, len(final_content), _STREAM_CHUNK_SIZE):
                        yield self._format_stream_chunk(
                            request_id=request_id,
                            delta={
                                "content": final_content[i : i + _STREAM_CHUNK_SIZE]
                            },
                        )

                    # Finalize: compute stats and yield final chunk
                    normalized_usage = self._normalize_usage(total_usage)
                    elapsed_seconds = time.perf_counter() - start_time
                    perf_stats = self._compute_perf_stats(
                        normalized_usage, elapsed_seconds
                    )

                    if self.verbose:
                        elapsed_ms = round(elapsed_seconds * 1000, 1)
                        logger.success(f"> Stream completed in {elapsed_ms}ms")

                    yield self._format_stream_chunk(
                        request_id=request_id,
                        delta={},
                        finish_reason="stop",
                        usage=normalized_usage,
                        perf_stats=perf_stats,
                    )
                    yield "[DONE]"
                    return
        else:
            # Max iterations exhausted — force content generation
            logger.warn(
                f"× Tool loop hit {resolved_iterations} iterations, "
                "forcing content generation"
            )
            full_messages.append({"role": "user", "content": _FORCE_CONTENT_NUDGE})

        # --- Phase 2: Real streaming for final content generation ---
        # Used after tool iterations completed (had_tools=True) or max iterations hit.
        # Makes a streaming LLM call so tokens arrive in real-time at the frontend.
        if self.verbose:
            logger.note("> Phase 2: real streaming for final content")

        final_content = ""
        stream_usage = {}
        for chunk in self.llm_client.chat_stream(
            messages=full_messages,
            temperature=temp,
        ):
            choices = chunk.get("choices", [])
            if not choices:
                # Accumulate usage from stream metadata chunks
                if chunk.get("usage"):
                    stream_usage = chunk["usage"]
                continue
            delta = choices[0].get("delta", {})

            # Stream reasoning_content delta (chain-of-thought)
            reasoning_delta = delta.get("reasoning_content", "")
            if reasoning_delta:
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta={"reasoning_content": reasoning_delta},
                )

            # Stream content delta (the actual answer)
            content_delta = delta.get("content", "")
            if content_delta:
                final_content += content_delta
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta={"content": content_delta},
                )

            # Check for usage in the final chunk
            if chunk.get("usage"):
                stream_usage = chunk["usage"]

        # Accumulate streaming usage
        if stream_usage:
            self._accumulate_usage(total_usage, stream_usage)

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

        elapsed_str = dt_to_str(elapsed_seconds, precision=0)

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
