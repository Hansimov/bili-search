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
import threading
import time
import uuid

from tclogger import logger, dt_to_str
from typing import Generator, Optional

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

# Inline tool command prefixes used for look-ahead detection during content streaming.
# When these appear in content, it means the LLM is issuing a tool call rather than
# producing a final answer, so we must stop streaming content to the client.
_TOOL_PREFIXES: tuple[str, ...] = ("<search_videos", "<check_author")
_MAX_TOOL_PREFIX_LEN: int = max(len(p) for p in _TOOL_PREFIXES)  # 14


def _find_tool_command_start(text: str) -> int | None:
    """Return the earliest index of any tool command prefix in *text*, or None."""
    pos = None
    for prefix in _TOOL_PREFIXES:
        idx = text.find(prefix)
        if idx >= 0 and (pos is None or idx < pos):
            pos = idx
    return pos


def _has_partial_tool_prefix(text: str) -> bool:
    """Return True if *text* ends with a partial match for any tool prefix.

    Used as a look-ahead guard: if the last few characters of the accumulated
    content *could* be the start of a tool tag, we withhold them from the
    client until more data arrives to confirm or deny the match.
    """
    for prefix in _TOOL_PREFIXES:
        for length in range(1, len(prefix)):
            if text.endswith(prefix[:length]):
                return True
    return False


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

    def _chat_interruptible(
        self,
        messages: list[dict],
        temperature: float = None,
        cancelled: Optional[threading.Event] = None,
    ) -> ChatResponse:
        """Chat with LLM, interruptible via cancelled event.

        Uses streaming internally so that cancellation can be checked between
        chunks rather than blocking for the entire response. Falls back to
        non-streaming chat() when no cancelled event is provided.

        Returns:
            ChatResponse with accumulated content and usage.
            If cancelled mid-stream, returns partial content with
            finish_reason="cancelled".
        """
        if cancelled is None:
            return self.llm_client.chat(messages=messages, temperature=temperature)

        accumulated_content = ""
        accumulated_reasoning = ""
        finish_reason = None
        usage = {}

        for chunk in self.llm_client.chat_stream(
            messages=messages,
            temperature=temperature,
        ):
            if cancelled.is_set():
                logger.warn("> LLM call interrupted by cancellation")
                return ChatResponse(
                    content=accumulated_content or None,
                    reasoning_content=accumulated_reasoning or None,
                    finish_reason="cancelled",
                    usage=usage,
                )

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
        last_usage = {}  # Track last LLM call's usage for prompt_tokens
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
            last_usage = response.usage

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
            last_usage = response.usage
            final_content = (
                response.content or final_content or "[抱歉，处理超时，请重试]"
            )

        final_content = _sanitize_content(final_content)

        elapsed_seconds = time.perf_counter() - start_time
        elapsed_ms = round(elapsed_seconds * 1000, 1)
        if self.verbose:
            logger.success(f"> Chat completed in {elapsed_ms}ms")

        # Use last call's prompt_tokens + accumulated completion_tokens
        final_usage = self._merge_final_usage(total_usage, last_usage)
        normalized_usage = self._normalize_usage(final_usage)
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
        cancelled: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """Handle a streaming chat completion request.

        All content (reasoning + final answer) is streamed in real-time:

        Phase 1: Tool loop — for each iteration, both reasoning_content AND
        content are yielded to the client token-by-token as they arrive.
        A look-ahead buffer detects tool command prefixes in content and stops
        streaming content to the client when a tool call is imminent (so the
        LLM's XML tags are never shown to the user).  When an iteration ends
        with tool commands, a ``retract_content`` event is sent to ask the
        frontend to clear any analysis text that was shown, because it belongs
        in the thinking section instead.  The full analysis (tool commands
        stripped) is then forwarded as ``reasoning_content`` so it appears in
        the thinking section.  Tool events are emitted and the conversation
        continues to the next iteration.

        When an iteration produces no tool commands, the final answer was
        already streamed in real-time.  Any look-ahead-buffered tail content
        is flushed, stats are computed, and the stream finishes.

        Phase 2: Reached only when max iterations are exhausted.  A nudge
        message forces content generation via real streaming (chat_stream).

        Token usage reporting: prompt_tokens reflects only the LAST LLM call
        (= actual context size for the final answer), while completion_tokens
        is accumulated across all iterations (= total output generated).

        Args:
            messages: User-provided conversation messages.
            temperature: Override temperature for this request.
            thinking: Enable thinking mode for deeper analysis.
            max_iterations: Override max iterations for this request.
            cancelled: Optional threading.Event for cooperative cancellation.
                When set, the generator will stop as soon as possible.

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
        last_usage = {}  # Track last LLM call's usage for prompt_tokens

        # First chunk: role + metadata
        yield self._format_stream_chunk(
            request_id=request_id,
            delta={"role": "assistant", "content": ""},
            thinking=thinking,
        )

        def _is_cancelled() -> bool:
            return cancelled is not None and cancelled.is_set()

        # --- Phase 1: Prompt-based tool loop with real-time reasoning streaming ---
        # Each iteration streams reasoning_content to the client in real-time
        # while accumulating content locally for tool command parsing.
        had_tools = False  # Track if any tool commands were executed
        # Collect analysis texts from tool-calling iterations so we can
        # strip duplicate leading text from the final answer.
        tool_analyses: list[str] = []

        for iteration in range(resolved_iterations):
            if _is_cancelled():
                logger.warn("> Chat cancelled before iteration")
                yield "[DONE]"
                return

            if self.verbose:
                logger.hint(f"> Stream iteration {iteration + 1}/{resolved_iterations}")

            # Stream from LLM: yield reasoning_content AND content in real-time.
            # Content is also accumulated so tool commands can be parsed after
            # the full response.  A look-ahead buffer prevents tool command tags
            # from being sent to the client.
            accumulated_content = ""
            # Pointer into accumulated_content: how many chars have already been
            # sent to the client as content deltas.
            content_sent_ptr = 0
            # Set to True once a tool command prefix is confirmed in the stream.
            tool_prefix_detected = False
            # Track whether the LLM already sent reasoning_content in this
            # iteration.  When True, we suppress content streaming because
            # the analysis text is already in the thinking section and will
            # be retracted/discarded when tools are detected.
            has_reasoning = False
            iter_usage = {}

            for chunk in self.llm_client.chat_stream(
                messages=full_messages,
                temperature=temp,
            ):
                if _is_cancelled():
                    logger.warn("> Chat cancelled during LLM streaming")
                    yield "[DONE]"
                    return

                choices = chunk.get("choices", [])
                if not choices:
                    if chunk.get("usage"):
                        iter_usage = chunk["usage"]
                    continue

                delta = choices[0].get("delta", {})

                # Stream reasoning_content to frontend in real-time (unchanged)
                reasoning_delta = delta.get("reasoning_content", "")
                if reasoning_delta:
                    has_reasoning = True
                    yield self._format_stream_chunk(
                        request_id=request_id,
                        delta={"reasoning_content": reasoning_delta},
                    )

                # Accumulate content and stream it in real-time with look-ahead.
                content_delta = delta.get("content") or ""
                if content_delta:
                    accumulated_content += content_delta

                # When the LLM already sent reasoning_content for this
                # iteration, suppress content streaming — the analysis
                # text is already in the thinking section.  We still
                # accumulate and run tool-prefix detection so we can
                # parse tool commands after the iteration.
                if content_delta and not tool_prefix_detected:
                    # Always run tool-prefix detection on accumulated content
                    tool_start = _find_tool_command_start(accumulated_content)
                    if tool_start is not None:
                        if not has_reasoning and tool_start > content_sent_ptr:
                            safe_text = accumulated_content[content_sent_ptr:tool_start]
                            if safe_text:
                                yield self._format_stream_chunk(
                                    request_id=request_id,
                                    delta={"content": safe_text},
                                )
                                content_sent_ptr = tool_start
                        tool_prefix_detected = True
                    elif not has_reasoning:
                        # No full tool prefix yet.  If the tail of accumulated
                        # content could be the start of a tool tag, withhold
                        # the last _MAX_TOOL_PREFIX_LEN chars as a look-ahead
                        # guard; otherwise yield everything accumulated so far.
                        if _has_partial_tool_prefix(accumulated_content):
                            safe_end = max(
                                content_sent_ptr,
                                len(accumulated_content) - _MAX_TOOL_PREFIX_LEN,
                            )
                        else:
                            safe_end = len(accumulated_content)

                        if safe_end > content_sent_ptr:
                            yield self._format_stream_chunk(
                                request_id=request_id,
                                delta={
                                    "content": accumulated_content[
                                        content_sent_ptr:safe_end
                                    ]
                                },
                            )
                            content_sent_ptr = safe_end

                if chunk.get("usage"):
                    iter_usage = chunk["usage"]

            if _is_cancelled():
                logger.warn("> Chat cancelled after LLM streaming")
                yield "[DONE]"
                return

            self._accumulate_usage(total_usage, iter_usage)
            last_usage = iter_usage

            content = accumulated_content
            commands = self._parse_tool_commands(content)

            if commands:
                had_tools = True

                if _is_cancelled():
                    logger.warn("> Chat cancelled after LLM response")
                    yield "[DONE]"
                    return

                # Derive the analysis text (tool commands stripped) for
                # bookkeeping regardless of whether we re-send it.
                analysis = self._strip_tool_commands(content).strip()
                if analysis:
                    tool_analyses.append(analysis)

                if has_reasoning:
                    # The LLM already streamed reasoning_content for this
                    # iteration — the analysis is already in the thinking
                    # section.  No retract or re-send needed.
                    pass
                else:
                    # No reasoning was sent; the analysis was streamed as
                    # content.  Retract it and re-send as reasoning.
                    if content_sent_ptr > 0:
                        yield self._format_stream_chunk(
                            request_id=request_id,
                            delta={"retract_content": True},
                        )
                    if analysis:
                        yield self._format_stream_chunk(
                            request_id=request_id,
                            delta={"reasoning_content": analysis},
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

                if _is_cancelled():
                    logger.warn("> Chat cancelled before tool execution")
                    yield "[DONE]"
                    return

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
                # No tool commands: the final answer.
                # Flush any content still in the look-ahead buffer, then
                # finalize the stream with stats.
                if has_reasoning:
                    # Content was buffered (reasoning was streamed instead).
                    # Strip leading text that duplicates a prior tool
                    # analysis to avoid echoing thinking into the answer.
                    final = _sanitize_content(content)
                    for ta in tool_analyses:
                        if final.startswith(ta):
                            final = final[len(ta) :].lstrip("\n")
                            break
                    if final:
                        if _is_cancelled():
                            yield "[DONE]"
                            return
                        yield self._format_stream_chunk(
                            request_id=request_id,
                            delta={"content": final},
                        )
                elif content_sent_ptr < len(content):
                    remaining = _sanitize_content(content[content_sent_ptr:])
                    if remaining:
                        if _is_cancelled():
                            yield "[DONE]"
                            return
                        yield self._format_stream_chunk(
                            request_id=request_id,
                            delta={"content": remaining},
                        )

                # Finalize: compute stats and yield final chunk
                final_usage = self._merge_final_usage(total_usage, last_usage)
                normalized_usage = self._normalize_usage(final_usage)
                elapsed_seconds = time.perf_counter() - start_time
                perf_stats = self._compute_perf_stats(normalized_usage, elapsed_seconds)

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

        # --- Phase 2: Real streaming for forced content generation ---
        # Only reached when max iterations exhausted. Normal tool→answer flow
        # completes in Phase 1 above.

        if _is_cancelled():
            logger.warn("> Chat cancelled before Phase 2")
            yield "[DONE]"
            return

        if self.verbose:
            logger.note("> Phase 2: real streaming for forced content")

        final_content = ""
        stream_usage = {}
        for chunk in self.llm_client.chat_stream(
            messages=full_messages,
            temperature=temp,
        ):
            if _is_cancelled():
                logger.warn("> Chat cancelled during Phase 2 streaming")
                break

            choices = chunk.get("choices", [])
            if not choices:
                if chunk.get("usage"):
                    stream_usage = chunk["usage"]
                continue
            delta = choices[0].get("delta", {})

            reasoning_delta = delta.get("reasoning_content", "")
            if reasoning_delta:
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta={"reasoning_content": reasoning_delta},
                )

            content_delta = delta.get("content", "")
            if content_delta:
                final_content += content_delta
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta={"content": content_delta},
                )

            if chunk.get("usage"):
                stream_usage = chunk["usage"]

        if stream_usage:
            self._accumulate_usage(total_usage, stream_usage)
            last_usage = stream_usage

        final_usage = self._merge_final_usage(total_usage, last_usage)
        normalized_usage = self._normalize_usage(final_usage)
        elapsed_seconds = time.perf_counter() - start_time
        perf_stats = self._compute_perf_stats(normalized_usage, elapsed_seconds)

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
