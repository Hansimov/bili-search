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

from tclogger import logger
from typing import Generator

from llms.llm_client import LLMClient, ChatResponse, create_llm_client
from llms.tools.defs import TOOL_DEFINITIONS
from llms.tools.executor import ToolExecutor
from llms.prompts.copilot import build_system_prompt

# Maximum tool-calling iterations to prevent infinite loops.
# Typical flow: check_author → search_videos → content = 2 tool calls + 1 content.
# 3 allows check_author + 2 searches before forcing content.
MAX_TOOL_ITERATIONS = 3

# Default chunk size for simulated streaming (chars per chunk)
STREAM_CHUNK_SIZE = 4

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
            llm_client=create_llm_client("deepseek"),
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
        """
        for key, value in new.items():
            if isinstance(value, (int, float)):
                total[key] = total.get(key, 0) + value

    def _run_tool_loop(
        self,
        full_messages: list[dict],
        temperature: float = None,
    ) -> tuple[str, dict]:
        """Run the tool-calling loop and return (content, usage).

        Shared between handle() and handle_stream() to avoid duplication.
        If the loop exhausts max_iterations without producing content,
        injects a nudge message and makes one final tools=None call.

        Returns:
            Tuple of (final_content, total_usage_dict).
        """
        final_content = None
        total_usage = {}

        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.hint(f"> Iteration {iteration + 1}/{self.max_iterations}")

            response = self.llm_client.chat(
                messages=full_messages,
                tools=self.tool_defs,
                temperature=temperature,
            )
            self._accumulate_usage(total_usage, response.usage)

            if response.has_tool_calls:
                if response.content:
                    final_content = response.content
                self._process_tool_calls(full_messages, response)
                continue
            else:
                final_content = response.content or ""
                break
        else:
            # Max iterations exhausted — nudge LLM to produce content
            logger.warn(
                f"× Tool loop hit {self.max_iterations} iterations, "
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

        return final_content, total_usage

    def handle(
        self,
        messages: list[dict],
        temperature: float = None,
    ) -> dict:
        """Handle a chat completion request (non-streaming).

        Runs the tool-calling loop synchronously and returns the final
        response in OpenAI-compatible format.

        Args:
            messages: User-provided conversation messages.
            temperature: Override temperature for this request.

        Returns:
            OpenAI-compatible chat completion response dict.
        """
        start_time = time.perf_counter()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        temp = temperature if temperature is not None else self.temperature

        full_messages = self._build_messages(messages)
        final_content, total_usage = self._run_tool_loop(full_messages, temp)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        if self.verbose:
            logger.success(f"> Chat completed in {elapsed_ms}ms")

        return self._format_completion(
            request_id=request_id,
            content=final_content,
            usage=total_usage,
        )

    def handle_stream(
        self,
        messages: list[dict],
        temperature: float = None,
    ) -> Generator[str, None, None]:
        """Handle a streaming chat completion request.

        Runs the tool-calling loop, then yields the final response as
        SSE-formatted chunks.

        Yields:
            SSE data strings (JSON-encoded chunks or "[DONE]").
        """
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        temp = temperature if temperature is not None else self.temperature

        full_messages = self._build_messages(messages)
        final_content, _usage = self._run_tool_loop(full_messages, temp)

        # Stream the final content as SSE chunks
        # First chunk: role
        yield self._format_stream_chunk(
            request_id=request_id,
            delta={"role": "assistant", "content": ""},
        )

        # Content chunks
        for i in range(0, len(final_content), STREAM_CHUNK_SIZE):
            chunk_text = final_content[i : i + STREAM_CHUNK_SIZE]
            yield self._format_stream_chunk(
                request_id=request_id,
                delta={"content": chunk_text},
            )

        # Final chunk
        yield self._format_stream_chunk(
            request_id=request_id,
            delta={},
            finish_reason="stop",
        )

        # Done signal
        yield "[DONE]"

    def _build_messages(self, user_messages: list[dict]) -> list[dict]:
        """Prepend system prompt to user messages."""
        system_message = {
            "role": "system",
            "content": build_system_prompt(),
        }
        return [system_message] + list(user_messages)

    def _process_tool_calls(
        self,
        messages: list[dict],
        response: ChatResponse,
    ):
        """Execute tool calls and append results to the message list.

        Follows the OpenAI conversation format:
        1. Append the assistant message with tool_calls
        2. Append each tool result as a tool message
        """
        # Append assistant message with tool_calls
        messages.append(response.to_message_dict())

        # Execute each tool call and append results
        for tool_call in response.tool_calls:
            result_message = self.tool_executor.execute(tool_call)
            messages.append(result_message)

            if self.verbose:
                logger.mesg(
                    f"  Tool '{tool_call.name}' → "
                    f"{len(result_message.get('content', ''))} chars"
                )

    def _format_completion(
        self,
        request_id: str,
        content: str,
        usage: dict = None,
    ) -> dict:
        """Format response as OpenAI-compatible chat completion."""
        return {
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

    def _format_stream_chunk(
        self,
        request_id: str,
        delta: dict,
        finish_reason: str = None,
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
        return json.dumps(chunk, ensure_ascii=False)
