"""Chat handler with tool-calling loop.

Orchestrates the conversation between user, LLM, and search tools.
Implements the iterative tool-calling pattern:
  1. Send user message + system prompt + tool defs to LLM
  2. If LLM returns tool_calls → execute tools → feed results back → repeat
  3. If LLM returns content → return as final response
"""

import json
import time
import uuid

from tclogger import logger
from typing import Generator

from llms.llm_client import LLMClient, ChatResponse, create_llm_client
from llms.search_service import SearchServiceClient
from llms.tools.defs import TOOL_DEFINITIONS
from llms.tools.executor import ToolExecutor
from llms.prompts.copilot import build_system_prompt

# Maximum tool-calling iterations to prevent infinite loops
MAX_TOOL_ITERATIONS = 8

# Default chunk size for simulated streaming (chars per chunk)
STREAM_CHUNK_SIZE = 4


class ChatHandler:
    """Main chat handler with tool-calling loop.

    Stateless per request - the client manages conversation history by
    sending the full message list with each request (OpenAI-compatible).

    Usage:
        handler = ChatHandler(
            llm_client=create_llm_client("deepseek"),
            search_client=SearchServiceClient("http://localhost:20001"),
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
        search_client: SearchServiceClient,
        max_iterations: int = MAX_TOOL_ITERATIONS,
        max_tool_results: int = 15,
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

        # Build full message list with system prompt
        full_messages = self._build_messages(messages)

        # Tool-calling loop
        final_content = None
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.hint(f"> Iteration {iteration + 1}/{self.max_iterations}")

            response = self.llm_client.chat(
                messages=full_messages,
                tools=self.tool_defs,
                temperature=temp,
            )

            # Accumulate usage
            for key in total_usage:
                total_usage[key] += response.usage.get(key, 0)

            if response.has_tool_calls:
                # Execute tools and append results to conversation
                self._process_tool_calls(full_messages, response)
                continue
            else:
                # Final content response
                final_content = response.content or ""
                break
        else:
            # Max iterations reached
            logger.warn(
                f"× Max iterations ({self.max_iterations}) reached, "
                "returning last available content"
            )
            final_content = final_content or "[抱歉，处理超时，请重试]"

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

        # Build full message list with system prompt
        full_messages = self._build_messages(messages)

        # Tool-calling loop (non-streaming, internal)
        final_content = None
        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.hint(f"> Stream iteration {iteration + 1}/{self.max_iterations}")

            response = self.llm_client.chat(
                messages=full_messages,
                tools=self.tool_defs,
                temperature=temp,
            )

            if response.has_tool_calls:
                self._process_tool_calls(full_messages, response)
                continue
            else:
                final_content = response.content or ""
                break
        else:
            final_content = final_content or "[抱歉，处理超时，请重试]"

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
