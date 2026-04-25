"""OpenAI-style chat handler backed by the multi-model orchestrator.

The current flow is:
    1. Build an intent profile from the latest conversation turn.
    2. Select graded prompt assets and the planner/response/delegate models.
    3. Use inline XML tool commands only, so planning can stream progressively.
    4. Keep raw tool results in a result store and only expose summaries/result_ids
         back to the model unless it explicitly asks for more detail.

This module owns the outward chat-completions/SSE surface while the planning
and tool loop live in llms.orchestration.engine.

Design constraint:
    bili-search must not rely on provider function calling. The active tool
    protocol is inline XML only because it is stream-friendly, user-visible in
    SSE, and more controllable for multi-step planning.
"""

import json
import re
import threading
import time
import uuid

from tclogger import logger
from typing import Optional

from llms.intent import build_intent_profile
from llms.intent.focus import select_primary_focus_term
from llms.messages import extract_message_text
from llms.chat.content import _sanitize_content
from llms.chat.response_context import ChatResponseContextMixin
from llms.chat.streaming import ChatStreamingMixin
from llms.orchestration import ChatOrchestrator
from llms.orchestration.tool_markup import parse_tool_argument
from llms.orchestration.tool_markup import parse_xml_commands
from llms.orchestration.tool_markup import strip_tool_commands
from llms.models import LLMClient, ChatResponse, create_llm_client
from llms.planning import OwnerResolutionMixin, ToolPlanningMixin
from llms.tools.executor import ToolExecutor
from llms.tools.video_lookup import coerce_search_video_lookup_arguments
from llms.prompts.copilot import build_system_prompt_profile
from llms.runtime.usage import accumulate_usage

# Maximum tool-calling iterations to prevent infinite loops.
MAX_TOOL_ITERATIONS = 4

# Thinking mode keeps a little more headroom, but still avoids long loops.
MAX_TOOL_ITERATIONS_THINKING = 7

# Nudge message injected before forcing content generation
_FORCE_CONTENT_NUDGE = (
    "你已经进行了充分的搜索。请根据以上搜索结果直接回答用户的问题。"
    "不要再输出任何工具命令（如 <search_videos/> 或 <search_owners/>）。"
    "如果搜索结果不完全匹配，就根据已有信息给出最佳回答。"
)

_OWNER_CONTEXT_MAX_RESULTS = 8
_OWNER_CONTEXT_TOPIC_MAX_RESULTS = 20
_VIDEO_CONTEXT_MAX_HITS = 20

_DUPLICATE_TOOL_NUDGE = (
    "相同或等价的搜索已经执行过，请不要重复输出同样的工具命令。"
    "请直接基于已有搜索结果回答用户问题；只有在查询条件明显不同或补充了新信息时，才继续搜索。"
)

_EMPTY_CONTENT_FALLBACK = "抱歉，我拿到了搜索结果，但这次没能整理成答案。请重试。"


class ChatHandler(
    ChatResponseContextMixin,
    ChatStreamingMixin,
    OwnerResolutionMixin,
    ToolPlanningMixin,
):
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
        small_llm_client: LLMClient | None = None,
        model_registry=None,
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
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.verbose = verbose
        self.search_capabilities = self.tool_executor.get_search_capabilities()
        self.small_llm_client = small_llm_client or llm_client
        self.model_registry = model_registry
        self.orchestrator = ChatOrchestrator(
            llm_client=llm_client,
            small_llm_client=self.small_llm_client,
            tool_executor=self.tool_executor,
            model_registry=model_registry,
            temperature=temperature,
            verbose=verbose,
        )

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
        accumulate_usage(total, new)

    @staticmethod
    def _parse_tool_argument(raw_value: str):
        return parse_tool_argument(raw_value)

    @staticmethod
    def _parse_tool_commands(content: str) -> list[dict]:
        """Parse inline XML tool commands from assistant content.

        The active bili-search contract is XML-only. Do not add provider
        function-calling parsing back into this layer.
        """
        return parse_xml_commands(content, tool_names=_SUPPORTED_TOOL_NAMES)

    @staticmethod
    def _get_latest_user_text(messages: list[dict]) -> str:
        recent_user_texts = ChatHandler._get_recent_user_texts(messages, limit=1)
        return recent_user_texts[0] if recent_user_texts else ""

    @staticmethod
    def _get_recent_user_texts(
        messages: list[dict], limit: int | None = None
    ) -> list[str]:
        texts = []
        for message in reversed(messages or []):
            if message.get("role") != "user":
                continue
            content = extract_message_text(message)
            if content.startswith("[搜索结果]"):
                continue
            if content in (_FORCE_CONTENT_NUDGE, _DUPLICATE_TOOL_NUDGE):
                continue
            if not content:
                continue
            texts.append(content)
            if limit is not None and len(texts) >= limit:
                break
        return texts

    @staticmethod
    def _has_tool_results_context(messages: list[dict]) -> bool:
        for message in messages or []:
            if message.get("role") != "user":
                continue
            content = extract_message_text(message)
            if content.startswith("[搜索结果]"):
                return True
        return False

    @classmethod
    def _extract_timeline_author_name(cls, messages: list[dict]) -> str | None:
        intent = build_intent_profile(messages)
        if intent.final_target != "videos" or intent.task_mode != "repeat":
            return None
        name = select_primary_focus_term(
            [
                *(intent.explicit_entities or []),
                *(intent.explicit_topics or []),
            ]
        )
        if not name:
            return None
        if len(name) > 24:
            return None
        return name

    @staticmethod
    def _extract_recent_window(text: str) -> str:
        source = "".join(str(text or "").split())
        if not source:
            return "30d"

        unit_scale = {"天": 1, "日": 1, "周": 7, "月": 30}
        chinese_digits = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        index = 0
        while index < len(source):
            value = None
            if source[index].isdigit():
                start = index
                while index < len(source) and source[index].isdigit():
                    index += 1
                value = int(source[start:index])
            elif source[index] in chinese_digits:
                value = chinese_digits[source[index]]
                index += 1
            else:
                index += 1
                continue

            if index < len(source) and source[index] == "个":
                index += 1
            if index < len(source) and source[index] in unit_scale:
                return f"{value * unit_scale[source[index]]}d"
        return "30d"

    @staticmethod
    def _split_search_query_tokens(text: str) -> list[str]:
        tokens: list[str] = []
        buffer: list[str] = []
        in_quotes = False
        for char in str(text or ""):
            if char == '"':
                buffer.append(char)
                in_quotes = not in_quotes
                continue
            if char.isspace() and not in_quotes:
                token = "".join(buffer).strip()
                if token:
                    tokens.append(token)
                buffer = []
                continue
            buffer.append(char)
        token = "".join(buffer).strip()
        if token:
            tokens.append(token)
        return tokens

    @classmethod
    def _normalize_search_video_query(cls, query: str) -> str | None:
        text = (query or "").strip()
        if not text:
            return None

        tokens = cls._split_search_query_tokens(text)
        cleaned_tokens: list[str] = []
        for token in tokens:
            raw = token.strip("，。！？?；;、（）()[]{}")
            if not raw:
                continue
            if (
                raw.startswith(":")
                or raw.startswith("q=")
                or (raw[0] in "+-" and len(raw) > 1)
            ):
                cleaned_tokens.append(raw)
                continue
            if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
                inner = cls._normalize_entity_focused_query_text(raw[1:-1])
                if inner:
                    cleaned_tokens.append(f'"{inner}"')
                continue

            raw = cls._normalize_entity_focused_query_text(raw)
            if not raw:
                continue
            cleaned_tokens.append(raw)

        if not cleaned_tokens:
            return None

        normalized = " ".join(" ".join(cleaned_tokens).split()).strip()
        if normalized in {"q=vwr", "q=wv"}:
            return None
        return normalized or None

    @classmethod
    def _normalize_search_video_commands(cls, commands: list[dict]) -> list[dict]:
        normalized_commands = []
        for command in commands or []:
            if command.get("type") != "search_videos":
                normalized_commands.append(command)
                continue

            args = dict(command.get("args") or {})
            lookup_args = coerce_search_video_lookup_arguments(args)
            if lookup_args is not None:
                normalized_commands.append(
                    {
                        "type": "search_videos",
                        "args": lookup_args,
                    }
                )
                continue
            queries = args.get("queries")
            if isinstance(queries, str):
                queries = [queries]
            elif queries is None and isinstance(args.get("query"), str):
                queries = [args.get("query")]

            if not isinstance(queries, list):
                normalized_commands.append(command)
                continue

            cleaned_queries: list[str] = []
            for query in queries:
                cleaned = cls._normalize_search_video_query(str(query))
                if cleaned and cleaned not in cleaned_queries:
                    cleaned_queries.append(cleaned)

            if not cleaned_queries:
                continue

            normalized_commands.append(
                {
                    "type": "search_videos",
                    "args": {
                        **args,
                        "queries": cleaned_queries,
                    },
                }
            )
        return normalized_commands

    @staticmethod
    def _strip_tool_commands(content: str) -> str:
        """Remove inline tool commands from content, keeping analysis text."""
        return strip_tool_commands(content, tool_names=_SUPPORTED_TOOL_NAMES).strip()

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
            handler = self.tool_executor._handlers.get(cmd_type)
            if handler is None:
                continue
            result = handler(args)
            results.append({"type": cmd_type, "args": args, "result": result})
            if self.verbose:
                result_str = json.dumps(result, ensure_ascii=False)
                logger.mesg(f"  {cmd_type} → {len(result_str)} chars")
        return results

    @staticmethod
    def _format_tool_args(args: dict) -> str:
        return ", ".join(
            f"{key}={json.dumps(value, ensure_ascii=False)}"
            for key, value in args.items()
        )

    @staticmethod
    def _format_results_message(results: list[dict]) -> str:
        """Format tool results as a user message for conversation injection."""
        parts = ["[搜索结果]"]
        for r in results:
            cmd_type = r["type"]
            args = r["args"]
            result = ChatHandler._compact_result_for_context(r["result"])
            formatted_args = ChatHandler._format_tool_args(args)
            parts.append(f"\n{cmd_type}({formatted_args}):")
            parts.append(json.dumps(result, ensure_ascii=False))
        return "\n".join(parts)

    @staticmethod
    def _compact_result_for_context(result: dict) -> dict:
        if not isinstance(result, dict):
            return result
        if "hits" in result:
            compact_hits = []
            visible_hits = (result.get("hits") or [])[:_VIDEO_CONTEXT_MAX_HITS]
            for hit in visible_hits:
                owner = hit.get("owner") or {}
                compact_hit = {
                    "title": hit.get("title", ""),
                    "bvid": hit.get("bvid", ""),
                    "owner": owner.get("name", ""),
                    "view": ((hit.get("stat") or {}).get("view")),
                    "pub": hit.get("pub_to_now_str") or hit.get("pubdate_str", ""),
                    "dur": hit.get("duration_str", ""),
                    "tags": hit.get("tags", ""),
                }
                compact_hits.append(
                    {k: v for k, v in compact_hit.items() if v not in (None, "")}
                )
            compact = {
                "query": result.get("query", ""),
                "total_hits": result.get("total_hits", len(compact_hits)),
                "hits": compact_hits,
            }
            omitted_hits = max(
                0,
                int(result.get("total_hits", len(result.get("hits") or [])))
                - len(compact_hits),
            )
            if omitted_hits:
                compact["omitted_hits"] = omitted_hits
            if result.get("error"):
                compact["error"] = result["error"]
            return compact
        if "results" in result and isinstance(result.get("results"), list):
            nested_results = result.get("results") or []
            if any(
                isinstance(item, dict) and ("hits" in item or "total_hits" in item)
                for item in nested_results
            ):
                compact_video_results = []
                for item in nested_results[:4]:
                    compact_video_results.append(
                        ChatHandler._compact_result_for_context(item)
                    )
                compact = {"results": compact_video_results}
                if result.get("error"):
                    compact["error"] = result["error"]
                return compact
        if "owners" in result:
            owner_limit = (
                _OWNER_CONTEXT_TOPIC_MAX_RESULTS
                if result.get("mode") == "topic"
                else _OWNER_CONTEXT_MAX_RESULTS
            )
            return {
                "text": result.get("text", ""),
                "total_owners": result.get(
                    "total_owners", len(result.get("owners") or [])
                ),
                "owners": [
                    {
                        "name": owner.get("name", ""),
                        "mid": owner.get("mid"),
                        "score": owner.get("score", 0),
                    }
                    for owner in (result.get("owners") or [])[:owner_limit]
                ],
            }
        if "options" in result:
            return {
                "text": result.get("text", ""),
                "total_options": result.get(
                    "total_options", len(result.get("options") or [])
                ),
                "options": [
                    {
                        "text": option.get("text", ""),
                        "score": option.get("score", 0),
                    }
                    for option in (result.get("options") or [])[:4]
                ],
            }
        if "results" in result and isinstance(result.get("results"), list):
            compact_results = []
            for item in (result.get("results") or [])[:3]:
                compact_item = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "domain": item.get("domain", ""),
                    "site_kind": item.get("site_kind", ""),
                    "snippet": item.get("snippet", ""),
                    "bvid": item.get("bvid"),
                    "mid": item.get("mid"),
                    "article_id": item.get("article_id", ""),
                }
                compact_results.append(
                    {k: v for k, v in compact_item.items() if v not in (None, "")}
                )
            compact = {"results": compact_results}
            if result.get("query"):
                compact["query"] = result["query"]
            if result.get("result_count") is not None:
                compact["result_count"] = result["result_count"]
            if result.get("backend"):
                compact["backend"] = result["backend"]
            if result.get("error"):
                compact["error"] = result["error"]
            return compact
        return result

    @staticmethod
    def _prune_old_results_messages(messages: list[dict]):
        result_indexes = [
            idx
            for idx, message in enumerate(messages)
            if message.get("role") == "user"
            and extract_message_text(message).startswith("[搜索结果]")
        ]
        for idx in reversed(result_indexes[:-1]):
            del messages[idx]

    @staticmethod
    def _canonicalize_command_value(value):
        if isinstance(value, dict):
            return {
                key: ChatHandler._canonicalize_command_value(val)
                for key, val in sorted(value.items())
            }
        if isinstance(value, list):
            canonical_items = [
                ChatHandler._canonicalize_command_value(item) for item in value
            ]
            sortable = all(
                isinstance(item, (str, int, float, bool)) or item is None
                for item in canonical_items
            )
            return sorted(canonical_items) if sortable else canonical_items
        return value

    @classmethod
    def _command_signature(cls, command: dict) -> str:
        payload = {
            "type": command.get("type", ""),
            "args": cls._canonicalize_command_value(command.get("args", {})),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @classmethod
    def _dedupe_commands(
        cls, commands: list[dict], executed_signatures: set[str] | None = None
    ) -> tuple[list[dict], int]:
        executed_signatures = executed_signatures or set()
        filtered = []
        removed_count = 0
        seen_signatures = set(executed_signatures)
        for command in commands or []:
            signature = cls._command_signature(command)
            if signature in seen_signatures:
                removed_count += 1
                continue
            seen_signatures.add(signature)
            filtered.append(command)
        return filtered, removed_count

    @classmethod
    def _record_executed_command_signatures(
        cls, commands: list[dict], executed_signatures: set[str]
    ):
        for command in commands or []:
            executed_signatures.add(cls._command_signature(command))

    @staticmethod
    def _summarize_context(messages: list[dict]) -> dict:
        summary = {
            "message_count": len(messages or []),
            "user_message_count": 0,
            "assistant_message_count": 0,
            "system_message_count": 0,
            "result_message_count": 0,
            "context_chars": 0,
            "result_context_chars": 0,
        }
        for message in messages or []:
            role = message.get("role") or ""
            content = extract_message_text(message)
            summary["context_chars"] += len(content)
            if role == "user":
                summary["user_message_count"] += 1
                if content.startswith("[搜索结果]"):
                    summary["result_message_count"] += 1
                    summary["result_context_chars"] += len(content)
            elif role == "assistant":
                summary["assistant_message_count"] += 1
            elif role == "system":
                summary["system_message_count"] += 1
        return summary

    @classmethod
    def _build_usage_trace_entry(
        cls,
        *,
        phase: str,
        iteration: int,
        messages: list[dict],
        usage: dict,
        commands: list[dict] | None = None,
    ) -> dict:
        normalized_usage = cls._normalize_usage(usage or {})
        tool_names = [cmd.get("type") for cmd in commands or [] if cmd.get("type")]
        entry = {
            "phase": phase,
            "iteration": iteration,
            **cls._summarize_context(messages),
            "prompt_tokens": normalized_usage.get("prompt_tokens", 0),
            "completion_tokens": normalized_usage.get("completion_tokens", 0),
            "total_tokens": normalized_usage.get("total_tokens", 0),
            "prompt_cache_hit_tokens": normalized_usage.get(
                "prompt_cache_hit_tokens", 0
            ),
            "prompt_cache_miss_tokens": normalized_usage.get(
                "prompt_cache_miss_tokens", 0
            ),
            "reasoning_tokens": normalized_usage.get("reasoning_tokens", 0),
            "tool_count": len(tool_names),
            "tool_names": tool_names,
        }
        return entry

    @staticmethod
    def _finalize_usage_trace(
        prompt_profile: dict,
        usage_trace_entries: list[dict],
        preflight_commands: list[dict] | None = None,
    ) -> dict:
        prompt_meta = dict(prompt_profile or {})
        preflight_tool_names = [
            cmd.get("type") for cmd in (preflight_commands or []) if cmd.get("type")
        ]
        prompt_meta["preflight_tools"] = preflight_tool_names

        peak_prompt_tokens = max(
            (entry.get("prompt_tokens", 0) for entry in usage_trace_entries), default=0
        )
        peak_context_chars = max(
            (entry.get("context_chars", 0) for entry in usage_trace_entries), default=0
        )
        tool_iterations = sum(
            1 for entry in usage_trace_entries if entry.get("tool_count", 0) > 0
        )
        return {
            "prompt": prompt_meta,
            "iterations": usage_trace_entries,
            "summary": {
                "llm_calls": len(usage_trace_entries),
                "tool_iterations": tool_iterations,
                "peak_prompt_tokens": peak_prompt_tokens,
                "peak_context_chars": peak_context_chars,
            },
        }

    def handle(
        self,
        messages: list[dict],
        temperature: float = None,
        thinking: bool = False,
        max_iterations: int = None,
    ) -> dict:
        """Handle a chat completion request (non-streaming).

        Delegates planning and tool execution to ChatOrchestrator,
        then formats the result into the outward chat-completions payload.

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
        result = self.orchestrator.run(
            messages=messages,
            thinking=thinking,
            max_iterations=max_iterations,
        )
        final_content = self._ensure_response_context(
            messages,
            _sanitize_content(result.content or ""),
        )
        elapsed_seconds = time.perf_counter() - start_time
        normalized_usage = self._normalize_usage(result.usage)
        perf_stats = self._compute_perf_stats(normalized_usage, elapsed_seconds)

        return self._format_completion(
            request_id=request_id,
            content=final_content,
            usage=normalized_usage,
            perf_stats=perf_stats,
            tool_events=result.tool_events,
            usage_trace=result.usage_trace,
            thinking=thinking,
        )
