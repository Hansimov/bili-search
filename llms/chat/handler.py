"""OpenAI-style chat handler backed by the multi-model orchestrator.

The current flow is:
    1. Build an intent profile from the latest conversation turn.
    2. Select graded prompt assets and the planner/response/delegate models.
    3. Run function calling when supported, with XML tool commands kept as fallback.
    4. Keep raw tool results in a result store and only expose summaries/result_ids
         back to the model unless it explicitly asks for more detail.

This module owns the outward chat-completions/SSE surface while the planning
and tool loop live in llms.orchestration.engine.
"""

import json
import re
import threading
import time
import uuid

from tclogger import logger
from typing import Generator, Optional

from llms.intent import build_intent_profile
from llms.intent.focus import build_focus_query
from llms.intent.focus import compact_focus_key
from llms.intent.focus import rewrite_known_term_aliases
from llms.intent.focus import select_primary_focus_term
from llms.orchestration import ChatOrchestrator
from llms.orchestration.tool_markup import EXTERNAL_TOOL_NAMES
from llms.orchestration.tool_markup import EXTERNAL_TOOL_PREFIXES
from llms.orchestration.tool_markup import parse_tool_argument
from llms.orchestration.tool_markup import parse_xml_commands
from llms.orchestration.tool_markup import sanitize_generated_content
from llms.orchestration.tool_markup import strip_tool_commands
from llms.models import LLMClient, ChatResponse, create_llm_client
from llms.planning import OwnerResolutionMixin, ToolPlanningMixin
from llms.tools.executor import ToolExecutor
from llms.prompts.copilot import build_system_prompt, build_system_prompt_profile
from llms.runtime.usage import accumulate_usage, compute_perf_stats, normalize_usage

# Maximum tool-calling iterations to prevent infinite loops.
MAX_TOOL_ITERATIONS = 4

# Thinking mode keeps a little more headroom, but still avoids long loops.
MAX_TOOL_ITERATIONS_THINKING = 7

_SUPPORTED_TOOL_NAMES: tuple[str, ...] = EXTERNAL_TOOL_NAMES
_SUPPORTED_TOOL_NAME_PATTERN = "|".join(_SUPPORTED_TOOL_NAMES)

# Patterns to strip echoed tool results that the LLM may copy from the
# conversation context into its content.  The format comes from
# _format_results_message: "search_videos(queries=[...]):\n{...json...}"
_RESULTS_HEADER_RE = re.compile(r"\[搜索结果\][ \t]*\n?")
_RESULTS_ECHO_RE = re.compile(
    rf"(?:{_SUPPORTED_TOOL_NAME_PATTERN})\([^\n)]*\):[ \t]*\n?\{{[^\n]*\}}",
)

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
_TOOL_PREFIXES: tuple[str, ...] = EXTERNAL_TOOL_PREFIXES


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


def _shared_prefix_len(left: str, right: str) -> int:
    """Return the length of the shared prefix between *left* and *right*."""
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def _leading_duplicate_prefix_len(text: str, *candidates: str) -> int:
    """Return the longest leading prefix in *text* duplicated by any candidate.

    Used while streaming content alongside reasoning_content: many providers
    start the content channel by echoing the same analysis text that is already
    available in reasoning_content or was shown in earlier tool iterations.
    Hiding that shared prefix lets us keep answer tokens incremental without
    flashing duplicate analysis text into the answer area.
    """
    max_len = 0
    for candidate in candidates:
        if not candidate:
            continue
        prefix_len = _shared_prefix_len(text, candidate)
        if prefix_len > max_len:
            max_len = prefix_len
    return max_len


def _sanitize_content(content: str) -> str:
    """Strip leaked markup and echoed tool results from content.

    Handles:
    - DeepSeek DSML function-calling tags
    - Inline XML tool commands (<search_videos/>, <search_owners/>, ...)
    - Echoed tool results in _format_results_message format
    """
    content = sanitize_generated_content(content, tool_names=_SUPPORTED_TOOL_NAMES)
    # Remove echoed tool results (e.g. search_videos(queries=[...]):\n{...})
    content = _RESULTS_HEADER_RE.sub("", content)
    content = _RESULTS_ECHO_RE.sub("", content)
    # Collapse runs of 3+ newlines left by stripped blocks
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = _dedupe_repeated_output(content)
    return content.strip()


def _dedupe_repeated_output(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if paragraphs:
        for repeat in (3, 2):
            if len(paragraphs) >= repeat and len(paragraphs) % repeat == 0:
                unit = len(paragraphs) // repeat
                chunks = [
                    "\n\n".join(paragraphs[index * unit : (index + 1) * unit]).strip()
                    for index in range(repeat)
                ]
                if len(set(chunks)) == 1:
                    return chunks[0]

        deduped: list[str] = []
        previous_norm = None
        for paragraph in paragraphs:
            paragraph_norm = re.sub(r"\s+", " ", paragraph)
            if paragraph_norm == previous_norm:
                continue
            deduped.append(paragraph)
            previous_norm = paragraph_norm
        return "\n\n".join(deduped)

    return text


def _anchor_subject_key(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", compact_focus_key(text))


class ChatHandler(OwnerResolutionMixin, ToolPlanningMixin):
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
        """Parse inline XML tool commands for providers that emit markup calls."""
        return parse_xml_commands(content, tool_names=_SUPPORTED_TOOL_NAMES)

    @staticmethod
    def _get_latest_user_text(messages: list[dict]) -> str:
        recent_user_texts = ChatHandler._get_recent_user_texts(messages, limit=1)
        return recent_user_texts[0] if recent_user_texts else ""

    @staticmethod
    def _get_recent_user_texts(messages: list[dict], limit: int | None = None) -> list[str]:
        texts = []
        for message in reversed(messages or []):
            if message.get("role") != "user":
                continue
            content = (message.get("content") or "").strip()
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
            content = (message.get("content") or "").strip()
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

    @classmethod
    def _ensure_author_timeline_context(
        cls,
        messages: list[dict],
        content: str,
        *,
        intent=None,
    ) -> str:
        final_content = (content or "").strip()
        if not final_content:
            return final_content

        resolved_intent = intent or build_intent_profile(messages)
        if resolved_intent.final_target != "videos" or resolved_intent.task_mode != "repeat":
            return final_content

        author_name = select_primary_focus_term(
            [
                *(resolved_intent.explicit_entities or []),
                *(resolved_intent.explicit_topics or []),
            ]
        )
        if not author_name or author_name in final_content:
            return final_content
        return f"{author_name}最近视频：\n{final_content}"

    @staticmethod
    def _extract_subject_from_latest_user_text(
        latest_user_text: str,
        candidate_keys: set[str],
    ) -> str:
        source = str(latest_user_text or "").strip()
        if not source or not candidate_keys:
            return ""

        segments = [
            (
                match.group(0),
                compact_focus_key(match.group(0)),
                match.start(),
                match.end(),
            )
            for match in re.finditer(
                r"[A-Za-z]+|\d+(?:[./]\d+)*|[\u4e00-\u9fff]+|[^A-Za-z0-9\u4e00-\u9fff\s]+",
                source,
            )
        ]
        if not segments:
            return ""

        best_subject = ""
        best_score = -1
        for index, (_, segment_key, start, end) in enumerate(segments):
            if segment_key not in candidate_keys:
                continue

            matched_keys = [segment_key]
            subject_end = end
            scan_index = index + 1
            while scan_index < len(segments):
                _, next_key, _, next_end = segments[scan_index]
                if not next_key:
                    scan_index += 1
                    continue
                if next_key not in candidate_keys:
                    break
                matched_keys.append(next_key)
                subject_end = next_end
                scan_index += 1

            subject = source[start:subject_end].strip(
                " ，。！？?；;：:、()[]{}<>《》\"'`~!@#$%^&*-+=|\\/"
            )
            score = sum(len(key) for key in dict.fromkeys(matched_keys))
            if score > best_score or (
                score == best_score and len(subject) > len(best_subject)
            ):
                best_subject = subject
                best_score = score

        return best_subject

    @staticmethod
    def _extract_leading_subject_phrase(latest_user_text: str) -> str:
        source = str(latest_user_text or "").strip()
        if not source:
            return ""

        segments = [
            (
                match.group(0),
                compact_focus_key(match.group(0)),
                match.start(),
                match.end(),
            )
            for match in re.finditer(
                r"[A-Za-z]+|\d+(?:[./]\d+)*|[\u4e00-\u9fff]+|[^A-Za-z0-9\u4e00-\u9fff\s]+",
                source,
            )
        ]
        if not segments:
            return ""

        subject_start = None
        subject_end = None
        saw_content = False
        for segment_text, segment_key, start, end in segments:
            is_long_cjk_clause = bool(re.fullmatch(r"[\u4e00-\u9fff]+", segment_text)) and len(segment_text) >= 5
            if not saw_content:
                if not segment_key:
                    continue
                subject_start = start
                subject_end = end
                saw_content = True
                continue
            if not segment_key:
                subject_end = end
                continue
            if is_long_cjk_clause:
                break
            subject_end = end

        if subject_start is None or subject_end is None:
            return ""
        return source[subject_start:subject_end].strip(
            " ，。！？?；;：:、()[]{}<>《》\"'`~!@#$%^&*-+=|\\/"
        )

    @classmethod
    def _ensure_primary_subject_context(
        cls,
        messages: list[dict],
        content: str,
        *,
        intent=None,
    ) -> str:
        final_content = (content or "").strip()
        if not final_content:
            return final_content

        resolved_intent = intent or build_intent_profile(messages)
        if (
            resolved_intent.final_target not in {"external", "mixed"}
            and not (
                resolved_intent.final_target == "videos"
                and resolved_intent.needs_term_normalization
            )
        ):
            return final_content

        candidate_texts = [
            *(resolved_intent.explicit_entities or []),
            *(resolved_intent.explicit_topics or []),
        ]
        candidate_keys = {
            compact_focus_key(candidate)
            for candidate in candidate_texts
            if len(compact_focus_key(candidate)) >= 2
        }
        subject = cls._extract_subject_from_latest_user_text(
            cls._get_latest_user_text(messages),
            candidate_keys,
        ) or select_primary_focus_term(candidate_texts)
        leading_subject = cls._extract_leading_subject_phrase(
            cls._get_latest_user_text(messages)
        )
        subject_key = _anchor_subject_key(subject)
        leading_subject_key = _anchor_subject_key(leading_subject)
        if leading_subject_key and (
            not subject_key or subject_key in leading_subject_key
        ):
            subject = leading_subject
        if resolved_intent.needs_term_normalization and subject:
            subject = rewrite_known_term_aliases(subject) or subject
        subject = " ".join(str(subject or "").split()).strip()
        subject_key = _anchor_subject_key(subject)
        if not subject_key or len(subject) > 48:
            return final_content
        if subject_key in _anchor_subject_key(final_content):
            return final_content
        return f"{subject}：\n{final_content}"

    @classmethod
    def _ensure_response_context(cls, messages: list[dict], content: str) -> str:
        resolved_intent = build_intent_profile(messages)
        final_content = cls._ensure_author_timeline_context(
            messages,
            content,
            intent=resolved_intent,
        )
        return cls._ensure_primary_subject_context(
            messages,
            final_content,
            intent=resolved_intent,
        )

    @staticmethod
    def _normalize_entity_focused_query_text(text: str) -> str:
        return build_focus_query(text)

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
            if raw.startswith(":") or raw.startswith("q=") or (raw[0] in "+-" and len(raw) > 1):
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
                isinstance(item, dict)
                and ("hits" in item or "total_hits" in item)
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
            and str(message.get("content") or "").startswith("[搜索结果]")
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
            canonical_items = [ChatHandler._canonicalize_command_value(item) for item in value]
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
            content = str(message.get("content") or "")
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
        final_content = self._ensure_response_context(messages, result.content)
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

        result = yield from self._relay_orchestration_stream(
            request_id=request_id,
            stream=self.orchestrator.run_stream(
                messages=messages,
                thinking=thinking,
                max_iterations=max_iterations,
                cancelled=cancelled,
            ),
        )

        final_content = self._ensure_response_context(messages, result.content)
        if final_content:
            should_replay_content = (
                not result.content_streamed or final_content != result.content
            )
            if result.content_streamed and final_content != result.content:
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
    ) -> Generator[str, None, object]:
        try:
            while True:
                event = next(stream)
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta=event.get("delta") or {},
                    tool_events=event.get("tool_events"),
                )
        except StopIteration as stop:
            return stop.value

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
