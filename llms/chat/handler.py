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

from llms.chat.owner_resolution import OwnerResolutionMixin
from llms.chat.tool_planning import ToolPlanningMixin
from llms.llm_client import LLMClient, ChatResponse, create_llm_client
from llms.tools.executor import ToolExecutor
from llms.prompts.copilot import build_system_prompt, build_system_prompt_profile

# Maximum tool-calling iterations to prevent infinite loops.
MAX_TOOL_ITERATIONS = 4

# Thinking mode keeps a little more headroom, but still avoids long loops.
MAX_TOOL_ITERATIONS_THINKING = 7

# Regex to strip leaked DeepSeek DSML function-calling markup from content
_DSML_PATTERN = re.compile(r"<｜.*?｜>")
_DSML_BLOCK_PATTERN = re.compile(
    r"<｜DSML｜function_calls>.*?</｜DSML｜function_calls>", re.DOTALL
)

_SUPPORTED_TOOL_NAMES: tuple[str, ...] = (
    "search_videos",
    "search_google",
    "search_owners",
    "related_tokens_by_tokens",
    "related_owners_by_tokens",
    "related_videos_by_videos",
    "related_owners_by_videos",
    "related_videos_by_owners",
    "related_owners_by_owners",
)
_SUPPORTED_TOOL_NAME_PATTERN = "|".join(_SUPPORTED_TOOL_NAMES)
_TOOL_ATTRS_PATTERN = r"(?:[^\"'/>]|\"[^\"]*\"|'[^']*')*"
_GENERIC_TOOL_CMD_RE = re.compile(
    rf"""<(?P<name>{_SUPPORTED_TOOL_NAME_PATTERN})\s*(?P<attrs>{_TOOL_ATTRS_PATTERN})/>""",
    re.DOTALL,
)
_TOOL_ATTR_RE = re.compile(r"""(\w+)=(['\"])(.*?)\2""", re.DOTALL)
_TOOL_CMD_PATTERN = re.compile(
    rf"""<(?:{_SUPPORTED_TOOL_NAME_PATTERN})\s{_TOOL_ATTRS_PATTERN}/>""",
    re.DOTALL,
)

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
_TOOL_PREFIXES: tuple[str, ...] = tuple(f"<{name}" for name in _SUPPORTED_TOOL_NAMES)
_MAX_TOOL_PREFIX_LEN: int = max(len(p) for p in _TOOL_PREFIXES)  # 14

_AUTHOR_TIMELINE_NAME_RE = re.compile(
    r"^(?:请问|麻烦问下|想看)?(?P<name>.+?)(?:最近|最新|近\d+[天日周月])"
)
_RECENT_VIDEO_INTENT_RE = re.compile(
    r"最近|最新|近\d+[天日周月]|近期|这几天|刚发|刚更新|新发"
)
_VIDEO_SEARCH_INTENT_RE = re.compile(
    r"视频|播放|剧情解析|解说|教程|攻略|推荐几条|找几条|热门|高播放|代表作"
)
_LEADING_QUERY_FILLER_RE = re.compile(
    r"^(?:请问|麻烦(?:帮我)?|帮我|给我|我想看|我想找|想看|想找|找一下|找找|搜一下|搜搜|查一下|查查|推荐(?:一下|几个|一些)?|看看)+"
)
_INLINE_QUERY_FILLER_RE = re.compile(
    r"(?:有没有|有无|有什么|有哪些|介绍一下|介绍下|讲一下|讲下|说说|来点|就行|即可|帮我|给我|我想看|我想找|相关的|相关|辅助的|辅助内容|口语化|口语的)"
)
_TRAILING_QUERY_FILLER_RE = re.compile(r"(?:一下|呢|吗|么|吧|呀|啊)+$")
_REDUNDANT_QUERY_TOKENS = {
    "视频",
    "一下",
    "介绍",
    "内容",
    "相关",
    "辅助",
    "辅助的",
}
_USER_FILTER_RE = re.compile(r":user=([^\s]+)")
_IDENTITY_OWNER_QUERY_RE = re.compile(
    r"^(?:请问|想知道|想了解|麻烦问下|麻烦问一下|能说说|告诉我)?\s*(?P<name>.+?)\s*(?:是谁|是什么人|是哪个up主|是哪位up主|是哪个博主|是哪位博主|是什么账号|是做什么的)(?:[呢吗嘛呀啊?？!！]*)$"
)
_TOKEN_OPTION_CANONICAL_RE = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff]+")
_GOOGLE_KEYWORD_BOOTSTRAP_RE = re.compile(
    r"怎么搜|怎么写标题|怎么写题目|搜什么关键词|摸一下关键词|摸关键词|标题写法"
)
_GOOGLE_CREATOR_BOOTSTRAP_RE = re.compile(
    r"不知道作者叫什么|不知道up主叫什么|不知道UP主叫什么|先帮我摸几个up主|先帮我摸几个UP主|先帮我找几个up主|先帮我找几个UP主|先帮我摸几个作者|先帮我找几个作者|谁在做|有哪些up主|有哪些UP主|有哪些作者|做这类内容的up主|做这类内容的UP主"
)


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
    # Remove full DSML blocks first
    content = _DSML_BLOCK_PATTERN.sub("", content)
    # Remove any remaining DSML tags
    content = _DSML_PATTERN.sub("", content)
    # Remove any leaked tool commands
    content = _TOOL_CMD_PATTERN.sub("", content)
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
    def _parse_tool_argument(raw_value: str):
        value = (raw_value or "").strip()
        if not value:
            return ""
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            if value[0] in ['[', '{', '"']:
                return json.loads(value)
        except Exception:
            pass
        if re.fullmatch(r"-?(?:0|[1-9]\d*)", value):
            return int(value)
        if re.fullmatch(r"-?\d+\.\d+", value):
            return float(value)
        return value

    @staticmethod
    def _parse_tool_commands(content: str) -> list[dict]:
        """Parse inline tool commands from LLM response text.

        Scans for supported XML tool tags.

        Returns:
            List of command dicts with 'type' and 'args' keys.
        """
        commands = []
        for match in _GENERIC_TOOL_CMD_RE.finditer(content):
            name = match.group("name")
            attrs = match.group("attrs") or ""
            args = {}
            for attr_match in _TOOL_ATTR_RE.finditer(attrs):
                key = attr_match.group(1)
                value = attr_match.group(3)
                args[key] = ChatHandler._parse_tool_argument(value)
            commands.append({"type": name, "args": args})
        return commands

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
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text:
            return None

        match = _AUTHOR_TIMELINE_NAME_RE.search(latest_user_text)
        if not match:
            return None

        name = match.group("name").strip(" ，。！？?：:")
        if not name:
            return None
        if any(sep in name for sep in ["和", "跟", "与", "、", ",", "，"]):
            return None
        if len(name) > 24:
            return None
        return name

    @staticmethod
    def _extract_recent_window(text: str) -> str:
        source = (text or "").strip()
        if re.search(r"一个月|1个?月|30天|近月", source):
            return "30d"
        if re.search(r"一周|1周|7天|近周", source):
            return "7d"
        day_match = re.search(r"(\d+)[天日]", source)
        if day_match:
            return f"{day_match.group(1)}d"
        week_match = re.search(r"(\d+)周", source)
        if week_match:
            return f"{int(week_match.group(1)) * 7}d"
        month_match = re.search(r"(\d+)个?月", source)
        if month_match:
            return f"{int(month_match.group(1)) * 30}d"
        return "15d"

    @classmethod
    def _ensure_author_timeline_context(
        cls, messages: list[dict], content: str
    ) -> str:
        final_content = (content or "").strip()
        if not final_content:
            return final_content

        author_name = cls._extract_timeline_author_name(messages)
        if not author_name or author_name in final_content:
            return final_content
        return f"{author_name}最近视频：\n{final_content}"

    @staticmethod
    def _normalize_entity_focused_query_text(text: str) -> str:
        query = (text or "").strip()
        if not query:
            return ""
        query = re.sub(r"[？?！!。，“”,、；;：:]", " ", query)
        query = re.sub(
            r"^(请问|麻烦|帮我|给我|我想看|想看|想找|找一下|找找|推荐一下|推荐几个|推荐一些|推荐|搜一下|搜搜|看看|了解一下)",
            "",
            query,
        )
        query = re.sub(r"(有什么|有哪些|有无|有没有|怎么|是什么)", " ", query)
        query = re.sub(r"(介绍一下|介绍下|讲一下|讲下|说说|就行|即可)", " ", query)
        query = re.sub(r"(一下|呢|吗|么|吧|呀|啊)$", "", query)
        query = re.sub(r"\bB站\b", " ", query)
        query = re.sub(r"\s+", " ", query)
        return query.strip(" ，。！？?：:")

    @classmethod
    def _normalize_search_video_query(cls, query: str) -> str | None:
        text = (query or "").strip()
        if not text:
            return None

        tokens = re.findall(r'"[^"]*"|\S+', text)
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

            raw = _LEADING_QUERY_FILLER_RE.sub("", raw)
            raw = _INLINE_QUERY_FILLER_RE.sub(" ", raw)
            raw = _TRAILING_QUERY_FILLER_RE.sub("", raw)
            raw = cls._normalize_entity_focused_query_text(raw)
            if not raw or raw in _REDUNDANT_QUERY_TOKENS:
                continue
            cleaned_tokens.append(raw)

        if not cleaned_tokens:
            return None

        normalized = re.sub(r"\s+", " ", " ".join(cleaned_tokens)).strip()
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
    def _clean_followup_refinement(text: str) -> str | None:
        refinement = (text or "").strip()
        if not refinement:
            return None
        refinement = re.sub(r"^(那|那就|那再|再|更偏|偏向|偏|更想看|想看|主要看)", "", refinement)
        refinement = re.sub(r"(有没有|有无).*$", "", refinement)
        refinement = refinement.strip(" ，。！？?：:")
        refinement = re.sub(r"(这类|这种|类似的|的呢|呢|吗|么|的话|还有吗)$", "", refinement)
        refinement = re.sub(r"\bB站\b", "", refinement)
        refinement = refinement.strip(" ，。！？?：:")
        refinement = re.sub(r"\s+", " ", refinement)
        if refinement.endswith("侧"):
            refinement = refinement[:-1].strip()
        if len(refinement) < 2:
            return None
        return refinement

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

    def _preflight_tool_commands(self, messages: list[dict]) -> list[dict]:
        latest_user_text = self._get_latest_user_text(messages)
        if not latest_user_text:
            return []

        match = _IDENTITY_OWNER_QUERY_RE.match(latest_user_text.strip())
        if not match:
            return []

        name = (match.group("name") or "").strip(" ，。！？?：:")
        if not name:
            return []

        if len(name) > 40:
            return []

        return [{"type": "search_owners", "args": {"text": name, "mode": "name"}}]

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
        enable_thinking_override = True if thinking else None

        # Resolve max_iterations: explicit override > thinking default > normal default
        resolved_iterations = max_iterations
        if resolved_iterations is None:
            resolved_iterations = (
                MAX_TOOL_ITERATIONS_THINKING if thinking else self.max_iterations
            )

        full_messages = self._build_messages(messages, thinking=thinking)
        prompt_profile = build_system_prompt_profile(self.search_capabilities)
        if thinking:
            prompt_profile = {
                **prompt_profile,
                "thinking_prompt_chars": len(_THINKING_PROMPT),
                "total_chars": prompt_profile.get("total_chars", 0)
                + len(_THINKING_PROMPT),
            }
        total_usage = {}
        last_usage = {}  # Track last LLM call's usage for prompt_tokens
        tool_events = []
        usage_trace_entries = []
        executed_signatures: set[str] = set()
        final_content = None
        last_tool_results: list[dict] | None = None
        owner_result_context: dict[str, dict] = {}
        duplicate_tool_nudged = False

        preflight_commands = self._preflight_tool_commands(full_messages)
        if preflight_commands:
            preflight_commands, _ = self._dedupe_commands(preflight_commands, executed_signatures)
            tool_names = [cmd["type"] for cmd in preflight_commands]
            tool_events.append(
                {"iteration": 0, "tools": tool_names, "preflight": True}
            )
            results = self._execute_tool_commands(preflight_commands)
            last_tool_results = results
            owner_result_context = self._merge_owner_result_context(
                owner_result_context,
                results,
            )
            self._record_executed_command_signatures(preflight_commands, executed_signatures)
            full_messages.append({"role": "assistant", "content": "我先检索相关结果。"})
            self._prune_old_results_messages(full_messages)
            full_messages.append(
                {
                    "role": "user",
                    "content": self._format_results_message(results),
                }
            )
            if self.verbose:
                logger.warn(
                    f"> Preflight tool execution: {', '.join(tool_names)}"
                )

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
            owner_result_scope = self._resolve_owner_result_scope(
                owner_result_context,
                last_tool_results,
            )
            commands = self._parse_tool_commands(content)
            commands = self._plan_tool_commands(
                commands,
                full_messages,
                last_tool_results,
                owner_result_scope,
            )
            commands, duplicate_count = self._dedupe_commands(commands, executed_signatures)
            usage_trace_entries.append(
                self._build_usage_trace_entry(
                    phase="tool_loop",
                    iteration=iteration + 1,
                    messages=full_messages,
                    usage=response.usage,
                    commands=commands,
                )
            )

            if duplicate_count and not commands:
                full_messages.append({"role": "assistant", "content": content})
                if duplicate_tool_nudged:
                    logger.warn(
                        "> Duplicate tool commands repeated after nudge, forcing content generation"
                    )
                    full_messages.append({"role": "user", "content": _FORCE_CONTENT_NUDGE})
                    response = self.llm_client.chat(
                        messages=full_messages,
                        temperature=temp,
                    )
                    self._accumulate_usage(total_usage, response.usage)
                    last_usage = response.usage
                    usage_trace_entries.append(
                        self._build_usage_trace_entry(
                            phase="forced_content",
                            iteration=iteration + 2,
                            messages=full_messages,
                            usage=response.usage,
                            commands=[],
                        )
                    )
                    final_content = response.content or final_content
                    break

                duplicate_tool_nudged = True
                full_messages.append({"role": "user", "content": _DUPLICATE_TOOL_NUDGE})
                if self.verbose:
                    logger.warn(
                        "> Suppressed duplicate tool commands and requested direct answer"
                    )
                continue

            if commands:
                duplicate_tool_nudged = False
                tool_names = [cmd["type"] for cmd in commands]
                tool_events.append({"iteration": iteration + 1, "tools": tool_names})
                results = self._execute_tool_commands(commands)
                last_tool_results = results
                owner_result_context = self._merge_owner_result_context(
                    owner_result_context,
                    results,
                )
                self._record_executed_command_signatures(commands, executed_signatures)
                full_messages.append({"role": "assistant", "content": content})
                self._prune_old_results_messages(full_messages)
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
                enable_thinking=enable_thinking_override,
            )
            self._accumulate_usage(total_usage, response.usage)
            last_usage = response.usage
            usage_trace_entries.append(
                self._build_usage_trace_entry(
                    phase="forced_content",
                    iteration=resolved_iterations + 1,
                    messages=full_messages,
                    usage=response.usage,
                    commands=[],
                )
            )
            final_content = (
                response.content or final_content or "[抱歉，处理超时，请重试]"
            )

        final_content = _sanitize_content(final_content)
        if not final_content and self._has_tool_results_context(full_messages):
            full_messages.append({"role": "user", "content": _FORCE_CONTENT_NUDGE})
            response = self.llm_client.chat(
                messages=full_messages,
                temperature=temp,
                enable_thinking=enable_thinking_override,
            )
            self._accumulate_usage(total_usage, response.usage)
            last_usage = response.usage
            usage_trace_entries.append(
                self._build_usage_trace_entry(
                    phase="empty_content_retry",
                    iteration=len(usage_trace_entries) + 1,
                    messages=full_messages,
                    usage=response.usage,
                    commands=[],
                )
            )
            final_content = _sanitize_content(response.content or "")
        if not final_content:
            final_content = _EMPTY_CONTENT_FALLBACK
        final_content = self._ensure_author_timeline_context(full_messages, final_content)

        elapsed_seconds = time.perf_counter() - start_time
        elapsed_ms = round(elapsed_seconds * 1000, 1)
        if self.verbose:
            logger.success(f"> Chat completed in {elapsed_ms}ms")

        # Use last call's prompt_tokens + accumulated completion_tokens
        final_usage = self._merge_final_usage(total_usage, last_usage)
        normalized_usage = self._normalize_usage(final_usage)
        perf_stats = self._compute_perf_stats(normalized_usage, elapsed_seconds)
        usage_trace = self._finalize_usage_trace(
            prompt_profile,
            usage_trace_entries,
            preflight_commands=preflight_commands,
        )

        return self._format_completion(
            request_id=request_id,
            content=final_content,
            usage=normalized_usage,
            perf_stats=perf_stats,
            tool_events=tool_events,
            usage_trace=usage_trace,
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
        enable_thinking_override = True if thinking else None

        # Resolve max_iterations: explicit override > thinking default > normal default
        resolved_iterations = max_iterations
        if resolved_iterations is None:
            resolved_iterations = (
                MAX_TOOL_ITERATIONS_THINKING if thinking else self.max_iterations
            )

        full_messages = self._build_messages(messages, thinking=thinking)
        prompt_profile = build_system_prompt_profile(self.search_capabilities)
        if thinking:
            prompt_profile = {
                **prompt_profile,
                "thinking_prompt_chars": len(_THINKING_PROMPT),
                "total_chars": prompt_profile.get("total_chars", 0)
                + len(_THINKING_PROMPT),
            }
        total_usage = {}
        last_usage = {}  # Track last LLM call's usage for prompt_tokens
        usage_trace_entries = []
        executed_signatures: set[str] = set()
        tool_events_summary: list[dict] = []
        last_tool_results: list[dict] | None = None
        owner_result_context: dict[str, dict] = {}
        preflight_commands = self._preflight_tool_commands(full_messages)

        # First chunk: role + metadata
        yield self._format_stream_chunk(
            request_id=request_id,
            delta={"role": "assistant", "content": ""},
            thinking=thinking,
        )

        if preflight_commands:
            preflight_commands, _ = self._dedupe_commands(
                preflight_commands,
                executed_signatures,
            )
            if preflight_commands:
                tool_names = [cmd["type"] for cmd in preflight_commands]
                tool_events_summary.append(
                    {"iteration": 0, "tools": tool_names, "preflight": True}
                )

                pending_calls = [
                    {
                        "type": cmd["type"],
                        "args": cmd.get("args", {}),
                        "status": "pending",
                    }
                    for cmd in preflight_commands
                ]
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta={},
                    tool_events=[
                        {
                            "iteration": 0,
                            "tools": tool_names,
                            "calls": pending_calls,
                            "preflight": True,
                        }
                    ],
                )

                results = self._execute_tool_commands(preflight_commands)
                last_tool_results = results
                owner_result_context = self._merge_owner_result_context(
                    owner_result_context,
                    results,
                )
                self._record_executed_command_signatures(
                    preflight_commands,
                    executed_signatures,
                )

                completed_calls = [
                    {
                        "type": result["type"],
                        "args": result["args"],
                        "status": "completed",
                        "result": result["result"],
                    }
                    for result in results
                ]
                yield self._format_stream_chunk(
                    request_id=request_id,
                    delta={},
                    tool_events=[
                        {
                            "iteration": 0,
                            "tools": tool_names,
                            "calls": completed_calls,
                            "preflight": True,
                        }
                    ],
                )

                full_messages.append({"role": "assistant", "content": "我先检索相关结果。"})
                self._prune_old_results_messages(full_messages)
                full_messages.append(
                    {
                        "role": "user",
                        "content": self._format_results_message(results),
                    }
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
        duplicate_tool_nudged = False

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
            # iteration.  When True, we still stream answer content, but hide
            # any leading prefix that duplicates the visible reasoning text or
            # earlier tool-analysis text.
            has_reasoning = False
            accumulated_reasoning = ""
            iter_usage = {}
            llm_finish_reason = None

            for chunk in self.llm_client.chat_stream(
                messages=full_messages,
                temperature=temp,
                enable_thinking=enable_thinking_override,
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

                # Track the LLM provider's finish_reason (e.g. "length" for truncation)
                chunk_finish = choices[0].get("finish_reason")
                if chunk_finish:
                    llm_finish_reason = chunk_finish

                # Stream reasoning_content to frontend in real-time (unchanged)
                reasoning_delta = delta.get("reasoning_content", "")
                if reasoning_delta:
                    has_reasoning = True
                    accumulated_reasoning += reasoning_delta
                    yield self._format_stream_chunk(
                        request_id=request_id,
                        delta={"reasoning_content": reasoning_delta},
                    )

                # Accumulate content and stream it in real-time with look-ahead.
                content_delta = delta.get("content") or ""
                if content_delta:
                    accumulated_content += content_delta

                if content_delta and not tool_prefix_detected:
                    hidden_prefix_len = _leading_duplicate_prefix_len(
                        accumulated_content,
                        accumulated_reasoning,
                        *tool_analyses,
                    )
                    visible_start = max(content_sent_ptr, hidden_prefix_len)

                    # Always run tool-prefix detection on accumulated content
                    tool_start = _find_tool_command_start(accumulated_content)
                    if tool_start is not None:
                        if tool_start > visible_start:
                            safe_text = accumulated_content[visible_start:tool_start]
                            if safe_text:
                                yield self._format_stream_chunk(
                                    request_id=request_id,
                                    delta={"content": safe_text},
                                )
                                content_sent_ptr = tool_start
                        tool_prefix_detected = True
                    else:
                        # No full tool prefix yet.  If the tail of accumulated
                        # content could be the start of a tool tag, withhold
                        # the last _MAX_TOOL_PREFIX_LEN chars as a look-ahead
                        # guard; otherwise yield everything accumulated so far.
                        if _has_partial_tool_prefix(accumulated_content):
                            safe_end = max(
                                visible_start,
                                len(accumulated_content) - _MAX_TOOL_PREFIX_LEN,
                            )
                        else:
                            safe_end = len(accumulated_content)

                        if safe_end > visible_start:
                            yield self._format_stream_chunk(
                                request_id=request_id,
                                delta={
                                    "content": accumulated_content[
                                        visible_start:safe_end
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
            owner_result_scope = self._resolve_owner_result_scope(
                owner_result_context,
                last_tool_results,
            )
            commands = self._parse_tool_commands(content)
            commands = self._plan_tool_commands(
                commands,
                full_messages,
                last_tool_results,
                owner_result_scope,
            )
            commands, duplicate_count = self._dedupe_commands(commands, executed_signatures)
            usage_trace_entries.append(
                self._build_usage_trace_entry(
                    phase="tool_loop",
                    iteration=iteration + 1,
                    messages=full_messages,
                    usage=iter_usage,
                    commands=commands,
                )
            )

            if duplicate_count and not commands:
                full_messages.append({"role": "assistant", "content": content})
                if duplicate_tool_nudged:
                    logger.warn(
                        "> Duplicate tool commands repeated after nudge, forcing content generation"
                    )
                    full_messages.append({"role": "user", "content": _FORCE_CONTENT_NUDGE})
                    break

                duplicate_tool_nudged = True
                full_messages.append({"role": "user", "content": _DUPLICATE_TOOL_NUDGE})
                if self.verbose:
                    logger.warn(
                        "> Suppressed duplicate tool commands and requested direct answer"
                    )
                continue

            if commands:
                duplicate_tool_nudged = False
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
                tool_events_summary.append(
                    {"iteration": iteration + 1, "tools": tool_names}
                )
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
                last_tool_results = results
                owner_result_context = self._merge_owner_result_context(
                    owner_result_context,
                    results,
                )
                self._record_executed_command_signatures(commands, executed_signatures)

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
                self._prune_old_results_messages(full_messages)
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
                hidden_prefix_len = _leading_duplicate_prefix_len(
                    content,
                    accumulated_reasoning,
                    *tool_analyses,
                )
                yielded_final_content = content_sent_ptr > hidden_prefix_len
                if content_sent_ptr < len(content):
                    remaining = _sanitize_content(
                        content[max(content_sent_ptr, hidden_prefix_len) :]
                    )
                    if remaining:
                        if _is_cancelled():
                            yield "[DONE]"
                            return
                        yield self._format_stream_chunk(
                            request_id=request_id,
                            delta={"content": remaining},
                        )
                        yielded_final_content = True

                if not yielded_final_content and self._has_tool_results_context(full_messages):
                    if self.verbose:
                        logger.warn(
                            "> Final answer was empty after tool results, forcing content generation"
                        )
                    full_messages.append({"role": "user", "content": _FORCE_CONTENT_NUDGE})
                    break

                # Finalize: compute stats and yield final chunk
                final_usage = self._merge_final_usage(total_usage, last_usage)
                normalized_usage = self._normalize_usage(final_usage)
                elapsed_seconds = time.perf_counter() - start_time
                perf_stats = self._compute_perf_stats(normalized_usage, elapsed_seconds)
                usage_trace = self._finalize_usage_trace(
                    prompt_profile,
                    usage_trace_entries,
                    preflight_commands=[],
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
                    usage_trace=usage_trace,
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
        phase2_finish_reason = None
        for chunk in self.llm_client.chat_stream(
            messages=full_messages,
            temperature=temp,
            enable_thinking=enable_thinking_override,
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
            chunk_finish = choices[0].get("finish_reason")
            if chunk_finish:
                phase2_finish_reason = chunk_finish

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
            usage_trace_entries.append(
                self._build_usage_trace_entry(
                    phase="forced_content",
                    iteration=resolved_iterations + 1,
                    messages=full_messages,
                    usage=stream_usage,
                    commands=[],
                )
            )

        if not _sanitize_content(final_content):
            final_content = _EMPTY_CONTENT_FALLBACK
            yield self._format_stream_chunk(
                request_id=request_id,
                delta={"content": final_content},
            )

        final_usage = self._merge_final_usage(total_usage, last_usage)
        normalized_usage = self._normalize_usage(final_usage)
        elapsed_seconds = time.perf_counter() - start_time
        perf_stats = self._compute_perf_stats(normalized_usage, elapsed_seconds)
        usage_trace = self._finalize_usage_trace(
            prompt_profile,
            usage_trace_entries,
            preflight_commands=[],
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
            usage_trace=usage_trace,
        )

        yield "[DONE]"

    def _build_messages(
        self, user_messages: list[dict], thinking: bool = False
    ) -> list[dict]:
        """Prepend system prompt to user messages.

        When thinking=True, prepends an additional thinking prompt
        to encourage deeper analysis.
        """
        system_content = build_system_prompt(capabilities=self.search_capabilities)
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
        normalized_usage = ChatHandler._normalize_usage(usage)
        completion_tokens = normalized_usage.get("completion_tokens", 0)

        tokens_per_second = (
            int(completion_tokens / elapsed_seconds)
            if elapsed_seconds > 0 and completion_tokens > 0
            else 0
        )

        elapsed_str = dt_to_str(elapsed_seconds, precision=0)

        result = {
            "tokens_per_second": tokens_per_second,
            "total_elapsed": elapsed_str,
            "total_elapsed_ms": round(elapsed_seconds * 1000, 1),
        }

        cache_hit_tokens = normalized_usage.get("prompt_cache_hit_tokens")
        cache_miss_tokens = normalized_usage.get("prompt_cache_miss_tokens")
        if cache_hit_tokens is not None:
            result["prompt_cache_hit_tokens"] = cache_hit_tokens
        if cache_miss_tokens is not None:
            result["prompt_cache_miss_tokens"] = cache_miss_tokens

        return result

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
