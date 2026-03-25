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

_AUTHOR_TIMELINE_HINT_RE = re.compile(
    r"最近|最新|新视频|新投稿|近\d+[天日周月]|时间线|投稿列表|更新了什么"
)
_AUTHOR_TIMELINE_NAME_RE = re.compile(
    r"^(?:请问|麻烦问下|想看)?(?P<name>.+?)(?:最近|最新|近\d+[天日周月])"
)
_AMBIGUOUS_AUTHOR_HINT_RE = re.compile(r"[A-Za-z0-9]")
_MULTI_CREATOR_COMPARE_RE = re.compile(
    r"^(?:请|帮我|麻烦)?(?:对比(?:一下)?|比较(?:一下)?|对比下)?(?P<left>.+?)(?:和|跟|与|vs|VS)(?P<right>.+?)(?:最近|近\d+[天日周月]|过去\d+[天日周月])"
)
_CREATOR_DISCOVERY_HINT_RE = re.compile(
    r"UP主|创作者|作者|博主|谁发的|谁做的|谁在做|推荐几个做|有哪些做|做.+内容的"
)
_CREATOR_DISCOVERY_REQUEST_RE = re.compile(
    r"^(推荐几个|推荐一些|有哪些|有没有|帮我找|找几个|找一些|谁在做|谁做的)"
)
_CREATOR_DISCOVERY_FOLLOWUP_RE = re.compile(
    r"更偏|偏.+(呢|吗)|这类|这种|类似的|再来几个|还有吗|主要看"
)
_CREATOR_META_HINT_RE = re.compile(
    r"关联账号|账号矩阵|矩阵号|主号|副号|小号|分身|马甲|别的号|其他账号|另一个号|另一个账号"
)
_CREATOR_META_FOLLOWUP_RE = re.compile(
    r"还有(别的|其他).*(号|账号)|另一个(号|账号)|主号呢|副号呢|小号呢|矩阵呢"
)
_GENERIC_CREATOR_REF_RE = re.compile(
    r"^(他|她|它|ta|TA|这位|这个|这个人|该(?:位)?)(?:UP主|up主|作者|创作者|博主|人)?$"
)
_SIMILAR_CREATOR_HINT_RE = re.compile(
    r"风格接近|类似的UP主|像.+的UP主|同类型.*UP主|同风格|接近的创作者"
)
_OFFICIAL_INFO_HINT_RE = re.compile(
    r"官方更新|更新日志|更新了什么|官方公告|官网|release notes|changelog|发布说明|API 更新|模型更新"
)
_BILIBILI_DECODE_HINT_RE = re.compile(
    r"B站.*解读|B站.*视频|有没有.*解读|有没有.*视频|相关解读|相关视频"
)
_EXTERNAL_SEARCH_FOLLOWUP_RE = re.compile(
    r"更偏开发者|更偏产品|API 侧|应用侧|官网|官方|B站.*解读|有没有.*解读|有没有.*视频"
)
_OFFICIAL_ONLY_FOLLOWUP_RE = re.compile(
    r"只看官网|只看官方|官网就行|官方就行|只要官网|只要官方|先不用B站|先不用视频|不用B站解读|不用看视频"
)
_MISSING_RESULTS_HINT_RE = re.compile(
    r"没收到|未收到|没有收到|搜索结果|工具链路|接口|重试|系统返回"
)
_VIDEO_SEARCH_INTENT_RE = re.compile(
    r"视频|播放|剧情解析|解说|教程|攻略|推荐几条|找几条|热门|高播放|代表作"
)
_EXPLICIT_VIDEO_REQUEST_RE = re.compile(
    r"^(推荐几条|找几条|帮我找|给我找|想看|推荐|找|有没有)"
)
_CREATOR_VIDEO_FOLLOWUP_RE = re.compile(
    r"^(?:那|那他|那她|那这个|那这位|他|她|这位|这个UP主|这个作者|这个博主).*(?:代表作|最近.*视频|最新.*视频|高播放|热门视频)"
)
_MULTI_CREATOR_COMPARE_HINT_RE = re.compile(
    r"谁更高产|谁发得更多|谁更新更多|谁更新更勤|谁更活跃|哪个更高产"
)
_SEARCH_PLEDGE_HINT_RE = re.compile(
    r"我来搜索|我先帮你搜|我来帮你搜|我先帮你把|我来帮你找|我先帮你找|我来先找|我先找一下|先找一下"
)
_EXTERNAL_SEARCH_PLEDGE_HINT_RE = re.compile(
    r"我先查|我来查|先查一下|先看一下|联网|Google|官网|官方"
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
        if re.fullmatch(r"-?\d+", value):
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

    @staticmethod
    def _normalize_name_key(text: str) -> str:
        return re.sub(r"\s+", "", (text or "").strip()).lower()

    @classmethod
    def _should_resolve_timeline_author_first(
        cls, author_name: str, latest_user_text: str = ""
    ) -> bool:
        normalized = cls._normalize_name_key(author_name)
        if not normalized:
            return False
        if _AMBIGUOUS_AUTHOR_HINT_RE.search(normalized):
            return True
        latest_text = (latest_user_text or "").strip()
        if re.search(r"对应|到底是|是不是|应该是|可能是|还是|节目|系列|作品", latest_text):
            return True
        return False

    @classmethod
    def _build_timeline_owner_lookup_commands(cls, messages: list[dict]) -> list[dict]:
        author_name = cls._extract_timeline_author_name(messages)
        if not author_name:
            return []
        latest_user_text = cls._get_latest_user_text(messages)
        if not cls._should_resolve_timeline_author_first(author_name, latest_user_text):
            return []
        return [
            {"type": "search_owners", "args": {"text": author_name, "mode": "name"}}
        ]

    @classmethod
    def _extract_multi_creator_compare_queries(cls, messages: list[dict]) -> list[str]:
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text or not _MULTI_CREATOR_COMPARE_HINT_RE.search(latest_user_text):
            return []

        match = _MULTI_CREATOR_COMPARE_RE.search(latest_user_text)
        if not match:
            return []

        left = match.group("left").strip(" ，。！？?：:")
        right = match.group("right").strip(" ，。！？?：:")
        right = re.sub(r"(?:最近|近\d+[天日周月]|过去\d+[天日周月]).*$", "", right).strip(
            " ，。！？?：:"
        )
        if not left or not right:
            return []

        window = cls._extract_recent_window(latest_user_text)
        return [f":user={left} :date<={window}", f":user={right} :date<={window}"]

    @classmethod
    def _normalize_author_timeline_commands(
        cls, commands: list[dict], messages: list[dict]
    ) -> list[dict]:
        author_name = cls._extract_timeline_author_name(messages)
        if not author_name:
            return commands

        latest_user_text = cls._get_latest_user_text(messages)
        if cls._should_resolve_timeline_author_first(author_name, latest_user_text):
            return commands

        normalized = []
        window = cls._extract_recent_window(cls._get_latest_user_text(messages) or "")
        timeline_query = f":user={author_name} :date<={window}"
        for command in commands or []:
            if command.get("type") != "search_videos":
                normalized.append(command)
                continue

            args = dict(command.get("args") or {})
            queries = args.get("queries")
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list) or not queries:
                normalized.append(command)
                continue

            normalized.append(
                {
                    "type": "search_videos",
                    "args": {
                        **args,
                        "queries": [timeline_query],
                    },
                }
            )
        return normalized

    @classmethod
    def _normalize_multi_creator_compare_commands(
        cls, commands: list[dict], messages: list[dict]
    ) -> list[dict]:
        compare_queries = cls._extract_multi_creator_compare_queries(messages)
        if not compare_queries:
            return commands

        normalized = []
        for command in commands or []:
            if command.get("type") != "search_videos":
                normalized.append(command)
                continue

            args = dict(command.get("args") or {})
            queries = args.get("queries")
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list) or not queries:
                normalized.append(command)
                continue
            if all(":user=" in str(query) for query in queries) and len(queries) >= 2:
                normalized.append(command)
                continue

            normalized.append(
                {
                    "type": "search_videos",
                    "args": {
                        **args,
                        "queries": compare_queries,
                    },
                }
            )
        return normalized

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

    @classmethod
    def _extract_token_rewrite_from_results(
        cls, results: list[dict] | None
    ) -> tuple[str, str] | None:
        for result_item in results or []:
            if result_item.get("type") != "related_tokens_by_tokens":
                continue
            result = result_item.get("result") or {}
            source_text = str(
                result.get("text") or result_item.get("args", {}).get("text") or ""
            ).strip()
            if not source_text:
                continue
            for option in result.get("options") or []:
                candidate = str(option.get("text") or "").strip()
                if candidate and candidate != source_text:
                    return source_text, candidate
        return None

    @classmethod
    def _extract_owner_rewrite_from_results(
        cls,
        results: list[dict] | None,
        messages: list[dict],
    ) -> dict | None:
        author_name = cls._extract_timeline_author_name(messages)
        if not author_name:
            return None
        author_key = cls._normalize_name_key(author_name)
        for result_item in results or []:
            if result_item.get("type") != "search_owners":
                continue
            result = result_item.get("result") or {}
            source_text = str(
                result.get("text") or result_item.get("args", {}).get("text") or ""
            ).strip()
            source_key = cls._normalize_name_key(source_text)
            if source_key and source_key != author_key:
                continue
            owners = result.get("owners") or []
            if not owners:
                continue
            top_owner = owners[0]
            owner_name = str(top_owner.get("name") or "").strip()
            owner_mid = top_owner.get("mid")
            if not owner_name and not owner_mid:
                continue
            return {
                "source_text": source_text or author_name,
                "name": owner_name,
                "mid": owner_mid,
            }
        return None

    @classmethod
    def _build_owner_resolved_timeline_query(
        cls,
        messages: list[dict],
        owner_rewrite: dict | None,
        query_text: str = "",
    ) -> str | None:
        if not owner_rewrite:
            return None
        latest_user_text = cls._get_latest_user_text(messages)
        author_name = cls._extract_timeline_author_name(messages)
        if not author_name or not latest_user_text:
            return None
        query_source = str(query_text or latest_user_text)
        window = cls._extract_recent_window(query_source)
        owner_mid = owner_rewrite.get("mid")
        owner_name = str(owner_rewrite.get("name") or "").strip()
        if owner_mid:
            return f":uid={int(owner_mid)} :date<={window}"
        if owner_name:
            return f":user={owner_name} :date<={window}"
        return None

    @classmethod
    def _build_token_assisted_video_query(
        cls, messages: list[dict], token_rewrite: tuple[str, str] | None
    ) -> str | None:
        if not token_rewrite:
            return None
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text or not _VIDEO_SEARCH_INTENT_RE.search(latest_user_text):
            return None

        source_text, canonical_text = token_rewrite
        rewritten = latest_user_text.replace(source_text, canonical_text)
        rewritten = cls._normalize_entity_focused_query_text(rewritten)
        if not rewritten:
            rewritten = canonical_text
        if canonical_text not in rewritten:
            rewritten = f"{canonical_text} {rewritten}".strip()
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        if not rewritten:
            return None
        if "q=" not in rewritten and ":user=" not in rewritten and ":uid=" not in rewritten:
            rewritten = f"{rewritten} q=vwr"
        return rewritten

    @classmethod
    def _normalize_token_assisted_search_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        token_rewrite = cls._extract_token_rewrite_from_results(last_tool_results)
        if not token_rewrite:
            return commands

        source_text, _ = token_rewrite
        replacement_query = cls._build_token_assisted_video_query(messages, token_rewrite)
        if not replacement_query:
            return commands

        normalized = []
        for command in commands or []:
            if command.get("type") != "search_videos":
                normalized.append(command)
                continue
            args = dict(command.get("args") or {})
            queries = args.get("queries")
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list) or not queries:
                normalized.append(command)
                continue
            if not any(source_text in str(query) for query in queries):
                normalized.append(command)
                continue
            normalized.append(
                {
                    "type": "search_videos",
                    "args": {
                        **args,
                        "queries": [replacement_query],
                    },
                }
            )
        return normalized

    @classmethod
    def _normalize_owner_assisted_timeline_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        owner_rewrite = cls._extract_owner_rewrite_from_results(last_tool_results, messages)
        if not owner_rewrite:
            return commands

        source_text = str(owner_rewrite.get("source_text") or "").strip()
        source_key = cls._normalize_name_key(source_text)
        normalized = []
        for command in commands or []:
            if command.get("type") != "search_videos":
                normalized.append(command)
                continue
            args = dict(command.get("args") or {})
            queries = args.get("queries")
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list) or not queries:
                normalized.append(command)
                continue

            should_replace = False
            for query in queries:
                query_text = str(query or "")
                query_key = cls._normalize_name_key(query_text)
                if source_key and source_key in query_key:
                    should_replace = True
                    break
                if ":user=" in query_text and source_text and source_text in query_text:
                    should_replace = True
                    break
            if not should_replace:
                normalized.append(command)
                continue

            resolved_query = cls._build_owner_resolved_timeline_query(
                messages,
                owner_rewrite,
                query_text=" ".join(str(query or "") for query in queries),
            )
            if not resolved_query:
                normalized.append(command)
                continue

            normalized.append(
                {
                    "type": "search_videos",
                    "args": {
                        **args,
                        "queries": [resolved_query],
                    },
                }
            )
        return normalized

    @classmethod
    def _fallback_token_assisted_search_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        if commands or not last_tool_results:
            return commands
        token_rewrite = cls._extract_token_rewrite_from_results(last_tool_results)
        query = cls._build_token_assisted_video_query(messages, token_rewrite)
        if not query:
            return commands
        return [{"type": "search_videos", "args": {"queries": [query]}}]

    @classmethod
    def _fallback_owner_assisted_timeline_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        if commands or not last_tool_results:
            return commands
        owner_rewrite = cls._extract_owner_rewrite_from_results(last_tool_results, messages)
        query = cls._build_owner_resolved_timeline_query(messages, owner_rewrite)
        if not query:
            return commands
        return [{"type": "search_videos", "args": {"queries": [query]}}]

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

    @classmethod
    def _extract_creator_topic_from_text(cls, text: str) -> str | None:
        topic = (text or "").strip()
        if not topic:
            return None
        meta_match = re.search(
            r"^(?P<name>.+?)(?:有哪[些个]|有没有|都有哪些|还有)?(?:关联账号|账号矩阵|矩阵号|主号|副号|小号|分身|马甲|别的号|其他账号|另一个号|另一个账号).*$",
            topic,
        )
        if meta_match:
            name = meta_match.group("name").strip(" ，。！？?：:")
            name = re.sub(r"^(那|那位|那这个|那这位)", "", name).strip(" ，。！？?：:")
            if _GENERIC_CREATOR_REF_RE.fullmatch(name):
                return None
            return name or None
        request_match = re.search(
            r"^(?P<topic>.+?)(?:有哪[些个]|有没有|找几个|找一些|推荐几个|推荐一些|谁在做|谁做的).*(?:UP主|创作者|作者|博主).*$",
            topic,
        )
        if request_match:
            candidate = request_match.group("topic").strip(" ，。！？?：:")
            candidate = re.sub(r"^(那|那位|那这个|那这位)", "", candidate).strip(" ，。！？?：:")
            if candidate:
                return candidate
        topic = re.sub(
            r"^(推荐几个|推荐一些|有哪些|有没有|谁发的|谁做的|帮我找|找一下|找找|找几个|找一些)",
            "",
            topic,
        )
        topic = re.sub(r"B站上.*$", "", topic)
        topic = re.sub(r"(做|讲|聊)", "", topic, count=1)
        topic = re.sub(r"(内容)?的?(UP主|创作者|作者|博主).*$", "", topic)
        topic = re.sub(r"是谁发的.*$", "", topic)
        topic = topic.strip(" ，。！？?：:")
        topic = re.sub(r"^(那|那位|那这个|那这位)", "", topic).strip(" ，。！？?：:")
        if _GENERIC_CREATOR_REF_RE.fullmatch(topic):
            return None
        if _CREATOR_META_FOLLOWUP_RE.search(topic) and re.match(r"^(他|她|它|ta|TA)", topic):
            return None
        return topic or None

    @classmethod
    def _extract_similar_creator_seed(cls, messages: list[dict]) -> str | None:
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text:
            return None

        patterns = [
            r"(?:和|跟)(?P<name>.+?)(?:风格接近|类似|同类型|同风格)",
            r"像(?P<name>.+?)这样的(?:UP主|创作者|作者|博主)",
        ]
        for pattern in patterns:
            match = re.search(pattern, latest_user_text)
            if not match:
                continue
            name = match.group("name").strip(" ，。！？?：:")
            if name:
                return name
        return None

    @classmethod
    def _extract_external_subject_from_text(cls, text: str) -> str | None:
        subject = (text or "").strip()
        if not subject:
            return None
        subject = re.sub(r"B站上有没有.*$", "", subject)
        subject = re.sub(r"有没有.*解读.*$", "", subject)
        subject = re.sub(r"最近有哪些官方更新.*$", "", subject)
        subject = re.sub(r"官方更新|更新日志|更新了什么|官方公告|官网", "", subject)
        subject = re.sub(r"release notes|changelog|发布说明|模型更新", "", subject, flags=re.I)
        subject = subject.strip(" ，。！？?：:")
        return subject or None

    @classmethod
    def _has_recent_user_intent(cls, messages: list[dict], pattern: re.Pattern, limit: int = 4) -> bool:
        recent_user_texts = cls._get_recent_user_texts(messages, limit=limit)
        return any(pattern.search(text) for text in recent_user_texts)

    def _fallback_tool_commands(
        self,
        commands: list[dict],
        messages: list[dict],
        content: str = "",
    ) -> list[dict]:
        if commands:
            return commands
        if self._has_tool_results_context(messages):
            return commands
        latest_user_text = self._get_latest_user_text(messages)
        if not latest_user_text:
            return commands
        if _OFFICIAL_INFO_HINT_RE.search(latest_user_text) or _BILIBILI_DECODE_HINT_RE.search(
            latest_user_text
        ):
            return commands
        if not _AUTHOR_TIMELINE_HINT_RE.search(latest_user_text):
            return commands
        if not _MISSING_RESULTS_HINT_RE.search(content or ""):
            return commands

        author_name = self._extract_timeline_author_name(messages)
        if not author_name:
            return commands

        owner_lookup_commands = self._build_timeline_owner_lookup_commands(messages)
        if owner_lookup_commands:
            if self.verbose:
                logger.warn(
                    "> Injecting owner-lookup fallback before author timeline video search"
                )
            return owner_lookup_commands

        fallback = [
            {
                "type": "search_videos",
                "args": {"queries": [f":user={author_name} :date<=15d"]},
            }
        ]
        if self.verbose:
            logger.warn(
                "> Injecting fallback tool commands for explicit author timeline request"
            )
        return fallback

    @classmethod
    def _extract_creator_discovery_topic(cls, messages: list[dict]) -> str | None:
        recent_user_texts = cls._get_recent_user_texts(messages, limit=4)
        if not recent_user_texts:
            return None

        latest_user_text = recent_user_texts[0]
        current_topic = cls._extract_creator_topic_from_text(latest_user_text)
        has_explicit_latest_creator_intent = bool(
            _CREATOR_DISCOVERY_HINT_RE.search(latest_user_text)
            or _CREATOR_DISCOVERY_REQUEST_RE.search(latest_user_text)
            or _CREATOR_META_HINT_RE.search(latest_user_text)
        )
        if current_topic and current_topic != latest_user_text and has_explicit_latest_creator_intent:
            return current_topic

        refinement = None
        has_creator_followup = bool(
            _CREATOR_DISCOVERY_FOLLOWUP_RE.search(latest_user_text)
        )
        has_meta_followup = bool(_CREATOR_META_FOLLOWUP_RE.search(latest_user_text))
        has_followup_refinement = has_creator_followup or has_meta_followup
        if has_creator_followup:
            refinement = cls._clean_followup_refinement(latest_user_text)

        if current_topic and has_explicit_latest_creator_intent and not has_followup_refinement:
            return current_topic

        for previous_text in recent_user_texts[1:] if has_followup_refinement else []:
            previous_topic = cls._extract_creator_topic_from_text(previous_text)
            if not previous_topic:
                continue
            if refinement and refinement not in previous_topic:
                return f"{previous_topic} {refinement}".strip()
            return previous_topic

        return current_topic or refinement

    def _fallback_creator_discovery_commands(
        self,
        commands: list[dict],
        messages: list[dict],
        content: str = "",
    ) -> list[dict]:
        if commands:
            return commands
        if self._has_tool_results_context(messages):
            return commands
        recent_user_texts = self._get_recent_user_texts(messages, limit=4)
        if not recent_user_texts:
            return commands
        latest_user_text = recent_user_texts[0]
        has_creator_context = bool(
            _CREATOR_DISCOVERY_HINT_RE.search(latest_user_text)
            or _CREATOR_META_HINT_RE.search(latest_user_text)
        )
        if not has_creator_context and (
            _CREATOR_DISCOVERY_FOLLOWUP_RE.search(latest_user_text)
            or _CREATOR_META_FOLLOWUP_RE.search(latest_user_text)
        ):
            has_creator_context = any(
                _CREATOR_DISCOVERY_HINT_RE.search(text)
                or _CREATOR_META_HINT_RE.search(text)
                for text in recent_user_texts[1:]
            )
        has_followup_context = bool(
            _CREATOR_DISCOVERY_FOLLOWUP_RE.search(latest_user_text)
            or _CREATOR_META_FOLLOWUP_RE.search(latest_user_text)
        ) and any(
            _CREATOR_DISCOVERY_HINT_RE.search(text)
            or _CREATOR_META_HINT_RE.search(text)
            for text in recent_user_texts[1:]
        )
        if not has_creator_context and not has_followup_context:
            return commands
        if (
            _VIDEO_SEARCH_INTENT_RE.search(latest_user_text)
            and not _CREATOR_DISCOVERY_HINT_RE.search(latest_user_text)
            and not _CREATOR_META_HINT_RE.search(latest_user_text)
            and not has_followup_context
        ):
            return commands
        is_explicit_creator_request = bool(
            _CREATOR_DISCOVERY_REQUEST_RE.search(latest_user_text)
            or _CREATOR_META_HINT_RE.search(latest_user_text)
        )
        if not (
            _MISSING_RESULTS_HINT_RE.search(content or "")
            or _SEARCH_PLEDGE_HINT_RE.search(content or "")
            or is_explicit_creator_request
            or has_followup_context
        ):
            return commands
        topic = self._extract_creator_discovery_topic(messages)
        if not topic:
            return commands
        owner_mode = "relation" if _CREATOR_META_HINT_RE.search(latest_user_text) else "topic"
        fallback = [{"type": "search_owners", "args": {"text": topic, "mode": owner_mode}}]
        if not _CREATOR_META_HINT_RE.search(latest_user_text):
            fallback.append(
                {"type": "search_videos", "args": {"queries": [f"{topic} q=vwr"]}}
            )
        if self.verbose:
            logger.warn(
                "> Injecting fallback relation command for creator-discovery request"
            )
        return fallback

    @classmethod
    def _extract_external_search_query(cls, messages: list[dict]) -> str | None:
        recent_user_texts = cls._get_recent_user_texts(messages, limit=4)
        if not recent_user_texts:
            return None
        latest_user_text = recent_user_texts[0]
        official_only_followup = cls._wants_official_only_followup(messages)
        if _OFFICIAL_INFO_HINT_RE.search(latest_user_text) and not cls._wants_official_only_followup(
            messages
        ):
            query = re.sub(r"B站上有没有.*$", "", latest_user_text)
            query = re.sub(r"有没有.*解读.*$", "", query)
            query = query.strip(" ，。！？?：:")
            return query or latest_user_text.strip(" ，。！？?：:") or None

        latest_focus = None if official_only_followup else cls._clean_followup_refinement(latest_user_text)
        for previous_text in recent_user_texts[1:]:
            previous_subject = cls._extract_external_subject_from_text(previous_text)
            if not previous_subject:
                continue
            if latest_focus and latest_focus not in previous_subject:
                return f"{previous_subject} {latest_focus} 最近有哪些官方更新".strip()
            return f"{previous_subject} 最近有哪些官方更新"

        latest_subject = cls._extract_external_subject_from_text(latest_user_text)
        if latest_subject:
            if latest_focus and latest_focus not in latest_subject:
                return f"{latest_subject} {latest_focus} 最近有哪些官方更新".strip()
            return f"{latest_subject} 最近有哪些官方更新"
        return None

    @classmethod
    def _wants_official_only_followup(cls, messages: list[dict]) -> bool:
        latest_user_text = cls._get_latest_user_text(messages)
        return bool(latest_user_text and _OFFICIAL_ONLY_FOLLOWUP_RE.search(latest_user_text))

    @classmethod
    def _extract_creator_video_followup_query(cls, messages: list[dict]) -> str | None:
        recent_user_texts = cls._get_recent_user_texts(messages, limit=4)
        if len(recent_user_texts) < 2:
            return None

        latest_user_text = recent_user_texts[0]
        if not _CREATOR_VIDEO_FOLLOWUP_RE.search(latest_user_text):
            return None

        subject_name = None
        for previous_text in recent_user_texts[1:]:
            subject_name = cls._extract_creator_topic_from_text(previous_text)
            if subject_name:
                break
        if not subject_name:
            return None

        query = f":user={subject_name}"
        if "最近" in latest_user_text or "最新" in latest_user_text:
            query = f"{query} :date<=15d"
        return query

    @classmethod
    def _extract_bilibili_topic_queries(cls, messages: list[dict]) -> list[str]:
        recent_user_texts = cls._get_recent_user_texts(messages, limit=4)
        if not recent_user_texts:
            return []
        latest_user_text = recent_user_texts[0]
        base_query = None
        refinement = None

        if _OFFICIAL_INFO_HINT_RE.search(latest_user_text):
            base_query = cls._extract_external_subject_from_text(latest_user_text)
        else:
            refinement = cls._clean_followup_refinement(latest_user_text)
            for previous_text in recent_user_texts[1:]:
                previous_subject = cls._extract_external_subject_from_text(previous_text)
                if previous_subject:
                    base_query = previous_subject
                    break
            if not base_query:
                base_query = cls._extract_external_subject_from_text(latest_user_text)
        if not base_query:
            return []

        query_candidates = []
        for raw_query in [
            base_query,
            f"{base_query} {refinement}".strip() if refinement else None,
        ]:
            if not raw_query:
                continue
            query = raw_query if "q=" in raw_query else f"{raw_query} q=vwr"
            if query not in query_candidates:
                query_candidates.append(query)
        return query_candidates

    def _fallback_external_search_commands(
        self,
        commands: list[dict],
        messages: list[dict],
        content: str = "",
    ) -> list[dict]:
        if commands:
            return commands
        if self._has_tool_results_context(messages):
            return commands
        recent_user_texts = self._get_recent_user_texts(messages, limit=4)
        if not recent_user_texts:
            return commands
        latest_user_text = recent_user_texts[0]
        has_external_context = bool(
            _OFFICIAL_INFO_HINT_RE.search(latest_user_text)
            or _BILIBILI_DECODE_HINT_RE.search(latest_user_text)
        )
        if not has_external_context and _EXTERNAL_SEARCH_FOLLOWUP_RE.search(latest_user_text):
            has_external_context = any(
                _OFFICIAL_INFO_HINT_RE.search(text)
                or _BILIBILI_DECODE_HINT_RE.search(text)
                for text in recent_user_texts[1:]
            )
        if not has_external_context:
            return commands
        if not (
            _MISSING_RESULTS_HINT_RE.search(content or "")
            or _EXTERNAL_SEARCH_PLEDGE_HINT_RE.search(content or "")
        ):
            return commands
        google_query = self._extract_external_search_query(messages)
        bilibili_queries = self._extract_bilibili_topic_queries(messages)
        fallback = []
        if google_query:
            fallback.append({"type": "search_google", "args": {"query": google_query}})
        wants_bilibili_decode = not self._wants_official_only_followup(messages) and (
            bool(_BILIBILI_DECODE_HINT_RE.search(latest_user_text))
            or any(_BILIBILI_DECODE_HINT_RE.search(text) for text in recent_user_texts[1:])
        )
        if bilibili_queries and wants_bilibili_decode:
            fallback.append({"type": "search_videos", "args": {"queries": bilibili_queries}})
        if self.verbose and fallback:
            logger.warn(
                "> Injecting fallback external-search commands for official-update request"
            )
        return fallback or commands

    def _fallback_similar_creator_commands(
        self,
        commands: list[dict],
        messages: list[dict],
        content: str = "",
    ) -> list[dict]:
        if commands:
            return commands
        if self._has_tool_results_context(messages):
            return commands
        latest_user_text = self._get_latest_user_text(messages)
        if not latest_user_text or not _SIMILAR_CREATOR_HINT_RE.search(latest_user_text):
            return commands
        seed_name = self._extract_similar_creator_seed(messages)
        if not seed_name:
            return commands
        if not (
            _MISSING_RESULTS_HINT_RE.search(content or "")
            or _SEARCH_PLEDGE_HINT_RE.search(content or "")
            or _CREATOR_DISCOVERY_REQUEST_RE.search(latest_user_text)
            or "推荐理由" in latest_user_text
        ):
            return commands
        fallback = [
            {"type": "search_owners", "args": {"text": seed_name, "mode": "relation"}},
            {"type": "search_videos", "args": {"queries": [f"{seed_name} q=vwr"]}},
        ]
        if self.verbose:
            logger.warn(
                "> Injecting fallback relation command for similar-creator request"
            )
        return fallback

    @staticmethod
    def _is_relation_command(cmd: dict) -> bool:
        return str(cmd.get("type", "")).startswith("related_")

    @classmethod
    def _derive_relation_search_queries(
        cls, commands: list[dict], messages: list[dict]
    ) -> list[str]:
        latest_user_text = cls._get_latest_user_text(messages)
        if latest_user_text and _CREATOR_META_HINT_RE.search(latest_user_text):
            return []

        queries = []
        topic = cls._extract_creator_discovery_topic(messages)
        if topic:
            queries.append(f"{topic} q=vwr")

        seed_name = cls._extract_similar_creator_seed(messages)
        if seed_name:
            queries.append(f"{seed_name} q=vwr")

        deduped = []
        for query in queries:
            if query and query not in deduped:
                deduped.append(query)
        return deduped[:2]

    @classmethod
    def _promote_relation_only_commands(
        cls, commands: list[dict], messages: list[dict]
    ) -> list[dict]:
        if not commands:
            return commands
        if any(cmd.get("type") == "search_videos" for cmd in commands):
            return commands
        relation_commands = [cmd for cmd in commands if cls._is_relation_command(cmd)]
        if not relation_commands:
            return commands
        if all(cmd.get("type") == "related_tokens_by_tokens" for cmd in relation_commands):
            return commands

        queries = cls._derive_relation_search_queries(commands, messages)
        if not queries:
            return commands
        return list(commands) + [{"type": "search_videos", "args": {"queries": queries}}]

    @classmethod
    def _should_fallback_video_search(cls, messages: list[dict], content: str) -> bool:
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text:
            return False
        if not _VIDEO_SEARCH_INTENT_RE.search(latest_user_text):
            return False
        if _EXPLICIT_VIDEO_REQUEST_RE.search(latest_user_text):
            return True
        return bool(_SEARCH_PLEDGE_HINT_RE.search(content or "")) or bool(
            _MISSING_RESULTS_HINT_RE.search(content or "")
        )

    @classmethod
    def _extract_video_search_query(cls, messages: list[dict]) -> str | None:
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text:
            return None
        query = re.sub(
            r"^(推荐几条|找几条|帮我找|给我找|想看|推荐|找)", "", latest_user_text
        )
        query = query.strip(" ，。！？?：:")
        query = re.sub(r"视频$", "", query).strip()
        if not query:
            return None
        if "q=" not in query:
            query = f"{query} q=vwr"
        return query

    def _fallback_video_search_commands(
        self,
        commands: list[dict],
        messages: list[dict],
        content: str = "",
    ) -> list[dict]:
        if commands:
            return commands
        if self._has_tool_results_context(messages):
            return commands
        compare_queries = self._extract_multi_creator_compare_queries(messages)
        if compare_queries and (
            _SEARCH_PLEDGE_HINT_RE.search(content or "")
            or _MISSING_RESULTS_HINT_RE.search(content or "")
            or _MULTI_CREATOR_COMPARE_HINT_RE.search(self._get_latest_user_text(messages) or "")
        ):
            return [{"type": "search_videos", "args": {"queries": compare_queries}}]
        creator_followup_query = self._extract_creator_video_followup_query(messages)
        if creator_followup_query and (
            _SEARCH_PLEDGE_HINT_RE.search(content or "")
            or _MISSING_RESULTS_HINT_RE.search(content or "")
            or _CREATOR_VIDEO_FOLLOWUP_RE.search(self._get_latest_user_text(messages) or "")
        ):
            return [{"type": "search_videos", "args": {"queries": [creator_followup_query]}}]
        if not self._should_fallback_video_search(messages, content):
            return commands

        query = self._extract_video_search_query(messages)
        if not query:
            return commands
        fallback = [{"type": "search_videos", "args": {"queries": [query]}}]
        if self.verbose:
            logger.warn(
                "> Injecting fallback search_videos command for explicit video search request"
            )
        return fallback

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
            for hit in (result.get("hits") or [])[:4]:
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
            if result.get("error"):
                compact["error"] = result["error"]
            return compact
        if "owners" in result:
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
                    for owner in (result.get("owners") or [])[:4]
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
            return {"results": result.get("results", [])[:2]}
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
        if self._has_tool_results_context(messages):
            return []
        recent_user_texts = self._get_recent_user_texts(messages, limit=4)
        if not recent_user_texts:
            return []
        latest_user_text = recent_user_texts[0]
        has_external_followup = bool(_EXTERNAL_SEARCH_FOLLOWUP_RE.search(latest_user_text)) and any(
            _OFFICIAL_INFO_HINT_RE.search(text)
            or _BILIBILI_DECODE_HINT_RE.search(text)
            for text in recent_user_texts[1:]
        )
        has_direct_external_request = bool(
            _OFFICIAL_INFO_HINT_RE.search(latest_user_text)
            or _BILIBILI_DECODE_HINT_RE.search(latest_user_text)
        )
        if not has_external_followup and not has_direct_external_request:
            timeline_lookup_commands = self._build_timeline_owner_lookup_commands(messages)
            if timeline_lookup_commands:
                return timeline_lookup_commands
            return []
        return self._fallback_external_search_commands(
            [], messages=messages, content="我先查一下官方更新，再看看 B 站解读。"
        )

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

        preflight_commands = self._preflight_tool_commands(full_messages)
        if preflight_commands:
            preflight_commands, _ = self._dedupe_commands(preflight_commands, executed_signatures)
            tool_names = [cmd["type"] for cmd in preflight_commands]
            tool_events.append(
                {"iteration": 0, "tools": tool_names, "preflight": True}
            )
            results = self._execute_tool_commands(preflight_commands)
            last_tool_results = results
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
            commands = self._fallback_tool_commands(
                self._parse_tool_commands(content),
                messages=full_messages,
                content=content,
            )
            commands = self._fallback_creator_discovery_commands(
                commands,
                messages=full_messages,
                content=content,
            )
            commands = self._fallback_external_search_commands(
                commands,
                messages=full_messages,
                content=content,
            )
            commands = self._fallback_similar_creator_commands(
                commands,
                messages=full_messages,
                content=content,
            )
            commands = self._fallback_video_search_commands(
                commands,
                messages=full_messages,
                content=content,
            )
            commands = self._normalize_token_assisted_search_commands(
                commands,
                full_messages,
                last_tool_results,
            )
            commands = self._normalize_owner_assisted_timeline_commands(
                commands,
                full_messages,
                last_tool_results,
            )
            commands = self._fallback_token_assisted_search_commands(
                commands,
                full_messages,
                last_tool_results,
            )
            commands = self._fallback_owner_assisted_timeline_commands(
                commands,
                full_messages,
                last_tool_results,
            )
            commands = self._normalize_author_timeline_commands(commands, full_messages)
            commands = self._normalize_multi_creator_compare_commands(commands, full_messages)
            commands = self._promote_relation_only_commands(commands, full_messages)
            commands = self._normalize_search_video_commands(commands)
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
                full_messages.append({"role": "user", "content": _DUPLICATE_TOOL_NUDGE})
                if self.verbose:
                    logger.warn(
                        "> Suppressed duplicate tool commands and requested direct answer"
                    )
                continue

            if commands:
                tool_names = [cmd["type"] for cmd in commands]
                tool_events.append({"iteration": iteration + 1, "tools": tool_names})
                results = self._execute_tool_commands(commands)
                last_tool_results = results
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
            commands = self._fallback_tool_commands(
                self._parse_tool_commands(content),
                messages=full_messages,
                content=content,
            )
            commands = self._fallback_creator_discovery_commands(
                commands,
                messages=full_messages,
                content=content,
            )
            commands = self._fallback_external_search_commands(
                commands,
                messages=full_messages,
                content=content,
            )
            commands = self._fallback_similar_creator_commands(
                commands,
                messages=full_messages,
                content=content,
            )
            commands = self._fallback_video_search_commands(
                commands,
                messages=full_messages,
                content=content,
            )
            commands = self._normalize_token_assisted_search_commands(
                commands,
                full_messages,
                last_tool_results,
            )
            commands = self._normalize_owner_assisted_timeline_commands(
                commands,
                full_messages,
                last_tool_results,
            )
            commands = self._fallback_token_assisted_search_commands(
                commands,
                full_messages,
                last_tool_results,
            )
            commands = self._fallback_owner_assisted_timeline_commands(
                commands,
                full_messages,
                last_tool_results,
            )
            commands = self._normalize_author_timeline_commands(commands, full_messages)
            commands = self._normalize_multi_creator_compare_commands(commands, full_messages)
            commands = self._promote_relation_only_commands(commands, full_messages)
            commands = self._normalize_search_video_commands(commands)
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
                full_messages.append({"role": "user", "content": _DUPLICATE_TOOL_NUDGE})
                if self.verbose:
                    logger.warn(
                        "> Suppressed duplicate tool commands and requested direct answer"
                    )
                continue

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
                yielded_final_content = bool(
                    not has_reasoning and content_sent_ptr > 0
                )
                if has_reasoning:
                    # Content was buffered (reasoning was streamed instead).
                    # Sanitize echoed results, then strip leading text that
                    # duplicates this iteration's reasoning or a prior tool
                    # analysis to avoid echoing thinking into the answer.
                    final = _sanitize_content(content)
                    # Strip leading text matching current reasoning
                    reasoning_text = accumulated_reasoning.strip()
                    if reasoning_text and final.startswith(reasoning_text):
                        final = final[len(reasoning_text) :].lstrip("\n")
                    # Strip leading text matching a prior tool analysis
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
                        yielded_final_content = True
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
