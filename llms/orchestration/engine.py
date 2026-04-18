from __future__ import annotations

import json
import re
import time

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generator, Optional

from tclogger import logger, ts_to_str

from llms.contracts import (
    IntentProfile,
    ModelSpec,
    OrchestrationResult,
    ToolCallRequest,
    ToolExecutionRecord,
)
from llms.intent import build_intent_profile
from llms.intent.focus import rewrite_known_term_aliases
from llms.intent.focus import select_primary_focus_term
from llms.messages import (
    extract_bvids,
    extract_message_text,
    extract_owner_mids,
    normalize_bvid_key,
)
from llms.models import ChatResponse
from llms.models import DEFAULT_SMALL_MODEL_CONFIG, ModelRegistry
from llms.orchestration.policies import FINAL_ANSWER_NUDGE
from llms.orchestration.policies import has_explicit_video_anchor
from llms.orchestration.policies import has_successful_tool_result
from llms.orchestration.policies import has_target_coverage
from llms.orchestration.policies import is_recent_timeline_request
from llms.orchestration.policies import needs_explicit_video_lookup_followup
from llms.orchestration.policies import select_blocked_request_nudge
from llms.orchestration.policies import select_post_execution_nudge
from llms.orchestration.policies import select_pre_execution_nudge
from llms.orchestration.result_store import ResultStore
from llms.orchestration.result_store import inspect_results
from llms.orchestration.result_store import summarize_result
from llms.orchestration.tool_markup import ALL_TOOL_PREFIXES
from llms.orchestration.tool_markup import command_signature
from llms.orchestration.tool_markup import find_tool_command_start
from llms.orchestration.tool_markup import parse_xml_tool_calls
from llms.orchestration.tool_markup import partial_tool_prefix_len
from llms.orchestration.tool_markup import sanitize_generated_content
from llms.planning.owner_resolution import OwnerResolutionMixin
from llms.prompts.assets import get_prompt_assets
from llms.prompts.copilot import build_system_prompt, build_system_prompt_profile
from llms.runtime.usage import accumulate_usage, normalize_usage
from llms.tools.names import canonical_tool_name


_THINKING_PROMPT = (
    "[思考模式] 你现在处于深度思考模式。请认真拆解任务、谨慎选择提示资产、"
    "尽量把复杂规划留给大模型，把窄任务交给小模型。"
)

_TRANSCRIPT_HINT_RE = re.compile(
    r"(讲了什么|讲啥|说了什么|说啥|主要讲|主要内容|总结|摘要|概括|梗概|重点|字幕|转写|音频)",
    re.IGNORECASE,
)
_EXPLICIT_TRANSCRIPT_REQUEST_RE = re.compile(
    r"(音频转写文本|音频转写|转写文本|完整转写|完整字幕|字幕原文|逐字稿|transcript|subtitles?)",
    re.IGNORECASE,
)
_TRANSCRIPT_CONTEXT_CHAR_LIMIT = 24000
_TRANSCRIPT_METADATA_DESC_LIMIT = 1200
_TRANSCRIPT_METADATA_TAG_LIMIT = 12
_TRANSCRIPT_SMALL_TASK_CANONICAL_TASK = "整理转写：主题概括 + 覆盖全片要点"
_TRANSCRIPT_SMALL_TASK_CANONICAL_OUTPUT_FORMAT = "主题概括+中文要点"
_BVID_LOOKUP_QUERY_RE = re.compile(
    r"^(?:bv\s*=\s*)?(BV[0-9A-Za-z]{10})$", re.IGNORECASE
)
_MID_LOOKUP_QUERY_RE = re.compile(
    r"^:?(?:uid|mid)\s*=\s*(\d{4,})(?:\s+:date<=([0-9]+[dwmy]))?$",
    re.IGNORECASE,
)
_RECENT_OWNER_QUERY_PATTERNS = (
    re.compile(
        r"^(?P<subject>.+?)(?:最近|近期|近况)(?:还)?(?:发了|发布了|上传了|更新了)?(?:哪些|什么)?(?:视频|作品).*$"
    ),
    re.compile(
        r"^(?P<subject>.+?)(?:还)?(?:发了|发布了|上传了|更新了)(?:哪些|什么)?(?:视频|作品).*$"
    ),
)
_RECENT_OWNER_SUBJECT_STOP_RE = re.compile(
    r"(最近|近期|近况|视频|作品|作者|发了|发布了|上传了|更新了|还发了|什么|哪些|谁|是谁)",
    re.IGNORECASE,
)


def _shared_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


@dataclass(frozen=True, slots=True)
class ModelDecision:
    client: object
    spec: ModelSpec
    reason: str
    factors: tuple[str, ...]


class ChatOrchestrator:
    def __init__(
        self,
        *,
        llm_client,
        tool_executor,
        small_llm_client=None,
        model_registry: ModelRegistry | None = None,
        temperature: float | None = None,
        verbose: bool = False,
    ):
        self.large_llm_client = llm_client
        self.small_llm_client = small_llm_client or llm_client
        self.tool_executor = tool_executor
        self.model_registry = model_registry or ModelRegistry.from_envs(
            primary_small_config=DEFAULT_SMALL_MODEL_CONFIG
        )
        self.temperature = temperature
        self.verbose = verbose

    @staticmethod
    def _owner_resolution_seed(intent: IntentProfile) -> str:
        return select_primary_focus_term(
            [
                *(intent.explicit_topics or []),
                *(intent.explicit_entities or []),
            ]
        )

    @staticmethod
    def _owner_request_text(arguments: dict) -> str:
        for key in ("text", "topic", "name", "relation", "query"):
            text = str(arguments.get(key, "") or "").strip()
            if text:
                return text
        queries = arguments.get("queries")
        if isinstance(queries, str):
            return queries.strip()
        if isinstance(queries, (list, tuple)):
            for item in queries:
                text = str(item or "").strip()
                if text:
                    return text
        return ""

    @staticmethod
    def _normalize_seed_values(values: Any) -> list[str]:
        if isinstance(values, str):
            text = values.strip()
            return [text] if text else []
        if isinstance(values, (list, tuple, set)):
            return [str(item).strip() for item in values if str(item or "").strip()]
        return []

    @staticmethod
    def _normalize_mid_values(values: Any) -> list[str]:
        normalized: list[str] = []
        for value in ChatOrchestrator._normalize_seed_values(values):
            try:
                normalized_value = str(int(value))
            except (TypeError, ValueError):
                continue
            if normalized_value not in normalized:
                normalized.append(normalized_value)
        return normalized

    @staticmethod
    def _parse_lookup_query(query: str) -> tuple[str, str | None, str | None] | None:
        query_text = str(query or "").strip()
        if not query_text:
            return None

        bvid_match = _BVID_LOOKUP_QUERY_RE.fullmatch(query_text)
        if bvid_match:
            return ("bvid", bvid_match.group(1), None)

        mid_match = _MID_LOOKUP_QUERY_RE.fullmatch(query_text)
        if mid_match:
            return ("mid", str(int(mid_match.group(1))), mid_match.group(2) or None)

        return None

    def _normalize_search_video_lookup_arguments(
        self,
        arguments: dict,
        intent: IntentProfile,
    ) -> dict:
        normalized = dict(arguments or {})
        mode = str(normalized.get("mode", "auto") or "auto").lower()
        if mode == "discover":
            return normalized

        raw_queries = normalized.get("queries")
        if isinstance(raw_queries, str):
            raw_queries = [raw_queries]
        elif not isinstance(raw_queries, list):
            single_query = str(normalized.get("query", "") or "").strip()
            raw_queries = [single_query] if single_query else []

        explicit_bvids: list[str] = []
        explicit_bvid_keys: set[str] = set()
        explicit_mids: list[str] = []
        date_window = str(normalized.get("date_window", "") or "").strip() or None

        for key in ("bv", "bvid"):
            for value in self._normalize_seed_values(normalized.get(key)):
                matches = extract_bvids({"content": value})
                for match in matches:
                    match_key = normalize_bvid_key(match)
                    if match_key not in explicit_bvid_keys:
                        explicit_bvid_keys.add(match_key)
                        explicit_bvids.append(match)

        for value in self._normalize_seed_values(normalized.get("bvids")):
            matches = extract_bvids({"content": value})
            for match in matches:
                match_key = normalize_bvid_key(match)
                if match_key not in explicit_bvid_keys:
                    explicit_bvid_keys.add(match_key)
                    explicit_bvids.append(match)

        for key in ("mid", "uid"):
            for value in self._normalize_mid_values(normalized.get(key)):
                if value not in explicit_mids:
                    explicit_mids.append(value)

        for value in self._normalize_mid_values(normalized.get("mids")):
            if value not in explicit_mids:
                explicit_mids.append(value)

        remaining_queries: list[str] = []
        for query in raw_queries:
            parsed = self._parse_lookup_query(query)
            if parsed is None:
                remaining_queries.append(str(query or "").strip())
                continue
            lookup_kind, lookup_value, query_window = parsed
            if (
                lookup_kind == "bvid"
                and normalize_bvid_key(lookup_value) not in explicit_bvid_keys
            ):
                explicit_bvid_keys.add(normalize_bvid_key(lookup_value))
                explicit_bvids.append(lookup_value)
            elif lookup_kind == "mid" and lookup_value not in explicit_mids:
                explicit_mids.append(lookup_value)
            if query_window and not date_window:
                date_window = query_window

        if remaining_queries or not (explicit_bvids or explicit_mids):
            return normalized

        normalized.pop("query", None)
        normalized.pop("queries", None)
        normalized["mode"] = "lookup"
        if date_window:
            normalized["date_window"] = date_window

        if explicit_bvids:
            if len(explicit_bvids) == 1:
                normalized["bv"] = explicit_bvids[0]
                normalized.pop("bvid", None)
                normalized.pop("bvids", None)
            else:
                normalized["bvids"] = explicit_bvids
                normalized.pop("bv", None)
                normalized.pop("bvid", None)

        if explicit_mids:
            if len(explicit_mids) == 1:
                normalized["mid"] = explicit_mids[0]
                normalized.pop("uid", None)
                normalized.pop("mids", None)
            else:
                normalized["mids"] = explicit_mids
                normalized.pop("mid", None)
                normalized.pop("uid", None)

        return normalized

    @staticmethod
    def _looks_like_generated_transcript_context(context: Any) -> bool:
        text = str(context or "").strip()
        if not text:
            return False
        lowered = text.lower()
        markers = (
            "chars=",
            "segments=",
            "preview=",
            "视频转写",
            "转写的chars=",
            "selected_text_length",
            "full_text_length",
        )
        matched = sum(1 for marker in markers if marker in text or marker in lowered)
        return matched >= 2

    @classmethod
    def _is_transcript_small_task_request(cls, arguments: dict) -> bool:
        if not (arguments or {}).get("result_ids"):
            return False
        task_text = str((arguments or {}).get("task") or "").strip()
        context_text = str((arguments or {}).get("context") or "").strip()
        joined_text = "\n".join(part for part in (task_text, context_text) if part)
        if not joined_text:
            return False
        if re.search(r"(转写|字幕|transcript)", joined_text, re.IGNORECASE):
            return True
        return cls._looks_like_generated_transcript_context(context_text)

    def _normalize_small_task_arguments(self, arguments: dict) -> dict:
        normalized = dict(arguments or {})
        if not self._is_transcript_small_task_request(normalized):
            return normalized

        normalized["task"] = _TRANSCRIPT_SMALL_TASK_CANONICAL_TASK
        normalized["output_format"] = _TRANSCRIPT_SMALL_TASK_CANONICAL_OUTPUT_FORMAT

        if self._looks_like_generated_transcript_context(normalized.get("context")):
            normalized.pop("context", None)
        elif not str(normalized.get("context", "") or "").strip():
            normalized.pop("context", None)

        return normalized

    def _normalize_request(
        self,
        request: ToolCallRequest,
        intent: IntentProfile,
        search_capabilities: dict,
        *,
        prefer_transcript_lookup: bool = False,
    ) -> ToolCallRequest:
        name = canonical_tool_name(request.name)
        arguments = dict(request.arguments or {})
        if name == "run_small_llm_task" and request.visibility == "internal":
            arguments = self._normalize_small_task_arguments(arguments)
        if request.visibility != "user":
            if arguments == request.arguments and name == request.name:
                return request
            return ToolCallRequest(
                id=request.id,
                name=name,
                arguments=arguments,
                visibility=request.visibility,
                source=request.source,
            )
        if name == "search_videos" and prefer_transcript_lookup:
            video_id = self._extract_transcript_video_id(arguments, intent)
            if video_id:
                transcript_args = {"video_id": video_id}
                page_index = arguments.get("page_index")
                if page_index not in (None, ""):
                    transcript_args["page_index"] = page_index
                return ToolCallRequest(
                    id=request.id,
                    name="get_video_transcript",
                    arguments=transcript_args,
                    visibility=request.visibility,
                    source=request.source,
                )
        if name == "search_videos":
            arguments = self._normalize_search_video_lookup_arguments(arguments, intent)
        if name == "search_owners":
            owner_text = self._owner_request_text(arguments)
            if not owner_text:
                owner_seed = self._owner_resolution_seed(intent)
                if owner_seed:
                    arguments["text"] = owner_seed
                    owner_text = owner_seed
            elif not str(arguments.get("text", "") or "").strip():
                arguments["text"] = owner_text
        elif name == "search_videos" and intent.needs_term_normalization:
            raw_queries = arguments.get("queries")
            if isinstance(raw_queries, str):
                raw_queries = [raw_queries]
            elif not isinstance(raw_queries, list):
                single_query = str(arguments.get("query", "") or "").strip()
                raw_queries = [single_query] if single_query else []

            rewritten_queries: list[str] = []
            query_changed = False
            for query in raw_queries:
                original_query = str(query or "")
                rewritten_query = rewrite_known_term_aliases(original_query)
                if rewritten_query != original_query:
                    query_changed = True
                if rewritten_query and rewritten_query not in rewritten_queries:
                    rewritten_queries.append(rewritten_query)

            if query_changed and rewritten_queries:
                arguments.pop("query", None)
                arguments["queries"] = rewritten_queries
        if name == "search_owners" and not str(arguments.get("text", "") or "").strip():
            owner_seed = self._owner_resolution_seed(intent)
            if owner_seed:
                arguments["text"] = owner_seed
        if arguments == request.arguments and name == request.name:
            return request
        return ToolCallRequest(
            id=request.id,
            name=name,
            arguments=arguments,
            visibility=request.visibility,
            source=request.source,
        )

    @staticmethod
    def _extract_transcript_video_id(
        arguments: dict,
        intent: IntentProfile,
    ) -> str:
        direct_keys = ("video_id", "bv", "bvid", "aid")
        for key in direct_keys:
            value = str(arguments.get(key, "") or "").strip()
            if value:
                matches = extract_bvids({"content": value})
                if matches:
                    return matches[0]
                return value

        bvids = arguments.get("bvids")
        if isinstance(bvids, str):
            bvid_matches = extract_bvids({"content": bvids})
            if bvid_matches:
                return bvid_matches[0]
        elif isinstance(bvids, (list, tuple)):
            for item in bvids:
                bvid_matches = extract_bvids({"content": str(item or "")})
                if bvid_matches:
                    return bvid_matches[0]

        for token in [
            *(intent.explicit_entities or []),
            *(intent.explicit_topics or []),
        ]:
            matches = extract_bvids({"content": str(token or "")})
            if matches:
                return matches[0]
        return ""

    def _collect_requests(
        self,
        response: ChatResponse,
        iteration: int,
    ) -> list[ToolCallRequest]:
        # bili-search intentionally uses inline XML only so tool planning can
        # stream to the UI. Provider function calling is not part of the active
        # orchestration contract.
        return parse_xml_tool_calls(response.content or "", iteration)

    def _result_summary(self, result_id: str, tool_name: str, result: dict) -> dict:
        return summarize_result(result_id, tool_name, result)

    def _inspect_result(self, result_store: ResultStore, args: dict) -> dict:
        return inspect_results(result_store, args)

    @staticmethod
    def _normalize_transcript_tags(raw_tags: Any) -> list[str]:
        if isinstance(raw_tags, str):
            values = re.split(r"[,/|、\n]+", raw_tags)
        elif isinstance(raw_tags, (list, tuple, set)):
            values = [str(item or "") for item in raw_tags]
        else:
            return []

        normalized: list[str] = []
        for value in values:
            tag = str(value or "").strip()
            if not tag or tag in normalized:
                continue
            normalized.append(tag)
            if len(normalized) >= _TRANSCRIPT_METADATA_TAG_LIMIT:
                break
        return normalized

    @staticmethod
    def _compact_transcript_metadata_text(
        value: Any,
        *,
        limit: int | None = None,
    ) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if limit is not None and len(text) > limit:
            return text[:limit].rstrip() + "..."
        return text

    @staticmethod
    def _extract_transcript_bvid(result: dict) -> str:
        for key in ("bvid", "requested_video_id", "video_id"):
            raw_value = str((result or {}).get(key) or "").strip()
            if not raw_value:
                continue
            matches = extract_bvids({"content": raw_value})
            if matches:
                return matches[0]
        return ""

    def _lookup_transcript_video_metadata(self, transcript_result: dict) -> dict:
        result = transcript_result if isinstance(transcript_result, dict) else {}
        bvid = self._extract_transcript_bvid(result)
        lookup_hit: dict[str, Any] = {}
        search_client = getattr(self.tool_executor, "search_client", None)
        lookup_method = getattr(search_client, "lookup_videos", None)

        if bvid and callable(lookup_method):
            lookup_result = None
            try:
                lookup_result = lookup_method(
                    bvids=[bvid],
                    limit=1,
                    verbose=self.verbose,
                )
            except TypeError:
                try:
                    lookup_result = lookup_method(bvids=[bvid], limit=1)
                except Exception:
                    lookup_result = None
            except Exception:
                lookup_result = None

            if isinstance(lookup_result, dict):
                hits = lookup_result.get("hits") or []
                if hits and isinstance(hits[0], dict):
                    lookup_hit = hits[0]

        owner = result.get("owner") if isinstance(result.get("owner"), dict) else {}
        if not owner and isinstance(lookup_hit.get("owner"), dict):
            owner = lookup_hit.get("owner") or {}

        title = self._compact_transcript_metadata_text(
            result.get("title") or lookup_hit.get("title")
        )
        author = self._compact_transcript_metadata_text(owner.get("name"))
        tags = self._normalize_transcript_tags(
            result.get("tags")
            or result.get("tag_names")
            or lookup_hit.get("tags")
            or lookup_hit.get("tag_names")
        )
        description = self._compact_transcript_metadata_text(
            result.get("desc")
            or result.get("description")
            or lookup_hit.get("desc")
            or lookup_hit.get("description"),
            limit=_TRANSCRIPT_METADATA_DESC_LIMIT,
        )

        published_at = self._compact_transcript_metadata_text(
            result.get("pubdate_str") or lookup_hit.get("pubdate_str")
        )
        if not published_at:
            pubdate = result.get("pubdate") or lookup_hit.get("pubdate")
            if pubdate not in (None, ""):
                try:
                    published_at = ts_to_str(int(pubdate))
                except (TypeError, ValueError):
                    published_at = self._compact_transcript_metadata_text(pubdate)

        return {
            "video_id": bvid
            or self._compact_transcript_metadata_text(
                result.get("bvid") or result.get("requested_video_id")
            ),
            "title": title,
            "author": author,
            "published_at": published_at,
            "tags": tags,
            "description": description,
        }

    def _build_transcript_context_block(
        self,
        record: ToolExecutionRecord,
        transcript_text: str,
        *,
        truncated: bool,
    ) -> str:
        metadata = self._lookup_transcript_video_metadata(record.result)
        lines = ["[转写信息]", f"result_id: {record.result_id}"]

        if metadata.get("video_id"):
            lines.append(f"视频ID: {metadata['video_id']}")
        if metadata.get("title"):
            lines.append(f"标题: {metadata['title']}")
        if metadata.get("author"):
            lines.append(f"作者: {metadata['author']}")
        if metadata.get("published_at"):
            lines.append(f"发布时间: {metadata['published_at']}")
        if metadata.get("tags"):
            lines.append(f"标签: {'、'.join(metadata['tags'])}")
        if metadata.get("description"):
            lines.append(f"简介: {metadata['description']}")

        lines.append(f"转写字数: {len(transcript_text)}")
        lines.append(f"转写被截断: {'是' if truncated else '否'}")
        return "\n".join(lines)

    def _build_small_task_messages(
        self,
        result_store: ResultStore,
        args: dict,
        intent: IntentProfile,
    ) -> tuple[str, ModelDecision, list[dict[str, str]]]:
        task = str(args.get("task", "")).strip()
        context_parts = []
        transcript_context_present = False
        transcript_context_truncated = False
        if args.get("context"):
            context_parts.append(str(args.get("context")))
        for result_id in list(args.get("result_ids") or []):
            record = result_store.get(result_id)
            if record is None:
                continue
            if canonical_tool_name(record.request.name) == "get_video_transcript":
                transcript_context_present = True
                transcript = (record.result.get("transcript") or {}).get("text") or ""
                transcript_text = str(transcript or "")
                trimmed_transcript = transcript_text[:_TRANSCRIPT_CONTEXT_CHAR_LIMIT]
                transcript_context_truncated = (
                    len(transcript_text) > _TRANSCRIPT_CONTEXT_CHAR_LIMIT
                )
                context_parts.append(
                    self._build_transcript_context_block(
                        record,
                        transcript_text,
                        truncated=transcript_context_truncated,
                    )
                )
                if trimmed_transcript:
                    context_parts.append(f"[转写正文]\n{trimmed_transcript}")
                continue
            context_parts.append(json.dumps(record.summary, ensure_ascii=False))
        context = "\n".join(context_parts).strip()
        decision = self._select_model(intent, stage="delegate", thinking=False)
        self._log_model_decision(
            phase="delegate",
            iteration=max(len(result_store.order), 1),
            decision=decision,
            intent=intent,
        )
        system_content = "你是搜索编排系统的小模型执行器。只完成窄任务，输出紧凑、结构化、可直接复用的结果。"
        extra_requirements = ""
        if transcript_context_present:
            system_content += (
                " 当前上下文来自单个视频的音频转写和元信息。"
                "请结合标题、标签、简介、作者、发布时间，提炼主题和覆盖全片的中文要点，避免只复述开头。"
            )
            extra_requirements = (
                "\n要求: 先用 1 句话概括主题，再整理覆盖全片的中文要点。"
                "如果视频是多人或多环节展示，优先按人物或环节归纳；"
                "只有在转写确实被截断时，才说明信息可能不完整。"
            )
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": (
                    f"任务: {task}\n"
                    f"输出格式: {args.get('output_format', '简洁中文要点')}\n"
                    f"当前意图: {intent.final_target} / {intent.task_mode}\n"
                    f"上下文:\n{context or '[无补充上下文]'}"
                    f"{extra_requirements}"
                ),
            },
        ]
        return task, decision, messages

    @staticmethod
    def _build_small_task_result(
        task: str,
        decision: ModelDecision,
        content: str,
        *,
        partial: bool = False,
    ) -> dict:
        payload = {
            "task": task,
            "model": decision.spec.config_name,
            "model_name": decision.spec.model_name,
            "model_reason": decision.reason,
            "result": sanitize_generated_content(content or ""),
        }
        if partial:
            payload["partial"] = True
        return payload

    def _run_small_task(
        self, result_store: ResultStore, args: dict, intent: IntentProfile
    ) -> dict:
        task = str(args.get("task", "")).strip()
        if not task:
            return {"error": "Missing task"}
        _, decision, messages = self._build_small_task_messages(
            result_store,
            args,
            intent,
        )
        response = decision.client.chat(
            messages=messages,
            temperature=0.2,
            enable_thinking=False,
        )
        return self._build_small_task_result(
            task,
            decision,
            response.content or "",
        )

    def _run_small_task_stream(
        self,
        result_store: ResultStore,
        args: dict,
        intent: IntentProfile,
        *,
        cancelled: Optional[object] = None,
    ) -> Generator[dict[str, Any], None, dict]:
        task = str(args.get("task", "")).strip()
        if not task:
            return {"error": "Missing task"}
        _, decision, messages = self._build_small_task_messages(
            result_store,
            args,
            intent,
        )
        # Emit an immediate streaming placeholder so the UI can switch from
        # "pending" to "streaming" before the small model produces its first
        # text delta.
        yield self._build_small_task_result(
            task,
            decision,
            "",
            partial=True,
        )
        stream = decision.client.chat_stream(
            messages=messages,
            temperature=0.2,
            enable_thinking=False,
        )
        accumulated_content = ""
        saw_content = False
        for chunk in stream or []:
            if cancelled is not None and getattr(cancelled, "is_set", lambda: False)():
                break
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content_delta = delta.get("content")
            if not content_delta:
                continue
            saw_content = True
            accumulated_content += content_delta
            yield self._build_small_task_result(
                task,
                decision,
                accumulated_content,
                partial=True,
            )
        if not saw_content:
            return self._run_small_task(result_store, args, intent)
        return self._build_small_task_result(task, decision, accumulated_content)

    @staticmethod
    def _select_transcript_post_execution_nudge(
        result_store: ResultStore,
        *,
        prefer_transcript_lookup: bool,
        prompted_nudges: set[str],
    ) -> tuple[str, str] | None:
        if not prefer_transcript_lookup:
            return None
        if "transcript_result_should_be_compressed" in prompted_nudges:
            return None
        if not has_successful_tool_result(result_store, "get_video_transcript"):
            return None
        if has_successful_tool_result(result_store, "run_small_llm_task"):
            return None
        return (
            "transcript_result_should_be_compressed",
            "你已经拿到 get_video_transcript 的结果。下一步请直接基于刚拿到的 result_id 调用 "
            "run_small_llm_task，把转写整理成主题概括和覆盖全片的中文要点，然后直接回答；"
            "不要改去 search_videos，也不要只复述 preview。",
        )

    @staticmethod
    def _reasoning_reset_delta(phase: str, iteration: int) -> dict[str, Any]:
        return {
            "reset_reasoning": True,
            "reasoning_phase": phase,
            "reasoning_iteration": iteration,
        }

    def _execute_internal_call(
        self,
        result_store: ResultStore,
        request: ToolCallRequest,
        intent: IntentProfile,
    ) -> dict:
        if request.name == "read_prompt_assets":
            assets = get_prompt_assets(
                ids=list(request.arguments.get("asset_ids") or []),
                tool_names=list(request.arguments.get("tool_names") or []),
                levels=list(request.arguments.get("levels") or []),
            )
            return {
                "assets": [
                    {
                        "asset_id": asset.asset_id,
                        "title": asset.title,
                        "section": asset.section,
                        "level": asset.level,
                        "content": asset.content,
                    }
                    for asset in assets
                ],
                "total_assets": len(assets),
            }
        if request.name == "inspect_tool_result":
            return self._inspect_result(result_store, request.arguments)
        if request.name == "run_small_llm_task":
            return self._run_small_task(result_store, request.arguments, intent)
        return {"error": f"Unknown internal tool: {request.name}"}

    def _execute_request(
        self,
        result_store: ResultStore,
        request: ToolCallRequest,
        intent: IntentProfile,
        *,
        result: dict | None = None,
    ) -> ToolExecutionRecord:
        resolved_result = result
        if resolved_result is None:
            if request.visibility == "internal":
                resolved_result = self._execute_internal_call(
                    result_store,
                    request,
                    intent,
                )
            else:
                resolved_result = self.tool_executor.execute_request(request)
        result_id = f"R{len(result_store.order) + 1}"
        summary = self._result_summary(result_id, request.name, resolved_result)
        record = ToolExecutionRecord(
            result_id=result_id,
            request=request,
            result=resolved_result,
            summary=summary,
            visibility=request.visibility,
        )
        result_store.add(record)
        return record

    def _execute_requests(
        self,
        result_store: ResultStore,
        requests: list[ToolCallRequest],
        intent: IntentProfile,
    ) -> list[ToolExecutionRecord]:
        if not requests:
            return []

        def run_one(request: ToolCallRequest) -> ToolExecutionRecord:
            return self._execute_request(result_store, request, intent)

        max_workers = min(max(len(requests), 1), 4)
        if max_workers == 1:
            return [run_one(request) for request in requests]

        records: list[ToolExecutionRecord] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, request) for request in requests]
            for future in futures:
                records.append(future.result())
        return records

    def _accumulate_usage(self, total: dict, usage: dict):
        accumulate_usage(total, usage)

    def _normalize_usage(self, usage: dict) -> dict:
        return normalize_usage(usage)

    @staticmethod
    def _latest_user_text(messages: list[dict]) -> str:
        return next(
            (
                extract_message_text(message)
                for message in reversed(messages or [])
                if message.get("role") == "user"
            ),
            "",
        )

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
    def _format_recent_window_label(window: str | None) -> str:
        value = str(window or "").strip().lower()
        match = re.fullmatch(r"(\d+)([dwmy])", value)
        if not match:
            return "30 天"
        amount = int(match.group(1))
        unit = match.group(2)
        unit_label = {
            "d": "天",
            "w": "周",
            "m": "个月",
            "y": "年",
        }.get(unit, "天")
        return f"{amount} {unit_label}"

    @staticmethod
    def _clean_subject_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip(
            " ，。！？?；;：:、()[]{}<>《》\"'`~!@#$%^&*-+=|\\/"
        )

    @classmethod
    def _extract_leading_subject_phrase(cls, latest_user_text: str) -> str:
        source = str(latest_user_text or "").strip()
        if not source:
            return ""

        segments = [
            (
                match.group(0),
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
        for segment_text, start, end in segments:
            is_long_cjk_clause = (
                bool(re.fullmatch(r"[\u4e00-\u9fff]+", segment_text))
                and len(segment_text) >= 5
            )
            normalized_segment = "".join(str(segment_text or "").split())
            is_short_question_clause = normalized_segment in {
                "是谁",
                "是什么",
                "谁",
                "什么",
                "哪个",
                "哪位",
                "哪种",
                "吗",
                "呢",
                "嘛",
                "么",
            }
            if not saw_content:
                if not normalized_segment:
                    continue
                subject_start = start
                subject_end = end
                saw_content = True
                continue
            if not normalized_segment:
                subject_end = end
                continue
            if is_long_cjk_clause or is_short_question_clause:
                break
            subject_end = end

        if subject_start is None or subject_end is None:
            return ""
        return cls._clean_subject_text(source[subject_start:subject_end])

    @classmethod
    def _extract_recent_owner_subject(
        cls,
        messages: list[dict],
        intent: IntentProfile,
    ) -> str:
        latest_user_text = cls._latest_user_text(messages)
        candidate_texts = [
            *(intent.explicit_entities or []),
            *(intent.explicit_topics or []),
        ]
        for candidate in candidate_texts:
            cleaned = cls._clean_subject_text(candidate)
            if (
                cleaned
                and len(cleaned) <= 32
                and not _RECENT_OWNER_SUBJECT_STOP_RE.search(cleaned)
            ):
                return rewrite_known_term_aliases(cleaned) or cleaned

        normalized_source = "".join(str(latest_user_text or "").split())
        for pattern in _RECENT_OWNER_QUERY_PATTERNS:
            match = pattern.match(normalized_source)
            if not match:
                continue
            subject = cls._clean_subject_text(match.group("subject"))
            if subject and len(subject) <= 32:
                return rewrite_known_term_aliases(subject) or subject

        leading_subject = cls._extract_leading_subject_phrase(latest_user_text)
        if (
            leading_subject
            and len(leading_subject) <= 32
            and not _RECENT_OWNER_SUBJECT_STOP_RE.search(leading_subject)
        ):
            return rewrite_known_term_aliases(leading_subject) or leading_subject
        return ""

    @classmethod
    def _select_recent_owner_candidate(
        cls,
        result_store: ResultStore,
        messages: list[dict],
        intent: IntentProfile,
    ) -> tuple[str, dict | None]:
        subject = cls._extract_recent_owner_subject(messages, intent)
        best_owner: dict | None = None
        best_rank = float("-inf")

        for result_id in result_store.order:
            record = result_store.get(result_id)
            if (
                record is None
                or canonical_tool_name(record.request.name) != "search_owners"
            ):
                continue
            result = record.result or {}
            owners = result.get("owners") or []
            source_text = cls._owner_request_text(result) or cls._owner_request_text(
                record.request.arguments or {}
            )
            seed_text = subject or source_text
            if not seed_text:
                continue
            matching_owners = [
                owner
                for owner in owners
                if OwnerResolutionMixin._owner_name_matches_source(
                    seed_text,
                    str(owner.get("name") or "").strip(),
                )
            ]
            if matching_owners:
                top_owner = matching_owners[0]
                try:
                    top_score = float(top_owner.get("score") or 0.0)
                except (TypeError, ValueError):
                    top_score = 0.0
                try:
                    next_score = float(matching_owners[1].get("score") or 0.0)
                except (IndexError, TypeError, ValueError):
                    next_score = float("-inf")
                if top_owner.get("mid") and (
                    len(matching_owners) == 1 or top_score - next_score >= 3.0
                ):
                    if top_score > best_rank:
                        best_owner = top_owner
                        best_rank = top_score
                    continue
            for index, owner in enumerate(owners):
                if not OwnerResolutionMixin._is_confident_owner_candidate(
                    seed_text,
                    owner,
                    owners=owners,
                ):
                    continue
                try:
                    service_score = float(owner.get("score") or 0.0)
                except (TypeError, ValueError):
                    service_score = 0.0
                candidate_rank = service_score - index * 0.001
                if candidate_rank > best_rank:
                    best_owner = owner
                    best_rank = candidate_rank
                break

        return subject, best_owner

    def _build_owner_recent_timeline_followup_requests(
        self,
        result_store: ResultStore,
        intent: IntentProfile,
        messages: list[dict] | None = None,
    ) -> list[ToolCallRequest]:
        if intent.final_target != "videos" or has_explicit_video_anchor(intent):
            return []
        if not is_recent_timeline_request(intent):
            return []

        message_list = list(messages or [])
        _subject, owner_candidate = self._select_recent_owner_candidate(
            result_store,
            message_list,
            intent,
        )
        if not owner_candidate:
            return []

        try:
            owner_mid = str(int(str(owner_candidate.get("mid") or "").strip()))
        except (TypeError, ValueError):
            return []

        arguments = {
            "mode": "lookup",
            "mid": owner_mid,
            "date_window": self._extract_recent_window(intent.raw_query),
            "limit": 10,
        }
        return [
            ToolCallRequest(
                id=f"auto_owner_recent_{len(result_store.order) + 1}",
                name="search_videos",
                arguments=arguments,
                visibility="user",
                source="deterministic_followup",
            )
        ]

    def _extract_recent_timeline_hits(self) -> tuple[list[dict], str | None, bool]:
        fallback_candidate: tuple[list[dict], str | None, bool] | None = None
        for result_id in reversed(self.result_store.order):
            record = self.result_store.get(result_id)
            if (
                record is None
                or canonical_tool_name(record.request.name) != "search_videos"
            ):
                continue

            args = record.request.arguments or {}
            result = record.result or {}
            mode = str(args.get("mode") or result.get("mode") or "").lower()
            lookup_by = str(
                result.get("lookup_by") or args.get("lookup_by") or ""
            ).lower()
            date_window = (
                str(result.get("date_window") or args.get("date_window") or "").strip()
                or None
            )

            if mode == "lookup" and (
                lookup_by in {"mid", "mids"}
                or args.get("mid")
                or args.get("mids")
                or args.get("uid")
            ):
                return list(result.get("hits") or []), date_window, True

            queries = args.get("queries")
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list):
                continue
            query_text = "\n".join(str(query or "") for query in queries)
            if ":date<=" not in query_text or not any(
                marker in query_text for marker in (":uid=", ":user=")
            ):
                continue
            if not date_window:
                date_match = re.search(r":date<=([0-9]+[dwmy])", query_text)
                if date_match:
                    date_window = date_match.group(1)
            if fallback_candidate is None:
                fallback_candidate = (
                    list(result.get("hits") or []),
                    date_window,
                    True,
                )

        if fallback_candidate is not None:
            return fallback_candidate

        return [], None, False

    def _build_owner_recent_timeline_answer(
        self,
        intent: IntentProfile,
        messages: list[dict],
    ) -> str | None:
        if intent.final_target != "videos" or has_explicit_video_anchor(intent):
            return None
        if not is_recent_timeline_request(intent):
            return None

        subject, owner_candidate = self._select_recent_owner_candidate(
            self.result_store,
            messages,
            intent,
        )
        recent_hits, date_window, recent_lookup_attempted = (
            self._extract_recent_timeline_hits()
        )
        if not owner_candidate and not recent_hits and not recent_lookup_attempted:
            return None

        owner_name = ""
        owner_mid = ""
        if owner_candidate:
            owner_name = str(owner_candidate.get("name") or "").strip()
            owner_mid = str(owner_candidate.get("mid") or "").strip()

        if recent_hits and (not owner_name or not owner_mid):
            owner = recent_hits[0].get("owner") or {}
            if not owner_name:
                owner_name = str(owner.get("name") or "").strip()
            if not owner_mid:
                owner_mid = str(owner.get("mid") or "").strip()

        lines: list[str] = []
        if subject and owner_name and subject != owner_name:
            if owner_mid:
                lines.append(
                    f"{subject} 对应的作者是 {owner_name}，UID 为 {owner_mid}。"
                )
            else:
                lines.append(f"{subject} 对应的作者是 {owner_name}。")
        elif owner_name and owner_mid:
            lines.append(f"作者是 {owner_name}，UID 为 {owner_mid}。")
        elif owner_name:
            lines.append(f"作者是 {owner_name}。")
        elif owner_mid:
            lines.append(f"作者 UID 为 {owner_mid}。")

        if owner_mid:
            lines.append(f"空间链接：https://space.bilibili.com/{owner_mid}")

        window = date_window or self._extract_recent_window(intent.raw_query)
        if recent_hits:
            timeline_owner = owner_name or subject or "该作者"
            lines.append(
                f"{timeline_owner}近 {self._format_recent_window_label(window)} 发布的视频包括："
            )
            for index, hit in enumerate(recent_hits[:5], start=1):
                hit_title = str(hit.get("title") or "").strip()
                hit_bvid = str(hit.get("bvid") or "").strip()
                if hit_title and hit_bvid:
                    lines.append(f"{index}. 《{hit_title}》({hit_bvid})")
                elif hit_title:
                    lines.append(f"{index}. 《{hit_title}》")
                elif hit_bvid:
                    lines.append(f"{index}. {hit_bvid}")
        elif recent_lookup_attempted:
            lines.append(
                f"当前 {self._format_recent_window_label(window)} 时间窗内未检索到该作者的公开视频。"
            )

        return "\n".join(line for line in lines if line).strip() or None

    def _build_deterministic_recovery_requests(
        self,
        messages: list[dict],
        intent: IntentProfile,
        *,
        prefer_transcript_lookup: bool = False,
    ) -> list[ToolCallRequest]:
        if intent.final_target != "videos" or prefer_transcript_lookup:
            return []

        if has_explicit_video_anchor(intent) and (
            intent.needs_owner_resolution or is_recent_timeline_request(intent)
        ):
            explicit_bvids = extract_bvids({"content": intent.raw_query})
            if explicit_bvids:
                arguments: dict[str, Any] = {"mode": "lookup"}
                if len(explicit_bvids) == 1:
                    arguments["bv"] = explicit_bvids[0]
                else:
                    arguments["bvids"] = explicit_bvids[:5]
                return [
                    ToolCallRequest(
                        id="auto_recover_explicit_bvid_1",
                        name="search_videos",
                        arguments=arguments,
                        visibility="user",
                        source="deterministic_recovery",
                    )
                ]

        if not is_recent_timeline_request(intent):
            return []

        explicit_mids = extract_owner_mids({"content": intent.raw_query})
        if explicit_mids:
            arguments = {
                "mode": "lookup",
                "date_window": self._extract_recent_window(intent.raw_query),
                "limit": 10,
            }
            if len(explicit_mids) == 1:
                arguments["mid"] = str(explicit_mids[0])
            else:
                arguments["mids"] = [str(mid) for mid in explicit_mids[:5]]
            return [
                ToolCallRequest(
                    id="auto_recover_owner_mid_1",
                    name="search_videos",
                    arguments=arguments,
                    visibility="user",
                    source="deterministic_recovery",
                )
            ]

        owner_subject = self._extract_recent_owner_subject(messages, intent)
        if not owner_subject:
            return []
        return [
            ToolCallRequest(
                id="auto_recover_owner_name_1",
                name="search_owners",
                arguments={"text": owner_subject, "mode": "name"},
                visibility="user",
                source="deterministic_recovery",
            )
        ]

    def _build_deterministic_followup_requests(
        self,
        result_store: ResultStore,
        intent: IntentProfile,
        messages: list[dict] | None = None,
    ) -> list[ToolCallRequest]:
        followup_requests: list[ToolCallRequest] = []

        if needs_explicit_video_lookup_followup(result_store, intent):
            mids: list[int] = []
            owner_names: list[str] = []
            anchor_bvids: list[str] = []
            seen_mids: set[int] = set()
            seen_owner_names: set[str] = set()
            seen_bvid_keys: set[str] = set()

            for result_id in result_store.order:
                record = result_store.get(result_id)
                if (
                    record is None
                    or canonical_tool_name(record.request.name) != "search_videos"
                ):
                    continue

                args = record.request.arguments or {}
                result = record.result or {}
                mode = str(args.get("mode") or result.get("mode") or "").lower()
                lookup_by = str(
                    result.get("lookup_by") or args.get("lookup_by") or ""
                ).lower()
                has_bvid_seed = bool(
                    args.get("bv") or args.get("bvid") or args.get("bvids")
                )
                if mode != "lookup" or (
                    lookup_by not in {"bvid", "bvids"} and not has_bvid_seed
                ):
                    continue

                for value in [*(args.get("bvids") or []), *(result.get("bvids") or [])]:
                    bvid = str(value or "").strip()
                    bvid_key = normalize_bvid_key(bvid)
                    if bvid_key and bvid_key not in seen_bvid_keys:
                        seen_bvid_keys.add(bvid_key)
                        anchor_bvids.append(bvid)
                for key in ("bv", "bvid"):
                    bvid = str(args.get(key, "") or "").strip()
                    bvid_key = normalize_bvid_key(bvid)
                    if bvid_key and bvid_key not in seen_bvid_keys:
                        seen_bvid_keys.add(bvid_key)
                        anchor_bvids.append(bvid)

                for hit in result.get("hits") or []:
                    owner = hit.get("owner") or {}
                    owner_name = str(owner.get("name") or "").strip()
                    if owner_name and owner_name not in seen_owner_names:
                        seen_owner_names.add(owner_name)
                        owner_names.append(owner_name)
                    try:
                        owner_mid = int(str(owner.get("mid") or "").strip())
                    except (TypeError, ValueError):
                        owner_mid = None
                    if owner_mid and owner_mid not in seen_mids:
                        seen_mids.add(owner_mid)
                        mids.append(owner_mid)

            if mids or owner_names:
                window = self._extract_recent_window(intent.raw_query)
                if mids:
                    arguments: dict[str, Any] = {
                        "mode": "lookup",
                        "date_window": window,
                        "limit": 10,
                    }
                    if anchor_bvids:
                        arguments["exclude_bvids"] = anchor_bvids[:10]
                    if len(mids) == 1:
                        arguments["mid"] = str(mids[0])
                    else:
                        arguments["mids"] = [str(mid) for mid in mids[:5]]
                else:
                    queries: list[str] = []
                    for owner_name in owner_names[:3]:
                        query = f":user={owner_name} :date<={window}"
                        if query not in queries:
                            queries.append(query)
                    if queries:
                        arguments = {"queries": queries}
                    else:
                        arguments = {}

                if arguments:
                    followup_requests.append(
                        ToolCallRequest(
                            id=f"auto_followup_{len(result_store.order) + 1}",
                            name="search_videos",
                            arguments=arguments,
                            visibility="user",
                            source="deterministic_followup",
                        )
                    )

        followup_requests.extend(
            self._build_owner_recent_timeline_followup_requests(
                result_store,
                intent,
                messages=messages,
            )
        )

        deduped_requests: list[ToolCallRequest] = []
        seen_signatures: set[str] = set()
        for request in followup_requests:
            signature = command_signature(request)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped_requests.append(request)
        return deduped_requests

    def _build_deterministic_final_answer(
        self,
        intent: IntentProfile,
        messages: list[dict],
    ) -> str | None:
        return self._build_explicit_video_lookup_answer(
            intent
        ) or self._build_owner_recent_timeline_answer(
            intent,
            messages,
        )

    def _build_explicit_video_lookup_answer(
        self,
        intent: IntentProfile,
    ) -> str | None:
        if intent.final_target != "videos" or not has_explicit_video_anchor(intent):
            return None
        if not (intent.needs_owner_resolution or is_recent_timeline_request(intent)):
            return None

        primary_hit: dict | None = None
        recent_hits: list[dict] = []

        for result_id in self.result_store.order:
            record = self.result_store.get(result_id)
            if (
                record is None
                or canonical_tool_name(record.request.name) != "search_videos"
            ):
                continue

            result = record.result or {}
            lookup_by = str(result.get("lookup_by") or "").lower()
            hits = result.get("hits") or []
            if not primary_hit and lookup_by in {"bvid", "bvids"} and hits:
                primary_hit = hits[0]
            if lookup_by in {"mid", "mids"} and not recent_hits:
                recent_hits = list(hits)

        if not primary_hit:
            return None

        owner = primary_hit.get("owner") or {}
        bvid = str(primary_hit.get("bvid") or "").strip()
        title = str(primary_hit.get("title") or "").strip()
        owner_name = str(owner.get("name") or "").strip()
        owner_mid = str(owner.get("mid") or "").strip()

        lines: list[str] = []
        title_text = f"《{title}》" if title else "该视频"
        if bvid and title:
            lines.append(f"{bvid} 这期视频的标题是 {title_text}。")
        elif title:
            lines.append(f"这期视频的标题是 {title_text}。")

        if owner_name and owner_mid:
            lines.append(f"作者是 {owner_name}，UID 为 {owner_mid}。")
        elif owner_name:
            lines.append(f"作者是 {owner_name}。")
        elif owner_mid:
            lines.append(f"作者 UID 为 {owner_mid}。")

        if is_recent_timeline_request(intent):
            if recent_hits:
                lines.append("该作者近 30 天发布的视频包括：")
                for index, hit in enumerate(recent_hits[:5], start=1):
                    hit_title = str(hit.get("title") or "").strip()
                    hit_bvid = str(hit.get("bvid") or "").strip()
                    if hit_title and hit_bvid:
                        lines.append(f"{index}. 《{hit_title}》({hit_bvid})")
                    elif hit_title:
                        lines.append(f"{index}. 《{hit_title}》")
                    elif hit_bvid:
                        lines.append(f"{index}. {hit_bvid}")
            else:
                lines.append("当前 30 天时间窗内未检索到该作者的其他公开视频。")

        return "\n".join(line for line in lines if line).strip() or None

    def _wants_transcript_lookup(
        self,
        messages: list[dict],
        intent: IntentProfile,
        search_capabilities: dict,
    ) -> bool:
        if not search_capabilities.get("supports_transcript_lookup", False):
            return False
        if intent.final_target != "videos":
            return False
        latest_user_text = self._latest_user_text(messages)
        if not latest_user_text or not _TRANSCRIPT_HINT_RE.search(latest_user_text):
            return False
        seen_bvids: set[str] = set()
        for message in messages or []:
            seen_bvids.update(extract_bvids(message))
        return bool(seen_bvids)

    def _is_explicit_transcript_request(
        self,
        messages: list[dict],
        intent: IntentProfile,
    ) -> bool:
        if intent.final_target != "videos":
            return False
        latest_user_text = self._latest_user_text(messages)
        if not latest_user_text or not _EXPLICIT_TRANSCRIPT_REQUEST_RE.search(
            latest_user_text
        ):
            return False
        seen_bvids: set[str] = set()
        for message in messages or []:
            seen_bvids.update(extract_bvids(message))
        return bool(seen_bvids)

    @staticmethod
    def _select_transcript_unavailable_nudge(
        *,
        transcript_requested_but_unavailable: bool,
        user_tool_names: list[str],
        prompted_nudges: set[str],
    ) -> tuple[str, str] | None:
        if not transcript_requested_but_unavailable:
            return None
        if "transcript_lookup_unavailable" in prompted_nudges:
            return None
        if user_tool_names and not any(
            tool_name in {"search_videos", "search_google"}
            for tool_name in user_tool_names
        ):
            return None
        return (
            "transcript_lookup_unavailable",
            "当前环境没有 get_video_transcript 工具，无法直接获取该视频的音频转写或字幕。"
            "不要改用 search_videos 的 bvids 或 search_google 充当转写读取；"
            "请直接明确告诉用户当前无法读取转写文本，并说明需要先配置 transcript 服务。",
        )

    def _extra_prompt_assets(
        self,
        messages: list[dict],
        intent: IntentProfile,
        search_capabilities: dict,
    ) -> list[str]:
        if not self._wants_transcript_lookup(messages, intent, search_capabilities):
            return []
        return [
            "tool.get_video_transcript.brief",
            "tool.get_video_transcript.detailed",
            "tool.get_video_transcript.examples",
        ]

    def _select_model(
        self,
        intent: IntentProfile,
        stage: str,
        thinking: bool,
        *,
        prefer_small: bool = False,
    ) -> ModelDecision:
        small_spec = self.model_registry.primary("small")
        large_spec = self.model_registry.primary("large")

        if stage == "delegate":
            return ModelDecision(
                client=self.small_llm_client,
                spec=small_spec,
                reason="窄任务统一下放给小模型执行。",
                factors=("delegate", "small_model"),
            )

        if thinking:
            return ModelDecision(
                client=self.large_llm_client,
                spec=large_spec,
                reason="思考模式开启时，规划与收口都优先使用大模型。",
                factors=("thinking", f"stage={stage}"),
            )

        if prefer_small and stage in {"planner", "response"}:
            return ModelDecision(
                client=self.small_llm_client,
                spec=small_spec,
                reason="已识别为具体视频的内容理解/转写任务，优先让小模型读取和总结。",
                factors=("known_video", "transcript_lookup", f"stage={stage}"),
            )

        large_factors: list[str] = []
        if intent.final_target == "mixed":
            large_factors.append("mixed_target")
        if intent.task_mode == "collect_compare":
            large_factors.append("collect_compare")
        if stage == "planner" and intent.needs_keyword_expansion:
            large_factors.append("needs_keyword_expansion")
        if stage == "planner" and intent.needs_term_normalization:
            large_factors.append("needs_term_normalization")
        if (
            stage == "planner"
            and intent.final_target == "relations"
            and intent.ambiguity >= 0.38
        ):
            large_factors.append("relation_ambiguity")
        if stage == "planner" and intent.complexity_score >= 0.58:
            large_factors.append("planner_complexity")
        if (
            stage == "response"
            and intent.final_target == "relations"
            and intent.ambiguity >= 0.42
        ):
            large_factors.append("relation_synthesis")
        if stage == "response" and intent.complexity_score >= 0.72:
            large_factors.append("response_complexity")

        if large_factors:
            return ModelDecision(
                client=self.large_llm_client,
                spec=large_spec,
                reason="任务复杂度或编排风险较高，使用大模型更稳妥。",
                factors=tuple(large_factors),
            )

        return ModelDecision(
            client=self.small_llm_client,
            spec=small_spec,
            reason="当前阶段是单目标、低复杂度任务，小模型足够完成。",
            factors=(
                f"stage={stage}",
                f"target={intent.final_target}",
                f"task={intent.task_mode}",
            ),
        )

    def _log_model_decision(
        self,
        *,
        phase: str,
        iteration: int,
        decision: ModelDecision,
        intent: IntentProfile,
    ) -> None:
        if not self.verbose:
            return
        factors = ", ".join(decision.factors) if decision.factors else "-"
        logger.note(
            "> LLM route"
            f"[{phase}#{iteration}]: "
            f"config={decision.spec.config_name}, model={decision.spec.model_name}, "
            f"reason={decision.reason}, factors={factors}, "
            f"intent={intent.final_target}/{intent.task_mode}"
        )

    def _usage_trace_entry(
        self,
        *,
        phase: str,
        iteration: int,
        decision: ModelDecision,
        usage: dict,
        tool_calls: list[ToolCallRequest] | None = None,
    ) -> dict:
        normalized_usage = self._normalize_usage(usage)
        return {
            "phase": phase,
            "iteration": iteration,
            "model_config": decision.spec.config_name,
            "model_name": decision.spec.model_name,
            "model_reason": decision.reason,
            "model_factors": list(decision.factors),
            "prompt_tokens": normalized_usage.get("prompt_tokens", 0),
            "completion_tokens": normalized_usage.get("completion_tokens", 0),
            "total_tokens": normalized_usage.get("total_tokens", 0),
            "tool_count": len(tool_calls or []),
            "tool_names": [call.name for call in tool_calls or []],
        }

    @staticmethod
    def _is_error_response(response) -> bool:
        content = str(getattr(response, "content", "") or "")
        return getattr(response, "finish_reason", "") == "error" or content.startswith(
            "[Error:"
        )

    def _build_fallback_response_messages(
        self,
        *,
        intent: IntentProfile,
        result_store: ResultStore,
    ) -> list[dict]:
        observations = result_store.render_observation(result_store.latest_ids(limit=8))
        return [
            {
                "role": "system",
                "content": (
                    "你是搜索问答的最终收口模型。只基于现有结果直接回答，不要继续规划，"
                    "不要再次调用工具，也不要把错误信息原样返回给用户。若信息不足，请明确说明。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题:\n{intent.raw_query}\n\n"
                    f"当前意图:\n"
                    + "\n".join(intent.to_prompt_lines())
                    + f"\n\n已有结果:\n{observations}\n\n请用自然中文直接给出最终回答。"
                ),
            },
        ]

    def _build_tool_messages(
        self,
        response,
        records: list[ToolExecutionRecord],
    ) -> list[dict]:
        messages: list[dict] = []
        analysis = sanitize_generated_content(response.content or "")
        if analysis:
            messages.append({"role": "assistant", "content": analysis})
        if records:
            result_ids = [record.result_id for record in records]
            messages.append(
                {
                    "role": "user",
                    "content": self.result_store.render_observation(result_ids),
                }
            )
        return messages

    def _build_tool_events(
        self, iteration: int, records: list[ToolExecutionRecord]
    ) -> dict:
        return {
            "iteration": iteration,
            "tools": [record.request.name for record in records],
            "calls": [record.to_tool_event_call() for record in records],
        }

    @staticmethod
    def _pending_tool_call_payload(request: ToolCallRequest) -> dict[str, Any]:
        return {
            "type": request.name,
            "args": request.arguments,
            "status": "pending",
            "visibility": request.visibility,
        }

    @staticmethod
    def _streaming_tool_call_payload(
        request: ToolCallRequest,
        result: dict,
    ) -> dict[str, Any]:
        payload = ChatOrchestrator._pending_tool_call_payload(request)
        payload["status"] = "streaming"
        payload["result"] = result
        result_text = str((result or {}).get("result", "") or "").strip()
        if result_text:
            payload["summary"] = {"summary_text": result_text}
        return payload

    @staticmethod
    def _build_tool_event_from_calls(
        iteration: int,
        calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "iteration": iteration,
            "tools": [str(call.get("type", "") or "") for call in calls],
            "calls": calls,
        }

    def _build_pending_tool_event(
        self,
        iteration: int,
        requests: list[ToolCallRequest],
    ) -> dict:
        return self._build_tool_event_from_calls(
            iteration,
            [self._pending_tool_call_payload(request) for request in requests],
        )

    def _execute_small_task_request_stream(
        self,
        *,
        result_store: ResultStore,
        request: ToolCallRequest,
        intent: IntentProfile,
        iteration: int,
        request_index: int,
        current_calls: list[dict[str, Any]],
        cancelled: Optional[object] = None,
    ) -> Generator[dict[str, Any], None, ToolExecutionRecord]:
        stream = self._run_small_task_stream(
            result_store,
            request.arguments,
            intent,
            cancelled=cancelled,
        )
        final_result: dict | None = None
        while True:
            try:
                partial_result = next(stream)
            except StopIteration as stop:
                final_result = stop.value
                break
            current_calls[request_index] = self._streaming_tool_call_payload(
                request,
                partial_result,
            )
            yield {
                "tool_events": [
                    self._build_tool_event_from_calls(iteration, list(current_calls))
                ]
            }
        return self._execute_request(
            result_store,
            request,
            intent,
            result=final_result,
        )

    def _execute_requests_with_stream_events(
        self,
        result_store: ResultStore,
        requests: list[ToolCallRequest],
        intent: IntentProfile,
        *,
        iteration: int,
        cancelled: Optional[object] = None,
    ) -> Generator[dict[str, Any], None, list[ToolExecutionRecord]]:
        if not requests:
            return []

        pending_event = self._build_pending_tool_event(iteration, requests)
        yield {"tool_events": [pending_event]}

        has_streaming_small_task = any(
            request.visibility == "internal" and request.name == "run_small_llm_task"
            for request in requests
        )
        if not has_streaming_small_task:
            records = self._execute_requests(result_store, requests, intent)
            yield {"tool_events": [self._build_tool_events(iteration, records)]}
            return records

        current_calls = [
            self._pending_tool_call_payload(request) for request in requests
        ]
        records: list[ToolExecutionRecord] = []
        for request_index, request in enumerate(requests):
            if (
                request.visibility == "internal"
                and request.name == "run_small_llm_task"
            ):
                record = yield from self._execute_small_task_request_stream(
                    result_store=result_store,
                    request=request,
                    intent=intent,
                    iteration=iteration,
                    request_index=request_index,
                    current_calls=current_calls,
                    cancelled=cancelled,
                )
            else:
                record = self._execute_request(result_store, request, intent)
            records.append(record)
            current_calls[request_index] = record.to_tool_event_call(status="completed")
            yield {
                "tool_events": [
                    self._build_tool_event_from_calls(iteration, list(current_calls))
                ]
            }
        return records

    def _stream_chat_response(
        self,
        *,
        client,
        messages: list[dict],
        thinking: bool,
        cancelled: Optional[object] = None,
    ) -> Generator[dict[str, Any], None, ChatResponse]:
        accumulated_content = ""
        accumulated_reasoning = ""
        finish_reason = None
        usage: dict[str, Any] = {}
        emitted_content_len = 0
        tool_command_start = None
        content_retracted = False
        mirrored_analysis = False

        stream = client.chat_stream(
            messages=messages,
            temperature=self.temperature,
            enable_thinking=True if thinking else None,
        )
        for chunk in stream or []:
            if cancelled is not None and getattr(cancelled, "is_set", lambda: False)():
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

            reasoning_delta = delta.get("reasoning_content")
            if reasoning_delta:
                accumulated_reasoning += reasoning_delta
                yield {"delta": {"reasoning_content": reasoning_delta}}

            content_delta = delta.get("content")
            if content_delta:
                accumulated_content += content_delta
                if tool_command_start is None:
                    tool_command_start = find_tool_command_start(
                        accumulated_content,
                        tool_prefixes=ALL_TOOL_PREFIXES,
                    )
                if tool_command_start is not None:
                    if emitted_content_len > 0 and not content_retracted:
                        yield {"delta": {"retract_content": True}}
                        content_retracted = True
                    if not mirrored_analysis:
                        analysis_prefix = accumulated_content[:tool_command_start]
                        overlap = _shared_prefix_len(
                            analysis_prefix,
                            accumulated_reasoning,
                        )
                        promoted_analysis = analysis_prefix[overlap:]
                        if promoted_analysis and promoted_analysis.strip():
                            yield {
                                "delta": {
                                    "reasoning_content": promoted_analysis,
                                }
                            }
                        mirrored_analysis = True
                    continue

                safe_end = len(accumulated_content) - partial_tool_prefix_len(
                    accumulated_content,
                    tool_prefixes=ALL_TOOL_PREFIXES,
                )
                start = emitted_content_len
                if start == 0 and accumulated_reasoning and safe_end > 0:
                    start = min(
                        safe_end,
                        _shared_prefix_len(
                            accumulated_content[:safe_end],
                            accumulated_reasoning,
                        ),
                    )
                if safe_end > start:
                    yield {"delta": {"content": accumulated_content[start:safe_end]}}
                emitted_content_len = max(emitted_content_len, safe_end)

            if chunk.get("usage"):
                usage = chunk["usage"]

        if (
            tool_command_start is None
            and len(accumulated_content) > emitted_content_len
        ):
            tail = accumulated_content[emitted_content_len:]
            if emitted_content_len == 0 and accumulated_reasoning:
                overlap = _shared_prefix_len(tail, accumulated_reasoning)
                tail = tail[overlap:]
            if tail:
                yield {"delta": {"content": tail}}

        return ChatResponse(
            content=accumulated_content or None,
            reasoning_content=accumulated_reasoning or None,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _prepare_orchestration_context(
        self,
        *,
        messages: list[dict],
        thinking: bool,
    ) -> tuple[
        dict, IntentProfile, list[str], str, dict, ModelDecision, ModelDecision, str
    ]:
        search_capabilities = self.tool_executor.get_search_capabilities()
        intent = build_intent_profile(messages)
        extra_asset_ids = self._extra_prompt_assets(
            messages,
            intent,
            search_capabilities,
        )
        prompt = build_system_prompt(
            capabilities=search_capabilities,
            intent=intent,
            extra_asset_ids=extra_asset_ids,
        )
        if thinking:
            prompt = _THINKING_PROMPT + "\n\n" + prompt
        prompt_profile = build_system_prompt_profile(
            capabilities=search_capabilities,
            intent=intent,
            extra_asset_ids=extra_asset_ids,
        )
        if thinking:
            prompt_profile = {
                **prompt_profile,
                "thinking_prompt_chars": len(_THINKING_PROMPT),
                "total_chars": prompt_profile.get("total_chars", 0)
                + len(_THINKING_PROMPT),
            }
        prefer_small = bool(extra_asset_ids)
        planner_decision = self._select_model(
            intent,
            stage="planner",
            thinking=thinking,
            prefer_small=prefer_small,
        )
        response_decision = self._select_model(
            intent,
            stage="response",
            thinking=thinking,
            prefer_small=prefer_small,
        )
        latest_user_message = next(
            (
                extract_message_text(message)
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        return (
            search_capabilities,
            intent,
            extra_asset_ids,
            prompt,
            prompt_profile,
            planner_decision,
            response_decision,
            latest_user_message,
        )

    def run(
        self,
        *,
        messages: list[dict],
        thinking: bool = False,
        max_iterations: int | None = None,
        cancelled: Optional[object] = None,
    ) -> OrchestrationResult:
        start_time = time.perf_counter()
        (
            search_capabilities,
            intent,
            extra_asset_ids,
            prompt,
            prompt_profile,
            planner_decision,
            response_decision,
            latest_user_message,
        ) = self._prepare_orchestration_context(messages=messages, thinking=thinking)
        prefer_transcript_lookup = bool(extra_asset_ids)
        transcript_requested_but_unavailable = self._is_explicit_transcript_request(
            messages,
            intent,
        ) and not search_capabilities.get("supports_transcript_lookup", False)

        planner_client = planner_decision.client
        planner_spec = planner_decision.spec
        response_client = response_decision.client
        response_spec = response_decision.spec
        conversation = [{"role": "system", "content": prompt}] + list(messages)
        total_usage: dict[str, Any] = {}
        usage_trace: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        executed_signatures: set[str] = set()
        prompted_nudges: set[str] = set()
        self.result_store = ResultStore()
        resolved_iterations = max_iterations or planner_spec.max_iterations
        final_content = ""
        final_reasoning = ""

        for iteration in range(1, resolved_iterations + 1):
            if cancelled is not None and getattr(cancelled, "is_set", lambda: False)():
                break

            self._log_model_decision(
                phase="planner",
                iteration=iteration,
                decision=planner_decision,
                intent=intent,
            )
            response = planner_client.chat(
                messages=conversation,
                temperature=self.temperature,
                enable_thinking=True if thinking else None,
            )
            self._accumulate_usage(total_usage, response.usage)

            requests = self._collect_requests(response, iteration)

            requests = [
                self._normalize_request(
                    request,
                    intent,
                    search_capabilities,
                    prefer_transcript_lookup=prefer_transcript_lookup,
                )
                for request in requests
            ]

            deduped_requests = []
            for request in requests:
                signature = command_signature(request)
                if signature in executed_signatures:
                    continue
                executed_signatures.add(signature)
                deduped_requests.append(request)
            requests = deduped_requests

            blocked_external_video_request = False
            if intent.final_target == "external":
                filtered_requests = []
                for request in requests:
                    if request.visibility == "user" and request.name == "search_videos":
                        blocked_external_video_request = True
                        continue
                    filtered_requests.append(request)
                requests = filtered_requests

            usage_trace.append(
                self._usage_trace_entry(
                    phase="planner",
                    iteration=iteration,
                    decision=planner_decision,
                    usage=response.usage,
                    tool_calls=requests,
                )
            )

            if not requests and self._is_error_response(response):
                recovery_requests = [
                    self._normalize_request(
                        request,
                        intent,
                        search_capabilities,
                        prefer_transcript_lookup=prefer_transcript_lookup,
                    )
                    for request in self._build_deterministic_recovery_requests(
                        messages,
                        intent,
                        prefer_transcript_lookup=prefer_transcript_lookup,
                    )
                ]
                deduped_recovery_requests: list[ToolCallRequest] = []
                for request in recovery_requests:
                    signature = command_signature(request)
                    if signature in executed_signatures:
                        continue
                    executed_signatures.add(signature)
                    deduped_recovery_requests.append(request)

                if deduped_recovery_requests:
                    recovery_records = self._execute_requests(
                        self.result_store,
                        deduped_recovery_requests,
                        intent,
                    )
                    recovery_followup_requests = (
                        self._build_deterministic_followup_requests(
                            self.result_store,
                            intent,
                            messages=messages,
                        )
                    )
                    deduped_recovery_followups: list[ToolCallRequest] = []
                    for request in recovery_followup_requests:
                        signature = command_signature(request)
                        if signature in executed_signatures:
                            continue
                        executed_signatures.add(signature)
                        deduped_recovery_followups.append(request)
                    if deduped_recovery_followups:
                        recovery_records.extend(
                            self._execute_requests(
                                self.result_store,
                                deduped_recovery_followups,
                                intent,
                            )
                        )
                    tool_events.append(
                        self._build_tool_events(iteration, recovery_records)
                    )

                final_content = (
                    self._build_deterministic_final_answer(intent, messages) or ""
                )
                if final_content:
                    break
                if deduped_recovery_requests:
                    break
                break

            user_tool_names = [
                request.name for request in requests if request.visibility == "user"
            ]
            transcript_unavailable_nudge = self._select_transcript_unavailable_nudge(
                transcript_requested_but_unavailable=transcript_requested_but_unavailable,
                user_tool_names=user_tool_names,
                prompted_nudges=prompted_nudges,
            )
            if transcript_unavailable_nudge:
                prompted_nudges.add(transcript_unavailable_nudge[0])
                stripped_content = sanitize_generated_content(response.content or "")
                if stripped_content:
                    conversation.append(
                        {"role": "assistant", "content": stripped_content}
                    )
                conversation.append(
                    {"role": "user", "content": transcript_unavailable_nudge[1]}
                )
                continue

            pre_execution_nudge = select_pre_execution_nudge(
                self.result_store,
                intent,
                user_tool_names,
                prompted_nudges,
            )
            if pre_execution_nudge:
                prompted_nudges.add(pre_execution_nudge[0])
                stripped_content = sanitize_generated_content(response.content or "")
                if stripped_content:
                    conversation.append(
                        {"role": "assistant", "content": stripped_content}
                    )
                conversation.append({"role": "user", "content": pre_execution_nudge[1]})
                continue

            if not requests:
                blocked_request_nudge = select_blocked_request_nudge(
                    intent,
                    blocked_external_video_request,
                    prompted_nudges,
                )
                if blocked_request_nudge:
                    prompted_nudges.add(blocked_request_nudge[0])
                    conversation.append(
                        {"role": "user", "content": blocked_request_nudge[1]}
                    )
                    continue
                final_content = sanitize_generated_content(response.content or "")
                final_reasoning = response.reasoning_content or ""
                if final_content:
                    break
                continue

            records = self._execute_requests(self.result_store, requests, intent)
            followup_requests = self._build_deterministic_followup_requests(
                self.result_store,
                intent,
                messages=messages,
            )
            deduped_followup_requests: list[ToolCallRequest] = []
            for request in followup_requests:
                signature = command_signature(request)
                if signature in executed_signatures:
                    continue
                executed_signatures.add(signature)
                deduped_followup_requests.append(request)
            if deduped_followup_requests:
                records.extend(
                    self._execute_requests(
                        self.result_store,
                        deduped_followup_requests,
                        intent,
                    )
                )
            tool_events.append(self._build_tool_events(iteration, records))

            stripped_content = sanitize_generated_content(response.content or "")
            if stripped_content:
                conversation.append({"role": "assistant", "content": stripped_content})
            conversation.append(
                {
                    "role": "user",
                    "content": self.result_store.render_observation(
                        [record.result_id for record in records]
                    ),
                }
            )

            blocked_request_nudge = select_blocked_request_nudge(
                intent,
                blocked_external_video_request,
                prompted_nudges,
            )
            if blocked_request_nudge:
                prompted_nudges.add(blocked_request_nudge[0])
                conversation.append(
                    {"role": "user", "content": blocked_request_nudge[1]}
                )

            deterministic_lookup_answer = self._build_deterministic_final_answer(
                intent,
                messages,
            )
            if deterministic_lookup_answer and has_target_coverage(
                self.result_store, intent
            ):
                final_content = deterministic_lookup_answer
                break

            if has_target_coverage(self.result_store, intent):
                break

            transcript_post_execution_nudge = (
                self._select_transcript_post_execution_nudge(
                    self.result_store,
                    prefer_transcript_lookup=prefer_transcript_lookup,
                    prompted_nudges=prompted_nudges,
                )
            )
            if transcript_post_execution_nudge:
                prompted_nudges.add(transcript_post_execution_nudge[0])
                conversation.append(
                    {"role": "user", "content": transcript_post_execution_nudge[1]}
                )
                continue

            post_execution_nudge = select_post_execution_nudge(
                self.result_store,
                intent,
                latest_user_message,
                prompted_nudges,
            )
            if post_execution_nudge:
                prompted_nudges.add(post_execution_nudge[0])
                conversation.append(
                    {"role": "user", "content": post_execution_nudge[1]}
                )

        if not final_content:
            conversation.append({"role": "user", "content": FINAL_ANSWER_NUDGE})
            self._log_model_decision(
                phase="response",
                iteration=len(usage_trace) + 1,
                decision=response_decision,
                intent=intent,
            )
            response = response_client.chat(
                messages=conversation,
                temperature=self.temperature,
                enable_thinking=True if thinking else None,
            )
            self._accumulate_usage(total_usage, response.usage)
            usage_trace.append(
                self._usage_trace_entry(
                    phase="response",
                    iteration=len(usage_trace) + 1,
                    decision=response_decision,
                    usage=response.usage,
                    tool_calls=[],
                )
            )
            if self._is_error_response(response):
                fallback_decision = self._select_model(
                    intent,
                    stage="delegate",
                    thinking=False,
                )
                self._log_model_decision(
                    phase="response_fallback",
                    iteration=len(usage_trace) + 1,
                    decision=fallback_decision,
                    intent=intent,
                )
                fallback_response = fallback_decision.client.chat(
                    messages=self._build_fallback_response_messages(
                        intent=intent,
                        result_store=self.result_store,
                    ),
                    temperature=0.2,
                )
                self._accumulate_usage(total_usage, fallback_response.usage)
                usage_trace.append(
                    self._usage_trace_entry(
                        phase="response_fallback",
                        iteration=len(usage_trace) + 1,
                        decision=fallback_decision,
                        usage=fallback_response.usage,
                        tool_calls=[],
                    )
                )
                if self._is_error_response(fallback_response):
                    final_content = "抱歉，最终整理结果时模型服务异常，请稍后重试。"
                else:
                    final_content = sanitize_generated_content(
                        fallback_response.content or ""
                    )
                    final_reasoning = fallback_response.reasoning_content or ""
            else:
                final_content = sanitize_generated_content(response.content or "")
                final_reasoning = response.reasoning_content or ""

        elapsed_seconds = time.perf_counter() - start_time
        normalized_usage = self._normalize_usage(total_usage)
        prompt_tokens = normalized_usage.get("prompt_tokens", 0)
        completion_tokens = normalized_usage.get("completion_tokens", 0)
        if prompt_tokens or completion_tokens:
            normalized_usage["total_tokens"] = prompt_tokens + completion_tokens

        return OrchestrationResult(
            content=final_content or "抱歉，这次没有生成有效回答。",
            reasoning_content=final_reasoning,
            usage=normalized_usage,
            tool_events=tool_events,
            usage_trace={
                "prompt": prompt_profile,
                "intent": {
                    "final_target": intent.final_target,
                    "task_mode": intent.task_mode,
                    "ambiguity": intent.ambiguity,
                    "complexity_score": intent.complexity_score,
                    "needs_term_normalization": intent.needs_term_normalization,
                    "motivation": intent.top_labels("motivation"),
                    "expected_payoff": intent.top_labels("expected_payoff"),
                    "constraints": intent.top_labels("constraints"),
                    "explicit_entities": intent.explicit_entities,
                    "explicit_topics": intent.explicit_topics,
                    "route_reason": intent.route_reason,
                    "prefers_transcript_lookup": prefer_transcript_lookup,
                },
                "models": {
                    "planner": {
                        "config": planner_spec.config_name,
                        "model": planner_spec.model_name,
                        "reason": planner_decision.reason,
                        "factors": list(planner_decision.factors),
                    },
                    "response": {
                        "config": response_spec.config_name,
                        "model": response_spec.model_name,
                        "reason": response_decision.reason,
                        "factors": list(response_decision.factors),
                    },
                    "delegate": {
                        "config": self.model_registry.primary("small").config_name,
                        "model": self.model_registry.primary("small").model_name,
                        "reason": "窄任务统一下放给小模型执行。",
                        "factors": ["delegate", "small_model"],
                    },
                },
                "iterations": usage_trace,
                "elapsed_seconds": elapsed_seconds,
                "result_ids": self.result_store.latest_ids(limit=32),
                "summary": {
                    "llm_calls": len(usage_trace),
                    "tool_iterations": len(tool_events),
                    "peak_prompt_tokens": max(
                        (entry.get("prompt_tokens", 0) for entry in usage_trace),
                        default=0,
                    ),
                    "peak_context_chars": len(prompt),
                },
            },
            prompt_profile=prompt_profile,
            thinking=thinking,
            content_streamed=False,
        )

    def run_stream(
        self,
        *,
        messages: list[dict],
        thinking: bool = False,
        max_iterations: int | None = None,
        cancelled: Optional[object] = None,
    ) -> Generator[dict[str, Any], None, OrchestrationResult]:
        start_time = time.perf_counter()
        (
            search_capabilities,
            intent,
            extra_asset_ids,
            prompt,
            prompt_profile,
            planner_decision,
            response_decision,
            latest_user_message,
        ) = self._prepare_orchestration_context(messages=messages, thinking=thinking)
        prefer_transcript_lookup = bool(extra_asset_ids)
        transcript_requested_but_unavailable = self._is_explicit_transcript_request(
            messages,
            intent,
        ) and not search_capabilities.get("supports_transcript_lookup", False)

        planner_client = planner_decision.client
        planner_spec = planner_decision.spec
        response_client = response_decision.client
        response_spec = response_decision.spec
        conversation = [{"role": "system", "content": prompt}] + list(messages)
        total_usage: dict[str, Any] = {}
        usage_trace: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        executed_signatures: set[str] = set()
        prompted_nudges: set[str] = set()
        self.result_store = ResultStore()
        resolved_iterations = max_iterations or planner_spec.max_iterations
        final_content = ""
        final_reasoning = ""
        content_streamed = False

        for iteration in range(1, resolved_iterations + 1):
            if cancelled is not None and getattr(cancelled, "is_set", lambda: False)():
                break

            self._log_model_decision(
                phase="planner",
                iteration=iteration,
                decision=planner_decision,
                intent=intent,
            )
            yield {
                "delta": self._reasoning_reset_delta("planner", iteration),
            }
            response = yield from self._stream_chat_response(
                client=planner_client,
                messages=conversation,
                thinking=thinking,
                cancelled=cancelled,
            )
            self._accumulate_usage(total_usage, response.usage)

            requests = self._collect_requests(response, iteration)
            requests = [
                self._normalize_request(
                    request,
                    intent,
                    search_capabilities,
                    prefer_transcript_lookup=prefer_transcript_lookup,
                )
                for request in requests
            ]

            deduped_requests = []
            for request in requests:
                signature = command_signature(request)
                if signature in executed_signatures:
                    continue
                executed_signatures.add(signature)
                deduped_requests.append(request)
            requests = deduped_requests

            blocked_external_video_request = False
            if intent.final_target == "external":
                filtered_requests = []
                for request in requests:
                    if request.visibility == "user" and request.name == "search_videos":
                        blocked_external_video_request = True
                        continue
                    filtered_requests.append(request)
                requests = filtered_requests

            usage_trace.append(
                self._usage_trace_entry(
                    phase="planner",
                    iteration=iteration,
                    decision=planner_decision,
                    usage=response.usage,
                    tool_calls=requests,
                )
            )

            if not requests and self._is_error_response(response):
                recovery_requests = [
                    self._normalize_request(
                        request,
                        intent,
                        search_capabilities,
                        prefer_transcript_lookup=prefer_transcript_lookup,
                    )
                    for request in self._build_deterministic_recovery_requests(
                        messages,
                        intent,
                        prefer_transcript_lookup=prefer_transcript_lookup,
                    )
                ]
                deduped_recovery_requests: list[ToolCallRequest] = []
                for request in recovery_requests:
                    signature = command_signature(request)
                    if signature in executed_signatures:
                        continue
                    executed_signatures.add(signature)
                    deduped_recovery_requests.append(request)

                if deduped_recovery_requests:
                    recovery_records = (
                        yield from self._execute_requests_with_stream_events(
                            self.result_store,
                            deduped_recovery_requests,
                            intent,
                            iteration=iteration,
                            cancelled=cancelled,
                        )
                    )
                    recovery_followup_requests = (
                        self._build_deterministic_followup_requests(
                            self.result_store,
                            intent,
                            messages=messages,
                        )
                    )
                    deduped_recovery_followups: list[ToolCallRequest] = []
                    for request in recovery_followup_requests:
                        signature = command_signature(request)
                        if signature in executed_signatures:
                            continue
                        executed_signatures.add(signature)
                        deduped_recovery_followups.append(request)
                    if deduped_recovery_followups:
                        followup_records = (
                            yield from self._execute_requests_with_stream_events(
                                self.result_store,
                                deduped_recovery_followups,
                                intent,
                                iteration=iteration,
                                cancelled=cancelled,
                            )
                        )
                        recovery_records.extend(followup_records)
                    tool_events.append(
                        self._build_tool_events(iteration, recovery_records)
                    )

                deterministic_lookup_answer = self._build_deterministic_final_answer(
                    intent,
                    messages,
                )
                if deterministic_lookup_answer and has_target_coverage(
                    self.result_store, intent
                ):
                    final_content = deterministic_lookup_answer
                    content_streamed = True
                    yield {"delta": {"content": deterministic_lookup_answer}}
                    break
                if deduped_recovery_requests:
                    break
                break

            user_tool_names = [
                request.name for request in requests if request.visibility == "user"
            ]
            transcript_unavailable_nudge = self._select_transcript_unavailable_nudge(
                transcript_requested_but_unavailable=transcript_requested_but_unavailable,
                user_tool_names=user_tool_names,
                prompted_nudges=prompted_nudges,
            )
            if transcript_unavailable_nudge:
                prompted_nudges.add(transcript_unavailable_nudge[0])
                stripped_content = sanitize_generated_content(response.content or "")
                if stripped_content:
                    conversation.append(
                        {"role": "assistant", "content": stripped_content}
                    )
                conversation.append(
                    {"role": "user", "content": transcript_unavailable_nudge[1]}
                )
                continue

            pre_execution_nudge = select_pre_execution_nudge(
                self.result_store,
                intent,
                user_tool_names,
                prompted_nudges,
            )
            if pre_execution_nudge:
                prompted_nudges.add(pre_execution_nudge[0])
                stripped_content = sanitize_generated_content(response.content or "")
                if stripped_content:
                    conversation.append(
                        {"role": "assistant", "content": stripped_content}
                    )
                conversation.append({"role": "user", "content": pre_execution_nudge[1]})
                continue

            if not requests:
                blocked_request_nudge = select_blocked_request_nudge(
                    intent,
                    blocked_external_video_request,
                    prompted_nudges,
                )
                if blocked_request_nudge:
                    prompted_nudges.add(blocked_request_nudge[0])
                    conversation.append(
                        {"role": "user", "content": blocked_request_nudge[1]}
                    )
                    continue
                final_content = sanitize_generated_content(response.content or "")
                final_reasoning = response.reasoning_content or ""
                content_streamed = bool(final_content)
                if final_content:
                    break
                continue

            records = yield from self._execute_requests_with_stream_events(
                self.result_store,
                requests,
                intent,
                iteration=iteration,
                cancelled=cancelled,
            )
            followup_requests = self._build_deterministic_followup_requests(
                self.result_store,
                intent,
                messages=messages,
            )
            deduped_followup_requests: list[ToolCallRequest] = []
            for request in followup_requests:
                signature = command_signature(request)
                if signature in executed_signatures:
                    continue
                executed_signatures.add(signature)
                deduped_followup_requests.append(request)
            if deduped_followup_requests:
                followup_records = yield from self._execute_requests_with_stream_events(
                    self.result_store,
                    deduped_followup_requests,
                    intent,
                    iteration=iteration,
                    cancelled=cancelled,
                )
                records.extend(followup_records)
            completed_event = self._build_tool_events(iteration, records)
            tool_events.append(completed_event)

            stripped_content = sanitize_generated_content(response.content or "")
            if stripped_content:
                conversation.append({"role": "assistant", "content": stripped_content})
            conversation.append(
                {
                    "role": "user",
                    "content": self.result_store.render_observation(
                        [record.result_id for record in records]
                    ),
                }
            )

            blocked_request_nudge = select_blocked_request_nudge(
                intent,
                blocked_external_video_request,
                prompted_nudges,
            )
            if blocked_request_nudge:
                prompted_nudges.add(blocked_request_nudge[0])
                conversation.append(
                    {"role": "user", "content": blocked_request_nudge[1]}
                )

            deterministic_lookup_answer = self._build_deterministic_final_answer(
                intent,
                messages,
            )
            if deterministic_lookup_answer and has_target_coverage(
                self.result_store, intent
            ):
                final_content = deterministic_lookup_answer
                content_streamed = True
                yield {"delta": {"content": deterministic_lookup_answer}}
                break

            if has_target_coverage(self.result_store, intent):
                break

            transcript_post_execution_nudge = (
                self._select_transcript_post_execution_nudge(
                    self.result_store,
                    prefer_transcript_lookup=prefer_transcript_lookup,
                    prompted_nudges=prompted_nudges,
                )
            )
            if transcript_post_execution_nudge:
                prompted_nudges.add(transcript_post_execution_nudge[0])
                conversation.append(
                    {"role": "user", "content": transcript_post_execution_nudge[1]}
                )
                continue

            post_execution_nudge = select_post_execution_nudge(
                self.result_store,
                intent,
                latest_user_message,
                prompted_nudges,
            )
            if post_execution_nudge:
                prompted_nudges.add(post_execution_nudge[0])
                conversation.append(
                    {"role": "user", "content": post_execution_nudge[1]}
                )

        if not final_content:
            conversation.append({"role": "user", "content": FINAL_ANSWER_NUDGE})
            self._log_model_decision(
                phase="response",
                iteration=len(usage_trace) + 1,
                decision=response_decision,
                intent=intent,
            )
            yield {
                "delta": self._reasoning_reset_delta(
                    "response",
                    len(usage_trace) + 1,
                ),
            }
            response = yield from self._stream_chat_response(
                client=response_client,
                messages=conversation,
                thinking=thinking,
                cancelled=cancelled,
            )
            self._accumulate_usage(total_usage, response.usage)
            usage_trace.append(
                self._usage_trace_entry(
                    phase="response",
                    iteration=len(usage_trace) + 1,
                    decision=response_decision,
                    usage=response.usage,
                    tool_calls=[],
                )
            )
            if self._is_error_response(response):
                fallback_decision = self._select_model(
                    intent,
                    stage="delegate",
                    thinking=False,
                )
                self._log_model_decision(
                    phase="response_fallback",
                    iteration=len(usage_trace) + 1,
                    decision=fallback_decision,
                    intent=intent,
                )
                fallback_response = fallback_decision.client.chat(
                    messages=self._build_fallback_response_messages(
                        intent=intent,
                        result_store=self.result_store,
                    ),
                    temperature=0.2,
                )
                self._accumulate_usage(total_usage, fallback_response.usage)
                usage_trace.append(
                    self._usage_trace_entry(
                        phase="response_fallback",
                        iteration=len(usage_trace) + 1,
                        decision=fallback_decision,
                        usage=fallback_response.usage,
                        tool_calls=[],
                    )
                )
                if self._is_error_response(fallback_response):
                    final_content = "抱歉，最终整理结果时模型服务异常，请稍后重试。"
                else:
                    final_content = sanitize_generated_content(
                        fallback_response.content or ""
                    )
                    final_reasoning = fallback_response.reasoning_content or ""
            else:
                final_content = sanitize_generated_content(response.content or "")
                final_reasoning = response.reasoning_content or ""
                content_streamed = bool(final_content)

        elapsed_seconds = time.perf_counter() - start_time
        normalized_usage = self._normalize_usage(total_usage)
        prompt_tokens = normalized_usage.get("prompt_tokens", 0)
        completion_tokens = normalized_usage.get("completion_tokens", 0)
        if prompt_tokens or completion_tokens:
            normalized_usage["total_tokens"] = prompt_tokens + completion_tokens

        return OrchestrationResult(
            content=final_content or "抱歉，这次没有生成有效回答。",
            reasoning_content=final_reasoning,
            usage=normalized_usage,
            tool_events=tool_events,
            usage_trace={
                "prompt": prompt_profile,
                "intent": {
                    "final_target": intent.final_target,
                    "task_mode": intent.task_mode,
                    "ambiguity": intent.ambiguity,
                    "complexity_score": intent.complexity_score,
                    "needs_term_normalization": intent.needs_term_normalization,
                    "motivation": intent.top_labels("motivation"),
                    "expected_payoff": intent.top_labels("expected_payoff"),
                    "constraints": intent.top_labels("constraints"),
                    "explicit_entities": intent.explicit_entities,
                    "explicit_topics": intent.explicit_topics,
                    "route_reason": intent.route_reason,
                    "prefers_transcript_lookup": prefer_transcript_lookup,
                },
                "models": {
                    "planner": {
                        "config": planner_spec.config_name,
                        "model": planner_spec.model_name,
                        "reason": planner_decision.reason,
                        "factors": list(planner_decision.factors),
                    },
                    "response": {
                        "config": response_spec.config_name,
                        "model": response_spec.model_name,
                        "reason": response_decision.reason,
                        "factors": list(response_decision.factors),
                    },
                    "delegate": {
                        "config": self.model_registry.primary("small").config_name,
                        "model": self.model_registry.primary("small").model_name,
                        "reason": "窄任务统一下放给小模型执行。",
                        "factors": ["delegate", "small_model"],
                    },
                },
                "iterations": usage_trace,
                "elapsed_seconds": elapsed_seconds,
                "result_ids": self.result_store.latest_ids(limit=32),
                "summary": {
                    "llm_calls": len(usage_trace),
                    "tool_iterations": len(tool_events),
                    "peak_prompt_tokens": max(
                        (entry.get("prompt_tokens", 0) for entry in usage_trace),
                        default=0,
                    ),
                    "peak_context_chars": len(prompt),
                },
            },
            prompt_profile=prompt_profile,
            thinking=thinking,
            content_streamed=content_streamed,
        )
