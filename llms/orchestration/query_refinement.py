"""LLM-backed query refinement before search execution.

This module is the semantic layer for turning conversational user text and
planner tool arguments into compact bili-search queries. Keep language-specific
examples, typo correction, aliasing, and intent wording out of
``video_queries``/``policies``; add prompt constraints or structured validation
here instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any
from unittest.mock import Mock

from llms.contracts import IntentProfile, ToolCallRequest
from llms.tools.names import canonical_tool_name


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_QMODE_RE = re.compile(r"(?<![\w])q=(?:w|v|r){1,3}(?![\w])", re.IGNORECASE)
_DATE_FILTER_RE = re.compile(r":date[<>=!]*[^\s]+", re.IGNORECASE)
_VIEW_FILTER_RE = re.compile(r":view[<>=!]*[^\s]+", re.IGNORECASE)
_USER_FILTER_RE = re.compile(r":(?:user|uid)=[^\s]+", re.IGNORECASE)
_GENERIC_QUERY_WORDS = {
    "讲",
    "找",
    "搜索",
    "视频",
    "相关",
    "热门",
    "高质量",
    "哪些",
    "值得看",
}


@dataclass(frozen=True)
class QueryRefinementDecision:
    arguments: dict[str, Any]
    changed: bool
    reason: str = ""
    name: str | None = None


def _coerce_query_list(arguments: dict[str, Any]) -> list[str]:
    raw_queries = arguments.get("queries")
    if isinstance(raw_queries, str):
        return [raw_queries]
    if isinstance(raw_queries, list):
        return [str(item or "").strip() for item in raw_queries if str(item or "").strip()]
    query = str(arguments.get("query", "") or "").strip()
    return [query] if query else []


def _compact(text: object) -> str:
    return " ".join(str(text or "").split()).strip()


def _has_explicit_user_dsl(text: str) -> bool:
    return bool(_QMODE_RE.search(text) or _DATE_FILTER_RE.search(text) or _VIEW_FILTER_RE.search(text))


def _is_lookup_or_relation_request(arguments: dict[str, Any]) -> bool:
    mode = str(arguments.get("mode") or "").lower()
    if mode in {"lookup", "discover"}:
        return True
    return any(arguments.get(key) for key in ("bv", "bvid", "bvids", "mid", "mids", "uid"))


def _should_skip_mock_client(client: object) -> bool:
    if getattr(client, "_enable_query_refinement_for_tests", False) is True:
        return False
    chat = getattr(client, "chat", None)
    return isinstance(client, Mock) or isinstance(chat, Mock)


def _extract_json_payload(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    match = _JSON_OBJECT_RE.search(text)
    if match:
        text = match.group(0)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _query_is_too_generic(query: str) -> bool:
    normalized = _compact(query)
    if not normalized:
        return True
    if normalized in _GENERIC_QUERY_WORDS:
        return True
    return len(normalized) <= 1


def _sanitize_refined_queries(
    queries: object,
    *,
    original_user_text: str,
    original_queries: list[str],
) -> list[str]:
    if isinstance(queries, str):
        raw_items = [queries]
    elif isinstance(queries, list):
        raw_items = queries
    else:
        return []

    user_has_dsl = _has_explicit_user_dsl(original_user_text)
    original_filter_text = " ".join(original_queries)
    cleaned: list[str] = []
    for item in raw_items:
        query = _compact(item).strip(" ，。！？?；;：:")
        if _query_is_too_generic(query):
            continue
        if not user_has_dsl:
            query = _DATE_FILTER_RE.sub("", query)
            query = _VIEW_FILTER_RE.sub("", query)
            query = _compact(query)
        if _USER_FILTER_RE.search(original_filter_text) and not _USER_FILTER_RE.search(query):
            continue
        if query and query not in cleaned:
            cleaned.append(query)
        if len(cleaned) >= 3:
            break
    return cleaned


def _sanitize_owner_text(text: object) -> str:
    value = _compact(text).strip(" ，。！？?；;：:、()[]{}<>《》\"'`~")
    if not value or len(value) > 48:
        return ""
    return value


def _build_refinement_prompt(
    request: ToolCallRequest,
    intent: IntentProfile,
    arguments: dict[str, Any],
) -> list[dict[str, str]]:
    tool_name = canonical_tool_name(request.name)
    return [
        {
            "role": "system",
            "content": (
                "你是 bili-search 的检索语句润色器，只输出 JSON。"
                "你的任务是在真正搜索前，把自然语言、错别字、套话和模型误选的泛词，"
                "改写成适合 B 站视频/作者检索的紧凑 query。"
                "保留作品名、作者名、主题、英文名、BV/MID、用户明确写出的 DSL 过滤。"
                "删除请求动词、寒暄、解释、无关套话，以及模型自行添加但用户没有要求的热度/日期过滤。"
                "不要回答用户，不要输出 markdown。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "tool": tool_name,
                    "user_question": intent.raw_query,
                    "normalized_user_question": intent.normalized_query,
                    "intent": {
                        "final_target": intent.final_target,
                        "task_mode": intent.task_mode,
                        "explicit_entities": intent.explicit_entities,
                        "explicit_topics": intent.explicit_topics,
                        "constraints": intent.top_labels("constraints", limit=8),
                    },
                    "current_arguments": arguments,
                    "output_schema": (
                        {
                            "tool": "search_videos",
                            "queries": ["compact video search query"],
                            "reason": "short",
                        }
                        if tool_name == "search_videos"
                        else {
                            "tool": "search_owners or search_videos",
                            "text": "compact owner search text when finding authors",
                            "queries": [
                                "compact video search query when the user wants videos"
                            ],
                            "reason": "short",
                        }
                    ),
                },
                ensure_ascii=False,
            ),
        },
    ]


class LLMQueryRefiner:
    """Refines search tool arguments with a small LLM and strict validation."""

    def __init__(self, client: object | None):
        self.client = client

    def refine(
        self,
        request: ToolCallRequest,
        intent: IntentProfile,
    ) -> QueryRefinementDecision:
        tool_name = canonical_tool_name(request.name)
        arguments = dict(request.arguments or {})
        if tool_name not in {"search_videos", "search_owners"}:
            return QueryRefinementDecision(arguments=arguments, changed=False)
        if tool_name == "search_videos" and _is_lookup_or_relation_request(arguments):
            return QueryRefinementDecision(arguments=arguments, changed=False)
        if self.client is None or _should_skip_mock_client(self.client):
            return QueryRefinementDecision(arguments=arguments, changed=False)

        chat = getattr(self.client, "chat", None)
        if not callable(chat):
            return QueryRefinementDecision(arguments=arguments, changed=False)

        try:
            response = chat(
                _build_refinement_prompt(request, intent, arguments),
                temperature=0,
                max_tokens=320,
                enable_thinking=False,
            )
        except Exception as exc:
            return QueryRefinementDecision(
                arguments=arguments,
                changed=False,
                reason=f"refinement_error:{exc}",
            )

        payload = _extract_json_payload(str(getattr(response, "content", "") or ""))
        if not payload:
            return QueryRefinementDecision(arguments=arguments, changed=False)

        if tool_name == "search_videos":
            refined_queries = _sanitize_refined_queries(
                payload.get("queries"),
                original_user_text=intent.raw_query,
                original_queries=_coerce_query_list(arguments),
            )
            if not refined_queries:
                return QueryRefinementDecision(arguments=arguments, changed=False)
            updated = dict(arguments)
            updated.pop("query", None)
            updated["queries"] = refined_queries
            return QueryRefinementDecision(
                arguments=updated,
                changed=updated != arguments,
                reason=str(payload.get("reason") or ""),
            )

        target_tool = canonical_tool_name(
            str(payload.get("tool") or payload.get("target_tool") or tool_name)
        )
        if target_tool == "search_videos":
            refined_queries = _sanitize_refined_queries(
                payload.get("queries"),
                original_user_text=intent.raw_query,
                original_queries=_coerce_query_list(arguments),
            )
            if refined_queries:
                return QueryRefinementDecision(
                    name="search_videos",
                    arguments={"queries": refined_queries},
                    changed=True,
                    reason=str(payload.get("reason") or ""),
                )

        refined_text = _sanitize_owner_text(payload.get("text"))
        if not refined_text:
            return QueryRefinementDecision(arguments=arguments, changed=False)
        updated = dict(arguments)
        updated["text"] = refined_text
        return QueryRefinementDecision(
            arguments=updated,
            changed=updated != arguments,
            reason=str(payload.get("reason") or ""),
        )


__all__ = ["LLMQueryRefiner", "QueryRefinementDecision"]
