from __future__ import annotations

import json
import re
import time
import uuid

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from tclogger import logger

from llms.config import DEFAULT_SMALL_MODEL_CONFIG, ModelRegistry
from llms.prompts.assets import get_prompt_assets
from llms.prompts.copilot import build_system_prompt, build_system_prompt_profile
from llms.protocol import (
    IntentProfile,
    ModelSpec,
    OrchestrationResult,
    ToolCallRequest,
    ToolExecutionRecord,
)
from llms.routing import build_intent_profile
from llms.tools.defs import build_tool_definitions


_SUPPORTED_TOOL_NAMES = (
    "search_videos",
    "search_google",
    "search_owners",
    "related_tokens_by_tokens",
    "related_owners_by_tokens",
    "related_videos_by_videos",
    "related_owners_by_videos",
    "related_videos_by_owners",
    "related_owners_by_owners",
    "read_prompt_assets",
    "inspect_tool_result",
    "run_small_llm_task",
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
_DSML_PATTERN = re.compile(r"<｜.*?｜>")
_DSML_BLOCK_PATTERN = re.compile(
    r"<｜DSML｜function_calls>.*?</｜DSML｜function_calls>", re.DOTALL
)
_FINAL_ANSWER_NUDGE = (
    "请基于已经拿到的结果摘要直接回答用户。不要继续规划，不要再次调用工具。"
)
_REPEATED_MIXED_TOOL_NUDGE = (
    "官方结果和站内视频候选已经拿到一轮。请直接基于现有 result_id 和摘要回答，"
    "不要再次同时调用 search_google 和 search_videos；若仍缺细节，只能 inspect_tool_result。"
)
_ZERO_HIT_MIXED_VIDEO_NUDGE = (
    "站内 search_videos 已经连续 0 命中。若还需要 B 站视频链接，请改用 search_google，"
    "并在 query 中显式加入 site:bilibili.com/video；拿到具体 BV 或链接后直接回答，"
    "不要继续重复 search_videos。"
)
_ZERO_HIT_VIDEO_GOOGLE_NUDGE = (
    "站内 search_videos 已经 0 命中。请改用 search_google，并在 query 中显式加入 "
    "site:bilibili.com/video；优先使用用户提到的明确实体名、作者名或最可能的规范写法，"
    "拿到具体 BV 或链接后直接回答，不要继续重复 search_videos。"
)
_EXTERNAL_NO_VIDEO_NUDGE = (
    "当前任务是只看官方/站外信息，不要调用 search_videos。"
    "请只使用 search_google，必要时再 inspect_tool_result，然后直接回答。"
)
_THINKING_PROMPT = (
    "[思考模式] 你现在处于深度思考模式。请认真拆解任务、谨慎选择提示资产、"
    "尽量把复杂规划留给大模型，把窄任务交给小模型。"
)


def _sanitize_content(content: str) -> str:
    sanitized = _DSML_BLOCK_PATTERN.sub("", content or "")
    sanitized = _DSML_PATTERN.sub("", sanitized)
    sanitized = _TOOL_CMD_PATTERN.sub("", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


def _is_recent_timeline_request(text: str) -> bool:
    normalized = str(text or "")
    return any(
        token in normalized for token in ["最近", "新视频", "近一个月", "最近一个月"]
    )


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
        if value[0] in ["[", "{", '"']:
            return json.loads(value)
    except Exception:
        pass
    if re.fullmatch(r"-?(?:0|[1-9]\d*)", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _parse_xml_tool_calls(content: str, iteration: int) -> list[ToolCallRequest]:
    commands: list[ToolCallRequest] = []
    content_text = str(content or "")
    for index, match in enumerate(_GENERIC_TOOL_CMD_RE.finditer(content_text)):
        args = {}
        attrs = match.group("attrs") or ""
        for attr_match in _TOOL_ATTR_RE.finditer(attrs):
            args[attr_match.group(1)] = _parse_tool_argument(attr_match.group(3))
        commands.append(
            ToolCallRequest(
                id=f"xmlcall_{iteration}_{index}",
                name=match.group("name"),
                arguments=args,
                visibility=(
                    "internal"
                    if match.group("name")
                    in {
                        "read_prompt_assets",
                        "inspect_tool_result",
                        "run_small_llm_task",
                    }
                    else "user"
                ),
                source="xml",
            )
        )
    return commands


def _canonicalize_value(value: Any):
    if isinstance(value, dict):
        return {key: _canonicalize_value(val) for key, val in sorted(value.items())}
    if isinstance(value, list):
        canonical_items = [_canonicalize_value(item) for item in value]
        sortable = all(
            isinstance(item, (str, int, float, bool)) or item is None
            for item in canonical_items
        )
        return sorted(canonical_items) if sortable else canonical_items
    return value


def _command_signature(call: ToolCallRequest) -> str:
    payload = {
        "name": call.name,
        "arguments": _canonicalize_value(call.arguments),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _make_video_url(bvid: str) -> str:
    bvid_text = str(bvid or "").strip()
    return f"https://www.bilibili.com/video/{bvid_text}" if bvid_text else ""


def _make_space_url(mid: Any) -> str:
    mid_text = str(mid or "").strip()
    return f"https://space.bilibili.com/{mid_text}" if mid_text else ""


def _compact_video_hit(hit: dict) -> dict:
    owner = hit.get("owner") or {}
    owner_name = owner.get("name", "") if isinstance(owner, dict) else str(owner or "")
    owner_mid = owner.get("mid") if isinstance(owner, dict) else None
    bvid = str(hit.get("bvid", "") or "")
    return {
        "title": hit.get("title", ""),
        "bvid": bvid,
        "url": _make_video_url(bvid),
        "owner": owner_name,
        "owner_mid": owner_mid,
        "view": ((hit.get("stat") or {}).get("view")),
    }


def _compact_owner(owner: dict) -> dict:
    mid = owner.get("mid")
    return {
        "name": owner.get("name", ""),
        "mid": mid,
        "url": _make_space_url(mid),
        "score": owner.get("score"),
    }


def _compact_google_row(row: dict) -> dict:
    return {
        "title": row.get("title", ""),
        "link": row.get("link", ""),
        "domain": row.get("domain", ""),
        "site_kind": row.get("site_kind", ""),
    }


def _compact_token_option(option: dict) -> dict:
    return {
        "text": option.get("text", ""),
        "score": option.get("score"),
    }


class _ResultStore:
    def __init__(self):
        self.records: dict[str, ToolExecutionRecord] = {}
        self.order: list[str] = []

    def add(self, record: ToolExecutionRecord) -> None:
        self.records[record.result_id] = record
        self.order.append(record.result_id)

    def get(self, result_id: str) -> ToolExecutionRecord | None:
        return self.records.get(result_id)

    def latest_ids(self, limit: int = 5) -> list[str]:
        return self.order[-limit:]

    def render_observation(self, result_ids: list[str]) -> str:
        lines = ["[TOOL_OBSERVATIONS]"]
        for result_id in result_ids:
            record = self.records.get(result_id)
            if record is None:
                continue
            lines.append(
                f"- {result_id} {record.request.name}: {record.summary.get('summary_text', '')}"
            )
        lines.append(
            "如需更细结果，可调用 inspect_tool_result(result_ids=[...])；如需压缩/对比，可调用 run_small_llm_task。"
        )
        lines.append("[/TOOL_OBSERVATIONS]")
        return "\n".join(lines)


def _has_successful_tool_result(result_store: _ResultStore, tool_name: str) -> bool:
    for result_id in reversed(result_store.order):
        record = result_store.get(result_id)
        if record is None or record.request.name != tool_name:
            continue
        result = record.result or {}
        if result.get("error"):
            continue
        if tool_name == "search_google":
            return bool(result.get("results")) or int(result.get("result_count", 0)) > 0
        if tool_name == "search_videos":
            if result.get("hits"):
                return True
            if int(result.get("total_hits", 0) or 0) > 0:
                return True
            nested_results = result.get("results") or []
            for item in nested_results:
                if item.get("error"):
                    continue
                if item.get("hits") or int(item.get("total_hits", 0) or 0) > 0:
                    return True
            continue
        return True
    return False


def _has_google_video_result(result_store: _ResultStore) -> bool:
    for result_id in reversed(result_store.order):
        record = result_store.get(result_id)
        if record is None or record.request.name != "search_google":
            continue
        for row in record.result.get("results") or []:
            link = str(row.get("link", "") or "")
            domain = str(row.get("domain", "") or "")
            site_kind = str(row.get("site_kind", "") or "")
            if "bilibili.com/video/" in link:
                return True
            if domain.endswith("bilibili.com") and site_kind == "video":
                return True
    return False


def _has_non_bilibili_google_result(result_store: _ResultStore) -> bool:
    for result_id in reversed(result_store.order):
        record = result_store.get(result_id)
        if record is None or record.request.name != "search_google":
            continue
        for row in record.result.get("results") or []:
            link = str(row.get("link", "") or "")
            domain = str(row.get("domain", "") or "")
            if "bilibili.com/video/" in link:
                continue
            if domain.endswith("bilibili.com"):
                continue
            return True
    return False


def _has_sufficient_mixed_coverage(result_store: _ResultStore) -> bool:
    return _has_non_bilibili_google_result(result_store) and (
        _has_successful_tool_result(result_store, "search_videos")
        or _has_google_video_result(result_store)
    )


def _has_video_coverage(result_store: _ResultStore) -> bool:
    return _has_successful_tool_result(
        result_store, "search_videos"
    ) or _has_google_video_result(result_store)


def _count_zero_hit_search_videos(result_store: _ResultStore) -> int:
    count = 0
    for result_id in result_store.order:
        record = result_store.get(result_id)
        if record is None or record.request.name != "search_videos":
            continue
        result = record.result or {}
        if result.get("error"):
            continue
        if result.get("hits") or int(result.get("total_hits", 0) or 0) > 0:
            continue
        nested_results = result.get("results") or []
        if nested_results and any(
            item.get("hits") or int(item.get("total_hits", 0) or 0) > 0
            for item in nested_results
            if not item.get("error")
        ):
            continue
        count += 1
    return count


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

    def _select_model(
        self, intent: IntentProfile, stage: str, thinking: bool
    ) -> tuple[object, ModelSpec]:
        needs_tool_orchestration = (
            stage in {"planner", "response"} and intent.final_target != "answer"
        )
        use_large = bool(
            thinking
            or needs_tool_orchestration
            or intent.final_target == "mixed"
            or intent.complexity_score >= 0.55
            or (stage == "planner" and intent.needs_external_search)
            or (stage == "planner" and intent.needs_keyword_expansion)
            or (stage == "response" and intent.task_mode == "collect_compare")
        )
        if stage == "delegate":
            spec = self.model_registry.primary("small")
            return self.small_llm_client, spec
        if use_large:
            spec = self.model_registry.primary("large")
            return self.large_llm_client, spec
        spec = self.model_registry.primary("small")
        return self.small_llm_client, spec

    def _build_internal_tool_defs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_prompt_assets",
                    "description": "读取更高层级的提示资产。用于按 tool_name / levels 获取 detailed 或 examples 说明。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_names": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "levels": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["brief", "detailed", "examples"],
                                },
                            },
                            "asset_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "inspect_tool_result",
                    "description": "读取已执行工具的更细结果摘要。通过 result_ids 和 focus 请求更具体的视图。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "result_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "focus": {"type": "string"},
                            "max_items": {"type": "integer", "default": 5},
                        },
                        "required": ["result_ids"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_small_llm_task",
                    "description": "将窄任务并行委托给小模型，例如关键词压缩、结果归纳、候选对比。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "context": {"type": "string"},
                            "result_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "output_format": {"type": "string"},
                        },
                        "required": ["task"],
                    },
                },
            },
        ]

    def _result_summary(self, result_id: str, tool_name: str, result: dict) -> dict:
        summary_text = ""
        if tool_name == "search_videos":
            if result.get("results"):
                query_summaries = []
                for item in result.get("results", [])[:3]:
                    hits = item.get("hits") or []
                    top_hits = [_compact_video_hit(hit) for hit in hits[:3]]
                    query_summaries.append(
                        {
                            "query": item.get("query", ""),
                            "resolved_query": item.get("resolved_query", ""),
                            "total_hits": item.get("total_hits", len(hits)),
                            "top_hits": top_hits,
                        }
                    )
                summary_text = "; ".join(
                    (
                        f"{item['query']}"
                        + (
                            f" (fallback={item['resolved_query']})"
                            if item.get("resolved_query")
                            else ""
                        )
                        + f" -> {item['total_hits']} hits, top="
                    )
                    + ", ".join(
                        f"{hit['title']}({hit['bvid']})"
                        for hit in item["top_hits"]
                        if hit.get("title")
                    )
                    for item in query_summaries
                )
                return {
                    "result_id": result_id,
                    "tool": tool_name,
                    "queries": query_summaries,
                    "summary_text": summary_text,
                }

            hits = result.get("hits") or []
            top_hits = [_compact_video_hit(hit) for hit in hits[:5]]
            summary_text = f"query={result.get('query', '')}" + (
                f", fallback={result.get('resolved_query', '')}"
                if result.get("resolved_query")
                else ""
            ) + f", total_hits={result.get('total_hits', len(hits))}, " "top_hits=" + ", ".join(
                f"{hit['title']}({hit['bvid']})" for hit in top_hits if hit.get("title")
            )
            return {
                "result_id": result_id,
                "tool": tool_name,
                "query": result.get("query", ""),
                "resolved_query": result.get("resolved_query", ""),
                "total_hits": result.get("total_hits", len(hits)),
                "top_hits": top_hits,
                "summary_text": summary_text,
            }

        if tool_name == "search_owners":
            owners = result.get("owners") or []
            owner_rows = [_compact_owner(owner) for owner in owners[:6]]
            summary_text = f"text={result.get('text', '')}, owners=" + ", ".join(
                owner["name"] for owner in owner_rows if owner.get("name")
            )
            return {
                "result_id": result_id,
                "tool": tool_name,
                "text": result.get("text", ""),
                "owners": owner_rows,
                "summary_text": summary_text,
            }

        if tool_name == "search_google":
            rows = result.get("results") or []
            top_results = [_compact_google_row(row) for row in rows[:5]]
            summary_text = f"query={result.get('query', '')}, top_results=" + "; ".join(
                row["title"] for row in top_results if row.get("title")
            )
            return {
                "result_id": result_id,
                "tool": tool_name,
                "query": result.get("query", ""),
                "results": top_results,
                "summary_text": summary_text,
            }

        if tool_name == "related_tokens_by_tokens":
            options = result.get("options") or []
            top_options = [_compact_token_option(item) for item in options[:8]]
            summary_text = f"text={result.get('text', '')}, token_options=" + ", ".join(
                option["text"] for option in top_options if option.get("text")
            )
            return {
                "result_id": result_id,
                "tool": tool_name,
                "text": result.get("text", ""),
                "options": top_options,
                "summary_text": summary_text,
            }

        if isinstance(result.get("owners"), list):
            owner_rows = [
                _compact_owner(owner) for owner in (result.get("owners") or [])[:8]
            ]
            summary_text = ", ".join(
                f"{owner['name']}({owner['mid']})"
                for owner in owner_rows
                if owner.get("name")
            )
            return {
                "result_id": result_id,
                "tool": tool_name,
                "owners": owner_rows,
                "summary_text": summary_text,
            }

        if isinstance(result.get("videos"), list):
            video_rows = [
                _compact_video_hit(video) for video in (result.get("videos") or [])[:8]
            ]
            summary_text = ", ".join(
                f"{video['title']}({video['bvid']})"
                for video in video_rows
                if video.get("title")
            )
            return {
                "result_id": result_id,
                "tool": tool_name,
                "videos": video_rows,
                "summary_text": summary_text,
            }

        summary_text = json.dumps(result, ensure_ascii=False)[:480]
        return {
            "result_id": result_id,
            "tool": tool_name,
            "summary_text": summary_text,
        }

    def _inspect_result(self, result_store: _ResultStore, args: dict) -> dict:
        result_ids = list(args.get("result_ids") or [])
        focus = str(args.get("focus", "summary") or "summary")
        max_items = int(args.get("max_items", 5) or 5)
        inspected = []
        for result_id in result_ids:
            record = result_store.get(result_id)
            if record is None:
                continue
            if record.request.name == "search_videos":
                payload = {"result_id": result_id, "focus": focus}
                if record.result.get("results"):
                    payload["queries"] = []
                    for item in record.result.get("results", [])[:max_items]:
                        payload["queries"].append(
                            {
                                "query": item.get("query", ""),
                                "resolved_query": item.get("resolved_query", ""),
                                "hits": [
                                    _compact_video_hit(hit)
                                    for hit in (item.get("hits") or [])[:max_items]
                                ],
                            }
                        )
                else:
                    payload["query"] = record.result.get("query", "")
                    if record.result.get("resolved_query"):
                        payload["resolved_query"] = record.result.get(
                            "resolved_query", ""
                        )
                    payload["hits"] = [
                        _compact_video_hit(hit)
                        for hit in (record.result.get("hits") or [])[:max_items]
                    ]
                inspected.append(payload)
                continue

            if record.request.name == "search_google":
                inspected.append(
                    {
                        "result_id": result_id,
                        "focus": focus,
                        "query": record.result.get("query", ""),
                        "results": [
                            _compact_google_row(row)
                            for row in (record.result.get("results") or [])[:max_items]
                        ],
                    }
                )
                continue

            if record.request.name == "search_owners" or isinstance(
                record.result.get("owners"), list
            ):
                inspected.append(
                    {
                        "result_id": result_id,
                        "focus": focus,
                        "text": record.result.get("text", ""),
                        "owners": [
                            _compact_owner(owner)
                            for owner in (record.result.get("owners") or [])[:max_items]
                        ],
                    }
                )
                continue

            if record.request.name == "related_tokens_by_tokens":
                inspected.append(
                    {
                        "result_id": result_id,
                        "focus": focus,
                        "text": record.result.get("text", ""),
                        "options": [
                            _compact_token_option(option)
                            for option in (record.result.get("options") or [])[
                                :max_items
                            ]
                        ],
                    }
                )
                continue

            inspected.append(
                {
                    "result_id": result_id,
                    "focus": focus,
                    "summary": record.summary,
                }
            )

        return {"focus": focus, "results": inspected}

    def _run_small_task(
        self, result_store: _ResultStore, args: dict, intent: IntentProfile
    ) -> dict:
        task = str(args.get("task", "")).strip()
        if not task:
            return {"error": "Missing task"}
        context_parts = []
        if args.get("context"):
            context_parts.append(str(args.get("context")))
        for result_id in list(args.get("result_ids") or []):
            record = result_store.get(result_id)
            if record is not None:
                context_parts.append(json.dumps(record.summary, ensure_ascii=False))
        context = "\n".join(context_parts).strip()
        client, spec = self._select_model(intent, stage="delegate", thinking=False)
        messages = [
            {
                "role": "system",
                "content": "你是搜索编排系统的小模型执行器。只完成窄任务，输出紧凑、结构化、可直接复用的结果。",
            },
            {
                "role": "user",
                "content": (
                    f"任务: {task}\n"
                    f"输出格式: {args.get('output_format', '简洁中文要点')}\n"
                    f"当前意图: {intent.final_target} / {intent.task_mode}\n"
                    f"上下文:\n{context or '[无补充上下文]'}"
                ),
            },
        ]
        response = client.chat(messages=messages, temperature=0.2)
        return {
            "task": task,
            "model": spec.config_name,
            "model_name": spec.model_name,
            "result": _sanitize_content(response.content or ""),
        }

    def _execute_internal_call(
        self,
        result_store: _ResultStore,
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

    def _execute_requests(
        self,
        result_store: _ResultStore,
        requests: list[ToolCallRequest],
        intent: IntentProfile,
    ) -> list[ToolExecutionRecord]:
        if not requests:
            return []

        def run_one(request: ToolCallRequest) -> ToolExecutionRecord:
            if request.visibility == "internal":
                result = self._execute_internal_call(result_store, request, intent)
            else:
                result = self.tool_executor.execute_request(request)
            result_id = f"R{len(result_store.order) + 1}"
            summary = self._result_summary(result_id, request.name, result)
            record = ToolExecutionRecord(
                result_id=result_id,
                request=request,
                result=result,
                summary=summary,
                visibility=request.visibility,
            )
            result_store.add(record)
            return record

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
        for key, value in (usage or {}).items():
            if isinstance(value, (int, float)):
                total[key] = total.get(key, 0) + value
            elif isinstance(value, dict):
                total.setdefault(key, {})
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        total[key][sub_key] = total[key].get(sub_key, 0) + sub_value

    def _normalize_usage(self, usage: dict) -> dict:
        result = dict(usage or {})
        prompt_details = result.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached = prompt_details.get("cached_tokens", 0)
            if cached and not result.get("prompt_cache_hit_tokens"):
                result["prompt_cache_hit_tokens"] = cached
                result["prompt_cache_miss_tokens"] = max(
                    0, result.get("prompt_tokens", 0) - cached
                )
        completion_details = result.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning = completion_details.get("reasoning_tokens", 0)
            if reasoning:
                result["reasoning_tokens"] = reasoning
        for key in list(result.keys()):
            if isinstance(result[key], dict):
                del result[key]
        return result

    def _usage_trace_entry(
        self,
        *,
        phase: str,
        iteration: int,
        model_spec: ModelSpec,
        usage: dict,
        tool_calls: list[ToolCallRequest] | None = None,
    ) -> dict:
        normalized_usage = self._normalize_usage(usage)
        return {
            "phase": phase,
            "iteration": iteration,
            "model_config": model_spec.config_name,
            "model_name": model_spec.model_name,
            "prompt_tokens": normalized_usage.get("prompt_tokens", 0),
            "completion_tokens": normalized_usage.get("completion_tokens", 0),
            "total_tokens": normalized_usage.get("total_tokens", 0),
            "tool_count": len(tool_calls or []),
            "tool_names": [call.name for call in tool_calls or []],
        }

    def _build_tool_messages(
        self,
        response,
        records: list[ToolExecutionRecord],
    ) -> list[dict]:
        messages: list[dict] = []
        if response.tool_calls:
            messages.append(response.to_message_dict())
            for tool_call, record in zip(response.tool_calls, records):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(record.summary, ensure_ascii=False),
                    }
                )
            return messages

        analysis = _sanitize_content(response.content or "")
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

    def run(
        self,
        *,
        messages: list[dict],
        thinking: bool = False,
        max_iterations: int | None = None,
        cancelled: Optional[object] = None,
    ) -> OrchestrationResult:
        start_time = time.perf_counter()
        search_capabilities = self.tool_executor.get_search_capabilities()
        intent = build_intent_profile(messages)
        prompt = build_system_prompt(
            capabilities=search_capabilities,
            intent=intent,
        )
        if thinking:
            prompt = _THINKING_PROMPT + "\n\n" + prompt
        prompt_profile = build_system_prompt_profile(
            capabilities=search_capabilities,
            intent=intent,
        )
        if thinking:
            prompt_profile = {
                **prompt_profile,
                "thinking_prompt_chars": len(_THINKING_PROMPT),
                "total_chars": prompt_profile.get("total_chars", 0)
                + len(_THINKING_PROMPT),
            }
        tool_defs = build_tool_definitions(
            search_capabilities,
            include_internal=True,
        )

        planner_client, planner_spec = self._select_model(
            intent,
            stage="planner",
            thinking=thinking,
        )
        response_client, response_spec = self._select_model(
            intent,
            stage="response",
            thinking=thinking,
        )

        conversation = [{"role": "system", "content": prompt}] + list(messages)
        latest_user_message = next(
            (
                str(message.get("content") or "")
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        total_usage: dict[str, Any] = {}
        usage_trace: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        executed_signatures: set[str] = set()
        self.result_store = _ResultStore()
        resolved_iterations = max_iterations or planner_spec.max_iterations
        final_content = ""
        mixed_video_fallback_prompted = False
        video_google_fallback_prompted = False
        external_video_block_prompted = False

        for iteration in range(1, resolved_iterations + 1):
            if cancelled is not None and getattr(cancelled, "is_set", lambda: False)():
                break

            response = planner_client.chat(
                messages=conversation,
                tools=tool_defs,
                temperature=self.temperature,
                enable_thinking=True if thinking else None,
            )
            self._accumulate_usage(total_usage, response.usage)

            requests = []
            for tool_call in response.tool_calls:
                visibility = (
                    "internal"
                    if tool_call.name
                    in {
                        "read_prompt_assets",
                        "inspect_tool_result",
                        "run_small_llm_task",
                    }
                    else "user"
                )
                requests.append(
                    ToolCallRequest(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.parse_arguments(),
                        visibility=visibility,
                        source="function_call",
                    )
                )
            if not requests:
                requests = _parse_xml_tool_calls(response.content or "", iteration)

            deduped_requests = []
            for request in requests:
                signature = _command_signature(request)
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
                    model_spec=planner_spec,
                    usage=response.usage,
                    tool_calls=requests,
                )
            )

            user_tool_names = [
                request.name for request in requests if request.visibility == "user"
            ]
            mixed_results_already_sufficient = (
                intent.final_target == "mixed"
                and _has_sufficient_mixed_coverage(self.result_store)
                and any(
                    tool_name in {"search_google", "search_videos"}
                    for tool_name in user_tool_names
                )
            )

            if mixed_results_already_sufficient:
                stripped_content = _sanitize_content(response.content or "")
                if stripped_content:
                    conversation.append(
                        {"role": "assistant", "content": stripped_content}
                    )
                conversation.append(
                    {"role": "user", "content": _REPEATED_MIXED_TOOL_NUDGE}
                )
                break

            if not requests:
                if blocked_external_video_request and not external_video_block_prompted:
                    conversation.append(
                        {"role": "user", "content": _EXTERNAL_NO_VIDEO_NUDGE}
                    )
                    external_video_block_prompted = True
                    continue
                final_content = _sanitize_content(response.content or "")
                if final_content:
                    break
                continue

            records = self._execute_requests(self.result_store, requests, intent)
            tool_events.append(self._build_tool_events(iteration, records))

            if response.tool_calls:
                conversation.append(response.to_message_dict())
                for tool_call, record in zip(response.tool_calls, records):
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(record.summary, ensure_ascii=False),
                        }
                    )
            else:
                stripped_content = _sanitize_content(response.content or "")
                if stripped_content:
                    conversation.append(
                        {"role": "assistant", "content": stripped_content}
                    )
                conversation.append(
                    {
                        "role": "user",
                        "content": self.result_store.render_observation(
                            [record.result_id for record in records]
                        ),
                    }
                )

            if blocked_external_video_request and not external_video_block_prompted:
                conversation.append(
                    {"role": "user", "content": _EXTERNAL_NO_VIDEO_NUDGE}
                )
                external_video_block_prompted = True

            if intent.final_target == "external" and _has_successful_tool_result(
                self.result_store, "search_google"
            ):
                break

            if intent.final_target == "mixed" and _has_sufficient_mixed_coverage(
                self.result_store
            ):
                break

            if intent.final_target == "videos" and _has_video_coverage(
                self.result_store
            ):
                break

            if (
                intent.final_target == "mixed"
                and not mixed_video_fallback_prompted
                and not _has_sufficient_mixed_coverage(self.result_store)
                and _has_successful_tool_result(self.result_store, "search_google")
                and not _has_google_video_result(self.result_store)
                and _count_zero_hit_search_videos(self.result_store) >= 1
            ):
                conversation.append(
                    {"role": "user", "content": _ZERO_HIT_MIXED_VIDEO_NUDGE}
                )
                mixed_video_fallback_prompted = True

            if (
                intent.final_target == "videos"
                and not video_google_fallback_prompted
                and not _has_video_coverage(self.result_store)
                and _count_zero_hit_search_videos(self.result_store) >= 1
                and (intent.explicit_entities or intent.explicit_topics)
                and not _is_recent_timeline_request(latest_user_message)
            ):
                conversation.append(
                    {"role": "user", "content": _ZERO_HIT_VIDEO_GOOGLE_NUDGE}
                )
                video_google_fallback_prompted = True

        if not final_content:
            conversation.append({"role": "user", "content": _FINAL_ANSWER_NUDGE})
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
                    model_spec=response_spec,
                    usage=response.usage,
                    tool_calls=[],
                )
            )
            final_content = _sanitize_content(response.content or "")

        elapsed_seconds = time.perf_counter() - start_time
        normalized_usage = self._normalize_usage(total_usage)
        prompt_tokens = normalized_usage.get("prompt_tokens", 0)
        completion_tokens = normalized_usage.get("completion_tokens", 0)
        if prompt_tokens or completion_tokens:
            normalized_usage["total_tokens"] = prompt_tokens + completion_tokens

        return OrchestrationResult(
            content=final_content or "抱歉，这次没有生成有效回答。",
            usage=normalized_usage,
            tool_events=tool_events,
            usage_trace={
                "prompt": prompt_profile,
                "intent": {
                    "final_target": intent.final_target,
                    "task_mode": intent.task_mode,
                    "ambiguity": intent.ambiguity,
                    "complexity_score": intent.complexity_score,
                    "motivation": intent.top_labels("motivation"),
                    "expected_payoff": intent.top_labels("expected_payoff"),
                    "constraints": intent.top_labels("constraints"),
                    "explicit_entities": intent.explicit_entities,
                    "explicit_topics": intent.explicit_topics,
                    "route_reason": intent.route_reason,
                },
                "models": {
                    "planner": {
                        "config": planner_spec.config_name,
                        "model": planner_spec.model_name,
                    },
                    "response": {
                        "config": response_spec.config_name,
                        "model": response_spec.model_name,
                    },
                    "delegate": {
                        "config": self.model_registry.primary("small").config_name,
                        "model": self.model_registry.primary("small").model_name,
                    },
                },
                "iterations": usage_trace,
                "elapsed_seconds": elapsed_seconds,
                "result_ids": self.result_store.latest_ids(limit=32),
                "summary": {
                    "llm_calls": len(usage_trace),
                    "tool_iterations": sum(
                        1 for entry in usage_trace if entry.get("tool_count", 0) > 0
                    ),
                    "peak_prompt_tokens": max(
                        (entry.get("prompt_tokens", 0) for entry in usage_trace),
                        default=0,
                    ),
                    "peak_context_chars": len(prompt),
                },
            },
            prompt_profile=prompt_profile,
            thinking=thinking,
        )
