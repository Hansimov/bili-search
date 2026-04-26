from __future__ import annotations

import json
import re
import time

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generator, Optional

from llms.contracts import (
    IntentProfile,
    OrchestrationResult,
    ToolCallRequest,
    ToolExecutionRecord,
)
from llms.intent import build_intent_profile
from llms.intent.focus import compact_focus_key
from llms.intent.focus import rewrite_known_term_aliases
from llms.intent.focus import select_primary_focus_term
from llms.messages import (
    extract_bvids,
    extract_message_text,
)
from llms.models import ChatResponse
from llms.models import DEFAULT_SMALL_MODEL_CONFIG, ModelRegistry
from llms.orchestration.policies import FINAL_ANSWER_NUDGE
from llms.orchestration.policies import has_explicit_video_anchor
from llms.orchestration.policies import has_successful_tool_result
from llms.orchestration.policies import has_target_coverage
from llms.orchestration.policies import is_recent_timeline_request
from llms.orchestration.policies import needs_short_ambiguous_dual_exploration
from llms.orchestration.policies import needs_short_identity_video_evidence
from llms.orchestration.policies import select_blocked_request_nudge
from llms.orchestration.policies import select_post_execution_nudge
from llms.orchestration.policies import select_pre_execution_nudge
from llms.orchestration.result_store import ResultStore
from llms.orchestration.result_store import inspect_results
from llms.orchestration.result_store import summarize_result
from llms.orchestration.deterministic import DeterministicOrchestrationMixin
from llms.orchestration.runtime import OrchestrationRuntimeMixin
from llms.orchestration.streaming import OrchestrationStreamingMixin
from llms.orchestration.tool_markup import command_signature
from llms.orchestration.tool_markup import parse_xml_tool_calls
from llms.orchestration.tool_markup import sanitize_generated_content
from llms.orchestration.transcripts import TranscriptOrchestrationMixin
from llms.orchestration.video_queries import VideoQueryNormalizer
from llms.prompts.assets import get_prompt_assets
from llms.tools.video_lookup import normalize_search_video_lookup_arguments
from llms.tools.names import canonical_tool_name


def _rewrite_known_aliases_in_video_search_arguments(arguments: dict) -> dict:
    normalized = dict(arguments or {})
    raw_queries = normalized.get("queries")
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    elif raw_queries is None and isinstance(normalized.get("query"), str):
        raw_queries = [normalized.get("query")]
    elif not isinstance(raw_queries, list):
        raw_queries = []

    rewritten: list[str] = []
    changed = False
    for query in raw_queries:
        original_query = str(query or "")
        rewritten_query = rewrite_known_term_aliases(original_query) or original_query
        if rewritten_query != original_query:
            changed = True
        if rewritten_query and rewritten_query not in rewritten:
            rewritten.append(rewritten_query)
    if not changed or not rewritten:
        return normalized

    normalized.pop("query", None)
    normalized["queries"] = rewritten
    return normalized


class ChatOrchestrator(
    DeterministicOrchestrationMixin,
    TranscriptOrchestrationMixin,
    OrchestrationRuntimeMixin,
    OrchestrationStreamingMixin,
):
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
    def _trim_after_last_alnum_anchor(text: str) -> str:
        stripped = VideoQueryNormalizer.clean_subject_text(text)
        if not stripped:
            return ""
        last_anchor = -1
        for index, char in enumerate(stripped):
            if char.isascii() and char.isalnum():
                last_anchor = index
        if last_anchor < 0 or last_anchor >= len(stripped) - 1:
            return stripped
        suffix = stripped[last_anchor + 1 :]
        if suffix[:1].isspace():
            return stripped
        if len(compact_focus_key(suffix)) <= 8:
            return VideoQueryNormalizer.clean_subject_text(stripped[: last_anchor + 1])
        return stripped

    @staticmethod
    def _owner_resolution_seed(intent: IntentProfile) -> str:
        candidates: list[str] = []
        for candidate in [
            *(intent.explicit_entities or []),
            *(intent.explicit_topics or []),
        ]:
            cleaned = ChatOrchestrator._trim_after_last_alnum_anchor(candidate)
            if any(mark in cleaned for mark in "？?。！，,；;：:"):
                cleaned = ChatOrchestrator._trim_after_last_alnum_anchor(cleaned)
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
        return select_primary_focus_term(candidates)

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

    def _augment_short_ambiguous_exploration_requests(
        self,
        requests: list[ToolCallRequest],
        intent: IntentProfile,
    ) -> list[ToolCallRequest]:
        if not needs_short_ambiguous_dual_exploration(intent):
            return requests
        user_tool_names = {
            canonical_tool_name(request.name)
            for request in requests
            if request.visibility == "user"
        }
        if "search_owners" in user_tool_names:
            return requests
        if not user_tool_names.intersection({"expand_query", "search_videos"}):
            return requests
        owner_seed = self._owner_resolution_seed(
            intent
        ) or VideoQueryNormalizer.clean_subject_text(intent.raw_query)
        if not owner_seed:
            return requests
        return [
            *requests,
            ToolCallRequest(
                id="auto_short_ambiguous_owner_probe_1",
                name="search_owners",
                arguments={"text": owner_seed, "size": 8},
                visibility="user",
                source="workflow_gate",
            ),
        ]

    def _augment_short_identity_video_evidence_requests(
        self,
        requests: list[ToolCallRequest],
        intent: IntentProfile,
    ) -> list[ToolCallRequest]:
        if not needs_short_identity_video_evidence(intent):
            return requests
        user_tool_names = {
            canonical_tool_name(request.name)
            for request in requests
            if request.visibility == "user"
        }
        if "search_owners" not in user_tool_names or "search_videos" in user_tool_names:
            return requests
        has_planned_owner_search = any(
            request.visibility == "user"
            and canonical_tool_name(request.name) == "search_owners"
            and request.source != "workflow_gate"
            for request in requests
        )
        if not has_planned_owner_search:
            return requests
        video_seed = self._owner_resolution_seed(
            intent
        ) or VideoQueryNormalizer.clean_subject_text(intent.raw_query)
        if not video_seed:
            return requests
        return [
            *requests,
            ToolCallRequest(
                id="auto_short_identity_video_evidence_1",
                name="search_videos",
                arguments={"queries": [video_seed], "limit": 8},
                visibility="user",
                source="workflow_gate",
            ),
        ]

    def _augment_followup_video_evidence_requests(
        self,
        requests: list[ToolCallRequest],
        intent: IntentProfile,
    ) -> list[ToolCallRequest]:
        if (
            intent.final_target != "videos"
            or not intent.is_followup
            or intent.task_mode != "exploration"
            or is_recent_timeline_request(intent)
        ):
            return requests
        user_tool_names = {
            canonical_tool_name(request.name)
            for request in requests
            if request.visibility == "user"
        }
        if "search_owners" not in user_tool_names or "search_videos" in user_tool_names:
            return requests
        video_seed = self._owner_resolution_seed(
            intent
        ) or VideoQueryNormalizer.clean_subject_text(intent.raw_query)
        if not video_seed:
            return requests
        return [
            *requests,
            ToolCallRequest(
                id="auto_followup_video_evidence_1",
                name="search_videos",
                arguments={"queries": [video_seed], "limit": 8},
                visibility="user",
                source="workflow_gate",
            ),
        ]

    @staticmethod
    def _query_text_matches_seed(query_text: str, seed_text: str) -> bool:
        query_key = compact_focus_key(query_text)
        seed_key = compact_focus_key(seed_text)
        if not query_key or not seed_key:
            return False
        return seed_key in query_key or query_key in seed_key

    def _clean_user_tool_text(self, text: str, intent: IntentProfile) -> str:
        cleaned = VideoQueryNormalizer.clean_subject_text(text)
        seed = self._owner_resolution_seed(intent)
        if not seed:
            return cleaned
        cleaned_key = compact_focus_key(cleaned)
        seed_key = compact_focus_key(seed)
        if not cleaned_key:
            return seed
        if self._query_text_matches_seed(cleaned, seed):
            return cleaned
        if len(cleaned_key) > 24 or len(seed_key) >= 2:
            return seed
        return cleaned

    def _clean_video_query_text(self, text: str, intent: IntentProfile) -> str:
        cleaned = VideoQueryNormalizer.clean_subject_text(text)
        seed = self._owner_resolution_seed(intent)
        if not seed:
            return cleaned
        cleaned_key = compact_focus_key(cleaned)
        if len(cleaned_key) <= 24 or self._query_text_matches_seed(cleaned, seed):
            return cleaned
        return seed

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
        title_like_video_query = ""
        explicit_dsl_video_query = ""
        if intent.final_target == "videos" or name in {"search_videos", "search_owners"}:
            title_like_video_query = (
                VideoQueryNormalizer.extract_title_like_video_query(intent.raw_query)
            )
            explicit_dsl_video_query = VideoQueryNormalizer.extract_explicit_dsl_query(
                intent.raw_query
            )
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
            arguments = normalize_search_video_lookup_arguments(arguments)
            if title_like_video_query or explicit_dsl_video_query:
                arguments = (
                    VideoQueryNormalizer.normalize_title_like_video_search_arguments(
                        arguments,
                        intent.raw_query,
                    )
                )
            if not any(
                arguments.get(key)
                for key in ("bv", "bvid", "bvids", "mid", "mids", "uid", "uids")
            ):
                raw_queries = arguments.get("queries")
                if isinstance(raw_queries, str):
                    queries = [raw_queries]
                elif isinstance(raw_queries, list):
                    queries = [str(query or "") for query in raw_queries]
                else:
                    query = str(arguments.get("query", "") or "").strip()
                    queries = [query] if query else []
                cleaned_queries = []
                changed = False
                for query in queries:
                    cleaned_query = self._clean_video_query_text(query, intent)
                    if cleaned_query != query:
                        changed = True
                    if cleaned_query and cleaned_query not in cleaned_queries:
                        cleaned_queries.append(cleaned_query)
                if changed and cleaned_queries:
                    arguments.pop("query", None)
                    arguments["queries"] = cleaned_queries
            arguments = _rewrite_known_aliases_in_video_search_arguments(arguments)
        if name == "search_owners":
            if (
                intent.final_target == "videos"
                and intent.is_followup
                and (
                    not intent.needs_owner_resolution
                    or intent.task_mode == "exploration"
                )
                and not is_recent_timeline_request(intent)
            ):
                video_seed = self._clean_user_tool_text(
                    self._owner_request_text(arguments) or intent.raw_query,
                    intent,
                )
                if video_seed:
                    return ToolCallRequest(
                        id=request.id,
                        name="search_videos",
                        arguments={"queries": [video_seed]},
                        visibility=request.visibility,
                        source=request.source,
                    )
            if title_like_video_query and intent.final_target not in {
                "owners",
                "relations",
            }:
                return ToolCallRequest(
                    id=request.id,
                    name="search_videos",
                    arguments={"queries": [title_like_video_query]},
                    visibility=request.visibility,
                    source=request.source,
                )
            owner_text = self._owner_request_text(arguments)
            if not owner_text:
                owner_seed = self._owner_resolution_seed(intent)
                if owner_seed:
                    arguments["text"] = owner_seed
                    owner_text = owner_seed
            elif not str(arguments.get("text", "") or "").strip():
                arguments["text"] = owner_text
            else:
                arguments["text"] = self._clean_user_tool_text(owner_text, intent)
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
                rewritten_query = (
                    rewrite_known_term_aliases(original_query) or original_query
                )
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
    def _is_scoped_or_lookup_video_request(request: ToolCallRequest) -> bool:
        arguments = request.arguments or {}
        mode = str(arguments.get("mode") or "").strip().lower()
        if mode == "lookup":
            return True
        if any(
            arguments.get(key)
            for key in ("bv", "bvid", "bvids", "mid", "mids", "uid", "uids")
        ):
            return True

        raw_queries = arguments.get("queries")
        if isinstance(raw_queries, str):
            queries = [raw_queries]
        elif isinstance(raw_queries, (list, tuple)):
            queries = [str(query or "") for query in raw_queries]
        else:
            query = str(arguments.get("query", "") or "").strip()
            queries = [query] if query else []

        scoped_markers = (":uid=", ":user=", ":mid=")
        return any(
            any(marker in str(query or "") for marker in scoped_markers)
            for query in queries
        )

    @classmethod
    def _is_exact_scoped_video_request(cls, request: ToolCallRequest) -> bool:
        arguments = request.arguments or {}
        mode = str(arguments.get("mode") or "").strip().lower()
        if mode == "lookup":
            return True
        if any(
            arguments.get(key)
            for key in ("bv", "bvid", "bvids", "mid", "mids", "uid", "uids")
        ):
            return True

        raw_queries = arguments.get("queries")
        if isinstance(raw_queries, str):
            queries = [raw_queries]
        elif isinstance(raw_queries, (list, tuple)):
            queries = [str(query or "") for query in raw_queries]
        else:
            query = str(arguments.get("query", "") or "").strip()
            queries = [query] if query else []
        return any(":uid=" in str(query or "") for query in queries)

    @staticmethod
    def _is_explicit_video_anchor_lookup_request(request: ToolCallRequest) -> bool:
        arguments = request.arguments or {}
        if any(arguments.get(key) for key in ("bv", "bvid", "bvids")):
            return True

        if str(arguments.get("mode") or "").strip().lower() != "lookup":
            return False
        return any(arguments.get(key) for key in ("bv", "bvid", "bvids"))

    @staticmethod
    def _owner_seed_from_video_requests(requests: list[ToolCallRequest]) -> str:
        for request in requests:
            if (
                request.visibility != "user"
                or canonical_tool_name(request.name) != "search_videos"
            ):
                continue
            arguments = request.arguments or {}
            raw_queries = arguments.get("queries")
            if isinstance(raw_queries, str):
                queries = [raw_queries]
            elif isinstance(raw_queries, (list, tuple)):
                queries = [str(query or "") for query in raw_queries]
            else:
                query = str(arguments.get("query", "") or "").strip()
                queries = [query] if query else []
            for query in queries:
                match = re.search(r":user=(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))", query)
                if not match:
                    continue
                owner_seed = next(
                    (group for group in match.groups() if group),
                    "",
                )
                owner_seed = VideoQueryNormalizer.clean_subject_text(owner_seed)
                if owner_seed:
                    return owner_seed
        return ""

    def _filter_owner_timeline_pre_resolution_requests(
        self,
        requests: list[ToolCallRequest],
        intent: IntentProfile,
        messages: list[dict],
    ) -> list[ToolCallRequest]:
        if intent.final_target != "videos":
            return requests
        has_owner_resolution = any(
            request.visibility == "user"
            and canonical_tool_name(request.name) == "search_owners"
            for request in requests
        )
        has_exact_video_lookup = any(
            request.visibility == "user"
            and canonical_tool_name(request.name) == "search_videos"
            and self._is_exact_scoped_video_request(request)
            for request in requests
        )
        requires_owner_first = (
            is_recent_timeline_request(intent)
            or intent.needs_owner_resolution
            or has_owner_resolution
        )
        if (
            requires_owner_first
            and not has_owner_resolution
            and not has_exact_video_lookup
        ):
            needs_owner_resolution = any(
                request.visibility == "user"
                and canonical_tool_name(request.name)
                in {"expand_query", "search_videos"}
                for request in requests
            )
            owner_subject = self._owner_seed_from_video_requests(
                requests
            ) or self._owner_resolution_seed(intent)
            if needs_owner_resolution and owner_subject:
                internal_requests = [
                    request for request in requests if request.visibility == "internal"
                ]
                return [
                    *internal_requests,
                    ToolCallRequest(
                        id="auto_owner_recent_pre_resolution_1",
                        name="search_owners",
                        arguments={"text": owner_subject, "size": 5},
                        visibility="user",
                        source="workflow_gate",
                    ),
                ]
        if not has_owner_resolution:
            return requests

        filtered: list[ToolCallRequest] = []
        for request in requests:
            if (
                request.visibility == "user"
                and canonical_tool_name(request.name) == "expand_query"
            ):
                continue
            if (
                request.visibility == "user"
                and canonical_tool_name(request.name) == "search_videos"
                and not self._is_explicit_video_anchor_lookup_request(request)
            ):
                continue
            filtered.append(request)
        return filtered

    def _build_direct_explicit_dsl_video_search_request(
        self,
        messages: list[dict],
        intent: IntentProfile,
        *,
        prefer_transcript_lookup: bool = False,
    ) -> ToolCallRequest | None:
        if prefer_transcript_lookup or intent.final_target != "videos":
            return None
        if has_explicit_video_anchor(intent) or is_recent_timeline_request(intent):
            return None

        latest_user_text = self._latest_user_text(messages)
        explicit_query = VideoQueryNormalizer.extract_explicit_dsl_query(
            latest_user_text
        ) or VideoQueryNormalizer.extract_explicit_dsl_query(intent.raw_query)
        if not explicit_query:
            return None

        return ToolCallRequest(
            id="auto_explicit_dsl_video_search_1",
            name="search_videos",
            arguments={"queries": [explicit_query]},
            visibility="user",
            source="deterministic_direct",
        )

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

        direct_request = self._build_direct_explicit_dsl_video_search_request(
            messages,
            intent,
            prefer_transcript_lookup=prefer_transcript_lookup,
        )
        if direct_request is not None:
            records = self._execute_requests(
                self.result_store, [direct_request], intent
            )
            tool_events.append(self._build_tool_events(1, records))
            final_content = (
                self._build_deterministic_final_answer(intent, messages)
                or "已完成搜索，但这次没有整理出可展示的视频结果。"
            )
            elapsed_seconds = time.perf_counter() - start_time
            return OrchestrationResult(
                content=final_content,
                reasoning_content="",
                usage={},
                tool_events=tool_events,
                usage_trace={
                    "prompt": prompt_profile,
                    "intent": {
                        "final_target": intent.final_target,
                        "task_mode": intent.task_mode,
                        "route_reason": intent.route_reason,
                        "deterministic_direct": "explicit_dsl_video_search",
                    },
                    "iterations": [
                        {
                            "phase": "deterministic_direct",
                            "tool_calls": [direct_request.name],
                        }
                    ],
                    "elapsed_seconds": elapsed_seconds,
                    "result_ids": self.result_store.latest_ids(limit=32),
                    "summary": {
                        "llm_calls": 0,
                        "tool_iterations": len(tool_events),
                        "peak_prompt_tokens": 0,
                        "peak_context_chars": len(prompt),
                    },
                },
                prompt_profile=prompt_profile,
                thinking=thinking,
                content_streamed=False,
            )

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
            requests = self._filter_owner_timeline_pre_resolution_requests(
                requests,
                intent,
                messages,
            )
            requests = self._augment_short_ambiguous_exploration_requests(
                requests,
                intent,
            )
            requests = self._augment_short_identity_video_evidence_requests(
                requests,
                intent,
            )
            requests = self._augment_followup_video_evidence_requests(
                requests,
                intent,
            )

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

            requests, blocked_google_request = self._filter_google_unavailable_requests(
                requests,
                search_capabilities,
            )

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
                    recovery_followup_requests = [
                        self._normalize_request(
                            request,
                            intent,
                            search_capabilities,
                            prefer_transcript_lookup=prefer_transcript_lookup,
                        )
                        for request in recovery_followup_requests
                    ]
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
                google_unavailable_nudge = self._select_google_unavailable_nudge(
                    blocked_google_request=blocked_google_request,
                    prompted_nudges=prompted_nudges,
                )
                if google_unavailable_nudge:
                    prompted_nudges.add(google_unavailable_nudge[0])
                    conversation.append(
                        {"role": "user", "content": google_unavailable_nudge[1]}
                    )
                    continue
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
                if not has_target_coverage(self.result_store, intent):
                    recovery_requests = []
                    if not transcript_requested_but_unavailable:
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
                        records = self._execute_requests(
                            self.result_store,
                            deduped_recovery_requests,
                            intent,
                        )
                        tool_events.append(self._build_tool_events(iteration, records))
                        stripped_content = sanitize_generated_content(
                            response.content or ""
                        )
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
                        deterministic_lookup_answer = (
                            self._build_deterministic_final_answer(intent, messages)
                        )
                        if deterministic_lookup_answer and has_target_coverage(
                            self.result_store, intent
                        ):
                            final_content = deterministic_lookup_answer
                            break
                        if has_target_coverage(self.result_store, intent):
                            break
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
            followup_requests = [
                self._normalize_request(
                    request,
                    intent,
                    search_capabilities,
                    prefer_transcript_lookup=prefer_transcript_lookup,
                )
                for request in followup_requests
            ]
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

            google_unavailable_nudge = self._select_google_unavailable_nudge(
                blocked_google_request=blocked_google_request,
                prompted_nudges=prompted_nudges,
            )
            if google_unavailable_nudge:
                prompted_nudges.add(google_unavailable_nudge[0])
                conversation.append(
                    {"role": "user", "content": google_unavailable_nudge[1]}
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
                        "provider": planner_spec.provider,
                        "api_format": planner_spec.api_format,
                        "model": planner_spec.model_name,
                        "reason": planner_decision.reason,
                        "factors": list(planner_decision.factors),
                    },
                    "response": {
                        "config": response_spec.config_name,
                        "provider": response_spec.provider,
                        "api_format": response_spec.api_format,
                        "model": response_spec.model_name,
                        "reason": response_decision.reason,
                        "factors": list(response_decision.factors),
                    },
                    "delegate": {
                        "config": self.model_registry.primary("small").config_name,
                        "provider": self.model_registry.primary("small").provider,
                        "api_format": self.model_registry.primary("small").api_format,
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
