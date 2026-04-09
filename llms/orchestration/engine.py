from __future__ import annotations

import json
import time

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generator, Optional

from tclogger import logger

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
from llms.models import ChatResponse
from llms.models import DEFAULT_SMALL_MODEL_CONFIG, ModelRegistry
from llms.orchestration.policies import FINAL_ANSWER_NUDGE
from llms.orchestration.policies import has_target_coverage
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
from llms.prompts.assets import get_prompt_assets
from llms.prompts.copilot import build_system_prompt, build_system_prompt_profile
from llms.runtime.usage import accumulate_usage, normalize_usage
from llms.tools.names import canonical_tool_name


_THINKING_PROMPT = (
    "[思考模式] 你现在处于深度思考模式。请认真拆解任务、谨慎选择提示资产、"
    "尽量把复杂规划留给大模型，把窄任务交给小模型。"
)

_INTERNAL_TOOL_NAMES = {
    "read_prompt_assets",
    "inspect_tool_result",
    "run_small_llm_task",
}


def _shared_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


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

    def _normalize_request(
        self,
        request: ToolCallRequest,
        intent: IntentProfile,
        search_capabilities: dict,
    ) -> ToolCallRequest:
        if request.visibility != "user":
            return request
        name = canonical_tool_name(request.name)
        arguments = dict(request.arguments or {})
        if name == "search_owners":
            owner_text = self._owner_request_text(arguments)
            if not owner_text:
                owner_seed = self._owner_resolution_seed(intent)
                if owner_seed:
                    arguments["text"] = owner_seed
                    owner_text = owner_seed
            elif not str(arguments.get("text", "") or "").strip():
                arguments["text"] = owner_text
            if (
                arguments.get("text")
                and str(arguments.get("mode", "auto") or "auto") == "auto"
                and intent.final_target in {"owners", "relations"}
                and intent.task_mode == "exploration"
            ):
                arguments["mode"] = "topic"
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

    def _collect_requests(
        self,
        response: ChatResponse,
        iteration: int,
    ) -> list[ToolCallRequest]:
        requests: list[ToolCallRequest] = []
        for tool_call in response.tool_calls:
            name = canonical_tool_name(tool_call.name)
            requests.append(
                ToolCallRequest(
                    id=tool_call.id,
                    name=name,
                    arguments=tool_call.parse_arguments(),
                    visibility=("internal" if name in _INTERNAL_TOOL_NAMES else "user"),
                    source="function_call",
                )
            )
        xml_requests = parse_xml_tool_calls(response.content or "", iteration)
        if not requests:
            return xml_requests
        seen_signatures = {command_signature(request) for request in requests}
        for request in xml_requests:
            signature = command_signature(request)
            if signature in seen_signatures:
                continue
            requests.append(request)
            seen_signatures.add(signature)
        return requests

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
        return summarize_result(result_id, tool_name, result)

    def _inspect_result(self, result_store: ResultStore, args: dict) -> dict:
        return inspect_results(result_store, args)

    def _run_small_task(
        self, result_store: ResultStore, args: dict, intent: IntentProfile
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
            "result": sanitize_generated_content(response.content or ""),
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

    def _execute_requests(
        self,
        result_store: ResultStore,
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
        accumulate_usage(total, usage)

    def _normalize_usage(self, usage: dict) -> dict:
        return normalize_usage(usage)

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

    def _build_pending_tool_event(
        self,
        iteration: int,
        requests: list[ToolCallRequest],
    ) -> dict:
        return {
            "iteration": iteration,
            "tools": [request.name for request in requests],
            "calls": [
                {
                    "type": request.name,
                    "args": request.arguments,
                    "status": "pending",
                    "visibility": request.visibility,
                }
                for request in requests
            ],
        }

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
        prompted_nudges: set[str] = set()
        self.result_store = ResultStore()
        resolved_iterations = max_iterations or planner_spec.max_iterations
        final_content = ""
        final_reasoning = ""

        for iteration in range(1, resolved_iterations + 1):
            if cancelled is not None and getattr(cancelled, "is_set", lambda: False)():
                break

            response = planner_client.chat(
                messages=conversation,
                temperature=self.temperature,
                enable_thinking=True if thinking else None,
            )
            self._accumulate_usage(total_usage, response.usage)

            requests = self._collect_requests(response, iteration)

            requests = [
                self._normalize_request(request, intent, search_capabilities)
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
                    model_spec=planner_spec,
                    usage=response.usage,
                    tool_calls=requests,
                )
            )

            user_tool_names = [
                request.name for request in requests if request.visibility == "user"
            ]
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
                stripped_content = sanitize_generated_content(response.content or "")
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

            if has_target_coverage(self.result_store, intent):
                break

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
            if self._is_error_response(response):
                fallback_client, fallback_spec = self._select_model(
                    intent,
                    stage="delegate",
                    thinking=False,
                )
                fallback_response = fallback_client.chat(
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
                        model_spec=fallback_spec,
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
        prompted_nudges: set[str] = set()
        self.result_store = ResultStore()
        resolved_iterations = max_iterations or planner_spec.max_iterations
        final_content = ""
        final_reasoning = ""
        content_streamed = False

        for iteration in range(1, resolved_iterations + 1):
            if cancelled is not None and getattr(cancelled, "is_set", lambda: False)():
                break

            response = yield from self._stream_chat_response(
                client=planner_client,
                messages=conversation,
                thinking=thinking,
                cancelled=cancelled,
            )
            self._accumulate_usage(total_usage, response.usage)

            requests = self._collect_requests(response, iteration)
            requests = [
                self._normalize_request(request, intent, search_capabilities)
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
                    model_spec=planner_spec,
                    usage=response.usage,
                    tool_calls=requests,
                )
            )

            user_tool_names = [
                request.name for request in requests if request.visibility == "user"
            ]
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

            yield {"tool_events": [self._build_pending_tool_event(iteration, requests)]}
            records = self._execute_requests(self.result_store, requests, intent)
            completed_event = self._build_tool_events(iteration, records)
            tool_events.append(completed_event)
            yield {"tool_events": [completed_event]}

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
                stripped_content = sanitize_generated_content(response.content or "")
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

            if has_target_coverage(self.result_store, intent):
                break

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
                    model_spec=response_spec,
                    usage=response.usage,
                    tool_calls=[],
                )
            )
            if self._is_error_response(response):
                fallback_client, fallback_spec = self._select_model(
                    intent,
                    stage="delegate",
                    thinking=False,
                )
                fallback_response = fallback_client.chat(
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
                        model_spec=fallback_spec,
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
            content_streamed=content_streamed,
        )
