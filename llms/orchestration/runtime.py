from __future__ import annotations

import re

from typing import Any, Generator, Optional

from tclogger import logger

from llms.contracts import IntentProfile, ToolCallRequest, ToolExecutionRecord
from llms.intent import build_intent_profile
from llms.messages import extract_bvids, extract_message_text
from llms.models import ChatResponse
from llms.orchestration.policies import has_successful_tool_result
from llms.orchestration.result_store import ResultStore
from llms.orchestration.tool_markup import ALL_TOOL_PREFIXES
from llms.orchestration.tool_markup import find_tool_command_start
from llms.orchestration.tool_markup import partial_tool_prefix_len
from llms.orchestration.tool_markup import sanitize_generated_content
from llms.orchestration.types import ModelDecision
from llms.prompts.assets import get_prompt_assets
from llms.prompts.copilot import build_system_prompt, build_system_prompt_profile
from llms.runtime.usage import accumulate_usage, normalize_usage
from llms.tools.names import canonical_tool_name


_TRANSCRIPT_HINT_RE = re.compile(
    r"(讲了什么|讲啥|说了什么|说啥|主要讲|主要内容|总结|摘要|概括|梗概|重点|字幕|转写|音频)",
    re.IGNORECASE,
)
_EXPLICIT_TRANSCRIPT_REQUEST_RE = re.compile(
    r"(音频转写文本|音频转写|转写文本|完整转写|完整字幕|字幕原文|逐字稿|transcript|subtitles?)",
    re.IGNORECASE,
)
_THINKING_PROMPT = (
    "[思考模式] 你现在处于深度思考模式。请认真拆解任务、谨慎选择提示资产、"
    "尽量把复杂规划留给大模型，把窄任务交给小模型。"
)


def _shared_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


class OrchestrationRuntimeMixin:
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

    def _accumulate_usage(self, total: dict, usage: dict):
        accumulate_usage(total, usage)

    def _normalize_usage(self, usage: dict) -> dict:
        return normalize_usage(usage)

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

    @staticmethod
    def _filter_google_unavailable_requests(
        requests: list[ToolCallRequest],
        search_capabilities: dict,
    ) -> tuple[list[ToolCallRequest], bool]:
        if search_capabilities.get("supports_google_search", False):
            return requests, False
        filtered_requests = []
        blocked_google_request = False
        for request in requests:
            if (
                request.visibility == "user"
                and canonical_tool_name(request.name) == "search_google"
            ):
                blocked_google_request = True
                continue
            filtered_requests.append(request)
        return filtered_requests, blocked_google_request

    @staticmethod
    def _select_google_unavailable_nudge(
        *,
        blocked_google_request: bool,
        prompted_nudges: set[str],
    ) -> tuple[str, str] | None:
        if not blocked_google_request:
            return None
        if "google_search_unavailable" in prompted_nudges:
            return None
        return (
            "google_search_unavailable",
            "当前环境没有 search_google 工具。请不要再调用 search_google；"
            "本轮只使用 search_videos、search_owners、expand_query 等站内工具，"
            "或基于已有站内结果直接回答。",
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
            intent.final_target == "videos"
            and intent.is_followup
            and intent.needs_owner_resolution
        ):
            large_factors.append("video_followup_context")
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
            f"config={decision.spec.config_name}, provider={decision.spec.provider}, model={decision.spec.model_name}, "
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
            "model_provider": decision.spec.provider,
            "model_api_format": decision.spec.api_format,
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
        payload = OrchestrationRuntimeMixin._pending_tool_call_payload(request)
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
