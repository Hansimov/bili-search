from __future__ import annotations

import time

from typing import Any, Generator, Optional

from llms.contracts import OrchestrationResult, ToolCallRequest
from llms.orchestration.policies import FINAL_ANSWER_NUDGE
from llms.orchestration.policies import has_target_coverage
from llms.orchestration.policies import select_blocked_request_nudge
from llms.orchestration.policies import select_post_execution_nudge
from llms.orchestration.policies import select_pre_execution_nudge
from llms.orchestration.result_store import ResultStore
from llms.orchestration.tool_markup import command_signature
from llms.orchestration.tool_markup import sanitize_generated_content


class OrchestrationStreamingMixin:
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

        direct_request = self._build_direct_explicit_dsl_video_search_request(
            messages,
            intent,
            prefer_transcript_lookup=prefer_transcript_lookup,
        )
        if direct_request is not None:
            records = yield from self._execute_requests_with_stream_events(
                self.result_store,
                [direct_request],
                intent,
                iteration=1,
                cancelled=cancelled,
            )
            completed_event = self._build_tool_events(1, records)
            tool_events.append(completed_event)
            final_content = (
                self._build_deterministic_final_answer(intent, messages)
                or "已完成搜索，但这次没有整理出可展示的视频结果。"
            )
            content_streamed = True
            yield {"delta": {"content": final_content}}
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
                content_streamed=content_streamed,
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
            requests = self._filter_owner_timeline_pre_resolution_requests(
                requests,
                intent,
                messages,
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
                    recovery_records = yield from self._execute_requests_with_stream_events(
                        self.result_store,
                        deduped_recovery_requests,
                        intent,
                        iteration=iteration,
                        cancelled=cancelled,
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
                        followup_records = yield from self._execute_requests_with_stream_events(
                            self.result_store,
                            deduped_recovery_followups,
                            intent,
                            iteration=iteration,
                            cancelled=cancelled,
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
            content_streamed=content_streamed,
        )
