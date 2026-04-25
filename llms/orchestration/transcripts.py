from __future__ import annotations

import json
import re

from typing import Any, Generator, Optional

from tclogger import ts_to_str

from llms.contracts import IntentProfile, ToolExecutionRecord
from llms.messages import extract_bvids
from llms.orchestration.result_store import ResultStore
from llms.orchestration.tool_markup import sanitize_generated_content
from llms.tools.names import canonical_tool_name


TRANSCRIPT_CONTEXT_CHAR_LIMIT = 24000
TRANSCRIPT_METADATA_DESC_LIMIT = 1200
TRANSCRIPT_METADATA_TAG_LIMIT = 12
TRANSCRIPT_SMALL_TASK_CANONICAL_TASK = "整理转写：主题概括 + 覆盖全片要点"
TRANSCRIPT_SMALL_TASK_CANONICAL_OUTPUT_FORMAT = "主题概括+中文要点"


class TranscriptOrchestrationMixin:
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

        normalized["task"] = TRANSCRIPT_SMALL_TASK_CANONICAL_TASK
        normalized["output_format"] = TRANSCRIPT_SMALL_TASK_CANONICAL_OUTPUT_FORMAT

        if self._looks_like_generated_transcript_context(normalized.get("context")):
            normalized.pop("context", None)
        elif not str(normalized.get("context", "") or "").strip():
            normalized.pop("context", None)

        return normalized

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
            if len(normalized) >= TRANSCRIPT_METADATA_TAG_LIMIT:
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
            limit=TRANSCRIPT_METADATA_DESC_LIMIT,
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
    ):
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
                trimmed_transcript = transcript_text[:TRANSCRIPT_CONTEXT_CHAR_LIMIT]
                transcript_context_truncated = (
                    len(transcript_text) > TRANSCRIPT_CONTEXT_CHAR_LIMIT
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
        decision,
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
