from __future__ import annotations

import re

from typing import Any

from llms.contracts import IntentProfile, ToolCallRequest
from llms.messages import (
    extract_bvids,
    extract_message_text,
    extract_owner_mids,
    normalize_bvid_key,
)
from llms.orchestration.policies import (
    has_explicit_video_anchor,
    is_recent_timeline_request,
    needs_explicit_video_lookup_followup,
)
from llms.orchestration.explicit_answers import ExplicitDeterministicAnswerMixin
from llms.orchestration.result_store import ResultStore
from llms.orchestration.tool_markup import command_signature
from llms.orchestration.video_queries import VideoQueryNormalizer
from llms.tools.names import canonical_tool_name


class DeterministicOrchestrationMixin(ExplicitDeterministicAnswerMixin):
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

    @classmethod
    def _intent_owner_seed(cls, intent: IntentProfile) -> str:
        candidate_texts: list[str] = [
            *(intent.explicit_entities or []),
            *(intent.explicit_topics or []),
        ]
        for candidate in candidate_texts:
            cleaned = VideoQueryNormalizer.clean_subject_text(candidate)
            if cleaned and len(cleaned) <= 32:
                return cleaned
        return ""

    def _build_deterministic_recovery_requests(
        self,
        messages: list[dict],
        intent: IntentProfile,
        *,
        prefer_transcript_lookup: bool = False,
    ) -> list[ToolCallRequest]:
        if intent.final_target in {"owners", "relations"}:
            owner_subject = self._intent_owner_seed(intent)
            if not owner_subject:
                return []
            return [
                ToolCallRequest(
                    id="auto_recover_owner_lookup_1",
                    name="search_owners",
                    arguments={"text": owner_subject, "mode": "name"},
                    visibility="user",
                    source="deterministic_recovery",
                )
            ]

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

        owner_subject = self._intent_owner_seed(intent)
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
        # Do not auto-promote search_owners results into video searches here.
        # Author candidates must be shown back to the planner so the large LLM can
        # decide whether the top owner is correct, whether multiple owners matter,
        # or whether another owner search is needed.
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

        deduped_requests: list[ToolCallRequest] = []
        seen_signatures: set[str] = set()
        for request in followup_requests:
            signature = command_signature(request)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped_requests.append(request)
        return deduped_requests
