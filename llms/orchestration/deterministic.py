from __future__ import annotations

import re
import unicodedata

from typing import Any

from llms.contracts import IntentProfile, ToolCallRequest
from llms.intent.focus import rewrite_known_term_aliases
from llms.messages import (
    extract_bvids,
    extract_message_text,
    extract_owner_mids,
    normalize_bvid_key,
)
from llms.orchestration.policies import (
    has_explicit_video_anchor,
    has_successful_tool_result,
    is_recent_timeline_request,
    needs_explicit_video_lookup_followup,
)
from llms.orchestration.result_store import ResultStore
from llms.orchestration.tool_markup import command_signature
from llms.orchestration.video_queries import VideoQueryNormalizer
from llms.planning.owner_resolution import OwnerResolutionMixin
from llms.tools.names import canonical_tool_name


RECENT_OWNER_QUERY_PATTERNS = (
    re.compile(
        r"^(?P<subject>.+?)(?:最近|近期|近况)(?:还)?(?:发了|发布了|上传了|更新了)?(?:哪些|什么)?(?:视频|作品).*$"
    ),
    re.compile(
        r"^(?P<subject>.+?)(?:还)?(?:发了|发布了|上传了|更新了)(?:哪些|什么)?(?:视频|作品).*$"
    ),
)
RECENT_OWNER_SUBJECT_STOP_RE = re.compile(
    r"(最近|近期|近况|视频|作品|作者|发了|发布了|上传了|更新了|还发了|什么|哪些|谁|是谁)",
    re.IGNORECASE,
)


class DeterministicOrchestrationMixin:
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
                digits: list[str] = []
                while index < len(source) and source[index].isdigit():
                    try:
                        digits.append(str(unicodedata.digit(source[index])))
                    except (TypeError, ValueError):
                        break
                    index += 1
                if not digits:
                    index = start + 1
                    continue
                value = int("".join(digits))
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

    @classmethod
    def _extract_leading_subject_phrase(cls, latest_user_text: str) -> str:
        source = str(latest_user_text or "").strip()
        if not source:
            return ""

        segments = [
            (match.group(0), match.start(), match.end())
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
        return VideoQueryNormalizer.clean_subject_text(
            source[subject_start:subject_end]
        )

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
            cleaned = VideoQueryNormalizer.clean_subject_text(candidate)
            if (
                cleaned
                and len(cleaned) <= 32
                and not RECENT_OWNER_SUBJECT_STOP_RE.search(cleaned)
            ):
                return rewrite_known_term_aliases(cleaned) or cleaned

        normalized_source = "".join(str(latest_user_text or "").split())
        for pattern in RECENT_OWNER_QUERY_PATTERNS:
            match = pattern.match(normalized_source)
            if not match:
                continue
            subject = VideoQueryNormalizer.clean_subject_text(match.group("subject"))
            if subject and len(subject) <= 32:
                return rewrite_known_term_aliases(subject) or subject

        leading_subject = cls._extract_leading_subject_phrase(latest_user_text)
        if (
            leading_subject
            and len(leading_subject) <= 32
            and not RECENT_OWNER_SUBJECT_STOP_RE.search(leading_subject)
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

    @classmethod
    def _trim_owner_from_focus_query(
        cls,
        focus_query: str,
        owner_texts: list[str],
    ) -> str:
        trimmed = " ".join(str(focus_query or "").split()).strip()
        if not trimmed:
            return ""

        for owner_text in owner_texts:
            normalized_owner = " ".join(str(owner_text or "").split()).strip()
            if not normalized_owner:
                continue
            if trimmed.startswith(normalized_owner):
                candidate = trimmed[len(normalized_owner) :].strip(" ，。！？?；;：:")
                if candidate:
                    trimmed = candidate
        return trimmed

    @classmethod
    def _select_video_followup_owner_candidate(
        cls,
        result_store: ResultStore,
        intent: IntentProfile,
    ) -> tuple[str, dict | None]:
        topic_hints = [
            hint
            for hint in (intent.explicit_topics or [])
            if len(VideoQueryNormalizer.clean_subject_text(hint)) >= 2
        ]
        best_source = ""
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
            source_text = VideoQueryNormalizer.clean_subject_text(source_text)
            if not source_text:
                continue

            candidate = OwnerResolutionMixin._select_best_owner_candidate(
                source_text,
                owners,
                hint_tokens=topic_hints,
            )
            if not OwnerResolutionMixin._is_confident_owner_candidate(
                source_text,
                candidate,
                hint_tokens=topic_hints,
                owners=owners,
            ):
                continue

            try:
                service_score = float((candidate or {}).get("score") or 0.0)
            except (TypeError, ValueError):
                service_score = 0.0
            if service_score > best_rank:
                best_rank = service_score
                best_source = source_text
                best_owner = candidate

        return best_source, best_owner

    def _build_video_gap_followup_requests(
        self,
        result_store: ResultStore,
        intent: IntentProfile,
        messages: list[dict] | None = None,
    ) -> list[ToolCallRequest]:
        if intent.final_target != "videos" or has_explicit_video_anchor(intent):
            return []
        if is_recent_timeline_request(intent):
            return []
        if has_successful_tool_result(result_store, "search_videos"):
            return []

        latest_user_text = self._latest_user_text(list(messages or []))
        focus_query = VideoQueryNormalizer.build_video_followup_focus_query(
            latest_user_text,
            explicit_entities=intent.explicit_entities,
            explicit_topics=intent.explicit_topics,
        )
        if not focus_query:
            return []

        owner_source, owner_candidate = self._select_video_followup_owner_candidate(
            result_store,
            intent,
        )
        queries: list[str] = []
        if owner_candidate and owner_candidate.get("mid"):
            owner_texts = [owner_source, str(owner_candidate.get("name") or "")]
            scoped_topic = self._trim_owner_from_focus_query(focus_query, owner_texts)
            scoped_query = f":uid={int(owner_candidate['mid'])}"
            if scoped_topic:
                scoped_query = f"{scoped_query} {scoped_topic}".strip()
            queries.append(scoped_query)

        if focus_query and focus_query not in queries:
            queries.append(focus_query)
        if not queries:
            return []

        return [
            ToolCallRequest(
                id=f"auto_video_gap_{len(result_store.order) + 1}",
                name="search_videos",
                arguments={"queries": queries[:2]},
                visibility="user",
                source="deterministic_followup",
            )
        ]

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
        followup_requests.extend(
            self._build_video_gap_followup_requests(
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
        return (
            self._build_explicit_dsl_video_search_answer(intent)
            or self._build_explicit_video_lookup_answer(intent)
            or self._build_owner_recent_timeline_answer(
                intent,
                messages,
            )
        )

    def _build_explicit_dsl_video_search_answer(
        self,
        intent: IntentProfile,
    ) -> str | None:
        explicit_query = VideoQueryNormalizer.extract_explicit_dsl_query(
            intent.raw_query
        )
        if not explicit_query:
            return None

        best_result: dict | None = None
        for result_id in self.result_store.order:
            record = self.result_store.get(result_id)
            if (
                record is None
                or canonical_tool_name(record.request.name) != "search_videos"
            ):
                continue
            args = record.request.arguments or {}
            queries = args.get("queries")
            if isinstance(queries, str):
                query_values = [queries]
            elif isinstance(queries, list):
                query_values = [str(item or "") for item in queries]
            else:
                query_values = []
            if explicit_query not in query_values:
                continue
            best_result = record.result or {}
            break

        if not best_result:
            return None
        hits = list(best_result.get("hits") or [])
        total_hits = int(best_result.get("total_hits") or len(hits))
        if not hits:
            return f"按 `{explicit_query}` 搜索后，当前没有找到可展示的视频结果。"

        lines = [f"按 `{explicit_query}` 找到这些相关视频："]
        for index, hit in enumerate(hits[:5], start=1):
            title = str(hit.get("title") or "").strip() or "未命名视频"
            bvid = str(hit.get("bvid") or "").strip()
            owner = hit.get("owner") or {}
            owner_name = owner.get("name", "") if isinstance(owner, dict) else ""
            suffix_parts = []
            if owner_name:
                suffix_parts.append(f"UP：{owner_name}")
            if bvid:
                suffix_parts.append(f"https://www.bilibili.com/video/{bvid}")
            suffix = f"（{'，'.join(suffix_parts)}）" if suffix_parts else ""
            lines.append(f"{index}. 《{title}》{suffix}")

        if total_hits > len(hits):
            lines.append(f"当前展示前 {len(hits)} 条，可检索总命中约 {total_hits} 条。")
        return "\n".join(lines).strip()

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
