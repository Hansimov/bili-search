"""Syntactic video-query guardrails for orchestration.

Do not grow this module into a catalog of natural-language phrases, examples,
aliases, or typo rules. Semantic cleanup belongs in the large-model planning
workflow before XML tool commands are emitted: the planner must think through
the user's intent and produce compact search DSL, not raw conversational text.
Execution code may add workflow gates for stable protocol mistakes such as
"resolve owner first, then use the returned mid", but regex here must stay
limited to syntax-level detection: explicit DSL markers, simple wrappers, and
safe fallback cleanup.
"""

from __future__ import annotations

import re

from llms.intent.focus import build_focus_query
from llms.intent.focus import compact_focus_key


_VIDEO_QUERY_BRACKET_RE = re.compile(
    r"^\s*[【\[](?P<prefix>[^】\]]{1,48})[】\]]\s*(?P<body>.+?)\s*$"
)
_VIDEO_QUERY_QUOTED_TITLE_RE = re.compile(
    r"(?P<prefix>[\u4e00-\u9fffA-Za-z0-9\s._+#/-]{0,32})[《\"“](?P<title>[^》\"”]{2,96})[》\"”](?P<tail>[\u4e00-\u9fffA-Za-z0-9\s._+#/『』-]{0,80})"
)
_EXPLICIT_QMODE_RE = re.compile(r"(?<![\w])q=(?:w|v|r){1,3}(?![\w])", re.IGNORECASE)


class VideoQueryNormalizer:
    @staticmethod
    def clean_subject_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip(
            " ，。！？?；;：:、()[]{}<>《》\"'`~!@#$%^&*-+=|\\/"
        )

    @classmethod
    def clean_video_query_body(cls, text: str) -> str:
        normalized = " ".join(str(text or "").split()).strip()
        normalized = normalized.split("【", 1)[0].split("[", 1)[0].strip()
        normalized = re.sub(r"[，。！？?；;：:、~]+", " ", normalized)
        return cls.clean_subject_text(normalized)

    @classmethod
    def _dedupe_focus_parts(cls, parts: list[str]) -> list[str]:
        deduped: list[str] = []
        seen_keys: set[str] = set()
        for part in parts:
            cleaned = cls.clean_subject_text(part)
            key = compact_focus_key(cleaned)
            if len(key) < 2 or key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(cleaned)
        return deduped

    @classmethod
    def _normalize_subject_match(cls, text: str) -> str:
        subject = cls.clean_subject_text(text)
        bracket_match = _VIDEO_QUERY_BRACKET_RE.match(subject)
        if bracket_match:
            parts = cls._dedupe_focus_parts(
                [
                    bracket_match.group("prefix"),
                    cls.clean_video_query_body(bracket_match.group("body")),
                ]
            )
            subject = " ".join(parts)
        return subject

    @classmethod
    def extract_title_like_video_query(cls, text: str) -> str:
        original = " ".join(str(text or "").split()).strip()
        normalized = original
        if not normalized:
            return ""

        bracket_match = _VIDEO_QUERY_BRACKET_RE.match(normalized)
        if bracket_match:
            parts = cls._dedupe_focus_parts(
                [
                    bracket_match.group("prefix"),
                    cls.clean_video_query_body(bracket_match.group("body")),
                ]
            )
            if parts:
                title_query = " ".join(parts)
                return title_query

        quoted_match = _VIDEO_QUERY_QUOTED_TITLE_RE.search(normalized)
        if quoted_match:
            tail = re.sub(r"[【\[].*$", "", quoted_match.group("tail") or "")
            parts = cls._dedupe_focus_parts(
                [
                    cls.clean_video_query_body(quoted_match.group("prefix")),
                    quoted_match.group("title"),
                    cls.clean_subject_text(tail),
                ]
            )
            if parts:
                title_query = " ".join(parts)
                return title_query
        return ""

    @classmethod
    def extract_explicit_dsl_query(cls, text: str) -> str:
        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            return ""
        match = _EXPLICIT_QMODE_RE.search(normalized)
        if not match:
            return ""

        query = normalized[: match.end()].strip()
        query = query.strip(" ，。！？?；;：:、()[]{}<>《》\"'`~")
        query = " ".join(query.split())
        if not query or query.lower() == match.group(0).lower():
            return ""
        return query

    @classmethod
    def normalize_title_like_video_search_arguments(
        cls,
        arguments: dict,
        raw_query: str,
    ) -> dict:
        title_query = cls.extract_title_like_video_query(raw_query)
        explicit_dsl_query = cls.extract_explicit_dsl_query(raw_query)
        normalized = dict(arguments or {})
        if explicit_dsl_query:
            normalized.pop("query", None)
            normalized["queries"] = [explicit_dsl_query]
            return normalized
        if not title_query:
            return normalized
        if str(normalized.get("mode") or "").lower() == "lookup":
            return normalized
        if any(normalized.get(key) for key in ("bv", "bvid", "bvids", "mid", "mids")):
            return normalized

        normalized.pop("query", None)
        normalized["queries"] = [title_query]
        return normalized

    @classmethod
    def build_video_followup_focus_query(
        cls,
        latest_user_text: str,
        *,
        explicit_entities: list[str] | None = None,
        explicit_topics: list[str] | None = None,
    ) -> str:
        title_query = cls.extract_title_like_video_query(latest_user_text)
        if title_query:
            return title_query

        candidate_parts: list[str] = []
        seen_keys: set[str] = set()
        for candidate in [
            *(explicit_entities or []),
            *(explicit_topics or []),
        ]:
            cleaned = cls.clean_subject_text(candidate)
            normalized_key = "".join(cleaned.split()).lower()
            if (
                len(normalized_key) < 2
                or len(normalized_key) > 24
                or normalized_key in seen_keys
            ):
                continue
            seen_keys.add(normalized_key)
            candidate_parts.append(cleaned)
        if candidate_parts:
            normalized = " ".join(candidate_parts[:4]).strip()
            return normalized

        focused = build_focus_query(latest_user_text)
        normalized = " ".join(str(focused or "").split()).strip()
        normalized = cls.clean_video_query_body(normalized)
        return normalized


__all__ = ["VideoQueryNormalizer"]
