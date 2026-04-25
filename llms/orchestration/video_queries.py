from __future__ import annotations

import re

from llms.intent.focus import build_focus_query
from llms.intent.focus import compact_focus_key
from llms.intent.focus import rewrite_known_term_aliases


_VIDEO_QUERY_PREFIX_RE = re.compile(
    r"^(?:我可能打错了字，?想找|忽略口播和套话，?帮我找和|帮我找和|帮我找|想找)\s*"
)
_VIDEO_QUERY_SUFFIX_RE = re.compile(
    r"(?:有哪些(?:值得看|适合直接上手看)?的视频|真\s*正?相关(?:的)?视频|相关(?:的)?视频|有关的视频|有哪些视频).*$"
)
_VIDEO_QUERY_NOISE_RE = re.compile(
    r"(?:忽略口播|套话|帮我找|想找|打错了字|真正相关|相关的视频|有关的视频|有哪些视频|值得看的视频)",
    re.IGNORECASE,
)
_VIDEO_QUERY_BODY_CUT_RE = re.compile(
    r"(?:\s+真\s*正相关.*|\s+相关.*|\s+有哪些.*|\s+有啥.*|\s+忽略口播.*|\s+套话.*|\s+帮我找.*|\s+想找.*)$"
)
_VIDEO_QUERY_BRACKET_RE = re.compile(
    r"^\s*[【\[](?P<prefix>[^】\]]{1,48})[】\]]\s*(?P<body>.+?)\s*$"
)
_VIDEO_QUERY_QUOTED_TITLE_RE = re.compile(
    r"(?P<prefix>[\u4e00-\u9fffA-Za-z0-9\s._+#/-]{0,32})[《\"](?P<title>[^》\"]{2,64})[》\"]"
)
_FOLLOWUP_META_QUERY_RE = re.compile(
    r"^(?:请|帮我|给我|麻烦你?)?(?:总结|概括|梳理|整理|分析)(?:一下|下)?$"
)
_EXPLICIT_QMODE_RE = re.compile(r"(?<![\w])q=(?:w|v|r){1,3}(?![\w])", re.IGNORECASE)


class VideoQueryNormalizer:
    @staticmethod
    def _is_meta_followup_query(text: str) -> bool:
        compact_text = "".join(str(text or "").split())
        if not compact_text:
            return False
        return bool(_FOLLOWUP_META_QUERY_RE.fullmatch(compact_text))

    @staticmethod
    def clean_subject_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip(
            " ，。！？?；;：:、()[]{}<>《》\"'`~!@#$%^&*-+=|\\/"
        )

    @classmethod
    def clean_video_query_body(cls, text: str) -> str:
        normalized = " ".join(str(text or "").split()).strip()
        normalized = normalized.split("【", 1)[0].split("[", 1)[0].strip()
        normalized = _VIDEO_QUERY_BODY_CUT_RE.sub("", normalized).strip()
        normalized = _VIDEO_QUERY_SUFFIX_RE.sub("", normalized).strip(
            " ，。！？?；;：:、~"
        )
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
    def extract_title_like_video_query(cls, text: str) -> str:
        normalized = " ".join(str(text or "").split()).strip()
        normalized = _VIDEO_QUERY_PREFIX_RE.sub("", normalized).strip()
        normalized = _VIDEO_QUERY_SUFFIX_RE.sub("", normalized).strip(
            " ，。！？?；;：:"
        )
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
                return rewrite_known_term_aliases(title_query) or title_query

        quoted_match = _VIDEO_QUERY_QUOTED_TITLE_RE.search(normalized)
        if quoted_match:
            parts = cls._dedupe_focus_parts(
                [
                    cls.clean_video_query_body(quoted_match.group("prefix")),
                    quoted_match.group("title"),
                ]
            )
            if parts:
                title_query = " ".join(parts)
                return rewrite_known_term_aliases(title_query) or title_query
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
        query = _VIDEO_QUERY_PREFIX_RE.sub("", query).strip()
        query = re.sub(r"^(?:请|麻烦|帮忙|给我|找一下|搜索一下)\s*", "", query)
        query = query.strip(" ，。！？?；;：:、()[]{}<>《》\"'`~")
        query = " ".join(query.split())
        if not query or query.lower() == match.group(0).lower():
            return ""
        return rewrite_known_term_aliases(query) or query

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
                or cls._is_meta_followup_query(cleaned)
                or _VIDEO_QUERY_NOISE_RE.search(cleaned)
            ):
                continue
            seen_keys.add(normalized_key)
            candidate_parts.append(cleaned)
        if candidate_parts:
            normalized = " ".join(candidate_parts[:4]).strip()
            if cls._is_meta_followup_query(normalized):
                return ""
            return rewrite_known_term_aliases(normalized) or normalized

        focused = build_focus_query(latest_user_text)
        normalized = " ".join(str(focused or "").split()).strip()
        normalized = _VIDEO_QUERY_PREFIX_RE.sub("", normalized).strip()
        normalized = _VIDEO_QUERY_SUFFIX_RE.sub("", normalized).strip(
            " ，。！？?；;：:"
        )
        normalized = cls.clean_video_query_body(normalized)
        if cls._is_meta_followup_query(normalized):
            return ""
        return rewrite_known_term_aliases(normalized) or normalized


__all__ = ["VideoQueryNormalizer"]
