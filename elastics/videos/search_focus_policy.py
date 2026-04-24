from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re


_ASSET_PATH = Path(__file__).with_name("assets") / "search_focus_policy.json"


@dataclass(frozen=True, slots=True)
class SearchFocusPolicy:
    enabled: bool
    query_min_length: int
    query_max_length: int
    min_segment_chars: int
    max_segments: int
    trigger_patterns: tuple[re.Pattern[str], ...]
    segment_split_pattern: re.Pattern[str]
    prefix_strip_patterns: tuple[re.Pattern[str], ...]
    suffix_strip_patterns: tuple[re.Pattern[str], ...]
    bracket_prefix_pattern: re.Pattern[str]

    def query_length_ok(self, text: str) -> bool:
        length = len(str(text or "").strip())
        return self.query_min_length <= length <= self.query_max_length

    def should_focus(self, text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized or not self.query_length_ok(normalized):
            return False
        return any(pattern.search(normalized) for pattern in self.trigger_patterns)


@lru_cache(maxsize=1)
def get_search_focus_policy() -> SearchFocusPolicy:
    payload = json.loads(_ASSET_PATH.read_text(encoding="utf-8"))
    query_length = payload.get("query_length") or {}
    return SearchFocusPolicy(
        enabled=bool(payload.get("enabled", True)),
        query_min_length=int(query_length.get("min", 6)),
        query_max_length=int(query_length.get("max", 96)),
        min_segment_chars=int(payload.get("min_segment_chars", 2)),
        max_segments=int(payload.get("max_segments", 2)),
        trigger_patterns=tuple(
            re.compile(pattern)
            for pattern in (
                payload.get("trigger_patterns")
                or [
                    r".*[《》【】（）()].*",
                    r".*\d{4}[./-]\d{1,2}[./-]\d{1,2}.*",
                    r".*\d{1,2}[:：]\d{2}.*",
                    r".*[｜|，,。！？!?~～].*",
                ]
            )
        ),
        segment_split_pattern=re.compile(
            payload.get("segment_split_pattern") or r"[｜|，,。！？!?~～]+"
        ),
        prefix_strip_patterns=tuple(
            re.compile(pattern)
            for pattern in (
                payload.get("prefix_strip_patterns")
                or [
                    r"^\s*\d{4}[./-]\d{1,2}[./-]\d{1,2}\s*",
                    r"^\s*[A-Za-z]\.\s*",
                ]
            )
        ),
        suffix_strip_patterns=tuple(
            re.compile(pattern)
            for pattern in (
                payload.get("suffix_strip_patterns")
                or [
                    r"[（(]\s*\d{1,2}[:：]\d{2}[^)）]*[)）]\s*$",
                    r"[~～]+\s*$",
                ]
            )
        ),
        bracket_prefix_pattern=re.compile(
            payload.get("bracket_prefix_pattern")
            or r"^\s*[【\[](?P<prefix>[^】\]]{2,48})[】\]]\s*(?P<body>.+?)\s*$"
        ),
    )


__all__ = ["SearchFocusPolicy", "get_search_focus_policy"]
