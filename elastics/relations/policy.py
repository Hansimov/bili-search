from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re


_ASSET_PATH = (
    Path(__file__).resolve().parent / "assets" / "related_token_cleaning_policy.json"
)


@dataclass(frozen=True, slots=True)
class ShortPrefixNoisePolicy:
    source_min_len: int
    option_min_len: int
    option_max_len: int
    candidate_types: frozenset[str]
    min_doc_freq: int
    shared_prefix_min: int


@dataclass(frozen=True, slots=True)
class SourceEchoNoisePolicy:
    candidate_types: frozenset[str]
    source_min_terms: int
    source_term_min_len: int
    source_term_max_len: int
    max_doc_freq: int
    generic_tails: frozenset[str]


@dataclass(frozen=True, slots=True)
class RelatedTokenCleaningPolicy:
    blocked_markers: tuple[str, ...]
    boilerplate_tail_patterns: tuple[re.Pattern[str], ...]
    punctuation_pattern: re.Pattern[str]
    normalize_strip_chars: str
    dedupe_min_key_len: int
    short_prefix_noise: ShortPrefixNoisePolicy
    digit_suffix_noise_pattern: re.Pattern[str]
    source_echo_noise: SourceEchoNoisePolicy


@lru_cache(maxsize=1)
def get_related_token_cleaning_policy() -> RelatedTokenCleaningPolicy:
    payload = json.loads(_ASSET_PATH.read_text(encoding="utf-8"))
    short_prefix = payload.get("short_prefix_noise") or {}
    source_echo = payload.get("source_echo_noise") or {}
    return RelatedTokenCleaningPolicy(
        blocked_markers=tuple(payload.get("blocked_markers") or []),
        boilerplate_tail_patterns=tuple(
            re.compile(pattern)
            for pattern in (payload.get("boilerplate_tail_patterns") or [])
        ),
        punctuation_pattern=re.compile(
            payload.get("punctuation_pattern") or r"[，。！？?；;：:、~]+"
        ),
        normalize_strip_chars=str(payload.get("normalize_strip_chars") or ""),
        dedupe_min_key_len=int(payload.get("dedupe_min_key_len", 2)),
        short_prefix_noise=ShortPrefixNoisePolicy(
            source_min_len=int(short_prefix.get("source_min_len", 6)),
            option_min_len=int(short_prefix.get("option_min_len", 2)),
            option_max_len=int(short_prefix.get("option_max_len", 3)),
            candidate_types=frozenset(
                short_prefix.get("candidate_types") or ["prefix"]
            ),
            min_doc_freq=int(short_prefix.get("min_doc_freq", 1000)),
            shared_prefix_min=int(short_prefix.get("shared_prefix_min", 2)),
        ),
        digit_suffix_noise_pattern=re.compile(
            payload.get("digit_suffix_noise_pattern")
            or r"^[\u4e00-\u9fffA-Za-z]{1,2}\d{1,2}$"
        ),
        source_echo_noise=SourceEchoNoisePolicy(
            candidate_types=frozenset(
                source_echo.get("candidate_types") or ["doc_cooccurrence"]
            ),
            source_min_terms=int(source_echo.get("source_min_terms", 2)),
            source_term_min_len=int(source_echo.get("source_term_min_len", 2)),
            source_term_max_len=int(source_echo.get("source_term_max_len", 16)),
            max_doc_freq=int(source_echo.get("max_doc_freq", 16)),
            generic_tails=frozenset(
                str(item or "").strip().lower()
                for item in (source_echo.get("generic_tails") or [])
                if str(item or "").strip()
            ),
        ),
    )


__all__ = [
    "RelatedTokenCleaningPolicy",
    "ShortPrefixNoisePolicy",
    "SourceEchoNoisePolicy",
    "get_related_token_cleaning_policy",
]
