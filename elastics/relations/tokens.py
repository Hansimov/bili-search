from __future__ import annotations

import re


_BLOCKED_MARKERS = (
    "__pyf__",
    "|关注不错过",
)
_BOILERPLATE_TAIL_PATTERNS = (
    re.compile(r"\s*点点关注不错过(?:\s*持续更新(?:系列)?中)?\s*$"),
    re.compile(r"\s*持续更新(?:系列)?中\s*$"),
    re.compile(r"\s*欢迎收看求三连\s*$"),
    re.compile(r"\s*求三连\s*$"),
)
_PUNCT_TO_SPACE_RE = re.compile(r"[，。！？?；;：:、~]+")


def normalize_related_token_text(text: str) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    for marker in _BLOCKED_MARKERS:
        if marker in normalized:
            return ""
    for pattern in _BOILERPLATE_TAIL_PATTERNS:
        normalized = pattern.sub("", normalized).strip()
    normalized = _PUNCT_TO_SPACE_RE.sub(" ", normalized)
    normalized = " ".join(normalized.split())
    return normalized.strip(" 【】[]()<>\"'`")


def compact_related_token_key(text: str) -> str:
    return "".join(
        ch for ch in normalize_related_token_text(text).lower() if not ch.isspace()
    )


def _shared_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def is_short_prefix_noise(source_text: str, option: dict) -> bool:
    source_key = compact_related_token_key(source_text)
    option_key = compact_related_token_key((option or {}).get("text") or "")
    if len(source_key) < 6 or len(option_key) > 3 or len(option_key) < 2:
        return False
    if str((option or {}).get("type") or "").lower() != "prefix":
        return False
    if int((option or {}).get("doc_freq") or 0) < 1000:
        return False
    return _shared_prefix_len(source_key, option_key) >= 2


def sanitize_related_token_options(
    source_text: str,
    options: list[dict] | None,
) -> list[dict]:
    sanitized: list[dict] = []
    seen_keys: set[str] = set()
    for option in options or []:
        text = normalize_related_token_text((option or {}).get("text") or "")
        if not text:
            continue
        if is_short_prefix_noise(source_text, option):
            continue
        key = compact_related_token_key(text)
        if len(key) < 2 or key in seen_keys:
            continue
        seen_keys.add(key)
        sanitized.append({**option, "text": text})
    return sanitized


def sanitize_related_token_result(source_text: str, result: dict | None) -> dict:
    payload = dict(result or {})
    payload["options"] = sanitize_related_token_options(
        source_text,
        payload.get("options") or [],
    )
    return payload
