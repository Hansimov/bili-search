from __future__ import annotations

from elastics.relations.policy import get_related_token_cleaning_policy


POLICY = get_related_token_cleaning_policy()


def normalize_related_token_text(text: str) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    for marker in POLICY.blocked_markers:
        if marker in normalized:
            return ""
    for pattern in POLICY.boilerplate_tail_patterns:
        normalized = pattern.sub("", normalized).strip()
    normalized = POLICY.punctuation_pattern.sub(" ", normalized)
    normalized = " ".join(normalized.split())
    return normalized.strip(POLICY.normalize_strip_chars)


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
    short_prefix_policy = POLICY.short_prefix_noise
    if (
        len(source_key) < short_prefix_policy.source_min_len
        or len(option_key) > short_prefix_policy.option_max_len
        or len(option_key) < short_prefix_policy.option_min_len
    ):
        return False
    if (
        str((option or {}).get("type") or "").lower()
        not in short_prefix_policy.candidate_types
    ):
        return False
    if int((option or {}).get("doc_freq") or 0) < short_prefix_policy.min_doc_freq:
        return False
    return (
        _shared_prefix_len(source_key, option_key)
        >= short_prefix_policy.shared_prefix_min
    )


def is_digit_suffix_prefix_noise(source_text: str, option: dict) -> bool:
    option_text = normalize_related_token_text((option or {}).get("text") or "")
    source_key = compact_related_token_key(source_text)
    option_key = compact_related_token_key(option_text)
    if not option_key:
        return False
    if not POLICY.digit_suffix_noise_pattern.fullmatch(option_text):
        return False
    short_prefix_policy = POLICY.short_prefix_noise
    if len(source_key) < short_prefix_policy.source_min_len:
        return False
    if (
        str((option or {}).get("type") or "").lower()
        not in short_prefix_policy.candidate_types
    ):
        return False
    if int((option or {}).get("doc_freq") or 0) < short_prefix_policy.min_doc_freq:
        return False
    stem = option_text.rstrip("0123456789").strip().lower()
    if not stem:
        return False
    return source_key.startswith(stem)


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
        if is_digit_suffix_prefix_noise(source_text, {**option, "text": text}):
            continue
        if is_short_prefix_noise(source_text, option):
            continue
        key = compact_related_token_key(text)
        if len(key) < POLICY.dedupe_min_key_len or key in seen_keys:
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
