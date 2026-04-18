from __future__ import annotations

import re

from typing import Any, Iterable


_BVID_RE = re.compile(r"\bBV[0-9A-Za-z]{10}\b", re.IGNORECASE)
_OWNER_MID_RE = re.compile(
    r"(?:(?<![A-Za-z0-9_])(?:uid|mid)\s*[:=：]?\s*(\d{4,})(?!\d))",
    re.IGNORECASE,
)
_SPACE_MID_RE = re.compile(r"space\.bilibili\.com/(\d{4,})(?:/|$)", re.IGNORECASE)


def normalize_bvid_key(bvid: Any) -> str:
    return str(bvid or "").strip().upper()


def _content_value(message_or_content: Any) -> Any:
    if isinstance(message_or_content, dict) and "content" in message_or_content:
        return message_or_content.get("content")
    return message_or_content


def iter_text_fragments(message_or_content: Any) -> Iterable[str]:
    content = _content_value(message_or_content)
    if content is None:
        return
    if isinstance(content, str):
        yield content
        return
    if isinstance(content, list):
        for item in content:
            yield from iter_text_fragments(item)
        return
    if isinstance(content, dict):
        part_type = str(content.get("type") or "").strip().lower()
        if part_type == "text":
            text = content.get("text")
            if isinstance(text, str) and text:
                yield text
        nested_content = content.get("content")
        if nested_content is not None:
            yield from iter_text_fragments(nested_content)


def extract_message_text(message_or_content: Any) -> str:
    fragments = [
        fragment.strip() for fragment in iter_text_fragments(message_or_content)
    ]
    return " ".join(fragment for fragment in fragments if fragment).strip()


def has_image_content(message_or_content: Any) -> bool:
    content = _content_value(message_or_content)
    if isinstance(content, list):
        return any(has_image_content(item) for item in content)
    if isinstance(content, dict):
        part_type = str(content.get("type") or "").strip().lower()
        if part_type == "image_url":
            return True
        nested_content = content.get("content")
        if nested_content is not None:
            return has_image_content(nested_content)
    return False


def extract_bvids(message_or_content: Any) -> list[str]:
    text = extract_message_text(message_or_content)
    results: list[str] = []
    seen: set[str] = set()
    for match in _BVID_RE.findall(text):
        normalized_key = normalize_bvid_key(match)
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        results.append(match)
    return results


def extract_owner_mids(message_or_content: Any) -> list[int]:
    text = extract_message_text(message_or_content)
    results: list[int] = []
    seen: set[int] = set()

    for pattern in (_OWNER_MID_RE, _SPACE_MID_RE):
        for match in pattern.findall(text):
            try:
                normalized = int(str(match).strip())
            except (TypeError, ValueError):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
    return results


__all__ = [
    "extract_bvids",
    "extract_message_text",
    "extract_owner_mids",
    "has_image_content",
    "iter_text_fragments",
    "normalize_bvid_key",
]
