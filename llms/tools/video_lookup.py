from __future__ import annotations

import re

from llms.messages import extract_bvids, normalize_bvid_key


_BVID_LOOKUP_QUERY_RE = re.compile(
    r"^(?:bv\s*=\s*)?(BV[0-9A-Za-z]{10})$",
    re.IGNORECASE,
)
_MID_LOOKUP_QUERY_RE = re.compile(
    r"^:?(?:uid|mid)\s*=\s*(\d{4,})(?:\s+:date\s*<=\s*([0-9]+[dwmy]))?$",
    re.IGNORECASE,
)


def _normalize_lookup_seed_values(values: object) -> list[str]:
    if isinstance(values, str):
        text = values.strip()
        return [text] if text else []
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        text = str(int(values)).strip()
        return [text] if text else []
    if isinstance(values, (list, tuple, set)):
        return [str(item).strip() for item in values if str(item or "").strip()]
    return []


def _normalize_lookup_mid_values(values: object) -> list[str]:
    normalized: list[str] = []
    for value in _normalize_lookup_seed_values(values):
        try:
            normalized_value = str(int(value))
        except (TypeError, ValueError):
            continue
        if normalized_value not in normalized:
            normalized.append(normalized_value)
    return normalized


def parse_search_video_lookup_query(
    query: str,
) -> tuple[str, str, str | None] | None:
    query_text = str(query or "").strip()
    if not query_text:
        return None

    bvid_match = _BVID_LOOKUP_QUERY_RE.fullmatch(query_text)
    if bvid_match:
        return ("bvid", bvid_match.group(1), None)

    mid_match = _MID_LOOKUP_QUERY_RE.fullmatch(query_text)
    if mid_match:
        return ("mid", str(int(mid_match.group(1))), mid_match.group(2) or None)

    return None


def coerce_search_video_lookup_arguments(arguments: dict) -> dict | None:
    normalized = dict(arguments or {})
    mode = str(normalized.get("mode", "auto") or "auto").lower()
    if mode == "discover":
        return None

    raw_queries = normalized.get("queries")
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    elif not isinstance(raw_queries, list):
        single_query = str(normalized.get("query", "") or "").strip()
        raw_queries = [single_query] if single_query else []

    explicit_bvids: list[str] = []
    explicit_bvid_keys: set[str] = set()
    explicit_mids: list[str] = []
    date_window = str(normalized.get("date_window", "") or "").strip() or None

    for key in ("bv", "bvid"):
        for value in _normalize_lookup_seed_values(normalized.get(key)):
            matches = extract_bvids({"content": value})
            for match in matches:
                match_key = normalize_bvid_key(match)
                if match_key not in explicit_bvid_keys:
                    explicit_bvid_keys.add(match_key)
                    explicit_bvids.append(match)

    for value in _normalize_lookup_seed_values(normalized.get("bvids")):
        matches = extract_bvids({"content": value})
        for match in matches:
            match_key = normalize_bvid_key(match)
            if match_key not in explicit_bvid_keys:
                explicit_bvid_keys.add(match_key)
                explicit_bvids.append(match)

    for key in ("mid", "uid"):
        for value in _normalize_lookup_mid_values(normalized.get(key)):
            if value not in explicit_mids:
                explicit_mids.append(value)

    for value in _normalize_lookup_mid_values(normalized.get("mids")):
        if value not in explicit_mids:
            explicit_mids.append(value)

    remaining_queries: list[str] = []
    for query in raw_queries:
        parsed = parse_search_video_lookup_query(query)
        if parsed is None:
            remaining_queries.append(str(query or "").strip())
            continue
        lookup_kind, lookup_value, query_window = parsed
        if (
            lookup_kind == "bvid"
            and normalize_bvid_key(lookup_value) not in explicit_bvid_keys
        ):
            explicit_bvid_keys.add(normalize_bvid_key(lookup_value))
            explicit_bvids.append(lookup_value)
        elif lookup_kind == "mid" and lookup_value not in explicit_mids:
            explicit_mids.append(lookup_value)
        if query_window and not date_window:
            date_window = query_window

    if remaining_queries or not (explicit_bvids or explicit_mids):
        return None

    normalized.pop("query", None)
    normalized.pop("queries", None)
    normalized["mode"] = "lookup"
    if date_window:
        normalized["date_window"] = date_window

    if explicit_bvids:
        if len(explicit_bvids) == 1:
            normalized["bv"] = explicit_bvids[0]
            normalized.pop("bvid", None)
            normalized.pop("bvids", None)
        else:
            normalized["bvids"] = explicit_bvids
            normalized.pop("bv", None)
            normalized.pop("bvid", None)

    if explicit_mids:
        if len(explicit_mids) == 1:
            normalized["mid"] = explicit_mids[0]
            normalized.pop("uid", None)
            normalized.pop("mids", None)
        else:
            normalized["mids"] = explicit_mids
            normalized.pop("mid", None)
            normalized.pop("uid", None)

    return normalized


def normalize_search_video_lookup_arguments(arguments: dict) -> dict:
    coerced = coerce_search_video_lookup_arguments(arguments)
    if coerced is not None:
        return coerced
    return dict(arguments or {})


__all__ = [
    "coerce_search_video_lookup_arguments",
    "normalize_search_video_lookup_arguments",
    "parse_search_video_lookup_query",
]
