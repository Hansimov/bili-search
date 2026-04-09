from __future__ import annotations


PUBLIC_EXTERNAL_TOOL_NAMES: tuple[str, ...] = (
    "search_videos",
    "search_google",
    "search_owners",
    "expand_query",
)


def canonical_tool_name(name: str) -> str:
    return str(name or "")


def is_owner_tool_name(name: str) -> bool:
    return canonical_tool_name(name) == "search_owners"


def is_video_tool_name(name: str) -> bool:
    return canonical_tool_name(name) == "search_videos"


def is_query_expansion_tool_name(name: str) -> bool:
    return canonical_tool_name(name) == "expand_query"


__all__ = [
    "PUBLIC_EXTERNAL_TOOL_NAMES",
    "canonical_tool_name",
    "is_owner_tool_name",
    "is_query_expansion_tool_name",
    "is_video_tool_name",
]
