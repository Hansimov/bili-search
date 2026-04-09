"""Shared XML/DSML tool-markup helpers for llms orchestration.

Keep tool-command parsing and cleanup centralized here so handler and engine do
not drift apart when providers change their markup behavior.
"""

from __future__ import annotations

import json
import re

from functools import lru_cache
from typing import Any

from llms.contracts import ToolCallRequest


EXTERNAL_TOOL_NAMES: tuple[str, ...] = (
    "search_videos",
    "search_google",
    "search_owners",
    "related_tokens_by_tokens",
    "related_owners_by_tokens",
    "related_videos_by_videos",
    "related_owners_by_videos",
    "related_videos_by_owners",
    "related_owners_by_owners",
)
INTERNAL_TOOL_NAMES: tuple[str, ...] = (
    "read_prompt_assets",
    "inspect_tool_result",
    "run_small_llm_task",
)
ALL_TOOL_NAMES: tuple[str, ...] = EXTERNAL_TOOL_NAMES + INTERNAL_TOOL_NAMES
_TOOL_ATTRS_PATTERN = r"(?:[^\"'/>]|\"[^\"]*\"|'[^']*')*"
_TOOL_ATTR_RE = re.compile(r"""(\w+)=(['\"])(.*?)\2""", re.DOTALL)
_DSML_PATTERN = re.compile(r"<｜.*?｜>")
_DSML_BLOCK_PATTERN = re.compile(
    r"<｜DSML｜function_calls>.*?</｜DSML｜function_calls>", re.DOTALL
)


@lru_cache(maxsize=None)
def _compiled_patterns(
    tool_names: tuple[str, ...],
) -> tuple[re.Pattern[str], re.Pattern[str]]:
    tool_name_pattern = "|".join(re.escape(name) for name in tool_names)
    generic_tool_cmd_re = re.compile(
        rf"""<(?P<name>{tool_name_pattern})\s*(?P<attrs>{_TOOL_ATTRS_PATTERN})/>""",
        re.DOTALL,
    )
    tool_cmd_pattern = re.compile(
        rf"""<(?:{tool_name_pattern})\s{_TOOL_ATTRS_PATTERN}/>""",
        re.DOTALL,
    )
    return generic_tool_cmd_re, tool_cmd_pattern


def build_tool_prefixes(tool_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"<{name}" for name in tool_names)


EXTERNAL_TOOL_PREFIXES = build_tool_prefixes(EXTERNAL_TOOL_NAMES)


def parse_tool_argument(raw_value: str):
    value = (raw_value or "").strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if value[0] in ["[", "{", '"']:
            return json.loads(value)
    except Exception:
        pass
    if re.fullmatch(r"-?(?:0|[1-9]\d*)", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def parse_xml_tool_calls(
    content: str,
    iteration: int,
    *,
    tool_names: tuple[str, ...] = ALL_TOOL_NAMES,
    internal_tool_names: tuple[str, ...] = INTERNAL_TOOL_NAMES,
) -> list[ToolCallRequest]:
    commands: list[ToolCallRequest] = []
    content_text = str(content or "")
    generic_tool_cmd_re, _ = _compiled_patterns(tuple(tool_names))
    internal_names = set(internal_tool_names)
    for index, match in enumerate(generic_tool_cmd_re.finditer(content_text)):
        args = {}
        attrs = match.group("attrs") or ""
        for attr_match in _TOOL_ATTR_RE.finditer(attrs):
            args[attr_match.group(1)] = parse_tool_argument(attr_match.group(3))
        name = match.group("name")
        commands.append(
            ToolCallRequest(
                id=f"xmlcall_{iteration}_{index}",
                name=name,
                arguments=args,
                visibility="internal" if name in internal_names else "user",
                source="xml",
            )
        )
    return commands


def parse_xml_commands(
    content: str,
    *,
    tool_names: tuple[str, ...] = EXTERNAL_TOOL_NAMES,
) -> list[dict[str, Any]]:
    return [
        {"type": call.name, "args": call.arguments}
        for call in parse_xml_tool_calls(
            content,
            0,
            tool_names=tool_names,
            internal_tool_names=(),
        )
    ]


def strip_tool_commands(
    content: str,
    *,
    tool_names: tuple[str, ...] = ALL_TOOL_NAMES,
) -> str:
    _, tool_cmd_pattern = _compiled_patterns(tuple(tool_names))
    return tool_cmd_pattern.sub("", content or "")


def sanitize_generated_content(
    content: str,
    *,
    tool_names: tuple[str, ...] = ALL_TOOL_NAMES,
) -> str:
    sanitized = _DSML_BLOCK_PATTERN.sub("", content or "")
    sanitized = _DSML_PATTERN.sub("", sanitized)
    sanitized = strip_tool_commands(sanitized, tool_names=tool_names)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


def _canonicalize_value(value: Any):
    if isinstance(value, dict):
        return {key: _canonicalize_value(val) for key, val in sorted(value.items())}
    if isinstance(value, list):
        canonical_items = [_canonicalize_value(item) for item in value]
        sortable = all(
            isinstance(item, (str, int, float, bool)) or item is None
            for item in canonical_items
        )
        return sorted(canonical_items) if sortable else canonical_items
    return value


def command_signature(call: ToolCallRequest) -> str:
    payload = {
        "name": call.name,
        "arguments": _canonicalize_value(call.arguments),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


__all__ = [
    "ALL_TOOL_NAMES",
    "EXTERNAL_TOOL_NAMES",
    "EXTERNAL_TOOL_PREFIXES",
    "INTERNAL_TOOL_NAMES",
    "build_tool_prefixes",
    "command_signature",
    "parse_tool_argument",
    "parse_xml_commands",
    "parse_xml_tool_calls",
    "sanitize_generated_content",
    "strip_tool_commands",
]
