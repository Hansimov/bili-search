from __future__ import annotations

from llms.intent import build_intent_profile
from llms.intent import select_prompt_asset_ids
from llms.prompts.assets import get_prompt_assets
from llms.prompts.system import get_date_prompt
from llms.contracts import IntentProfile, PromptSelection


def _capabilities_block(capabilities: dict | None = None) -> str:
    caps = capabilities or {}
    external_tools = ["search_videos"]
    if caps.get("supports_owner_search"):
        external_tools.append("search_owners")
    if caps.get("supports_google_search"):
        external_tools.append("search_google")
    for endpoint in caps.get("relation_endpoints") or []:
        if endpoint not in external_tools:
            external_tools.append(endpoint)
    tools_text = ", ".join(external_tools)
    return (
        "[TOOL_OVERVIEW]\n"
        f"外部终局工具: {tools_text}\n"
        "内部工具: read_prompt_assets, inspect_tool_result, run_small_llm_task\n"
        "读取高层级提示时优先 read_prompt_assets；读取结果细节时优先 inspect_tool_result；\n"
        "需要并行压缩或归纳时优先 run_small_llm_task。\n"
        "[/TOOL_OVERVIEW]"
    )


def _intent_block(intent: IntentProfile | None) -> str:
    if intent is None:
        return ""
    return (
        "[INTENT_PROFILE]\n"
        + "\n".join(intent.to_prompt_lines())
        + "\n[/INTENT_PROFILE]"
    )


def build_prompt_selection(
    capabilities: dict | None = None,
    intent: IntentProfile | None = None,
    extra_asset_ids: list[str] | None = None,
) -> PromptSelection:
    resolved_intent = intent or IntentProfile(raw_query="", normalized_query="")
    asset_ids = select_prompt_asset_ids(resolved_intent)
    for asset_id in extra_asset_ids or []:
        if asset_id not in asset_ids:
            asset_ids.append(asset_id)

    assets = get_prompt_assets(ids=asset_ids)
    capabilities_block = _capabilities_block(capabilities)
    intent_block = _intent_block(intent)
    date_block = get_date_prompt()
    blocks = [capabilities_block]
    section_chars: dict[str, int] = {
        "tool_overview": len(capabilities_block),
    }
    if intent_block:
        section_chars["intent_profile"] = len(intent_block)

    # Keep the longest shared guidance at the front so provider prompt caches
    # can reuse it across requests that only differ in date or intent details.
    for asset in assets:
        section_key = asset.section.lower()
        section_chars[section_key] = section_chars.get(section_key, 0) + len(
            asset.content
        )
        blocks.append(f"[{asset.section}]\n{asset.content}\n[/{asset.section}]")

    blocks.extend([date_block, intent_block])

    prompt = "\n\n".join(block for block in blocks if block)
    return PromptSelection(
        prompt=prompt,
        assets=assets,
        section_chars=section_chars,
    )


def build_system_prompt(
    capabilities: dict | None = None,
    intent: IntentProfile | None = None,
    messages: list[dict] | None = None,
    extra_asset_ids: list[str] | None = None,
) -> str:
    resolved_intent = intent
    if resolved_intent is None and messages is not None:
        resolved_intent = build_intent_profile(messages)
    return build_prompt_selection(
        capabilities=capabilities,
        intent=resolved_intent,
        extra_asset_ids=extra_asset_ids,
    ).prompt


def build_system_prompt_profile(
    capabilities: dict | None = None,
    intent: IntentProfile | None = None,
    messages: list[dict] | None = None,
    extra_asset_ids: list[str] | None = None,
) -> dict:
    resolved_intent = intent
    if resolved_intent is None and messages is not None:
        resolved_intent = build_intent_profile(messages)
    selection = build_prompt_selection(
        capabilities=capabilities,
        intent=resolved_intent,
        extra_asset_ids=extra_asset_ids,
    )
    return {
        "asset_ids": [asset.asset_id for asset in selection.assets],
        "section_chars": {
            **selection.section_chars,
            "tool_commands": selection.section_chars.get("tool_overview", 0),
        },
        "total_chars": len(selection.prompt),
    }


def get_prompt_assets_payload(
    *,
    tool_names: list[str] | None = None,
    levels: list[str] | None = None,
    asset_ids: list[str] | None = None,
) -> dict:
    assets = get_prompt_assets(
        ids=asset_ids,
        tool_names=tool_names,
        levels=levels,
    )
    return {
        "assets": [
            {
                "asset_id": asset.asset_id,
                "title": asset.title,
                "section": asset.section,
                "level": asset.level,
                "tool_name": asset.tool_name,
                "content": asset.content,
            }
            for asset in assets
        ],
        "total_assets": len(assets),
    }
