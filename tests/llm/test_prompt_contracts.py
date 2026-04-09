"""Contract tests for the rewritten llms prompt/tool system."""

from datetime import datetime

from llms.contracts import FacetScore, IntentProfile


def _video_intent() -> IntentProfile:
    return IntentProfile(
        raw_query="黑神话教程",
        normalized_query="黑神话教程",
        final_target="videos",
        task_mode="lookup_entity",
        motivation=[FacetScore("information", 0.9)],
        expected_payoff=[FacetScore("clear_explanation", 0.8)],
        explicit_entities=["黑神话"],
        explicit_topics=["教程"],
    )


def test_build_system_prompt_contains_new_core_sections():
    from llms.prompts.copilot import build_system_prompt

    prompt = build_system_prompt(
        {
            "default_query_mode": "wv",
            "rerank_query_mode": "vwr",
            "supports_multi_query": True,
            "supports_google_search": True,
            "supports_owner_search": True,
            "relation_endpoints": ["related_tokens_by_tokens"],
        },
        intent=_video_intent(),
    )

    for section in [
        "[TOOL_OVERVIEW]",
        "[INTENT_PROFILE]",
        "[PROMPT_LOADING]",
        "[RESULT_ISOLATION]",
        "[ROUTING_EXAMPLES]",
        "[OUTPUT_PROTOCOL]",
        "[ROUTE_VIDEOS]",
        "[DSL_QUICKREF]",
    ]:
        assert section in prompt


def test_build_system_prompt_mentions_internal_prompt_loading_and_result_isolation():
    from llms.prompts.copilot import build_system_prompt

    prompt = build_system_prompt(intent=_video_intent())

    assert "read_prompt_assets" in prompt
    assert "inspect_tool_result" in prompt
    assert "run_small_llm_task" in prompt
    assert "原始工具结果会保留在独立结果仓库和前端工具面板中" in prompt
    assert "默认只加载与当前任务最相关的 brief guidance" in prompt
    assert "不要重复扩搜" in prompt
    assert "若摘要里已有 BV、mid 或链接，就直接回答" in prompt
    assert "共享路由样例" in prompt


def test_build_system_prompt_profile_tracks_new_sections():
    from llms.prompts.copilot import build_system_prompt_profile

    profile = build_system_prompt_profile(intent=_video_intent())
    section_chars = profile["section_chars"]

    assert "output_protocol" in section_chars
    assert "prompt_loading" in section_chars
    assert "result_isolation" in section_chars
    assert "routing_examples" in section_chars
    assert "tool_commands" in section_chars
    assert "dsl_quickref" in section_chars
    assert profile["total_chars"] >= sum(section_chars.values())


def test_date_prompt_is_stable_within_same_day():
    from llms.prompts.system import get_date_prompt

    morning = datetime(2026, 4, 9, 9, 1, 3)
    evening = datetime(2026, 4, 9, 23, 59, 59)

    prompt = get_date_prompt(now=morning)

    assert prompt == get_date_prompt(now=evening)
    assert "当前日期：2026-04-09（周四）" in prompt
    assert "昨天日期：2026-04-08" in prompt


def test_build_system_prompt_places_dynamic_context_after_shared_guidance():
    from llms.prompts.copilot import build_system_prompt

    prompt = build_system_prompt(intent=_video_intent())

    assert prompt.index("[PROMPT_LOADING]") < prompt.index("[SYSTEM_TIME]")
    assert prompt.index("[OUTPUT_PROTOCOL]") < prompt.index("[INTENT_PROFILE]")


def test_get_prompt_assets_payload_returns_requested_levels():
    from llms.prompts.copilot import get_prompt_assets_payload

    payload = get_prompt_assets_payload(
        tool_names=["search_videos"],
        levels=["detailed", "examples"],
    )

    assert payload["total_assets"] == 2
    levels = {asset["level"] for asset in payload["assets"]}
    assert levels == {"detailed", "examples"}
    assert all(asset["tool_name"] == "search_videos" for asset in payload["assets"])


def test_tool_definitions_include_internal_tools_when_requested():
    from llms.tools.defs import build_tool_definitions

    tools = build_tool_definitions(
        {
            "default_query_mode": "wv",
            "rerank_query_mode": "vwr",
            "supports_multi_query": True,
            "supports_owner_search": True,
            "supports_google_search": True,
            "relation_endpoints": ["related_tokens_by_tokens"],
            "docs": ["search_syntax"],
        },
        include_read_spec=True,
        include_internal=True,
    )
    names = [tool["function"]["name"] for tool in tools]

    assert "search_videos" in names
    assert "search_google" in names
    assert "search_owners" in names
    assert "expand_query" in names
    assert "read_spec" in names
    assert "read_prompt_assets" in names
    assert "inspect_tool_result" in names
    assert "run_small_llm_task" in names


def test_tool_definitions_describe_new_routing_boundaries():
    from llms.tools.defs import build_tool_definitions

    tools = build_tool_definitions(
        {
            "default_query_mode": "wv",
            "rerank_query_mode": "vwr",
            "supports_multi_query": True,
            "supports_owner_search": True,
            "supports_google_search": True,
            "relation_endpoints": ["related_tokens_by_tokens"],
        }
    )
    by_name = {tool["function"]["name"]: tool for tool in tools}

    assert "默认终局工具" in by_name["search_videos"]["function"]["description"]
    assert "DSL 搜索语句" in by_name["search_videos"]["function"]["description"]
    assert (
        "作者名不稳时先 search_owners"
        in by_name["search_videos"]["function"]["description"]
    )
    assert "关键词侦察层" in by_name["search_google"]["function"]["description"]
    assert (
        "site:bilibili.com/video" in by_name["search_google"]["function"]["description"]
    )
    assert "作者问题优先用它" in by_name["search_owners"]["function"]["description"]
    assert (
        "抽象 query 的语义展开工具"
        in by_name["expand_query"]["function"]["description"]
    )
