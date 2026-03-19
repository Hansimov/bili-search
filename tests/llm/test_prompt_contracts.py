"""Contract tests for copilot prompts and tool definitions.

These tests keep the prompt/tool orchestration structure stable as we iterate.
"""


def test_build_system_prompt_contains_core_orchestration_sections():
    from llms.prompts.copilot import build_system_prompt

    prompt = build_system_prompt(
        {
            "default_query_mode": "wv",
            "rerank_query_mode": "vwr",
            "supports_multi_query": True,
            "supports_google_search": True,
            "relation_endpoints": [
                "related_tokens_by_tokens",
                "related_owners_by_tokens",
                "related_videos_by_videos",
            ],
            "docs": ["search_syntax"],
        }
    )

    for section in [
        "[OUTPUT_PROTOCOL]",
        "[TOOL_ROUTING]",
        "[DSL_PLANNING]",
        "[ANTI_PATTERNS]",
        "[SEARCH_SYNTAX]",
        "[EXAMPLES]",
    ]:
        assert section in prompt


def test_build_system_prompt_examples_cover_major_tool_mix_scenarios():
    from llms.prompts.copilot import build_system_prompt

    prompt = build_system_prompt()

    assert "何同学有哪些关联账号" in prompt
    assert "那他的代表作有哪些" in prompt
    assert '<related_owners_by_tokens text="何同学"/>' in prompt
    assert "<search_videos queries=" "'" '[":user=何同学"]' "'" "/>" in prompt
    assert '<search_google query="Gemini 2.5 最近有哪些官方更新"/>' in prompt
    assert "先只看官网就行" in prompt
    assert '<related_tokens_by_tokens text="ComfyUI" mode="auto"/>' in prompt
    assert '<related_tokens_by_tokens text="康夫UI" mode="auto"/>' in prompt
    assert "对比一下老番茄和影视飓风最近一个月发布的视频，谁更高产" in prompt
    assert (
        "<search_videos queries="
        "'"
        '[":user=老番茄 :date<=30d", ":user=影视飓风 :date<=30d"]'
        "'"
        "/>" in prompt
    )
    assert "<related_videos_by_videos bvids='[\"BV1xx\"]'/>" in prompt


def test_build_system_prompt_profile_tracks_new_sections():
    from llms.prompts.copilot import build_system_prompt_profile

    profile = build_system_prompt_profile()
    section_chars = profile["section_chars"]

    assert "output_protocol" in section_chars
    assert "tool_routing" in section_chars
    assert "dsl_planning" in section_chars
    assert "anti_patterns" in section_chars
    assert profile["total_chars"] >= sum(section_chars.values())


def test_tool_definitions_explain_routing_boundaries():
    from llms.tools.defs import build_tool_definitions

    tools = build_tool_definitions(
        {
            "default_query_mode": "wv",
            "rerank_query_mode": "vwr",
            "supports_multi_query": True,
            "supports_google_search": True,
            "relation_endpoints": [
                "related_tokens_by_tokens",
                "related_owners_by_tokens",
                "related_videos_by_videos",
            ],
        }
    )
    by_name = {tool["function"]["name"]: tool for tool in tools}

    assert "不是用户原话整句" in by_name["search_videos"]["function"]["description"]
    assert "账号关系问题" in by_name["search_videos"]["function"]["description"]
    assert "官方更新和B站解读" in by_name["search_google"]["function"]["description"]
    assert (
        "通常还应继续调用 search_videos"
        in by_name["related_tokens_by_tokens"]["function"]["description"]
    )
    assert (
        "关联账号/矩阵号/主副号"
        in by_name["related_owners_by_tokens"]["function"]["description"]
    )
