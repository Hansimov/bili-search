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
    assert '"Gemini 2.5" API 教程 q=vwr' in prompt
    assert "剧情解析 +黑神话 -广告 :date<=30d :view>=1w q=vwr" in prompt
    assert "对比一下老番茄和影视飓风最近一个月发布的视频，谁更高产" in prompt
    assert (
        "<search_videos queries="
        "'"
        '[":user=老番茄 :date<=30d", ":user=影视飓风 :date<=30d"]'
        "'"
        "/>" in prompt
    )
    assert "<related_videos_by_videos bvids='[\"BV1xx\"]'/>" in prompt


def test_build_system_prompt_emphasizes_search_videos_as_primary_route():
    from llms.prompts.copilot import build_system_prompt

    prompt = build_system_prompt()

    assert "`search_videos`：主力工具，也是大多数 B 站问题的默认首选" in prompt
    assert "默认先尝试用 `search_videos` 这个终局工具" in prompt
    assert "`search_videos` 仍然是主力；大多数 B 站问题先考虑它" in prompt


def test_build_system_prompt_requires_entity_focused_video_queries():
    from llms.prompts.copilot import build_system_prompt

    prompt = build_system_prompt()

    assert "query 里优先保留关键实体" in prompt
    assert "主动删掉口语词、助词和功能词" in prompt
    assert "尽量只保留关键实体和检索条件" in prompt or "只保留关键实体" in prompt
    assert "如果用户说“必须有/不要有/排除”，是否应该用 `+词` / `-词`" in prompt
    assert "如果是多个作者或多个 UID 的并列对比，是否应该用数组过滤器" in prompt


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

    assert "主力工具和默认首选" in by_name["search_videos"]["function"]["description"]
    assert "关键实体和检索条件" in by_name["search_videos"]["function"]["description"]
    assert "不是用户原话整句" in by_name["search_videos"]["function"]["description"]
    assert "账号关系问题" in by_name["search_videos"]["function"]["description"]
    assert "官方更新和B站解读" in by_name["search_google"]["function"]["description"]
    assert "辅助工具" in by_name["related_tokens_by_tokens"]["function"]["description"]
    assert (
        "通常还应继续调用 search_videos"
        in by_name["related_tokens_by_tokens"]["function"]["description"]
    )
    assert "辅助工具" in by_name["related_owners_by_tokens"]["function"]["description"]
    assert (
        "关联账号/矩阵号/主副号"
        in by_name["related_owners_by_tokens"]["function"]["description"]
    )
