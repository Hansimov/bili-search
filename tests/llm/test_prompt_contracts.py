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

    assert "这个作者有哪些关联账号" in prompt
    assert '<search_owners text="目标作者" mode="relation"/>' in prompt
    assert (
        "<search_videos queries=" "'" '[":user=目标作者 :date<=15d"]' "'" "/>" in prompt
    )
    assert '<search_google query="Gemini 2.5 最近有哪些官方更新"/>' in prompt
    assert '<related_tokens_by_tokens text="ComfyUI" mode="auto"/>' in prompt
    assert '<related_tokens_by_tokens text="康夫UI" mode="auto"/>' in prompt
    assert "Gemini 2.5 q=vwr" in prompt
    assert "对比一下作者甲和作者乙最近一个月发布的视频，谁更高产" in prompt
    assert "红警08最近发了什么视频" in prompt
    assert '<search_owners text="红警08" mode="name"/>' in prompt
    assert '<search_owners text="硬件评测" mode="topic"/>' in prompt
    assert (
        "<search_videos queries="
        "'"
        '[":user=作者甲 :date<=30d", ":user=作者乙 :date<=30d"]'
        "'"
        "/>" in prompt
    )
    assert '<search_owners text="黑神话悟空" mode="topic"/>' in prompt


def test_build_system_prompt_emphasizes_search_videos_as_primary_route():
    from llms.prompts.copilot import build_system_prompt

    prompt = build_system_prompt()

    assert "大多数 B 站视频需求优先 `search_videos`" in prompt
    assert (
        "用户最终要视频、代表作、时间线、热门、教程、解读、对比时，默认先用它" in prompt
    )
    assert "不要直接写 `:user=原词`，先用 `search_owners` 确认作者" in prompt
    assert "不要补写猜测的作者主页链接或空链接" in prompt


def test_build_system_prompt_requires_entity_focused_video_queries():
    from llms.prompts.copilot import build_system_prompt

    prompt = build_system_prompt()

    assert "query 是否只剩下关键实体和检索条件" in prompt
    assert "尽量只保留关键实体和检索条件" in prompt or "只保留关键实体" in prompt
    assert "如果仍像一句完整口语句子，说明你还没整理好，先不要搜索" in prompt


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
            "supports_owner_search": True,
            "supports_google_search": True,
            "relation_endpoints": [
                "related_tokens_by_tokens",
            ],
        }
    )
    by_name = {tool["function"]["name"]: tool for tool in tools}

    assert "主力工具和默认首选" in by_name["search_videos"]["function"]["description"]
    assert "关键实体和检索条件" in by_name["search_videos"]["function"]["description"]
    assert "不是用户原话整句" in by_name["search_videos"]["function"]["description"]
    assert "账号关系问题" in by_name["search_videos"]["function"]["description"]
    assert (
        "不要直接把原词写成 :user=xxx"
        in by_name["search_videos"]["function"]["description"]
    )
    assert "官方更新和B站解读" in by_name["search_google"]["function"]["description"]
    assert "搜索作者/UP主" in by_name["search_owners"]["function"]["description"]
    assert "关联账号/矩阵号" in by_name["search_owners"]["function"]["description"]
    assert "最近发了什么视频" in by_name["search_owners"]["function"]["description"]
    assert "辅助工具" in by_name["related_tokens_by_tokens"]["function"]["description"]
    assert (
        "通常还应继续调用 search_videos"
        in by_name["related_tokens_by_tokens"]["function"]["description"]
    )
