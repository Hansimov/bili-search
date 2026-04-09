"""Tests for capability-aware llms tool definitions."""


def test_build_tool_definitions_uses_capabilities():
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
    )

    assert len(tools) == 5
    assert "q=vwr" in tools[0]["function"]["description"]
    assert tools[1]["function"]["name"] == "search_google"
    assert tools[2]["function"]["name"] == "search_owners"
    assert tools[3]["function"]["name"] == "expand_query"
    assert tools[4]["function"]["name"] == "read_spec"


def test_build_tool_prompt_overview_uses_capabilities():
    from llms.tools.defs import build_tool_prompt_overview

    overview = build_tool_prompt_overview(
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

    assert "search_google" in overview
    assert "search_owners" in overview
    assert "expand_query" in overview
    assert "read_spec" in overview
    assert "run_small_llm_task" in overview
    assert "XML 示例" in overview
