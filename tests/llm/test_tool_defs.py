"""Tests for capability-aware llms tool definitions."""


def test_build_tool_definitions_uses_capabilities():
    from llms.tools.defs import build_tool_definitions

    tools = build_tool_definitions(
        {
            "default_query_mode": "wv",
            "rerank_query_mode": "vwr",
            "supports_multi_query": True,
            "supports_google_search": True,
            "relation_endpoints": ["related_owners_by_tokens"],
            "docs": ["search_syntax"],
        },
        include_read_spec=True,
    )

    assert len(tools) == 4
    assert "q=vwr" in tools[0]["function"]["description"]
    assert tools[1]["function"]["name"] == "search_google"
    assert tools[2]["function"]["name"] == "related_owners_by_tokens"
    assert tools[3]["function"]["name"] == "read_spec"
