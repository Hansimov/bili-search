"""Tests for llms.tools — tool definitions and executor.

Uses mocked search service for unit tests.

Run:
    python -m tests.llm.test_tools
"""

import json
from unittest.mock import MagicMock, patch
from tclogger import logger

from llms.llm_client import ToolCall
from llms.tools.defs import TOOL_DEFINITIONS
from llms.tools.executor import ToolExecutor


# ============================================================
# Test data
# ============================================================

MOCK_EXPLORE_RESULT = {
    "query": "黑神话",
    "status": "finished",
    "data": [
        {
            "step": 0,
            "name": "most_relevant_search",
            "output": {
                "hits": [
                    {
                        "bvid": "BV1abc",
                        "title": "黑神话悟空全流程",
                        "desc": "完整攻略",
                        "owner": {"mid": 100, "name": "游戏UP主"},
                        "pubdate": 1708700000,
                        "stat": {"view": 500000, "coin": 10000, "danmaku": 2000},
                    },
                    {
                        "bvid": "BV1def",
                        "title": "黑神话评测",
                        "desc": "深度评测",
                        "owner": {"mid": 200, "name": "测评达人"},
                        "pubdate": 1708600000,
                        "stat": {"view": 200000, "coin": 5000, "danmaku": 1000},
                    },
                ],
                "total_hits": 42,
            },
        },
    ],
}

MOCK_RELATED_OWNERS_RESULT = {
    "text": "影视飓风",
    "owners": [
        {
            "mid": 946974,
            "name": "影视飓风",
            "doc_freq": 18,
            "score": 71.3,
        },
        {
            "mid": 1780480185,
            "name": "飓多多StormCrew",
            "doc_freq": 3,
            "score": 12.5,
        },
    ],
}

MOCK_SEARCH_OWNERS_RESULT = {
    "text": "红警08",
    "mode": "name",
    "owners": [
        {
            "mid": 546195,
            "name": "红警HBK08",
            "score": 168.2,
            "sources": ["name"],
            "sample_title": "红警月亮3对战复盘",
            "sample_bvid": "BV1owner1",
        },
        {
            "mid": 946974,
            "name": "红警月亮3",
            "score": 88.5,
            "sources": ["topic"],
        },
    ],
}

# ============================================================
# Tests
# ============================================================


def test_tool_definitions_format():
    """Test that tool definitions follow OpenAI schema."""
    logger.note("=" * 60)
    logger.note("[TEST] tool definitions format")

    assert len(TOOL_DEFINITIONS) == 1

    for tool_def in TOOL_DEFINITIONS:
        assert tool_def["type"] == "function"
        func = tool_def["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        params = func["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

    # Check specific tools exist
    names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert "search_videos" in names

    # Verify search_videos uses queries array
    search_tool = [
        t for t in TOOL_DEFINITIONS if t["function"]["name"] == "search_videos"
    ][0]
    search_params = search_tool["function"]["parameters"]
    assert "queries" in search_params["properties"]
    assert search_params["properties"]["queries"]["type"] == "array"

    logger.success("[PASS] tool definitions format")


def test_execute_search_videos():
    """Test search_videos tool execution."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_videos")

    mock_client = MagicMock()
    mock_client.explore.return_value = MOCK_EXPLORE_RESULT

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_1",
        name="search_videos",
        arguments=json.dumps({"queries": ["黑神话"]}),
    )
    result_msg = executor.execute(tc)

    assert result_msg["role"] == "tool"
    assert result_msg["tool_call_id"] == "call_test_1"

    result_data = json.loads(result_msg["content"])
    assert result_data["query"] == "黑神话"
    assert result_data["total_hits"] == 42
    assert len(result_data["hits"]) == 2
    assert result_data["hits"][0]["bvid"] == "BV1abc"
    assert "link" in result_data["hits"][0]  # Links should be added
    assert "pub_to_now_str" in result_data["hits"][0]  # Time strings should be added

    # Verify explore was called
    mock_client.explore.assert_called_once_with(query="黑神话")

    logger.success("[PASS] execute search_videos")


def test_execute_related_owners_by_tokens():
    """Test related_owners_by_tokens tool execution."""
    logger.note("=" * 60)
    logger.note("[TEST] execute related_owners_by_tokens")

    mock_client = MagicMock()
    mock_client.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_2",
        name="related_owners_by_tokens",
        arguments=json.dumps({"text": "影视飓风"}),
    )
    result_msg = executor.execute(tc)

    assert result_msg["role"] == "tool"
    assert result_msg["tool_call_id"] == "call_test_2"

    result_data = json.loads(result_msg["content"])
    assert result_data["text"] == "影视飓风"
    assert result_data["total_owners"] == 2
    owner = result_data["owners"][0]
    assert owner["mid"] == 946974
    assert owner["name"] == "影视飓风"
    assert "link" not in owner
    assert "doc_freq" not in owner

    logger.success("[PASS] execute related_owners_by_tokens")


def test_execute_search_owners():
    """Test search_owners tool execution."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_owners")

    mock_client = MagicMock()
    mock_client.search_owners.return_value = MOCK_SEARCH_OWNERS_RESULT

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_owners",
        name="search_owners",
        arguments=json.dumps({"text": "红警08", "mode": "name"}),
    )
    result_msg = executor.execute(tc)

    assert result_msg["role"] == "tool"
    assert result_msg["tool_call_id"] == "call_test_owners"

    result_data = json.loads(result_msg["content"])
    assert result_data["text"] == "红警08"
    assert result_data["mode"] == "name"
    assert result_data["total_owners"] == 2
    assert result_data["owners"][0]["name"] == "红警HBK08"
    assert result_data["owners"][0]["sources"] == ["name"]
    assert result_data["owners"][0]["sample_title"] == "红警月亮3对战复盘"
    mock_client.search_owners.assert_called_once_with(
        text="红警08", mode="name", size=8
    )

    logger.success("[PASS] execute search_owners")


def test_execute_unknown_tool():
    """Test unknown tool name handling."""
    logger.note("=" * 60)
    logger.note("[TEST] execute unknown tool")

    mock_client = MagicMock()
    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_3",
        name="nonexistent_tool",
        arguments="{}",
    )
    result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    assert "error" in result_data

    logger.success("[PASS] execute unknown tool")


def test_tool_executor_merges_google_capability():
    """Google Hub capability should be exposed when health check passes."""
    logger.note("=" * 60)
    logger.note("[TEST] tool executor merges google capability")

    mock_client = MagicMock()
    mock_client.capabilities.return_value = {
        "service_name": "search-app",
        "supports_multi_query": True,
        "relation_endpoints": ["related_owners_by_tokens"],
    }
    mock_google = MagicMock()
    mock_google.base_url = "http://mock-google:18100"

    executor = ToolExecutor(search_client=mock_client, google_client=mock_google)

    # Mock the health check HTTP call to return 200
    mock_response = MagicMock()
    mock_response.status_code = 200
    with patch("llms.tools.executor.requests.get", return_value=mock_response):
        capabilities = executor.get_search_capabilities()

    assert capabilities["supports_google_search"] is True
    assert capabilities["relation_endpoints"] == ["related_owners_by_tokens"]

    logger.success("[PASS] tool executor merges google capability")


def test_execute_empty_query():
    """Test search with empty query."""
    logger.note("=" * 60)
    logger.note("[TEST] execute empty query")

    mock_client = MagicMock()
    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_4",
        name="search_videos",
        arguments=json.dumps({"queries": [""]}),
    )
    result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    assert "error" in result_data

    logger.success("[PASS] execute empty query")


def test_execute_search_error():
    """Test search error handling."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search error")

    mock_client = MagicMock()
    mock_client.explore.return_value = {
        "error": "connection_failed",
        "hits": [],
        "total_hits": 0,
    }

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_5",
        name="search_videos",
        arguments=json.dumps({"queries": ["test"]}),
    )
    result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    assert "error" in result_data
    assert result_data["total_hits"] == 0

    logger.success("[PASS] execute search error")


def test_execute_read_spec():
    """Test read_spec tool execution."""
    logger.note("=" * 60)
    logger.note("[TEST] execute read_spec")

    mock_client = MagicMock()
    executor = ToolExecutor(search_client=mock_client)

    # Test valid spec name
    tc = ToolCall(
        id="call_test_spec_1",
        name="read_spec",
        arguments=json.dumps({"name": "search_syntax"}),
    )
    result_msg = executor.execute(tc)

    assert result_msg["role"] == "tool"
    assert result_msg["tool_call_id"] == "call_test_spec_1"

    result_data = json.loads(result_msg["content"])
    assert result_data["name"] == "search_syntax"
    assert "SEARCH_SYNTAX" in result_data["content"]
    assert ":view>=1w" in result_data["content"]  # Contains syntax examples

    # Test unknown spec name
    tc_bad = ToolCall(
        id="call_test_spec_2",
        name="read_spec",
        arguments=json.dumps({"name": "nonexistent"}),
    )
    result_bad = executor.execute(tc_bad)
    result_bad_data = json.loads(result_bad["content"])
    assert "error" in result_bad_data
    assert "available" in result_bad_data

    logger.success("[PASS] execute read_spec")


def test_max_results_limit():
    """Test that executor limits the number of results."""
    logger.note("=" * 60)
    logger.note("[TEST] max_results limit")

    # Build result with 30 hits
    many_hits = [
        {
            "bvid": f"BV{i}",
            "title": f"Video {i}",
            "owner": {"mid": i, "name": f"User{i}"},
        }
        for i in range(30)
    ]
    explore_result = {
        "query": "test",
        "status": "finished",
        "data": [
            {
                "step": 0,
                "name": "most_relevant_search",
                "output": {"hits": many_hits, "total_hits": 100},
            }
        ],
    }

    mock_client = MagicMock()
    mock_client.explore.return_value = explore_result

    executor = ToolExecutor(search_client=mock_client, max_results=10)

    tc = ToolCall(
        id="call_test_6",
        name="search_videos",
        arguments=json.dumps({"queries": ["test"]}),
    )
    result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    assert len(result_data["hits"]) == 10  # Limited to max_results
    assert result_data["total_hits"] == 100  # Total is preserved

    logger.success("[PASS] max_results limit")


def test_multi_query_search():
    """Test search_videos with multiple queries."""
    logger.note("=" * 60)
    logger.note("[TEST] multi-query search")

    mock_client = MagicMock()
    mock_client.explore.return_value = MOCK_EXPLORE_RESULT

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_mq1",
        name="search_videos",
        arguments=json.dumps(
            {"queries": ["黑神话 :view>=1w", ":user=影视飓风 :date<=7d"]}
        ),
    )
    result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    # Multi-query returns { "results": [...] }
    assert "results" in result_data
    assert len(result_data["results"]) == 2
    assert result_data["results"][0]["query"] == "黑神话 :view>=1w"
    assert result_data["results"][1]["query"] == ":user=影视飓风 :date<=7d"

    # explore should be called twice
    assert mock_client.explore.call_count == 2

    logger.success("[PASS] multi-query search")


def test_legacy_query_string():
    """Test search_videos backward compatibility with single 'query' string."""
    logger.note("=" * 60)
    logger.note("[TEST] legacy query string")

    mock_client = MagicMock()
    mock_client.explore.return_value = MOCK_EXPLORE_RESULT

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_legacy",
        name="search_videos",
        arguments=json.dumps({"query": "黑神话"}),
    )
    result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    # Single legacy query returns flat result
    assert result_data["query"] == "黑神话"
    assert "total_hits" in result_data
    assert "hits" in result_data

    mock_client.explore.assert_called_once_with(query="黑神话")

    logger.success("[PASS] legacy query string")


if __name__ == "__main__":
    tests = [
        ("tool_definitions_format", test_tool_definitions_format),
        ("execute_search_videos", test_execute_search_videos),
        ("execute_search_owners", test_execute_search_owners),
        ("execute_related_owners_by_tokens", test_execute_related_owners_by_tokens),
        (
            "tool_executor_merges_google_capability",
            test_tool_executor_merges_google_capability,
        ),
        ("execute_read_spec", test_execute_read_spec),
        ("execute_unknown_tool", test_execute_unknown_tool),
        ("execute_empty_query", test_execute_empty_query),
        ("execute_search_error", test_execute_search_error),
        ("max_results_limit", test_max_results_limit),
        ("multi_query_search", test_multi_query_search),
        ("legacy_query_string", test_legacy_query_string),
    ]

    results = {}
    for name, test_func in tests:
        try:
            test_func()
            results[name] = "PASS"
        except Exception as e:
            logger.warn(f"\n[FAIL] {name}: {e}")
            results[name] = f"FAIL: {e}"

    logger.note("\n" + "=" * 60)
    logger.note("[SUMMARY]")
    logger.note("=" * 60)
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    for name, result in results.items():
        status = logger.success if result == "PASS" else logger.warn
        status(f"  {name}: {result}")
    logger.note(f"\n  {passed}/{total} tests passed")

    # python -m tests.llm.test_tools
