"""Tests for llms.tools — tool definitions and executor.

Uses mocked search service for unit tests.

Run:
    python -m tests.llm.test_tools
"""

import json
import requests
import time
from unittest.mock import MagicMock, patch
from tclogger import logger

from llms.models import ToolCall
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
            "face": "https://example.com/owner-face.jpg",
            "sample_title": "红警月亮3对战复盘",
            "sample_bvid": "BV1owner1",
            "sample_pic": "https://example.com/sample-cover.jpg",
            "sample_view": 234567,
        },
        {
            "mid": 946974,
            "name": "红警月亮3",
            "score": 88.5,
            "sources": ["topic"],
        },
    ],
}

MOCK_TRANSCRIPT_RESULT = {
    "ok": True,
    "requested_video_id": "BV1YXZPB1Erc",
    "bvid": "BV1YXZPB1Erc",
    "title": "示例视频",
    "page_index": 1,
    "selection": {
        "selected_text_length": 128,
        "full_text_length": 512,
    },
    "transcript": {
        "text": "这是一个示例转写，用来验证 transcript tool 的执行链路。",
        "text_length": 128,
        "segment_count": 3,
    },
}

MOCK_VIDEO_LOOKUP_RESULT = {
    "lookup_by": "bvids",
    "bvids": ["BV1e9cfz5EKj"],
    "hits": [
        {
            "bvid": "BV1e9cfz5EKj",
            "title": "人在柬埔寨，刚下飞机，现在跑还来得及吗？",
            "desc": "示例描述",
            "owner": {"mid": 39627524, "name": "食贫道"},
            "pubdate": 1708700000,
            "stat": {"view": 1234567},
        }
    ],
    "total_hits": 1,
    "source_counts": {"mongo": 1, "es": 1},
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
    assert "bv" in search_params["properties"]
    assert "mid" in search_params["properties"]

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


def test_execute_search_videos_prefers_direct_search_when_available():
    logger.note("=" * 60)
    logger.note("[TEST] execute search_videos prefers direct search")

    mock_client = MagicMock()
    mock_client.search.return_value = {
        "hits": [
            {
                "bvid": "BV1direct",
                "title": "ComfyUI 工作流讲解",
                "owner": {"mid": 1, "name": "教程作者"},
            }
        ],
        "total_hits": 12,
    }

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_direct_search",
        name="search_videos",
        arguments=json.dumps({"queries": ["康夫 UI 工作流"]}),
    )
    result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    assert result_data["query"] == "康夫 UI 工作流"
    assert result_data["total_hits"] == 12
    assert result_data["hits"][0]["bvid"] == "BV1direct"
    mock_client.search.assert_called_once_with(
        query="康夫 UI 工作流",
        limit=15,
        verbose=False,
    )
    mock_client.explore.assert_not_called()

    logger.success("[PASS] execute search_videos prefers direct search")


def test_execute_search_videos_falls_back_to_explore_when_direct_search_has_no_hits():
    logger.note("=" * 60)
    logger.note("[TEST] execute search_videos falls back to explore")

    mock_client = MagicMock()
    mock_client.search.return_value = {"hits": [], "total_hits": 0}
    mock_client.explore.return_value = MOCK_EXPLORE_RESULT

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_direct_fallback",
        name="search_videos",
        arguments=json.dumps({"queries": ["黑神话"]}),
    )
    result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    assert result_data["query"] == "黑神话"
    assert result_data["total_hits"] == 42
    assert result_data["hits"][0]["bvid"] == "BV1abc"
    mock_client.search.assert_called_once_with(
        query="黑神话",
        limit=15,
        verbose=False,
    )
    mock_client.explore.assert_called_once_with(query="黑神话")

    logger.success("[PASS] execute search_videos falls back to explore")


def test_execute_search_videos_filters_off_topic_interview_hits_for_llm():
    """Interview-style queries should not expose off-topic hits without interview anchors."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_videos filters off-topic interview hits")

    mock_client = MagicMock()
    mock_client.explore.return_value = {
        "query": "袁启 专访",
        "status": "finished",
        "data": [
            {
                "step": 0,
                "name": "most_relevant_search",
                "output": {
                    "hits": [
                        {
                            "bvid": "BV1offtopic",
                            "title": "雪王看袁启聪《看了就知道，老头乐到底有多么危险》",
                            "desc": "搞笑 reaction",
                            "owner": {"mid": 100, "name": "雪绘Yukie"},
                            "pubdate": 1708700000,
                            "stat": {"view": 46000},
                        }
                    ],
                    "total_hits": 40,
                },
            },
        ],
    }

    executor = ToolExecutor(search_client=mock_client)
    tc = ToolCall(
        id="call_test_interview_filter",
        name="search_videos",
        arguments=json.dumps({"queries": ["袁启 专访"]}),
    )

    result_msg = executor.execute(tc)
    result_data = json.loads(result_msg["content"])

    assert result_data["query"] == "袁启 专访"
    assert result_data["total_hits"] == 40
    assert result_data["hits"] == []
    assert result_data["theme_filter"]["theme_tokens"] == ["采访", "专访", "访谈"]
    assert "warning" in result_data

    logger.success("[PASS] execute search_videos filters off-topic interview hits")


def test_execute_search_videos_keeps_on_topic_interview_hits_for_llm():
    """Interview-style queries should keep hits whose title carries interview anchors."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_videos keeps on-topic interview hits")

    mock_client = MagicMock()
    mock_client.explore.return_value = {
        "query": "袁启 专访",
        "status": "finished",
        "data": [
            {
                "step": 0,
                "name": "most_relevant_search",
                "output": {
                    "hits": [
                        {
                            "bvid": "BV1topic",
                            "title": "袁启聪专访：聊聊汽车媒体行业",
                            "desc": "真正的专访内容",
                            "owner": {"mid": 100, "name": "采访频道"},
                            "pubdate": 1708700000,
                            "stat": {"view": 120000},
                        }
                    ],
                    "total_hits": 3,
                },
            },
        ],
    }

    executor = ToolExecutor(search_client=mock_client)
    tc = ToolCall(
        id="call_test_interview_keep",
        name="search_videos",
        arguments=json.dumps({"queries": ["袁启 专访"]}),
    )

    result_msg = executor.execute(tc)
    result_data = json.loads(result_msg["content"])

    assert result_data["query"] == "袁启 专访"
    assert result_data["total_hits"] == 3
    assert len(result_data["hits"]) == 1
    assert result_data["hits"][0]["bvid"] == "BV1topic"
    assert result_data["theme_filter"]["dropped_hits"] == 0
    assert "warning" not in result_data

    logger.success("[PASS] execute search_videos keeps on-topic interview hits")


def test_execute_search_videos_explicit_bv_query_uses_lookup():
    """Explicit BV-only video queries should be coerced into exact lookup."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_videos explicit bv lookup")

    mock_client = MagicMock()
    mock_client.lookup_videos.return_value = MOCK_VIDEO_LOOKUP_RESULT

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_lookup_1",
        name="search_videos",
        arguments=json.dumps({"queries": ["BV1e9cfz5EKj"]}),
    )
    result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    assert result_data["mode"] == "lookup"
    assert result_data["lookup_by"] == "bvids"
    assert result_data["bvids"] == ["BV1e9cfz5EKj"]
    assert result_data["total_hits"] == 1
    assert result_data["hits"][0]["bvid"] == "BV1e9cfz5EKj"
    assert result_data["hits"][0]["owner"]["name"] == "食贫道"
    assert result_data["source_counts"] == {"mongo": 1, "es": 1}

    mock_client.lookup_videos.assert_called_once_with(
        bvids=["BV1e9cfz5EKj"],
        mids=None,
        limit=10,
        date_window=None,
        exclude_bvids=None,
        verbose=False,
    )

    logger.success("[PASS] execute search_videos explicit bv lookup")


def test_execute_get_video_transcript():
    """Test get_video_transcript tool execution."""
    logger.note("=" * 60)
    logger.note("[TEST] execute get_video_transcript")

    mock_client = MagicMock()
    mock_client.get_video_transcript.return_value = MOCK_TRANSCRIPT_RESULT

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_transcript",
        name="get_video_transcript",
        arguments=json.dumps(
            {
                "video_id": "BV1YXZPB1Erc",
                "head_chars": 6000,
                "include_segments": True,
            }
        ),
    )
    result_msg = executor.execute(tc)

    assert result_msg["role"] == "tool"
    assert result_msg["tool_call_id"] == "call_test_transcript"

    result_data = json.loads(result_msg["content"])
    assert result_data["bvid"] == "BV1YXZPB1Erc"
    assert result_data["transcript"]["text"].startswith("这是一个示例转写")
    mock_client.get_video_transcript.assert_called_once_with(
        video_id="BV1YXZPB1Erc",
        head_chars=6000,
        include_segments=True,
    )

    logger.success("[PASS] execute get_video_transcript")


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


def test_execute_search_videos_tries_fallback_when_primary_hits_fail_relevance_floor():
    """Weak primary hits should not block the fallback query path."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_videos weak-hit fallback")

    weak_primary = {
        "query": "黑神话 更新",
        "status": "finished",
        "data": [
            {
                "step": 0,
                "name": "most_relevant_search",
                "output": {
                    "hits": [
                        {
                            "bvid": "BV1weak1",
                            "title": "低相关候选1",
                            "owner": {"mid": 1, "name": "A"},
                            "relevance_score": 0.12,
                        },
                        {
                            "bvid": "BV1weak2",
                            "title": "低相关候选2",
                            "owner": {"mid": 2, "name": "B"},
                            "relevance_score": 0.08,
                        },
                    ],
                    "total_hits": 7,
                },
            }
        ],
    }
    strong_fallback = {
        "query": "黑神话",
        "status": "finished",
        "data": [
            {
                "step": 0,
                "name": "most_relevant_search",
                "output": {
                    "hits": [
                        {
                            "bvid": "BV1strong",
                            "title": "黑神话悟空全流程",
                            "owner": {"mid": 100, "name": "游戏UP主"},
                            "pubdate": 1708700000,
                            "stat": {"view": 500000},
                            "relevance_score": 0.84,
                        }
                    ],
                    "total_hits": 12,
                },
            }
        ],
    }

    mock_client = MagicMock()
    mock_client.explore.side_effect = [weak_primary, strong_fallback]

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_fallback_weak_hits",
        name="search_videos",
        arguments=json.dumps({"queries": ["黑神话 更新"]}),
    )
    result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    assert result_data["query"] == "黑神话 更新"
    assert result_data["fallback_applied"] is True
    assert result_data["resolved_query"] == "黑神话"
    assert result_data["hits"][0]["bvid"] == "BV1strong"
    assert mock_client.explore.call_count == 2

    logger.success("[PASS] execute search_videos weak-hit fallback")


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
        (
            "execute_search_owners_relation_discovery",
            test_execute_search_owners_relation_discovery,
        ),
        (
            "tool_executor_merges_google_capability",
            test_tool_executor_merges_google_capability,
        ),
        ("execute_read_spec", test_execute_read_spec),
        (
            "execute_expand_query_prefers_semantic_mode",
            test_execute_expand_query_prefers_semantic_mode,
        ),
        (
            "execute_expand_query_falls_back_to_auto_when_semantic_unsupported",
            test_execute_expand_query_falls_back_to_auto_when_semantic_unsupported,
        ),
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
