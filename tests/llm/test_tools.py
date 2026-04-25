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


def test_execute_search_owners_relation_discovery():
    """Test search_owners relation discovery via owner seeds."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_owners relation discovery")

    mock_client = MagicMock()
    mock_client.related_owners_by_owners.return_value = {
        "relation": "related_owners_by_owners",
        "owners": MOCK_RELATED_OWNERS_RESULT["owners"],
    }

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_2",
        name="search_owners",
        arguments=json.dumps({"mids": [946974], "mode": "relation"}),
    )
    result_msg = executor.execute(tc)

    assert result_msg["role"] == "tool"
    assert result_msg["tool_call_id"] == "call_test_2"

    result_data = json.loads(result_msg["content"])
    assert result_data["mids"] == ["946974"]
    assert result_data["total_owners"] == 2
    owner = result_data["owners"][0]
    assert owner["mid"] == 946974
    assert owner["name"] == "影视飓风"
    assert "link" not in owner
    assert "doc_freq" not in owner

    logger.success("[PASS] execute search_owners relation discovery")


def test_execute_search_owners():
    """search_owners should aggregate multiple owner sources for text queries."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_owners")

    mock_client = MagicMock()

    def fake_search_owners(*, text: str, mode: str, size: int):
        assert text == "红警08"
        assert size == 8
        if mode == "name":
            return MOCK_SEARCH_OWNERS_RESULT
        if mode == "topic":
            return {
                "text": "红警08",
                "mode": "topic",
                "owners": [
                    {
                        "mid": 546195,
                        "name": "红警HBK08",
                        "score": 92.0,
                        "sources": ["topic"],
                    }
                ],
            }
        if mode == "relation":
            return {"text": "红警08", "mode": "relation", "owners": []}
        raise AssertionError(f"unexpected mode: {mode}")

    mock_client.search_owners.side_effect = fake_search_owners

    executor = ToolExecutor(search_client=mock_client)

    tc = ToolCall(
        id="call_test_owners",
        name="search_owners",
        arguments=json.dumps({"text": "红警08", "mode": "name"}),
    )
    with patch.object(executor, "_is_google_available", return_value=False):
        result_msg = executor.execute(tc)

    assert result_msg["role"] == "tool"
    assert result_msg["tool_call_id"] == "call_test_owners"

    result_data = json.loads(result_msg["content"])
    assert result_data["text"] == "红警08"
    assert result_data["mode"] == "aggregate"
    assert result_data["requested_mode"] == "name"
    assert result_data["total_owners"] == 2
    assert result_data["owners"][0]["name"] == "红警HBK08"
    assert result_data["owners"][0]["sources"] == ["name", "topic"]
    assert result_data["owners"][0]["face"] == "https://example.com/owner-face.jpg"
    assert result_data["owners"][0]["sample_title"] == "红警月亮3对战复盘"
    assert (
        result_data["owners"][0]["sample_pic"] == "https://example.com/sample-cover.jpg"
    )
    assert result_data["owners"][0]["sample_view"] == 234567
    assert result_data["source_counts"]["name"] == 2
    assert result_data["source_counts"]["topic"] == 1
    assert result_data["source_counts"]["relation"] == 0
    assert mock_client.search_owners.call_count == 3
    called_modes = {
        call.kwargs["mode"] for call in mock_client.search_owners.call_args_list
    }
    assert called_modes == {"name", "topic", "relation"}

    logger.success("[PASS] execute search_owners")


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


def test_execute_search_owners_topic_uses_wider_default_limit():
    """Aggregated owner search should fan out fixed per-source limits and trim merged output."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_owners topic default size")

    mock_client = MagicMock()
    mock_client.search_owners.return_value = {
        "text": "红警",
        "mode": "topic",
        "owners": [
            {"mid": index, "name": f"作者{index}", "score": 100 - index}
            for index in range(25)
        ],
    }

    executor = ToolExecutor(search_client=mock_client, max_results=15)

    tc = ToolCall(
        id="call_test_owners_topic",
        name="search_owners",
        arguments=json.dumps({"text": "红警", "mode": "topic"}),
    )
    with patch.object(executor, "_is_google_available", return_value=False):
        result_msg = executor.execute(tc)
    result_data = json.loads(result_msg["content"])

    assert mock_client.search_owners.call_count == 3
    assert {
        call.kwargs["mode"] for call in mock_client.search_owners.call_args_list
    } == {
        "name",
        "topic",
        "relation",
    }
    assert {
        call.kwargs["size"] for call in mock_client.search_owners.call_args_list
    } == {8}
    assert result_data["mode"] == "aggregate"
    assert result_data["requested_mode"] == "topic"
    assert result_data["total_owners"] == 25
    assert len(result_data["owners"]) == 15

    logger.success("[PASS] execute search_owners topic default size")


def test_execute_search_owners_accepts_queries_alias_for_topic_lookup():
    """Creator-discovery style tool calls may send queries instead of text."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_owners queries alias")

    mock_client = MagicMock()
    mock_client.search_owners.return_value = {
        "text": "黑神话悟空",
        "mode": "topic",
        "owners": [
            {"mid": 1, "name": "作者1", "score": 100},
            {"mid": 2, "name": "作者2", "score": 80},
        ],
    }

    executor = ToolExecutor(search_client=mock_client, max_results=10)

    tc = ToolCall(
        id="call_test_owners_alias",
        name="search_owners",
        arguments=json.dumps({"queries": ["黑神话悟空 :view>=1w"]}),
    )
    with patch.object(executor, "_is_google_available", return_value=False):
        result_msg = executor.execute(tc)
    result_data = json.loads(result_msg["content"])

    assert mock_client.search_owners.call_count == 3
    assert {
        call.kwargs["mode"] for call in mock_client.search_owners.call_args_list
    } == {
        "name",
        "topic",
        "relation",
    }
    assert result_data["text"] == "黑神话悟空"
    assert result_data["mode"] == "aggregate"
    assert result_data["requested_mode"] == "topic"
    assert result_data["total_owners"] == 2

    logger.success("[PASS] execute search_owners queries alias")


def test_execute_search_owners_accepts_topic_alias():
    """Owner discovery calls may send topic instead of text."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_owners topic alias")

    mock_client = MagicMock()
    mock_client.search_owners.return_value = {
        "text": "黑神话悟空",
        "mode": "topic",
        "owners": [{"mid": 1, "name": "作者1", "score": 100}],
    }

    executor = ToolExecutor(search_client=mock_client, max_results=10)

    tc = ToolCall(
        id="call_test_owners_topic_alias",
        name="search_owners",
        arguments=json.dumps({"mode": "topic", "topic": "黑神话悟空"}),
    )
    with patch.object(executor, "_is_google_available", return_value=False):
        result_msg = executor.execute(tc)
    result_data = json.loads(result_msg["content"])

    assert mock_client.search_owners.call_count == 3
    assert {
        call.kwargs["mode"] for call in mock_client.search_owners.call_args_list
    } == {
        "name",
        "topic",
        "relation",
    }
    assert result_data["text"] == "黑神话悟空"
    assert result_data["mode"] == "aggregate"
    assert result_data["requested_mode"] == "topic"
    assert result_data["total_owners"] == 1

    logger.success("[PASS] execute search_owners topic alias")


def test_execute_search_owners_aggregates_related_tokens_and_google_space_results():
    """Aggregated owner search should merge related-token and Google space results."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_owners aggregates related tokens and google")

    mock_client = MagicMock()
    mock_client.search_owners.return_value = {
        "text": "黑神话悟空",
        "owners": [],
    }
    mock_client.related_owners_by_tokens.return_value = {
        "text": "黑神话悟空",
        "owners": [{"mid": 1, "name": "作者1", "score": 100}],
    }
    mock_google = MagicMock()
    mock_google.base_url = "http://mock-google:18100"
    mock_google.search.return_value = {
        "backend": "mock-google",
        "result_count": 2,
        "results": [
            {
                "title": "作者2的个人空间_哔哩哔哩_Bilibili",
                "link": "https://space.bilibili.com/2",
                "snippet": "作者2空间主页",
                "display_link": "space.bilibili.com",
            },
            {
                "title": "无关视频",
                "link": "https://www.bilibili.com/video/BV1abc",
                "snippet": "不是空间页",
                "display_link": "www.bilibili.com",
            },
        ],
    }

    executor = ToolExecutor(
        search_client=mock_client,
        google_client=mock_google,
        max_results=10,
    )

    tc = ToolCall(
        id="call_test_owners_aggregate",
        name="search_owners",
        arguments=json.dumps({"text": "黑神话悟空", "mode": "auto"}),
    )
    with patch.object(executor, "_is_google_available", return_value=True):
        result_msg = executor.execute(tc)
    result_data = json.loads(result_msg["content"])

    assert result_data["mode"] == "aggregate"
    assert result_data["requested_mode"] == "auto"
    assert result_data["total_owners"] == 2
    assert result_data["source_counts"]["related_tokens"] == 1
    assert result_data["source_counts"]["google_space"] == 1
    assert result_data["google_results"][0]["site_kind"] == "space"
    mock_client.related_owners_by_tokens.assert_called_once_with(
        text="黑神话悟空", size=8
    )
    mock_google.search.assert_called_once_with(
        query="黑神话悟空 site:space.bilibili.com",
        num=8,
        lang="zh-CN",
    )

    logger.success("[PASS] execute search_owners aggregates related tokens and google")


def test_execute_search_owners_survives_partial_source_failures():
    """One failing owner source should not break the aggregated response."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_owners survives partial source failures")

    mock_client = MagicMock()

    def flaky_search_owners(*, text: str, mode: str, size: int):
        assert text == "黑神话悟空"
        assert size == 8
        if mode == "name":
            raise RuntimeError("name source failed")
        return {"text": text, "mode": mode, "owners": []}

    mock_client.search_owners.side_effect = flaky_search_owners
    mock_client.related_owners_by_tokens.return_value = {
        "text": "黑神话悟空",
        "owners": [{"mid": 1, "name": "作者1", "score": 100}],
    }

    executor = ToolExecutor(search_client=mock_client, max_results=10)

    tc = ToolCall(
        id="call_test_owners_related_fallback",
        name="search_owners",
        arguments=json.dumps({"text": "黑神话悟空", "mode": "auto"}),
    )
    with patch.object(executor, "_is_google_available", return_value=False):
        result_msg = executor.execute(tc)
    result_data = json.loads(result_msg["content"])

    assert mock_client.search_owners.call_count == 3
    mock_client.related_owners_by_tokens.assert_called_once_with(
        text="黑神话悟空", size=8
    )
    assert result_data["mode"] == "aggregate"
    assert result_data["total_owners"] == 1
    name_group = next(
        group for group in result_data["source_groups"] if group["source"] == "name"
    )
    assert name_group["error"] == "name source failed"

    logger.success("[PASS] execute search_owners survives partial source failures")


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


def test_execute_search_google_formats_bilibili_site_results():
    """search_google should preserve useful bilibili site metadata for downstream reasoning."""
    logger.note("=" * 60)
    logger.note("[TEST] execute search_google formats bilibili site results")

    mock_client = MagicMock()
    mock_google = MagicMock()
    mock_google.base_url = "http://mock-google:18100"
    mock_google.search.return_value = {
        "backend": "mock-google",
        "result_count": 3,
        "results": [
            {
                "title": "Gemini CLI MCP 工作流实战",
                "link": "https://www.bilibili.com/video/BV1abc123xyz",
                "snippet": "MCP 工作流和 Gemini CLI 的实战视频。",
                "display_link": "www.bilibili.com",
            },
            {
                "title": "MCP 开发者主页",
                "link": "https://space.bilibili.com/12345678",
                "snippet": "长期做 MCP / Agent 内容。",
                "display_link": "space.bilibili.com",
            },
            {
                "title": "Gemini CLI 专栏",
                "link": "https://www.bilibili.com/read/cv24680",
                "snippet": "B 站专栏文章。",
                "display_link": "www.bilibili.com",
            },
        ],
    }

    executor = ToolExecutor(search_client=mock_client, google_client=mock_google)

    with patch.object(executor, "_is_google_available", return_value=True):
        tc = ToolCall(
            id="call_test_google",
            name="search_google",
            arguments=json.dumps(
                {
                    "query": "Gemini CLI MCP site:bilibili.com/video",
                    "num": 3,
                    "lang": "zh-CN",
                }
            ),
        )
        result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    assert result_data["query"] == "Gemini CLI MCP site:bilibili.com/video"
    assert result_data["backend"] == "mock-google"
    assert result_data["result_count"] == 3
    assert result_data["results"][0]["site_kind"] == "video"
    assert result_data["results"][0]["bvid"] == "BV1abc123xyz"
    assert result_data["results"][1]["site_kind"] == "space"
    assert result_data["results"][1]["mid"] == 12345678
    assert result_data["results"][2]["site_kind"] == "read"
    assert result_data["results"][2]["article_id"] == "cv24680"
    mock_google.search.assert_called_once_with(
        query="Gemini CLI MCP site:bilibili.com/video",
        num=3,
        lang="zh-CN",
    )

    logger.success("[PASS] execute search_google formats bilibili site results")


def test_execute_search_google_error_temporarily_disables_capability():
    mock_client = MagicMock()
    mock_google = MagicMock()
    mock_google.base_url = "http://mock-google:18100"
    mock_google.search.return_value = {
        "success": False,
        "error": "read timed out",
        "results": [],
    }

    executor = ToolExecutor(search_client=mock_client, google_client=mock_google)
    executor._google_available = True
    executor._google_available_ts = time.monotonic()

    tc = ToolCall(
        id="call_test_google_timeout",
        name="search_google",
        arguments=json.dumps({"query": "AI site:bilibili.com/video", "num": 3}),
    )
    result_msg = executor.execute(tc)

    result_data = json.loads(result_msg["content"])
    assert result_data["error"] == "read timed out"
    assert executor.get_search_capabilities()["supports_google_search"] is False


def test_google_search_client_retries_502_then_succeeds():
    logger.note("=" * 60)
    logger.note("[TEST] google search client retries transient 502")

    from llms.tools.executor import GoogleSearchClient

    client = GoogleSearchClient(base_url="http://mock-google:18100", timeout=3)
    bad_response = MagicMock()
    bad_response.status_code = 502
    good_response = MagicMock()
    good_response.raise_for_status.return_value = None
    good_response.json.return_value = {
        "backend": "mock-google",
        "result_count": 1,
        "results": [],
    }

    def fake_request(method, url, params=None, timeout=None):
        if fake_request.calls == 0:
            fake_request.calls += 1
            raise requests.HTTPError("502 Server Error", response=bad_response)
        return good_response

    fake_request.calls = 0

    with patch(
        "llms.tools.executor.requests.request", side_effect=fake_request
    ) as mock_request:
        result = client.search(query="site:bilibili.com test", num=5)

    assert result["backend"] == "mock-google"
    assert mock_request.call_count == 2
    logger.success("[PASS] google search client retries transient 502")


def test_create_google_search_client_supports_fallback_urls():
    from llms.tools.executor import create_google_search_client

    client = create_google_search_client(
        base_url="http://primary:18100, http://secondary:18100",
    )

    assert client is not None
    assert client.base_url == "http://primary:18100"
    assert client.base_urls == ["http://primary:18100", "http://secondary:18100"]


def test_create_google_search_client_reads_secrets_when_env_missing():
    from llms.tools.executor import create_google_search_client

    with (
        patch.dict(
            "os.environ",
            {
                "BILI_GOOGLE_HUB_BASE_URL": "",
                "BILI_GOOGLE_HUB_TIMEOUT": "",
            },
            clear=False,
        ),
        patch(
            "llms.tools.executor.GOOGLE_HUB_ENVS",
            {"endpoint": "http://127.0.0.1:18100", "timeout": 17},
        ),
    ):
        client = create_google_search_client()

    assert client is not None
    assert client.base_url == "http://127.0.0.1:18100"
    assert client.timeout == 17.0


def test_create_google_search_client_can_be_disabled_by_env():
    from llms.tools.executor import create_google_search_client

    with patch.dict("os.environ", {"BILI_GOOGLE_HUB_DISABLED": "1"}, clear=False):
        client = create_google_search_client(base_url="http://127.0.0.1:18100")

    assert client is None


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


def test_execute_expand_query_prefers_semantic_mode():
    logger.note("=" * 60)
    logger.note("[TEST] execute expand_query prefers semantic")

    mock_client = MagicMock()
    mock_client.capabilities.return_value = {
        "relation_endpoints": ["related_tokens_by_tokens"],
        "supports_google_search": False,
    }
    mock_client.related_tokens_by_tokens.return_value = {
        "text": "袁启 专访",
        "mode": "semantic",
        "options": [
            {
                "text": "袁启 采访",
                "doc_freq": 1,
                "score": 128.0,
                "type": "rewrite",
                "shard_count": 1,
            }
        ],
    }

    executor = ToolExecutor(search_client=mock_client)
    tc = ToolCall(
        id="call_expand_semantic",
        name="expand_query",
        arguments=json.dumps({"text": "袁启 专访"}),
    )

    result_msg = executor.execute(tc)
    result_data = json.loads(result_msg["content"])

    assert result_data["mode"] == "semantic"
    assert result_data["options"][0]["text"] == "袁启 采访"
    mock_client.related_tokens_by_tokens.assert_called_once_with(
        text="袁启 专访",
        mode="semantic",
        size=8,
    )

    logger.success("[PASS] execute expand_query prefers semantic")

    def test_execute_expand_query_keeps_text_without_alias_rules(self):
        logger.note("=" * 60)
        logger.note("[TEST] execute expand_query normalizes aliases")

        mock_client = MagicMock()
        mock_client.capabilities.return_value = {
            "relation_endpoints": ["related_tokens_by_tokens"],
            "supports_google_search": False,
        }
        mock_client.related_tokens_by_tokens.return_value = {
            "text": "康夫 UI 工作流",
            "mode": "semantic",
            "options": [
                {
                    "text": "comfyui教程工作流",
                    "doc_freq": 1,
                    "score": 256.0,
                    "type": "rewrite",
                    "shard_count": 1,
                }
            ],
        }

        executor = ToolExecutor(search_client=mock_client)
        tc = ToolCall(
            id="call_expand_alias",
            name="expand_query",
            arguments=json.dumps({"text": "康夫 UI 工作流"}),
        )

        result_msg = executor.execute(tc)
        result_data = json.loads(result_msg["content"])

        assert result_data["text"] == "康夫 UI 工作流"
        assert result_data["normalized_text"] == "康夫 UI 工作流"
        assert result_data["options"][0]["text"] == "comfyui教程工作流"
        mock_client.related_tokens_by_tokens.assert_called_once_with(
            text="康夫 UI 工作流",
            mode="semantic",
            size=8,
        )

        logger.success("[PASS] execute expand_query normalizes aliases")


def test_execute_expand_query_falls_back_to_auto_when_semantic_unsupported():
    logger.note("=" * 60)
    logger.note("[TEST] execute expand_query fallback to auto")

    mock_client = MagicMock()
    mock_client.capabilities.return_value = {
        "relation_endpoints": ["related_tokens_by_tokens"],
        "supports_google_search": False,
    }
    mock_client.related_tokens_by_tokens.side_effect = [
        {
            "text": "袁启 专访",
            "error": "mode must be 'prefix', 'associate', 'correction' or 'auto'; semantic unsupported",
            "options": [],
        },
        {
            "text": "袁启 专访",
            "mode": "auto",
            "options": [
                {
                    "text": "袁启 采访",
                    "doc_freq": 1,
                    "score": 96.0,
                    "type": "associate",
                    "shard_count": 1,
                }
            ],
        },
    ]

    executor = ToolExecutor(search_client=mock_client)
    tc = ToolCall(
        id="call_expand_fallback",
        name="expand_query",
        arguments=json.dumps({"text": "袁启 专访"}),
    )

    result_msg = executor.execute(tc)
    result_data = json.loads(result_msg["content"])

    assert result_data["mode"] == "auto"
    assert result_data["options"][0]["text"] == "袁启 采访"
    assert mock_client.related_tokens_by_tokens.call_count == 2
    assert mock_client.related_tokens_by_tokens.call_args_list[0].kwargs == {
        "text": "袁启 专访",
        "mode": "semantic",
        "size": 8,
    }
    assert mock_client.related_tokens_by_tokens.call_args_list[1].kwargs == {
        "text": "袁启 专访",
        "mode": "auto",
        "size": 8,
    }

    logger.success("[PASS] execute expand_query fallback to auto")


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
