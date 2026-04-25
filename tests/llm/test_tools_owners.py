"""Owner-search tests for llms.tools executor."""

import json
from unittest.mock import MagicMock, patch

from tclogger import logger

from llms.models import ToolCall
from llms.tools.executor import ToolExecutor
from test_tools import MOCK_RELATED_OWNERS_RESULT, MOCK_SEARCH_OWNERS_RESULT


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
