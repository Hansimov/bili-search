"""Google-search tests for llms.tools executor."""

import json
import requests
import time
from unittest.mock import MagicMock, patch

from tclogger import logger

from llms.models import ToolCall
from llms.tools.executor import ToolExecutor


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
