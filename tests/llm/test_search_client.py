"""Tests for llms.search_service — HTTP client for Search App service.

Uses mocked HTTP responses for unit tests.

Run:
    python -m tests.llm.test_search_client
"""

import json
from unittest.mock import patch, MagicMock
from tclogger import logger

from llms.search_service import SearchServiceClient


# ============================================================
# Mock helpers
# ============================================================


def make_mock_response(data: dict, status_code: int = 200):
    """Create a mock requests.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests

        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code}"
        )
    return mock_resp


# ============================================================
# Tests
# ============================================================


@patch("llms.search_service.requests.post")
def test_explore(mock_post):
    """Test explore() sends correct request."""
    logger.note("=" * 60)
    logger.note("[TEST] explore()")

    mock_data = {
        "query": "黑神话",
        "status": "finished",
        "data": [
            {
                "step": 0,
                "name": "most_relevant_search",
                "output": {
                    "hits": [{"bvid": "BV1a", "title": "Test"}],
                    "total_hits": 1,
                },
            }
        ],
    }
    mock_post.return_value = make_mock_response(mock_data)

    client = SearchServiceClient("http://localhost:20001")
    result = client.explore("黑神话")

    assert result["query"] == "黑神话"
    assert result["status"] == "finished"

    # Verify request
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert "/explore" in call_args[0][0] or "/explore" in str(call_args)

    logger.success("[PASS] explore()")


@patch("llms.search_service.requests.post")
def test_explore_with_qmod(mock_post):
    """Test explore() with qmod parameter."""
    logger.note("=" * 60)
    logger.note("[TEST] explore() with qmod")

    mock_post.return_value = make_mock_response({"data": []})

    client = SearchServiceClient("http://localhost:20001")
    client.explore("黑神话", qmod="wv")

    call_args = mock_post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert payload["qmod"] == "wv"

    logger.success("[PASS] explore() with qmod")


@patch("llms.search_service.requests.post")
def test_search(mock_post):
    """Test search() sends correct request."""
    logger.note("=" * 60)
    logger.note("[TEST] search()")

    mock_data = {
        "query": "Python教程",
        "total_hits": 100,
        "hits": [{"bvid": "BV1x", "title": "Python入门"}],
    }
    mock_post.return_value = make_mock_response(mock_data)

    client = SearchServiceClient("http://localhost:20001")
    result = client.search("Python教程", limit=10)

    assert result["query"] == "Python教程"
    assert result["total_hits"] == 100

    call_args = mock_post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert payload["limit"] == 10

    logger.success("[PASS] search()")


@patch("llms.search_service.requests.post")
def test_suggest(mock_post):
    """Test suggest() sends correct request."""
    logger.note("=" * 60)
    logger.note("[TEST] suggest()")

    mock_data = {
        "query": "影视飓风",
        "total_hits": 25,
        "hits": [
            {
                "bvid": "BV1a",
                "title": "影视飓风测评",
                "owner": {"mid": 946974, "name": "影视飓风"},
            }
        ],
    }
    mock_post.return_value = make_mock_response(mock_data)

    client = SearchServiceClient("http://localhost:20001")
    result = client.suggest("影视飓风")

    assert result["query"] == "影视飓风"
    assert len(result["hits"]) == 1

    logger.success("[PASS] suggest()")


@patch("llms.search_service.requests.post")
def test_connection_error(mock_post):
    """Test graceful handling of connection errors."""
    logger.note("=" * 60)
    logger.note("[TEST] connection error handling")

    import requests

    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    client = SearchServiceClient("http://localhost:99999")
    result = client.explore("test")

    assert "error" in result
    assert result["hits"] == []
    assert result["total_hits"] == 0

    logger.success("[PASS] connection error handling")


@patch("llms.search_service.requests.post")
def test_timeout_error(mock_post):
    """Test graceful handling of timeout errors."""
    logger.note("=" * 60)
    logger.note("[TEST] timeout error handling")

    import requests

    mock_post.side_effect = requests.exceptions.Timeout("Timed out")

    client = SearchServiceClient("http://localhost:20001", timeout=1)
    result = client.suggest("test")

    assert "error" in result
    assert result["hits"] == []

    logger.success("[PASS] timeout error handling")


@patch("llms.search_service.requests.get")
def test_is_available(mock_get):
    """Test is_available() health check."""
    logger.note("=" * 60)
    logger.note("[TEST] is_available()")

    # Available
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp

    client = SearchServiceClient("http://localhost:20001")
    assert client.is_available() is True

    # Not available
    import requests

    mock_get.side_effect = requests.exceptions.ConnectionError()
    assert client.is_available() is False

    logger.success("[PASS] is_available()")


def test_url_construction():
    """Test URL construction with base_url."""
    logger.note("=" * 60)
    logger.note("[TEST] URL construction")

    # With trailing slash
    client1 = SearchServiceClient("http://localhost:20001/")
    assert client1.base_url == "http://localhost:20001"

    # Without trailing slash
    client2 = SearchServiceClient("http://localhost:20001")
    assert client2.base_url == "http://localhost:20001"

    logger.success("[PASS] URL construction")


if __name__ == "__main__":
    tests = [
        ("explore", test_explore),
        ("explore_with_qmod", test_explore_with_qmod),
        ("search", test_search),
        ("suggest", test_suggest),
        ("connection_error", test_connection_error),
        ("timeout_error", test_timeout_error),
        ("is_available", test_is_available),
        ("url_construction", test_url_construction),
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

    # python -m tests.llm.test_search_client
