"""Tests for llms.tools search service adapters."""

from unittest.mock import MagicMock, patch


@patch("llms.tools.executor.requests.post")
def test_search_service_client_explore(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"hits": [{"title": "黑神话"}]}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    from llms.tools.executor import SearchServiceClient

    client = SearchServiceClient("http://127.0.0.1:21031", timeout=5)
    result = client.explore("黑神话")

    assert result["hits"][0]["title"] == "黑神话"
    assert mock_post.call_args.args[0] == "http://127.0.0.1:21031/explore"


def test_create_search_service_prefers_http_client():
    from llms.tools.executor import SearchServiceClient, create_search_service

    service = create_search_service(base_url="http://127.0.0.1:21031")
    assert isinstance(service, SearchServiceClient)


@patch("llms.tools.executor.requests.get")
def test_search_service_client_capabilities(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "service_name": "Bili Search App",
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_author_check": True,
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    from llms.tools.executor import SearchServiceClient

    client = SearchServiceClient("http://127.0.0.1:21031", timeout=5)
    capabilities = client.capabilities()

    assert capabilities["service_name"] == "Bili Search App"
    assert capabilities["default_query_mode"] == "wv"
    assert mock_get.call_args.args[0] == "http://127.0.0.1:21031/capabilities"
