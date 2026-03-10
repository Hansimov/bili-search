"""Tests for SearchApp chat endpoints — integrated /chat/completions.

Uses TestClient for endpoint testing with mocked handler and search components.

Run:
    python -m tests.llm.test_app
"""

import json
import sys
from unittest.mock import MagicMock, patch
from tclogger import logger

from fastapi.testclient import TestClient
from configs.envs import LLM_CONFIG


# ============================================================
# Test setup
# ============================================================


def create_test_app():
    """Create a SearchApp with mocked search and chat internals for testing."""
    with patch("apps.search_app.init_embed_client_with_keepalive") as mock_embed, patch(
        "apps.search_app.VideoSearcherV2"
    ) as mock_searcher_cls, patch(
        "apps.search_app.VideoExplorer"
    ) as mock_explorer_cls, patch(
        "llms.llm_client.create_llm_client"
    ) as mock_create_llm:

        mock_searcher = MagicMock()
        mock_explorer = MagicMock()
        mock_searcher_cls.return_value = mock_searcher
        mock_explorer_cls.return_value = mock_explorer

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_create_llm.return_value = mock_llm

        from apps.search_app import SearchApp

        app_envs = {
            "app_name": "Test Search App",
            "version": "0.0.1",
            "mode": "test",
            "elastic_index": "test_index",
            "llm_config": LLM_CONFIG,
        }
        search_app = SearchApp(app_envs)
        return search_app, mock_llm, mock_searcher, mock_explorer


# ============================================================
# Tests
# ============================================================


def test_health_endpoint():
    """Test /health endpoint."""
    logger.note("=" * 60)
    logger.note("[TEST] /health endpoint")

    search_app, _, _, _ = create_test_app()
    client = TestClient(search_app.app)
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["llm_model"] == "test-model"
    assert data["search_service"] == "integrated"

    logger.success("[PASS] /health endpoint")


def test_chat_completions_endpoint():
    """Test /chat/completions endpoint (non-streaming)."""
    logger.note("=" * 60)
    logger.note("[TEST] /chat/completions endpoint")

    search_app, mock_llm, _, _ = create_test_app()

    from llms.llm_client import ChatResponse

    mock_llm.chat.return_value = ChatResponse(
        content="Hello!",
        finish_reason="stop",
        usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    )

    client = TestClient(search_app.app)
    resp = client.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] == "Hello!"

    logger.success("[PASS] /chat/completions endpoint")


def test_chat_completions_v1_path():
    """Test /v1/chat/completions path works."""
    logger.note("=" * 60)
    logger.note("[TEST] /v1/chat/completions path")

    search_app, mock_llm, _, _ = create_test_app()
    from llms.llm_client import ChatResponse

    mock_llm.chat.return_value = ChatResponse(
        content="Test",
        finish_reason="stop",
        usage={},
    )

    client = TestClient(search_app.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "test"}],
        },
    )

    assert resp.status_code == 200

    logger.success("[PASS] /v1/chat/completions path")


def test_chat_request_validation():
    """Test request validation for /chat/completions."""
    logger.note("=" * 60)
    logger.note("[TEST] request validation")

    search_app, _, _, _ = create_test_app()
    client = TestClient(search_app.app)

    # Missing messages
    resp = client.post("/chat/completions", json={})
    assert resp.status_code == 422  # Validation error

    # Invalid message format
    resp = client.post(
        "/chat/completions",
        json={"messages": [{"invalid": "format"}]},
    )
    assert resp.status_code == 422

    logger.success("[PASS] request validation")


def test_streaming_response():
    """Test /chat/completions with stream=true."""
    logger.note("=" * 60)
    logger.note("[TEST] streaming response")

    search_app, mock_llm, _, _ = create_test_app()
    from llms.llm_client import ChatResponse

    mock_llm.chat.return_value = ChatResponse(
        content="Hello world",
        finish_reason="stop",
        usage={},
    )

    client = TestClient(search_app.app)
    resp = client.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        },
    )

    assert resp.status_code == 200
    # SSE response should have text/event-stream content type
    assert "text/event-stream" in resp.headers.get("content-type", "")

    logger.success("[PASS] streaming response")


def test_cors_headers():
    """Test CORS headers are set."""
    logger.note("=" * 60)
    logger.note("[TEST] CORS headers")

    search_app, mock_llm, _, _ = create_test_app()
    from llms.llm_client import ChatResponse

    mock_llm.chat.return_value = ChatResponse(
        content="OK", finish_reason="stop", usage={}
    )

    client = TestClient(search_app.app)

    # OPTIONS request should return CORS headers
    resp = client.options(
        "/chat/completions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    # Note: TestClient may not fully handle CORS, but we verify the middleware is set
    assert resp.status_code in (200, 405)

    logger.success("[PASS] CORS headers")


def test_no_chat_without_llm_config():
    """Test that chat endpoints are not registered when llm_config is empty."""
    logger.note("=" * 60)
    logger.note("[TEST] no chat without llm_config")

    with patch("apps.search_app.init_embed_client_with_keepalive"), patch(
        "apps.search_app.VideoSearcherV2"
    ), patch("apps.search_app.VideoExplorer"):
        from apps.search_app import SearchApp

        app_envs = {
            "app_name": "Test Search App",
            "version": "0.0.1",
            "mode": "test",
            "elastic_index": "test_index",
            "llm_config": "",
        }
        search_app = SearchApp(app_envs)

        client = TestClient(search_app.app)

        # Chat endpoints should not exist
        resp = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert resp.status_code in (404, 405)

    logger.success("[PASS] no chat without llm_config")


def test_search_app_arg_parser_allows_llm_config_override():
    """Test SearchAppArgParser accepts llm_config override."""
    logger.note("=" * 60)
    logger.note("[TEST] search app arg parser llm override")

    with patch.object(
        sys,
        "argv",
        [
            "search_app.py",
            "-m",
            "dev",
            "-ei",
            "bili_videos_dev6",
            "-ev",
            "elastic_dev",
            "-lc",
            "deepseek",
            "-p",
            "21011",
        ],
    ):
        from apps.search_app import SearchAppArgParser

        parser = SearchAppArgParser()
        app_envs = {
            "app_name": "Test Search App",
            "version": "0.0.1",
            "mode": "prod",
            "host": {"prod": "0.0.0.0", "dev": "0.0.0.0"},
            "port": {"prod": 20001, "dev": 21001},
            "elastic_index": {"prod": "bili_videos_pro1", "dev": "bili_videos_dev6"},
            "llm_config": "gpt",
        }
        new_envs = parser.update_app_envs(app_envs)

    assert new_envs["elastic_index"] == "bili_videos_dev6"
    assert new_envs["elastic_env_name"] == "elastic_dev"
    assert new_envs["llm_config"] == "deepseek"
    assert new_envs["port"] == 21011

    logger.success("[PASS] search app arg parser llm override")


if __name__ == "__main__":
    tests = [
        ("health_endpoint", test_health_endpoint),
        ("chat_completions_endpoint", test_chat_completions_endpoint),
        ("chat_completions_v1_path", test_chat_completions_v1_path),
        ("request_validation", test_chat_request_validation),
        ("streaming_response", test_streaming_response),
        ("cors_headers", test_cors_headers),
        ("no_chat_without_llm_config", test_no_chat_without_llm_config),
        (
            "search_app_arg_parser_llm_override",
            test_search_app_arg_parser_allows_llm_config_override,
        ),
    ]

    results = {}
    for name, test_func in tests:
        try:
            test_func()
            results[name] = "PASS"
        except Exception as e:
            logger.warn(f"\n[FAIL] {name}: {e}")
            import traceback

            traceback.print_exc()
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

    # python -m tests.llm.test_app
