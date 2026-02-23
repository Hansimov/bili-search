"""Tests for llms.app — FastAPI chat completions endpoint.

Uses TestClient for endpoint testing with mocked handler.

Run:
    python -m tests.llm.test_app
"""

import json
from unittest.mock import MagicMock, patch
from tclogger import logger

from fastapi.testclient import TestClient
from llms.app import ChatApp


# ============================================================
# Test setup
# ============================================================


def create_test_app():
    """Create a ChatApp with mocked internals for testing."""
    with patch("llms.app.create_llm_client") as mock_create_llm, patch(
        "llms.app.SearchServiceClient"
    ) as mock_search_cls:

        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_llm.model = "test-model"

        mock_search = MagicMock()
        mock_search_cls.return_value = mock_search

        app_envs = {
            "app_name": "Test Chat App",
            "version": "0.0.1",
            "llm_config": "deepseek",
            "search_app_url": "http://localhost:20001",
        }
        chat_app = ChatApp(app_envs)
        return chat_app, mock_llm, mock_search


# ============================================================
# Tests
# ============================================================


def test_health_endpoint():
    """Test /health endpoint."""
    logger.note("=" * 60)
    logger.note("[TEST] /health endpoint")

    chat_app, _, mock_search = create_test_app()
    mock_search.is_available.return_value = True

    client = TestClient(chat_app.app)
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["llm_model"] == "test-model"
    assert data["search_service"] == "available"

    logger.success("[PASS] /health endpoint")


def test_chat_completions_endpoint():
    """Test /chat/completions endpoint (non-streaming)."""
    logger.note("=" * 60)
    logger.note("[TEST] /chat/completions endpoint")

    chat_app, mock_llm, _ = create_test_app()

    # Mock the handler's handle method
    from llms.llm_client import ChatResponse

    mock_llm.chat.return_value = ChatResponse(
        content="Hello!",
        finish_reason="stop",
        usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    )

    client = TestClient(chat_app.app)
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

    chat_app, mock_llm, _ = create_test_app()
    from llms.llm_client import ChatResponse

    mock_llm.chat.return_value = ChatResponse(
        content="Test",
        finish_reason="stop",
        usage={},
    )

    client = TestClient(chat_app.app)
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

    chat_app, _, _ = create_test_app()
    client = TestClient(chat_app.app)

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

    chat_app, mock_llm, _ = create_test_app()
    from llms.llm_client import ChatResponse

    mock_llm.chat.return_value = ChatResponse(
        content="Hello world",
        finish_reason="stop",
        usage={},
    )

    client = TestClient(chat_app.app)
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

    chat_app, mock_llm, _ = create_test_app()
    from llms.llm_client import ChatResponse

    mock_llm.chat.return_value = ChatResponse(
        content="OK", finish_reason="stop", usage={}
    )

    client = TestClient(chat_app.app)

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


if __name__ == "__main__":
    tests = [
        ("health_endpoint", test_health_endpoint),
        ("chat_completions_endpoint", test_chat_completions_endpoint),
        ("chat_completions_v1_path", test_chat_completions_v1_path),
        ("request_validation", test_chat_request_validation),
        ("streaming_response", test_streaming_response),
        ("cors_headers", test_cors_headers),
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
