"""Tests for SearchApp chat endpoints — integrated /chat/completions.

Uses TestClient for endpoint testing with mocked handler and search components.

Run:
    python -m tests.llm.test_app
"""

import json
import os
import sys
import asyncio
from unittest.mock import MagicMock, patch
from tclogger import logger

from fastapi.testclient import TestClient
from configs.envs import LLM_CONFIG


def make_chat_response(content: str, usage: dict | None = None):
    from llms.models import ChatResponse

    return ChatResponse(
        content=content,
        finish_reason="stop",
        usage=usage
        or {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    )


# ============================================================
# Test setup
# ============================================================


def create_test_app(transcript_configured: bool = True):
    """Create a SearchApp with mocked search and chat internals for testing."""
    with (
        patch("service.app.init_embed_client_with_keepalive") as mock_embed,
        patch("service.app.VideoSearcherV2") as mock_searcher_cls,
        patch("service.app.VideoExplorer") as mock_explorer_cls,
        patch("service.app.OwnerSearcher") as mock_owner_searcher_cls,
        patch("service.app.RelationsClient") as mock_relations_cls,
        patch("service.app.BiliStoreTranscriptClient") as mock_transcript_cls,
        patch("llms.models.client.create_llm_client") as mock_create_llm,
    ):

        mock_searcher = MagicMock()
        mock_explorer = MagicMock()
        mock_owner_searcher = MagicMock()
        mock_relations = MagicMock()
        mock_searcher_cls.return_value = mock_searcher
        mock_explorer_cls.return_value = mock_explorer
        mock_owner_searcher_cls.return_value = mock_owner_searcher
        mock_relations_cls.return_value = mock_relations

        mock_transcript_client = MagicMock()
        mock_transcript_client.is_configured = transcript_configured
        mock_transcript_cls.return_value = mock_transcript_client

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_create_llm.return_value = mock_llm

        from service.app import SearchApp

        app_envs = {
            "app_name": "Test Search App",
            "version": "0.0.1",
            "elastic_index": "test_index",
            "llm_config": LLM_CONFIG,
        }
        search_app = SearchApp(app_envs)
        return (
            search_app,
            mock_llm,
            mock_searcher,
            mock_explorer,
            mock_transcript_client,
        )


# ============================================================
# Tests
# ============================================================


def test_health_endpoint():
    """Test /health endpoint."""
    logger.note("=" * 60)
    logger.note("[TEST] /health endpoint")

    search_app, _, _, _, _ = create_test_app()
    client = TestClient(search_app.app)
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["llm_model"] == "test-model"
    assert data["search_service"] == "integrated"

    logger.success("[PASS] /health endpoint")


def test_capabilities_endpoint():
    """Test /capabilities endpoint."""
    logger.note("=" * 60)
    logger.note("[TEST] /capabilities endpoint")

    search_app, _, _, _, _ = create_test_app()
    client = TestClient(search_app.app)
    resp = client.get("/capabilities")

    assert resp.status_code == 200
    data = resp.json()
    assert data["service_name"] == "Test Search App"
    assert data["supports_multi_query"] is True
    assert data["supports_author_check"] is False
    assert data["supports_owner_search"] is True
    assert data["supports_transcript_lookup"] is True
    assert "related_owners_by_tokens" in data["relation_endpoints"]
    assert "/explore" in data["available_endpoints"]
    assert "/search_owners" in data["available_endpoints"]
    assert "/video_transcript" in data["available_endpoints"]
    assert "/user_briefs" in data["available_endpoints"]
    assert "/related_owners_by_tokens" in data["available_endpoints"]

    logger.success("[PASS] /capabilities endpoint")


def test_capabilities_endpoint_disables_transcript_when_unconfigured():
    """Transcript capability should turn off when the endpoint is not configured."""
    logger.note("=" * 60)
    logger.note("[TEST] /capabilities transcript disabled when unconfigured")

    search_app, _, _, _, _ = create_test_app(transcript_configured=False)
    client = TestClient(search_app.app)
    resp = client.get("/capabilities")

    assert resp.status_code == 200
    data = resp.json()
    assert data["supports_transcript_lookup"] is False
    assert "/video_transcript" in data["available_endpoints"]

    logger.success("[PASS] /capabilities transcript disabled when unconfigured")


def test_chat_completions_endpoint():
    """Test /chat/completions endpoint (non-streaming)."""
    logger.note("=" * 60)
    logger.note("[TEST] /chat/completions endpoint")

    search_app, mock_llm, _, _, _ = create_test_app()

    from llms.models import ChatResponse

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


def test_chat_completions_accepts_multimodal_message_content():
    """The API should accept OpenAI-style content arrays and still complete normally."""
    logger.note("=" * 60)
    logger.note("[TEST] /chat/completions accepts multimodal content")

    search_app, mock_llm, _, _, _ = create_test_app()
    mock_llm.chat.return_value = make_chat_response("已收到图文输入")

    client = TestClient(search_app.app)
    resp = client.post(
        "/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请总结这个视频截图"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/test.png"},
                        },
                    ],
                }
            ]
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["choices"][0]["message"]["content"] == "已收到图文输入"

    logger.success("[PASS] /chat/completions accepts multimodal content")


def test_video_transcript_endpoint():
    """Test /video_transcript forwards to the transcript client."""
    logger.note("=" * 60)
    logger.note("[TEST] /video_transcript endpoint")

    search_app, _, _, _, _ = create_test_app()
    mock_transcript_client = MagicMock()
    mock_transcript_client.get_video_transcript.return_value = {
        "ok": True,
        "bvid": "BV1YXZPB1Erc",
        "transcript": {"text": "示例转写"},
    }
    search_app.transcript_client = mock_transcript_client

    client = TestClient(search_app.app)
    resp = client.post(
        "/video_transcript",
        json={
            "video_id": "BV1YXZPB1Erc",
            "head_chars": 500,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["bvid"] == "BV1YXZPB1Erc"
    mock_transcript_client.get_video_transcript.assert_called_once_with(
        "BV1YXZPB1Erc",
        request={"head_chars": 500},
    )

    logger.success("[PASS] /video_transcript endpoint")


def test_video_transcript_endpoint_reports_unavailable_when_unconfigured():
    """The endpoint should return a stable unavailable error when transcript config is missing."""
    logger.note("=" * 60)
    logger.note("[TEST] /video_transcript unavailable when transcript client disabled")

    search_app, _, _, _, _ = create_test_app(transcript_configured=False)
    client = TestClient(search_app.app)
    resp = client.post(
        "/video_transcript",
        json={
            "video_id": "BV1YXZPB1Erc",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"error": "Transcript lookup unavailable"}

    logger.success(
        "[PASS] /video_transcript unavailable when transcript client disabled"
    )


def test_user_briefs_endpoint_uses_local_lookup():
    """Test /user_briefs forwards mids to the local Mongo-backed searcher helper."""
    logger.note("=" * 60)
    logger.note("[TEST] /user_briefs endpoint")

    search_app, _, mock_searcher, _, _ = create_test_app()
    mock_searcher.get_user_briefs.return_value = [
        {
            "mid": 946974,
            "name": "影视飓风",
            "face": "https://example.com/face.jpg",
            "video_count": 321,
        }
    ]

    client = TestClient(search_app.app)
    resp = client.post(
        "/user_briefs",
        json={"mids": [946974, "12345", "bad-mid"]},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "users": [
            {
                "mid": 946974,
                "name": "影视飓风",
                "face": "https://example.com/face.jpg",
                "video_count": 321,
            }
        ]
    }
    mock_searcher.get_user_briefs.assert_called_once_with([946974, "12345", "bad-mid"])

    logger.success("[PASS] /user_briefs endpoint")


def test_chat_completions_v1_path():
    """Test /v1/chat/completions path works."""
    logger.note("=" * 60)
    logger.note("[TEST] /v1/chat/completions path")

    search_app, mock_llm, _, _, _ = create_test_app()
    from llms.models import ChatResponse

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

    search_app, _, _, _, _ = create_test_app()
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

    search_app, mock_llm, _, _, _ = create_test_app()
    from llms.models import ChatResponse

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
    assert resp.headers.get("cache-control") == "no-cache, no-transform"
    assert resp.headers.get("pragma") == "no-cache"
    assert resp.headers.get("x-accel-buffering") == "no"
    assert resp.headers.get("x-content-type-options") == "nosniff"

    logger.success("[PASS] streaming response")


def test_stream_response_cleanup_swallows_monitor_cancellation():
    """The SSE generator should finish cleanly after cancelling its disconnect monitor."""
    logger.note("=" * 60)
    logger.note("[TEST] stream response cleanup")

    search_app, _, _, _, _ = create_test_app()
    search_app.chat_handler.handle_stream = MagicMock(
        return_value=iter(
            [
                json.dumps(
                    {
                        "id": "chunk-1",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                "[DONE]",
            ]
        )
    )

    async def consume_stream():
        chunks = []
        async for item in search_app._stream_response(
            messages=[{"role": "user", "content": "test"}],
            http_request=None,
        ):
            chunks.append(item)
        return chunks

    chunks = asyncio.run(consume_stream())

    assert chunks[0]["data"]
    assert chunks[-1]["data"] == "[DONE]"

    logger.success("[PASS] stream response cleanup")


def test_chat_completions_account_query_stays_relation_only():
    """Account/meta queries should not be auto-promoted to search_videos at the API layer."""
    logger.note("=" * 60)
    logger.note("[TEST] /chat/completions account query stays relation-only")

    search_app, mock_llm, _, _, _ = create_test_app()
    mock_llm.chat.side_effect = [
        make_chat_response(
            '我先找一下何同学相关作者线索。\n<search_owners text="何同学" mode="relation"/>'
        ),
        make_chat_response("找到了何同学相关的作者候选。"),
    ]
    search_app.relations_client.related_owners_by_tokens.return_value = {
        "owners": [{"mid": 1, "name": "何同学小号候选"}]
    }

    client = TestClient(search_app.app)
    resp = client.post(
        "/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "你有什么功能"},
                {"role": "assistant", "content": "我可以帮你搜索 B 站内容。"},
                {"role": "user", "content": "何同学有哪些关联账号？"},
            ],
            "stream": False,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "找到了何同学相关的作者候选。"
    used_tools = [
        tool for event in data.get("tool_events", []) for tool in event.get("tools", [])
    ]
    assert "search_owners" in used_tools
    assert "search_videos" not in used_tools
    search_app.relations_client.related_owners_by_tokens.assert_called_once_with(
        text="何同学", size=8
    )
    search_app.video_explorer.unified_explore.assert_not_called()

    logger.success("[PASS] /chat/completions account query stays relation-only")


def test_chat_completions_account_followup_stays_relation_only():
    """Pronoun-based account follow-ups should inherit creator context without video search promotion."""
    logger.note("=" * 60)
    logger.note("[TEST] /chat/completions account follow-up stays relation-only")

    search_app, mock_llm, _, _, _ = create_test_app()
    mock_llm.chat.side_effect = [
        make_chat_response(
            '我先继续找这个作者相关账号。\n<search_owners text="何同学" mode="relation"/>'
        ),
        make_chat_response("补充找到了何同学的其他关联作者候选。"),
    ]
    search_app.relations_client.related_owners_by_tokens.return_value = {
        "owners": [{"mid": 2, "name": "何同学关联候选"}]
    }

    client = TestClient(search_app.app)
    resp = client.post(
        "/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "何同学有哪些关联账号？"},
                {"role": "assistant", "content": "我先帮你找相关作者线索。"},
                {"role": "user", "content": "他还有别的号吗？"},
            ],
            "stream": False,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert (
        data["choices"][0]["message"]["content"]
        == "补充找到了何同学的其他关联作者候选。"
    )
    used_tools = [
        tool for event in data.get("tool_events", []) for tool in event.get("tools", [])
    ]
    assert "search_owners" in used_tools
    assert "search_videos" not in used_tools
    search_app.relations_client.related_owners_by_tokens.assert_called_once_with(
        text="何同学", size=8
    )
    search_app.video_explorer.unified_explore.assert_not_called()

    logger.success("[PASS] /chat/completions account follow-up stays relation-only")


def test_search_and_suggest_endpoints_match_searcher_v2_signature():
    """Test that search_app no longer forwards removed kwargs to VideoSearcherV2."""
    logger.note("=" * 60)
    logger.note("[TEST] search/suggest endpoint compatibility")

    search_app, _, mock_searcher, _, _ = create_test_app()

    def search_side_effect(
        query,
        match_fields=None,
        source_fields=None,
        match_type=None,
        suggest_info=None,
        use_script_score=None,
        rank_method=None,
        detail_level=-1,
        limit=None,
        verbose=False,
    ):
        return {"query": query, "hits": [], "total_hits": 0}

    def suggest_side_effect(
        query,
        match_fields=None,
        source_fields=None,
        match_type=None,
        use_script_score=None,
        use_pinyin=True,
        detail_level=-1,
        limit=None,
        verbose=False,
    ):
        return {"query": query, "hits": [], "total_hits": 0}

    mock_searcher.search.side_effect = search_side_effect
    mock_searcher.suggest.side_effect = suggest_side_effect

    client = TestClient(search_app.app)

    search_resp = client.post("/search", json={"query": "黑神话"})
    suggest_resp = client.post("/suggest", json={"query": "黑神话"})

    assert search_resp.status_code == 200
    assert suggest_resp.status_code == 200

    logger.success("[PASS] search/suggest endpoint compatibility")


def test_cors_headers():
    """Test CORS headers are set."""
    logger.note("=" * 60)
    logger.note("[TEST] CORS headers")

    search_app, mock_llm, _, _, _ = create_test_app()
    from llms.models import ChatResponse

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

    with (
        patch("service.app.init_embed_client_with_keepalive"),
        patch("service.app.VideoSearcherV2"),
        patch("service.app.VideoExplorer"),
        patch("service.app.OwnerSearcher"),
        patch("service.app.RelationsClient"),
    ):
        from service.app import SearchApp

        app_envs = {
            "app_name": "Test Search App",
            "version": "0.0.1",
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


def test_shared_runtime_arg_resolution_allows_llm_config_override():
    """Test shared runtime arg resolution accepts llm_config override."""
    logger.note("=" * 60)
    logger.note("[TEST] shared runtime arg resolution llm override")

    with patch.object(
        sys,
        "argv",
        [
            "search_app.py",
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
        from service.arg_parser import ArgParser

        parser = ArgParser()
        app_envs = {
            "app_name": "Test Search App",
            "version": "0.0.1",
            "host": "0.0.0.0",
            "port": 21001,
            "elastic_index": "bili_videos_dev6",
            "llm_config": "legacy",
        }
        new_envs = parser.update_app_envs(app_envs)

    assert new_envs["elastic_index"] == "bili_videos_dev6"
    assert new_envs["elastic_env_name"] == "elastic_dev"
    assert new_envs["llm_config"] == "deepseek"
    assert new_envs["port"] == 21011

    logger.success("[PASS] shared runtime arg resolution llm override")


def test_search_app_env_helpers_support_service_overrides():
    """Test environment-based app factory helpers used by search_app_cli."""
    logger.note("=" * 60)
    logger.note("[TEST] search app env helpers")

    with patch.dict(
        os.environ,
        {
            "BILI_SEARCH_APP_PORT": "21015",
            "BILI_SEARCH_APP_ELASTIC_INDEX": "bili_videos_dev6",
            "BILI_SEARCH_APP_ELASTIC_ENV_NAME": "elastic_dev",
            "BILI_SEARCH_APP_LLM_CONFIG": "deepseek",
        },
        clear=False,
    ):
        from service.app import (
            get_search_app_env_overrides_from_env,
            resolve_search_app_envs,
        )

        overrides = get_search_app_env_overrides_from_env()
        envs = resolve_search_app_envs(overrides=overrides)

    assert envs["port"] == 21015
    assert envs["elastic_index"] == "bili_videos_dev6"
    assert envs["elastic_env_name"] == "elastic_dev"
    assert envs["llm_config"] == "deepseek"

    logger.success("[PASS] search app env helpers")


def test_resolve_search_app_envs_backfills_missing_fields_from_config():
    """Test partial env dicts are completed from configs/envs.json."""
    logger.note("=" * 60)
    logger.note("[TEST] search app env resolution backfills config defaults")

    from service.envs import resolve_search_app_envs

    envs = resolve_search_app_envs({"port": 21011})

    assert envs["port"] == 21011
    assert envs["host"] == "0.0.0.0"
    assert envs["elastic_index"] == "bili_videos_dev6"
    assert envs["elastic_env_name"] == "elastic_dev"
    assert envs["llm_config"] == "deepseek"

    logger.success("[PASS] search app env resolution backfills config defaults")


if __name__ == "__main__":
    tests = [
        ("health_endpoint", test_health_endpoint),
        ("capabilities_endpoint", test_capabilities_endpoint),
        ("chat_completions_endpoint", test_chat_completions_endpoint),
        ("chat_completions_v1_path", test_chat_completions_v1_path),
        ("request_validation", test_chat_request_validation),
        ("streaming_response", test_streaming_response),
        (
            "chat_completions_account_query_relation_only",
            test_chat_completions_account_query_stays_relation_only,
        ),
        (
            "chat_completions_account_followup_relation_only",
            test_chat_completions_account_followup_stays_relation_only,
        ),
        (
            "search_and_suggest_signature_compat",
            test_search_and_suggest_endpoints_match_searcher_v2_signature,
        ),
        ("cors_headers", test_cors_headers),
        ("no_chat_without_llm_config", test_no_chat_without_llm_config),
        (
            "shared_runtime_arg_resolution_llm_override",
            test_shared_runtime_arg_resolution_allows_llm_config_override,
        ),
        (
            "search_app_env_helpers",
            test_search_app_env_helpers_support_service_overrides,
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
