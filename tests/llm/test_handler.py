"""Tests for llms.chat.handler — chat handler with tool-calling loop.

Uses mocked LLM client and search client for unit tests.

Run:
    python -m tests.llm.test_handler
"""

import json
from unittest.mock import MagicMock, patch, call
from tclogger import logger
import threading

from llms.llm_client import LLMClient, ChatResponse
from llms.chat.handler import ChatHandler, _FORCE_CONTENT_NUDGE


# ============================================================
# Mock helpers
# ============================================================


def make_content_response(content: str, extra_usage: dict = None) -> ChatResponse:
    """Build a ChatResponse with content."""
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    if extra_usage:
        usage.update(extra_usage)
    return ChatResponse(
        content=content,
        finish_reason="stop",
        usage=usage,
    )


def make_stream_chunks(
    content: str, extra_usage: dict = None, reasoning: str = None
) -> list[dict]:
    """Build streaming chunks that accumulate to the given content.

    Used for mocking chat_stream() when _chat_interruptible() consumes streaming.
    """
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    if extra_usage:
        usage.update(extra_usage)
    chunks = []
    if reasoning:
        chunks.append(
            {
                "choices": [
                    {"delta": {"reasoning_content": reasoning}, "finish_reason": None}
                ]
            }
        )
    chunks.append({"choices": [{"delta": {"content": content}, "finish_reason": None}]})
    chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage})
    return chunks


def make_tool_cmd_response(
    analysis: str,
    commands: str,
    extra_usage: dict = None,
) -> ChatResponse:
    """Build a ChatResponse with analysis text + inline XML tool commands.

    Args:
        analysis: Short analysis text before commands.
        commands: XML tool command string(s).
        extra_usage: Additional usage fields to merge.
    """
    content = f"{analysis}\n{commands}" if analysis else commands
    usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    if extra_usage:
        usage.update(extra_usage)
    return ChatResponse(
        content=content,
        finish_reason="stop",
        usage=usage,
    )


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
                        "title": "黒神话全流程",
                        "owner": {"mid": 100, "name": "GameUP"},
                        "pubdate": 1708700000,
                        "stat": {"view": 500000, "coin": 10000, "danmaku": 2000},
                    }
                ],
                "total_hits": 10,
            },
        }
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


# ============================================================
# Tests
# ============================================================


def test_direct_content_response():
    """Test handler when LLM returns content directly (no tool calls)."""
    logger.note("=" * 60)
    logger.note("[TEST] direct content response")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response("你好！我是AI助手。")

    mock_search = MagicMock()

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "你好"}])

    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["choices"][0]["message"]["content"] == "你好！我是AI助手。"
    assert result["choices"][0]["finish_reason"] == "stop"

    # LLM should be called exactly once
    assert mock_llm.chat.call_count == 1

    logger.success("[PASS] direct content response")


def test_system_prompt_includes_search_capabilities():
    """Test ChatHandler injects runtime search capabilities into system prompt."""
    logger.note("=" * 60)
    logger.note("[TEST] system prompt capabilities")

    mock_llm = MagicMock(spec=LLMClient)

    class SearchClientWithCapabilities:
        def capabilities(self, refresh: bool = False) -> dict:
            return {
                "service_name": "runtime-search",
                "service_type": "remote",
                "default_query_mode": "wv",
                "rerank_query_mode": "vwr",
                "supports_multi_query": True,
                "supports_google_search": True,
                "relation_endpoints": ["related_owners_by_tokens"],
                "available_endpoints": ["/explore", "/related_owners_by_tokens"],
                "docs": ["search_syntax"],
            }

    handler = ChatHandler(
        llm_client=mock_llm,
        search_client=SearchClientWithCapabilities(),
    )
    messages = handler._build_messages([{"role": "user", "content": "test"}])

    assert "[SEARCH_CAPABILITIES]" in messages[0]["content"]
    assert "runtime-search" in messages[0]["content"]

    logger.success("[PASS] system prompt capabilities")


def test_single_tool_call():
    """Test handler with one tool command → content response."""
    logger.note("=" * 60)
    logger.note("[TEST] single tool call")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        # First call: analysis + tool command in content
        make_tool_cmd_response(
            "我来搜索黑神话相关视频。",
            "<search_videos queries='[\"黑神话\"]'/>",
        ),
        # Second call: content response (after results injected)
        make_content_response("找到了10个黑神话相关视频。"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "搜索黑神话"}])

    assert result["choices"][0]["message"]["content"] == "找到了10个黑神话相关视频。"

    # LLM called twice (tool command + content)
    assert mock_llm.chat.call_count == 2

    # Search client was called
    mock_search.explore.assert_called_once_with(query="黑神话")

    # Verify tool results were injected as a user message (not tool role)
    second_call_messages = mock_llm.chat.call_args_list[1].kwargs.get("messages")
    result_messages = [
        m
        for m in second_call_messages
        if m.get("role") == "user" and "[搜索结果]" in m.get("content", "")
    ]
    assert len(result_messages) == 1

    logger.success("[PASS] single tool call")


def test_multi_tool_calls():
    """Test handler with related_owners_by_tokens + search_videos in single response → content."""
    logger.note("=" * 60)
    logger.note("[TEST] multi tool calls")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        # 1. Both commands in one response
        make_tool_cmd_response(
            "我来搜索影视飓风的视频，同时找相关创作者。",
            '<related_owners_by_tokens text="影视飓风"/>\n'
            "<search_videos queries='[\":user=影视飓风 :date<=7d\"]'/>",
        ),
        # 2. Final content
        make_content_response("影视飓风最近7天发布了以下视频..."),
    ]

    mock_search = MagicMock()
    mock_search.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "影视飓风最近有什么新视频？"}]
    )

    content = result["choices"][0]["message"]["content"]
    assert "影视飓风" in content

    # LLM called 2 times (commands + content)
    assert mock_llm.chat.call_count == 2

    # Both tools were called
    mock_search.related_owners_by_tokens.assert_called_once()
    mock_search.explore.assert_called_once()

    # Final usage uses last prompt_tokens + accumulated completion_tokens.
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 15
    assert result["usage"]["total_tokens"] == 25

    logger.success("[PASS] multi tool calls")


def test_fallback_tool_commands_for_single_author_timeline():
    """Single-author timeline requests should get deterministic fallback tools."""
    logger.note("=" * 60)
    logger.note("[TEST] fallback tool commands for author timeline")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_tool_commands(
        [],
        messages=[{"role": "user", "content": "影视飓风最近有什么新视频"}],
        content="抱歉，我还没收到系统返回的搜索结果，请再试一次。",
    )

    assert fallback == [
        {"type": "search_videos", "args": {"queries": [":user=影视飓风 :date<=15d"]}},
    ]

    logger.success("[PASS] fallback tool commands for author timeline")


def test_normalize_author_timeline_search_command():
    """Single-author timeline search commands should be normalized to :user queries."""
    logger.note("=" * 60)
    logger.note("[TEST] normalize author timeline search command")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    commands = handler._normalize_author_timeline_commands(
        [{"type": "search_videos", "args": {"queries": ["影视飓风 q=vwr"]}}],
        messages=[{"role": "user", "content": "影视飓风最近有什么新视频？"}],
    )

    assert commands == [
        {"type": "search_videos", "args": {"queries": [":user=影视飓风 :date<=15d"]}},
    ]

    logger.success("[PASS] normalize author timeline search command")


def test_fallback_creator_discovery_commands():
    """Creator discovery requests should fall back to relation search."""
    logger.note("=" * 60)
    logger.note("[TEST] fallback creator discovery commands")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_creator_discovery_commands(
        [],
        messages=[{"role": "user", "content": "推荐几个做黑神话悟空内容的UP主"}],
        content="我来先找一下相关创作者。",
    )

    assert fallback == [
        {"type": "related_owners_by_tokens", "args": {"text": "黑神话悟空"}},
        {"type": "search_videos", "args": {"queries": ["黑神话悟空 q=vwr"]}},
    ]

    logger.success("[PASS] fallback creator discovery commands")


def test_fallback_creator_discovery_commands_for_direct_request():
    """Direct creator recommendation requests should fall back even without pledge text."""
    logger.note("=" * 60)
    logger.note("[TEST] fallback creator discovery direct request")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_creator_discovery_commands(
        [],
        messages=[{"role": "user", "content": "推荐几个做黑神话悟空内容的UP主"}],
        content="我推荐下面这些UP主。",
    )

    assert fallback == [
        {"type": "related_owners_by_tokens", "args": {"text": "黑神话悟空"}},
        {"type": "search_videos", "args": {"queries": ["黑神话悟空 q=vwr"]}},
    ]

    logger.success("[PASS] fallback creator discovery direct request")


def test_fallback_creator_discovery_commands_for_followup_dialogue():
    """Creator-discovery follow-up turns should inherit the earlier topic."""
    logger.note("=" * 60)
    logger.note("[TEST] fallback creator discovery follow-up dialogue")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_creator_discovery_commands(
        [],
        messages=[
            {"role": "user", "content": "推荐几个做黑神话悟空内容的UP主"},
            {"role": "assistant", "content": "可以，我先给你找一批。"},
            {"role": "user", "content": "更偏剧情解析和世界观考据的呢？"},
        ],
        content="我给你筛一批更合适的创作者。",
    )

    assert fallback == [
        {
            "type": "related_owners_by_tokens",
            "args": {"text": "黑神话悟空 剧情解析和世界观考据"},
        },
        {
            "type": "search_videos",
            "args": {"queries": ["黑神话悟空 剧情解析和世界观考据 q=vwr"]},
        },
    ]

    logger.success("[PASS] fallback creator discovery follow-up dialogue")


def test_fallback_external_search_commands():
    """Official-update requests should fall back to Google + Bilibili search."""
    logger.note("=" * 60)
    logger.note("[TEST] fallback external search commands")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_external_search_commands(
        [],
        messages=[
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            }
        ],
        content="我先查一下官方更新，再看看 B 站有没有解读。",
    )

    assert fallback == [
        {"type": "search_google", "args": {"query": "Gemini 2.5 最近有哪些官方更新"}},
        {"type": "search_videos", "args": {"queries": ["Gemini 2.5 q=vwr"]}},
    ]

    logger.success("[PASS] fallback external search commands")


def test_fallback_external_search_commands_for_followup_dialogue():
    """Official-update follow-up turns should inherit the earlier product topic."""
    logger.note("=" * 60)
    logger.note("[TEST] fallback external search follow-up dialogue")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_external_search_commands(
        [],
        messages=[
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            },
            {"role": "assistant", "content": "我先查一下官方更新。"},
            {"role": "user", "content": "更偏开发者 API 侧，有没有 B 站解读？"},
        ],
        content="我先查一下开发者侧的官方更新，再看看 B 站解读。",
    )

    assert fallback == [
        {
            "type": "search_google",
            "args": {"query": "Gemini 2.5 开发者 API 最近有哪些官方更新"},
        },
        {
            "type": "search_videos",
            "args": {"queries": ["Gemini 2.5 q=vwr", "Gemini 2.5 开发者 API q=vwr"]},
        },
    ]

    logger.success("[PASS] fallback external search follow-up dialogue")


def test_fallback_external_search_commands_for_official_only_followup():
    """Official-only follow-ups should stop inheriting the earlier Bilibili decode request."""
    logger.note("=" * 60)
    logger.note("[TEST] fallback external search official-only follow-up")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_external_search_commands(
        [],
        messages=[
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            },
            {"role": "assistant", "content": "我先查一下官方更新。"},
            {"role": "user", "content": "先只看官网就行"},
        ],
        content="我继续只查官网。",
    )

    assert fallback == [
        {"type": "search_google", "args": {"query": "Gemini 2.5 最近有哪些官方更新"}},
    ]

    logger.success("[PASS] fallback external search official-only follow-up")


def test_preflight_tool_commands_for_direct_external_request():
    """Direct official-update requests should preflight deterministic external tools."""
    logger.note("=" * 60)
    logger.note("[TEST] preflight direct external request")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    commands = handler._preflight_tool_commands(
        [
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            }
        ]
    )

    assert commands == [
        {"type": "search_google", "args": {"query": "Gemini 2.5 最近有哪些官方更新"}},
        {"type": "search_videos", "args": {"queries": ["Gemini 2.5 q=vwr"]}},
    ]

    logger.success("[PASS] preflight direct external request")


def test_preflight_tool_commands_for_followup_external_request():
    """Official-update follow-up dialogue should also preflight deterministic external tools."""
    logger.note("=" * 60)
    logger.note("[TEST] preflight follow-up external request")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    commands = handler._preflight_tool_commands(
        [
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            },
            {"role": "assistant", "content": "我先查一下官方更新。"},
            {"role": "user", "content": "更偏开发者 API 侧，有没有 B 站解读？"},
        ]
    )

    assert commands == [
        {
            "type": "search_google",
            "args": {"query": "Gemini 2.5 开发者 API 最近有哪些官方更新"},
        },
        {
            "type": "search_videos",
            "args": {"queries": ["Gemini 2.5 q=vwr", "Gemini 2.5 开发者 API q=vwr"]},
        },
    ]

    logger.success("[PASS] preflight follow-up external request")


def test_duplicate_search_commands_are_suppressed_after_preflight():
    """A command already executed by preflight should not be executed again."""
    logger.note("=" * 60)
    logger.note("[TEST] duplicate search commands suppressed after preflight")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我再搜一下 B 站相关视频。",
            "<search_videos queries='[\"Gemini 2.5 q=vwr\"]' />",
        ),
        make_content_response("这里是 Gemini 2.5 的官方更新和对应 B 站解读。"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT
    mock_search.search_google.return_value = {
        "results": [{"title": "Gemini 2.5 更新", "link": "https://example.com"}]
    }

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)
    handler.tool_executor._handlers["search_google"] = lambda args: {
        "results": [{"title": "Gemini 2.5 更新", "link": "https://example.com"}]
    }

    result = handler.handle(
        messages=[
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            }
        ]
    )

    assert "Gemini 2.5" in result["choices"][0]["message"]["content"]
    assert mock_search.explore.call_count == 1
    assert len(result["tool_events"]) == 1
    assert result["tool_events"][0]["preflight"] is True
    assert result["tool_events"][0]["tools"] == ["search_google", "search_videos"]

    logger.success("[PASS] duplicate search commands suppressed after preflight")


def test_duplicate_commands_are_deduped_within_single_iteration():
    """Identical tool commands emitted in one response should execute only once."""
    logger.note("=" * 60)
    logger.note("[TEST] duplicate commands deduped within single iteration")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我来搜索黑神话相关视频。",
            "<search_videos queries='[\"黑神话 q=vwr\"]'/>\n"
            "<search_videos queries='[\"黑神话 q=vwr\"]'/>",
        ),
        make_content_response("找到了黑神话相关视频。"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "搜索黑神话"}])

    assert "黑神话" in result["choices"][0]["message"]["content"]
    assert mock_search.explore.call_count == 1
    assert result["tool_events"][0]["tools"] == ["search_videos"]

    logger.success("[PASS] duplicate commands deduped within single iteration")


def test_author_timeline_fallback_skips_official_update_queries():
    """Official-update queries should not be mistaken for creator timeline requests."""
    logger.note("=" * 60)
    logger.note("[TEST] author timeline fallback skips official update queries")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_tool_commands(
        [],
        messages=[
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            }
        ],
        content="我先查一下官方更新。",
    )

    assert fallback == []

    logger.success("[PASS] author timeline fallback skips official update queries")


def test_fallback_video_search_commands_for_explicit_video_request():
    """Explicit video-search intents should get a deterministic search fallback."""
    logger.note("=" * 60)
    logger.note("[TEST] fallback video search commands")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_video_search_commands(
        [],
        messages=[{"role": "user", "content": "找几条黑神话悟空剧情解析视频"}],
        content="好的，我先帮你把黑神话悟空剧情解析视频搜出来。",
    )

    assert fallback == [
        {
            "type": "search_videos",
            "args": {"queries": ["黑神话悟空剧情解析 q=vwr"]},
        }
    ]

    logger.success("[PASS] fallback video search commands")


def test_fallback_video_search_commands_for_creator_video_followup():
    """Creator video follow-ups should inherit the earlier creator and search by :user."""
    logger.note("=" * 60)
    logger.note("[TEST] fallback video search creator follow-up")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_video_search_commands(
        [],
        messages=[
            {"role": "user", "content": "何同学有哪些关联账号？"},
            {"role": "assistant", "content": "我先帮你找相关作者线索。"},
            {"role": "user", "content": "那他的代表作有哪些？"},
        ],
        content="我来搜索这位作者的代表作。",
    )

    assert fallback == [
        {"type": "search_videos", "args": {"queries": [":user=何同学"]}},
    ]

    logger.success("[PASS] fallback video search creator follow-up")


def test_fallback_similar_creator_commands():
    """Similar-creator requests should fall back to relation search."""
    logger.note("=" * 60)
    logger.note("[TEST] fallback similar creator commands")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_similar_creator_commands(
        [],
        messages=[
            {
                "role": "user",
                "content": "和影视飓风风格接近的UP主有哪些？各给我一句推荐理由。",
            }
        ],
        content="我先给你推荐几位风格接近的创作者。",
    )

    assert fallback == [
        {"type": "related_owners_by_tokens", "args": {"text": "影视飓风"}},
        {"type": "search_videos", "args": {"queries": ["影视飓风 q=vwr"]}},
    ]

    logger.success("[PASS] fallback similar creator commands")


def test_relation_only_commands_are_promoted_to_search_videos():
    """Relation-only plans should be auto-promoted to include search_videos."""
    logger.note("=" * 60)
    logger.note("[TEST] relation-only commands promoted")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先找一些相关创作者。",
            '<related_owners_by_tokens text="黑神话悟空"/>',
        ),
        make_content_response("这些创作者近期相关视频里，黑神话内容最活跃的是..."),
    ]

    mock_search = MagicMock()
    mock_search.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "推荐几个做黑神话悟空内容的UP主"}]
    )

    assert "黑神话" in result["choices"][0]["message"]["content"]
    mock_search.related_owners_by_tokens.assert_called_once_with(
        text="黑神话悟空", size=8
    )
    mock_search.explore.assert_called_once_with(query="黑神话悟空 q=vwr")
    assert result["tool_events"][0]["tools"] == [
        "related_owners_by_tokens",
        "search_videos",
    ]

    logger.success("[PASS] relation-only commands promoted")


def test_old_results_messages_are_pruned():
    """Only the newest injected tool result block should remain in context."""
    logger.note("=" * 60)
    logger.note("[TEST] old results messages pruned")

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问题"},
        {"role": "user", "content": "[搜索结果]\nsearch_videos():\n{}"},
        {"role": "assistant", "content": "继续分析"},
        {"role": "user", "content": "[搜索结果]\nrelated_owners_by_tokens():\n{}"},
    ]

    ChatHandler._prune_old_results_messages(messages)

    result_messages = [
        m
        for m in messages
        if m.get("role") == "user" and "[搜索结果]" in m.get("content", "")
    ]
    assert len(result_messages) == 1
    assert "related_owners_by_tokens" in result_messages[0]["content"]

    logger.success("[PASS] old results messages pruned")


def test_cache_token_accumulation():
    """Test that cache hit/miss tokens are accumulated in usage."""
    logger.note("=" * 60)
    logger.note("[TEST] cache token accumulation")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "搜索中",
            "<search_videos queries='[\"test\"]'/>",
            extra_usage={
                "prompt_cache_hit_tokens": 100,
                "prompt_cache_miss_tokens": 50,
            },
        ),
        make_content_response(
            "结果如下...",
            extra_usage={
                "prompt_cache_hit_tokens": 200,
                "prompt_cache_miss_tokens": 30,
            },
        ),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "test"}])

    usage = result["usage"]
    assert usage["prompt_cache_hit_tokens"] == 200  # last call input context
    assert usage["prompt_cache_miss_tokens"] == 30  # last call input context
    assert usage["prompt_tokens"] == 10  # last call prompt only
    assert usage["completion_tokens"] == 15  # 10 + 5
    assert usage["total_tokens"] == 25

    logger.success(f"  Cache hit: {usage['prompt_cache_hit_tokens']}")
    logger.success(f"  Cache miss: {usage['prompt_cache_miss_tokens']}")
    logger.success("[PASS] cache token accumulation")


def test_max_iterations():
    """Test that handler stops at max iterations and forces content."""
    logger.note("=" * 60)
    logger.note("[TEST] max iterations")

    mock_llm = MagicMock(spec=LLMClient)
    # First 3 calls: always tool commands (simulates infinite loop)
    # 4th call: forced content response after nudge
    mock_llm.chat.side_effect = [
        make_tool_cmd_response("搜索1", "<search_videos queries='[\"test\"]'/>"),
        make_tool_cmd_response("搜索2", "<search_videos queries='[\"test\"]'/>"),
        make_tool_cmd_response("搜索3", "<search_videos queries='[\"test\"]'/>"),
        make_content_response("根据搜索结果，找到了以下视频..."),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(
        llm_client=mock_llm, search_client=mock_search, max_iterations=3
    )

    result = handler.handle(messages=[{"role": "user", "content": "test"}])

    # 3 iterations + 1 forced content call = 4
    assert mock_llm.chat.call_count == 4
    # Should have real content, not a timeout
    content = result["choices"][0]["message"]["content"]
    assert "搜索结果" in content

    # Verify force nudge was injected in the last call
    last_call_messages = mock_llm.chat.call_args_list[-1].kwargs.get("messages")
    nudge_msg = last_call_messages[-1]
    assert "搜索命令" in nudge_msg["content"] or "回答" in nudge_msg["content"]

    logger.success("[PASS] max iterations")


def test_dsml_sanitization():
    """Test that leaked DSML markup is stripped from content."""
    logger.note("=" * 60)
    logger.note("[TEST] DSML sanitization")

    mock_llm = MagicMock(spec=LLMClient)
    # Simulate DSML leakage in forced content response
    dsml_content = (
        "让我搜索一下：\n\n"
        '<｜DSML｜function_calls><｜DSML｜invoke name="search_videos">'
        '<｜DSML｜parameter name="query" string="true">test</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜function_calls>"
    )
    mock_llm.chat.side_effect = [
        make_tool_cmd_response("搜索1", "<search_videos queries='[\"test\"]'/>"),
        make_tool_cmd_response("搜索2", "<search_videos queries='[\"test\"]'/>"),
        make_tool_cmd_response("搜索3", "<search_videos queries='[\"test\"]'/>"),
        make_content_response(dsml_content),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(
        llm_client=mock_llm, search_client=mock_search, max_iterations=3
    )

    result = handler.handle(messages=[{"role": "user", "content": "test"}])

    content = result["choices"][0]["message"]["content"]
    # DSML markup should be stripped
    assert "DSML" not in content
    assert "function_calls" not in content
    # Real text should be preserved
    assert "让我搜索一下" in content

    logger.success("[PASS] DSML sanitization")


def test_streaming_response():
    """Test streaming mode returns proper SSE chunks using chat() + chunked content."""
    logger.note("=" * 60)
    logger.note("[TEST] streaming response")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_stream.return_value = iter(make_stream_chunks("你好世界！"))

    mock_search = MagicMock()

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = list(handler.handle_stream(messages=[{"role": "user", "content": "你好"}]))

    # Should have multiple chunks
    assert len(chunks) > 2  # At least: role chunk + content chunks + final + [DONE]

    # First chunk should have role
    first = json.loads(chunks[0])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"

    # Last chunk should be [DONE]
    assert chunks[-1] == "[DONE]"

    # Second-to-last should have finish_reason=stop and usage + perf_stats
    last_data = json.loads(chunks[-2])
    assert last_data["choices"][0]["finish_reason"] == "stop"
    assert "perf_stats" in last_data
    assert "usage" in last_data
    assert "usage_trace" in last_data
    # perf_stats should NOT contain token counts (they're in usage)
    assert "completion_tokens" not in last_data["perf_stats"]
    assert "total_tokens" not in last_data["perf_stats"]
    # tokens_per_second should be int
    assert isinstance(last_data["perf_stats"]["tokens_per_second"], int)
    assert last_data["usage_trace"]["summary"]["llm_calls"] == 1

    # Reconstruct content from chunks
    content = ""
    for chunk_str in chunks[:-1]:  # Skip [DONE]
        chunk = json.loads(chunk_str)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            content += delta["content"]
    assert content == "你好世界！"

    assert mock_llm.chat.call_count == 0
    assert mock_llm.chat_stream.call_count == 1

    logger.success("[PASS] streaming response")


def test_streaming_with_tools():
    """Test streaming mode with tool commands using chat()."""
    logger.note("=" * 60)
    logger.note("[TEST] streaming with tools")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_stream.side_effect = [
        iter(
            make_stream_chunks(
                "我来搜索相关视频。\n<search_videos queries='[\"test\"]'/>",
                extra_usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            )
        ),
        iter(
            make_stream_chunks(
                "找到了结果。",
                extra_usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            )
        ),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = list(handler.handle_stream(messages=[{"role": "user", "content": "test"}]))

    # Reconstruct content and check for tool event + thinking
    content = ""
    thinking = ""
    tool_event_found = False
    for chunk_str in chunks[:-1]:
        chunk = json.loads(chunk_str)
        delta = chunk["choices"][0]["delta"]
        if delta.get("retract_content"):
            content = ""
        if "content" in delta:
            content += delta["content"]
        if "reasoning_content" in delta:
            thinking += delta["reasoning_content"]
        if chunk.get("tool_events"):
            tool_event_found = True
            assert chunk["tool_events"][0]["tools"] == ["search_videos"]
    assert content == "找到了结果。"
    assert "搜索" in thinking  # Analysis should appear as thinking
    assert tool_event_found, "Tool event chunk should be present"

    assert mock_llm.chat.call_count == 0
    assert mock_llm.chat_stream.call_count == 2

    logger.success("[PASS] streaming with tools")


def test_system_prompt_included():
    """Test that system prompt is prepended to messages."""
    logger.note("=" * 60)
    logger.note("[TEST] system prompt included")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response("OK")

    mock_search = MagicMock()

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    handler.handle(messages=[{"role": "user", "content": "test"}])

    # Check that system message was prepended
    call_args = mock_llm.chat.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get(
        "messages", call_args[0][0]
    )
    assert messages[0]["role"] == "system"
    assert "blbl.copilot" in messages[0]["content"]
    assert messages[1]["role"] == "user"

    logger.success("[PASS] system prompt included")


def test_response_format():
    """Test that response follows OpenAI format."""
    logger.note("=" * 60)
    logger.note("[TEST] response format")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response("Test response")

    mock_search = MagicMock()

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "test"}])

    # Verify OpenAI format
    assert "id" in result
    assert result["id"].startswith("chatcmpl-")
    assert result["object"] == "chat.completion"
    assert "choices" in result
    assert len(result["choices"]) == 1
    assert result["choices"][0]["index"] == 0
    assert "message" in result["choices"][0]
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert "usage" in result
    assert "usage_trace" in result
    assert result["usage_trace"]["summary"]["llm_calls"] == 1

    logger.success("[PASS] response format")


def test_usage_trace_reports_prompt_and_iterations():
    """Test usage_trace exposes prompt profile and per-iteration diagnostics."""
    logger.note("=" * 60)
    logger.note("[TEST] usage trace diagnostics")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我来搜索黑神话相关视频。",
            "<search_videos queries='[\"黑神话 q=vwr\"]'/>",
        ),
        make_content_response("找到了几条黑神话相关视频。"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "搜索黑神话"}])
    usage_trace = result["usage_trace"]

    assert usage_trace["prompt"]["total_chars"] > 0
    assert usage_trace["prompt"]["section_chars"]["tool_commands"] > 0
    assert usage_trace["summary"]["llm_calls"] == 2
    assert usage_trace["summary"]["tool_iterations"] == 1
    assert len(usage_trace["iterations"]) == 2

    first_iter = usage_trace["iterations"][0]
    second_iter = usage_trace["iterations"][1]
    assert first_iter["tool_names"] == ["search_videos"]
    assert first_iter["result_message_count"] == 0
    assert second_iter["tool_names"] == []
    assert second_iter["result_message_count"] == 1
    assert second_iter["context_chars"] > first_iter["context_chars"]

    logger.success("[PASS] usage trace diagnostics")


def test_multi_turn_conversation():
    """Test handler with multi-turn conversation history."""
    logger.note("=" * 60)
    logger.note("[TEST] multi-turn conversation")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response("好的，这是最近3天的视频。")

    mock_search = MagicMock()

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    messages = [
        {"role": "user", "content": "影视飓风最近有什么新视频？"},
        {"role": "assistant", "content": "影视飓风最近15天发布了以下视频..."},
        {"role": "user", "content": "能不能再近一点？"},
    ]

    result = handler.handle(messages=messages)

    # All user messages should be preserved in the LLM call
    call_args = mock_llm.chat.call_args
    sent_messages = call_args.kwargs.get("messages") or call_args[0][0]
    # system + 3 user/assistant messages
    assert len(sent_messages) == 4
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1]["role"] == "user"
    assert sent_messages[2]["role"] == "assistant"
    assert sent_messages[3]["role"] == "user"

    logger.success("[PASS] multi-turn conversation")


def test_thinking_mode():
    """Test handler with thinking=True (deeper analysis mode)."""
    logger.note("=" * 60)
    logger.note("[TEST] thinking mode")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response("经过深入分析...")

    mock_search = MagicMock()

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "分析黑神话的视频趋势"}],
        thinking=True,
    )

    # Verify result has thinking flag
    assert result.get("thinking") is True
    assert result["choices"][0]["message"]["content"] == "经过深入分析..."

    # Verify thinking prompt is prepended to system message
    call_args = mock_llm.chat.call_args
    sent_messages = call_args.kwargs.get("messages") or call_args[0][0]
    system_content = sent_messages[0]["content"]
    assert "[思考模式]" in system_content

    logger.success("[PASS] thinking mode")


def test_thinking_mode_max_iterations():
    """Test that thinking mode uses higher max_iterations by default."""
    logger.note("=" * 60)
    logger.note("[TEST] thinking mode max iterations")

    mock_llm = MagicMock(spec=LLMClient)
    # Simulate max iterations being hit (all tool commands, never plain content)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response("搜索", "<search_videos queries='[\"q\"]'/>"),
    ] * 10 + [
        make_content_response("最终结果"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(
        llm_client=mock_llm, search_client=mock_search, max_iterations=5
    )

    # Non-thinking: max 5 iterations (will hit forced content)
    result_normal = handler.handle(
        messages=[{"role": "user", "content": "test"}],
        thinking=False,
    )
    # Should have forced content after 5 iterations (5 tool + 1 forced = 6 calls)
    normal_calls = mock_llm.chat.call_count
    assert normal_calls == 6  # 5 tool iterations + 1 forced

    # Reset mock
    mock_llm.reset_mock()
    mock_llm.chat.side_effect = [
        make_tool_cmd_response("搜索", "<search_videos queries='[\"q\"]'/>"),
    ] * 10 + [
        make_content_response("思考后的最终结果"),
    ]

    # Thinking: max 7 iterations (default thinking max)
    result_thinking = handler.handle(
        messages=[{"role": "user", "content": "test"}],
        thinking=True,
    )
    thinking_calls = mock_llm.chat.call_count
    assert thinking_calls == 8  # 7 tool iterations + 1 forced

    logger.success("[PASS] thinking mode max iterations")


def test_explicit_max_iterations_override():
    """Test per-request max_iterations override."""
    logger.note("=" * 60)
    logger.note("[TEST] explicit max iterations override")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response("搜索", "<search_videos queries='[\"q\"]'/>"),
    ] * 3 + [
        make_content_response("结果"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    # Override to max 3 iterations
    result = handler.handle(
        messages=[{"role": "user", "content": "test"}],
        max_iterations=3,
    )
    # 3 tool iterations + 1 forced content
    assert mock_llm.chat.call_count == 4

    logger.success("[PASS] explicit max iterations override")


def test_tool_events_tracking():
    """Test that tool_events are correctly tracked and returned."""
    logger.note("=" * 60)
    logger.note("[TEST] tool events tracking")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        # Iteration 1: multiple commands in one response
        make_tool_cmd_response(
            "我来搜索何同学的视频。",
            '<related_owners_by_tokens text="何同学"/>\n'
            "<search_videos queries='[\":user=何同学 :date<=15d\"]'/>",
        ),
        # Iteration 2: refined search
        make_tool_cmd_response(
            "根据结果，我再搜索一下何同学最近一周的视频。",
            "<search_videos queries='[\"何同学 :date<=7d\"]'/>",
        ),
        # Final content
        make_content_response("何同学最近的视频..."),
    ]

    mock_search = MagicMock()
    mock_search.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "何同学最近的视频"}])

    # tool_events should be returned in the response
    assert "tool_events" in result
    tool_events = result["tool_events"]
    assert len(tool_events) == 2

    # First iteration: related_owners_by_tokens + search_videos
    assert tool_events[0]["iteration"] == 1
    assert "related_owners_by_tokens" in tool_events[0]["tools"]
    assert "search_videos" in tool_events[0]["tools"]

    # Second iteration
    assert tool_events[1]["iteration"] == 2
    assert "search_videos" in tool_events[1]["tools"]

    logger.success("[PASS] tool events tracking")


def test_streaming_with_thinking():
    """Test streaming response with thinking mode and tool events."""
    logger.note("=" * 60)
    logger.note("[TEST] streaming with thinking")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_stream.side_effect = [
        iter(
            make_stream_chunks(
                "让我深入搜索。\n<search_videos queries='[\"test\"]'/>",
                extra_usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            )
        ),
        iter(
            make_stream_chunks(
                "思考后回答",
                extra_usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            )
        ),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = list(
        handler.handle_stream(
            messages=[{"role": "user", "content": "深度分析"}],
            thinking=True,
        )
    )

    # Parse first chunk (role + metadata)
    first_chunk = json.loads(chunks[0])
    assert first_chunk["choices"][0]["delta"]["role"] == "assistant"
    assert first_chunk.get("thinking") is True

    # Find tool event chunks (pending + completed)
    tool_event_chunks = [json.loads(c) for c in chunks[:-1] if "tool_events" in c]
    assert len(tool_event_chunks) == 2  # pending + completed

    # Last real chunk should have finish_reason and stats
    done_chunk = json.loads(chunks[-2])  # -1 is [DONE]
    assert done_chunk["choices"][0]["finish_reason"] == "stop"
    assert "perf_stats" in done_chunk
    assert "usage" in done_chunk
    assert "usage_trace" in done_chunk
    assert done_chunk["usage_trace"]["summary"]["tool_iterations"] == 1

    # Last element is [DONE]
    assert chunks[-1] == "[DONE]"

    logger.success("[PASS] streaming with thinking")


def test_parallel_tool_calls():
    """Test handler with multiple commands in one response (parallel execution)."""
    logger.note("=" * 60)
    logger.note("[TEST] parallel tool calls")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        # 1. Multiple owner relation + search in one response
        make_tool_cmd_response(
            "我来搜索这两位UP主的视频。",
            '<related_owners_by_tokens text="何同学"/>\n'
            '<related_owners_by_tokens text="影视飓风"/>\n'
            '<search_videos queries=\'[":user=何同学 :date<=15d", ":user=影视飓风 :date<=15d"]\'/>',
        ),
        # 2. Final content
        make_content_response("这是何同学和影视飓风最近的视频..."),
    ]

    mock_search = MagicMock()
    mock_search.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "何同学和影视飓风最近有什么新视频？"}]
    )

    content = result["choices"][0]["message"]["content"]
    assert "何同学" in content or "影视飓风" in content

    # LLM called 2 times (commands + content)
    assert mock_llm.chat.call_count == 2

    # Both related_owners_by_tokens calls should have been made
    assert mock_search.related_owners_by_tokens.call_count == 2

    # Search was called (from the multi-query)
    assert mock_search.explore.call_count >= 1

    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 15
    assert result["usage"]["total_tokens"] == 25

    logger.success("[PASS] parallel tool calls")


def test_fallback_video_search_commands_for_multi_creator_productivity_comparison():
    """Multi-creator productivity comparisons should fall back to parallel :user queries."""
    logger.note("=" * 60)
    logger.note("[TEST] fallback video search multi-creator comparison")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    fallback = handler._fallback_video_search_commands(
        [],
        messages=[
            {
                "role": "user",
                "content": "对比一下老番茄和影视飓风最近一个月发布的视频，谁更高产？",
            }
        ],
        content="我来分别搜一下两位作者最近一个月的视频。",
    )

    assert fallback == [
        {
            "type": "search_videos",
            "args": {
                "queries": [":user=老番茄 :date<=30d", ":user=影视飓风 :date<=30d"]
            },
        }
    ]

    logger.success("[PASS] fallback video search multi-creator comparison")


def test_normalize_multi_creator_compare_search_command():
    """Multi-creator comparison search commands should be normalized to per-author :user queries."""
    logger.note("=" * 60)
    logger.note("[TEST] normalize multi-creator comparison search command")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    commands = handler._normalize_multi_creator_compare_commands(
        [
            {
                "type": "search_videos",
                "args": {"queries": ["老番茄 影视飓风 谁更高产 q=vwr"]},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": "对比一下老番茄和影视飓风最近一个月发布的视频，谁更高产？",
            }
        ],
    )

    assert commands == [
        {
            "type": "search_videos",
            "args": {
                "queries": [":user=老番茄 :date<=30d", ":user=影视飓风 :date<=30d"]
            },
        }
    ]

    logger.success("[PASS] normalize multi-creator comparison search command")


def test_multi_query_search():
    """Test handler with multi-query search_videos (queries array)."""
    logger.note("=" * 60)
    logger.note("[TEST] multi-query search")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        # Multi-query search command
        make_tool_cmd_response(
            "我来搜索黑神话和原神的热门视频。",
            '<search_videos queries=\'["黑神话 :view>=1w", "原神 :view>=1w"]\'/>',
        ),
        # Content
        make_content_response("黑神话和原神的热门视频如下..."),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "黑神话和原神播放量最高的视频对比"}]
    )

    assert mock_llm.chat.call_count == 2
    # explore should be called twice (once per query)
    assert mock_search.explore.call_count == 2

    logger.success("[PASS] multi-query search")


def test_empty_final_content_retries_with_force_content_nudge():
    """Tool-backed answers should retry once when the final content is empty."""
    logger.note("=" * 60)
    logger.note("[TEST] empty final content retry")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先找相关作者。",
            '<related_owners_by_tokens text="何同学"/>',
        ),
        make_content_response(""),
        make_content_response("何同学相关账号候选如下。"),
    ]

    mock_search = MagicMock()
    mock_search.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "何同学有哪些关联账号？"}]
    )

    assert result["choices"][0]["message"]["content"] == "何同学相关账号候选如下。"
    assert mock_llm.chat.call_count == 3

    retry_call_messages = mock_llm.chat.call_args_list[-1].kwargs["messages"]
    assert retry_call_messages[-1]["content"] == _FORCE_CONTENT_NUDGE

    phases = [entry["phase"] for entry in result["usage_trace"]["iterations"]]
    assert "empty_content_retry" in phases

    logger.success("[PASS] empty final content retry")


def test_streaming_reasoning_content():
    """Test streaming with reasoning_content (DeepSeek chain-of-thought)."""
    logger.note("=" * 60)
    logger.note("[TEST] streaming reasoning content")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_stream.return_value = iter(
        make_stream_chunks(
            "答案是42。",
            extra_usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            reasoning="让我思考一下这个问题...",
        )
    )

    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = list(
        handler.handle_stream(
            messages=[{"role": "user", "content": "思考"}],
            thinking=True,
        )
    )

    # Reconstruct reasoning and content
    reasoning = ""
    content = ""
    for chunk_str in chunks[:-1]:
        chunk = json.loads(chunk_str)
        delta = chunk["choices"][0]["delta"]
        if "reasoning_content" in delta:
            reasoning += delta["reasoning_content"]
        if "content" in delta:
            content += delta["content"]

    assert reasoning == "让我思考一下这个问题..."
    assert content == "答案是42。"

    logger.success("[PASS] streaming reasoning content")


def test_perf_stats_no_overlap_with_usage():
    """Test that perf_stats does not duplicate fields from usage."""
    logger.note("=" * 60)
    logger.note("[TEST] perf_stats no overlap with usage")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response("Test")

    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "test"}])

    perf_stats = result.get("perf_stats", {})
    usage = result.get("usage", {})

    # perf_stats should only have timing/rate metrics
    assert "tokens_per_second" in perf_stats
    assert "total_elapsed" in perf_stats
    assert "total_elapsed_ms" in perf_stats
    # perf_stats should NOT have token counts
    assert "completion_tokens" not in perf_stats
    assert "total_tokens" not in perf_stats
    assert "prompt_cache_hit_tokens" not in perf_stats
    assert "prompt_cache_miss_tokens" not in perf_stats

    # usage should have token counts
    assert "completion_tokens" in usage
    assert "total_tokens" in usage

    # tokens_per_second should be int
    assert isinstance(perf_stats["tokens_per_second"], int)

    logger.success("[PASS] perf_stats no overlap with usage")


def test_usage_normalization_gpt_nested():
    """Test that GPT nested usage format is normalized to flat format."""
    logger.note("=" * 60)
    logger.note("[TEST] usage normalization GPT nested")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response(
        "result",
        extra_usage={
            "prompt_tokens_details": {"cached_tokens": 500},
            "completion_tokens_details": {"reasoning_tokens": 100},
            "prompt_tokens": 600,
        },
    )

    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "test"}])
    usage = result["usage"]

    # GPT nested should be flattened
    assert usage.get("prompt_cache_hit_tokens") == 500
    assert usage.get("prompt_cache_miss_tokens") == 100  # 600 - 500
    assert usage.get("reasoning_tokens") == 100
    # Nested dicts should be removed
    assert "prompt_tokens_details" not in usage
    assert "completion_tokens_details" not in usage

    logger.success("[PASS] usage normalization GPT nested")


def test_stream_cancellation_before_iteration():
    """Test that handle_stream stops when cancelled before first iteration."""
    logger.note("=" * 60)
    logger.note("[TEST] stream cancellation before iteration")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    # Set cancelled immediately
    cancelled = threading.Event()
    cancelled.set()

    chunks = list(
        handler.handle_stream(
            messages=[{"role": "user", "content": "test"}],
            cancelled=cancelled,
        )
    )

    # Should get role chunk + [DONE], but no LLM calls
    assert chunks[-1] == "[DONE]"
    assert mock_llm.chat.call_count == 0
    assert mock_llm.chat_stream.call_count == 0

    logger.success("[PASS] stream cancellation before iteration")


def test_stream_cancellation_during_tool_loop():
    """Test that handle_stream stops between tool execution and next iteration."""
    logger.note("=" * 60)
    logger.note("[TEST] stream cancellation during tool loop")

    mock_llm = MagicMock(spec=LLMClient)
    cancelled = threading.Event()

    # _chat_interruptible uses chat_stream when cancelled is provided.
    # Return streaming chunks that accumulate to tool command content.
    tool_content = "分析中\n<search_videos queries='[\"test\"]'/>"
    tool_usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}

    def chat_stream_side_effect(**kwargs):
        return iter(make_stream_chunks(tool_content, extra_usage=tool_usage))

    mock_llm.chat_stream.side_effect = chat_stream_side_effect

    mock_search = MagicMock()
    mock_search.explore.return_value = {
        "data": [
            {
                "step": 0,
                "name": "search",
                "output": {"hits": [], "total_hits": 0},
            }
        ]
    }

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = []
    for chunk in handler.handle_stream(
        messages=[{"role": "user", "content": "test"}],
        cancelled=cancelled,
    ):
        chunks.append(chunk)
        # After first completed tool event, cancel
        if '"status": "completed"' in chunk:
            cancelled.set()

    # Should stop after first iteration
    assert chunks[-1] == "[DONE]"
    assert mock_llm.chat_stream.call_count == 1

    logger.success("[PASS] stream cancellation during tool loop")


def test_stream_cancellation_during_phase2():
    """Test that handle_stream stops during Phase 2 streaming."""
    logger.note("=" * 60)
    logger.note("[TEST] stream cancellation during Phase 2")

    mock_llm = MagicMock(spec=LLMClient)
    cancelled = threading.Event()

    # _chat_interruptible uses chat_stream when cancelled is provided.
    # Call 1 (Phase 1 iter 1): tool command content
    tool_content = "搜索中\n<search_videos queries='[\"test\"]'/>"
    tool_chunks = make_stream_chunks(
        tool_content,
        extra_usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    )

    # Call 2 (Phase 2): forced content after max_iterations exhaustion
    phase2_chunks = [
        {"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "World"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]},
    ]

    mock_llm.chat_stream.side_effect = [
        iter(tool_chunks),
        iter(phase2_chunks),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = {
        "data": [
            {
                "step": 0,
                "name": "search",
                "output": {"hits": [], "total_hits": 0},
            }
        ]
    }

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = []
    content_count = 0
    for chunk in handler.handle_stream(
        messages=[{"role": "user", "content": "test"}],
        max_iterations=1,
        cancelled=cancelled,
    ):
        chunks.append(chunk)
        # Cancel after receiving first content chunk in Phase 2
        if '"content":' in chunk and "Hello" in chunk:
            content_count += 1
            cancelled.set()

    # Should have stopped during Phase 2
    assert chunks[-1] == "[DONE]"
    assert content_count >= 1

    logger.success("[PASS] stream cancellation during Phase 2")


def test_stream_no_cancellation_completes_normally():
    """Test that handle_stream completes normally when not cancelled."""
    logger.note("=" * 60)
    logger.note("[TEST] stream no cancellation completes normally")

    mock_llm = MagicMock(spec=LLMClient)

    # _chat_interruptible uses chat_stream when cancelled is provided (even if not set)
    mock_llm.chat_stream.return_value = iter(make_stream_chunks("全部完成"))

    mock_search = MagicMock()

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    # Pass cancelled event but never set it
    cancelled = threading.Event()

    chunks = list(
        handler.handle_stream(
            messages=[{"role": "user", "content": "完成测试"}],
            cancelled=cancelled,
        )
    )

    # Should complete normally with [DONE]
    assert chunks[-1] == "[DONE]"
    # Should have content
    content_chunks = [c for c in chunks if '"content":' in c and "全部完成" in c]
    assert len(content_chunks) > 0
    # Should have used chat_stream (not chat) since cancelled event is provided
    assert mock_llm.chat_stream.call_count == 1

    logger.success("[PASS] stream no cancellation completes normally")


def test_explicit_video_request_injects_fallback_search_when_model_skips_tools():
    """Explicit video search requests should not bypass search_videos."""
    logger.note("=" * 60)
    logger.note("[TEST] explicit video request fallback search")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_content_response("以下是我为你筛到的黑神话悟空剧情解析视频。"),
        make_content_response("找到了几条黑神话悟空剧情解析视频。"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "找几条黑神话悟空剧情解析视频"}]
    )

    assert (
        result["choices"][0]["message"]["content"]
        == "找到了几条黑神话悟空剧情解析视频。"
    )
    mock_search.explore.assert_called_once_with(query="黑神话悟空剧情解析 q=vwr")
    assert mock_llm.chat.call_count == 2

    logger.success("[PASS] explicit video request fallback search")


def test_extract_creator_discovery_topic_does_not_leak_unrelated_history():
    """Creator topic extraction should not inherit unrelated previous turns."""
    logger.note("=" * 60)
    logger.note("[TEST] creator topic extraction ignores unrelated history")

    topic = ChatHandler._extract_creator_discovery_topic(
        [
            {"role": "user", "content": "你有什么功能"},
            {"role": "assistant", "content": "我可以帮你搜索 B 站内容。"},
            {"role": "user", "content": "何同学有哪些关联账号？"},
        ]
    )

    assert topic == "何同学"

    logger.success("[PASS] creator topic extraction ignores unrelated history")


def test_extract_creator_discovery_topic_for_real_account_dialogues():
    """Creator topic extraction should work for realistic account/matrix dialogues."""
    logger.note("=" * 60)
    logger.note("[TEST] creator topic extraction for real account dialogues")

    cases = [
        (
            [
                {"role": "user", "content": "你支持哪些搜索方式？"},
                {"role": "assistant", "content": "我可以搜索视频、作者和关系。"},
                {"role": "user", "content": "半佛仙人有没有小号？"},
            ],
            "半佛仙人",
        ),
        (
            [
                {"role": "user", "content": "你能做什么？"},
                {"role": "assistant", "content": "我能做视频和作者搜索。"},
                {"role": "user", "content": "影视飓风账号矩阵有哪些？"},
            ],
            "影视飓风",
        ),
        (
            [
                {"role": "user", "content": "何同学有哪些关联账号？"},
                {"role": "assistant", "content": "我先帮你找相关作者线索。"},
                {"role": "user", "content": "他还有别的号吗？"},
            ],
            "何同学",
        ),
        (
            [
                {"role": "user", "content": "先说说你能帮我做什么"},
                {"role": "assistant", "content": "我能帮你搜索视频、作者和关系。"},
                {"role": "user", "content": "老番茄有没有其他账号？"},
            ],
            "老番茄",
        ),
        (
            [
                {"role": "user", "content": "老师我想先了解下你的能力边界"},
                {"role": "assistant", "content": "可以，我支持 B 站内容搜索。"},
                {"role": "user", "content": "那影视飓风主号和副号分别是什么？"},
            ],
            "影视飓风",
        ),
    ]

    for messages, expected in cases:
        assert ChatHandler._extract_creator_discovery_topic(messages) == expected

    logger.success("[PASS] creator topic extraction for real account dialogues")


def test_relation_only_account_query_does_not_promote_to_search_videos():
    """Relation-only account queries should not be auto-promoted to video search."""
    logger.note("=" * 60)
    logger.note("[TEST] relation-only account query skips search_videos promotion")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先找一下何同学相关作者线索。",
            '<related_owners_by_tokens text="何同学"/>',
        ),
        make_content_response("找到了何同学相关的作者候选。"),
    ]

    mock_search = MagicMock()
    mock_search.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[
            {"role": "user", "content": "你有什么功能"},
            {"role": "assistant", "content": "我可以帮你搜索 B 站内容。"},
            {"role": "user", "content": "何同学有哪些关联账号？"},
        ]
    )

    assert result["choices"][0]["message"]["content"] == "找到了何同学相关的作者候选。"
    mock_search.related_owners_by_tokens.assert_called_once_with(text="何同学", size=8)
    mock_search.explore.assert_not_called()
    assert mock_llm.chat.call_count == 2

    logger.success("[PASS] relation-only account query skips search_videos promotion")


def test_direct_account_query_injects_relation_fallback_when_model_skips_tools():
    """Direct account/meta queries should still trigger relation fallback when the model emits no tools."""
    logger.note("=" * 60)
    logger.note(
        "[TEST] direct account query injects relation fallback when model skips tools"
    )

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_content_response("我来帮你查一下。"),
        make_content_response("找到了何同学相关的作者候选。"),
    ]

    mock_search = MagicMock()
    mock_search.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[
            {"role": "user", "content": "你有什么功能"},
            {"role": "assistant", "content": "我可以帮你搜索 B 站内容。"},
            {"role": "user", "content": "何同学有哪些关联账号？"},
        ]
    )

    assert result["choices"][0]["message"]["content"] == "找到了何同学相关的作者候选。"
    mock_search.related_owners_by_tokens.assert_called_once_with(text="何同学", size=8)
    mock_search.explore.assert_not_called()
    assert mock_llm.chat.call_count == 2

    logger.success(
        "[PASS] direct account query injects relation fallback when model skips tools"
    )


def test_relation_only_account_followup_still_skips_search_videos():
    """Pronoun-based account follow-ups should inherit creator context but avoid video search."""
    logger.note("=" * 60)
    logger.note("[TEST] relation-only account follow-up skips search_videos promotion")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先继续找这个作者相关账号。",
            '<related_owners_by_tokens text="何同学"/>',
        ),
        make_content_response("补充找到了何同学的其他关联作者候选。"),
    ]

    mock_search = MagicMock()
    mock_search.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[
            {"role": "user", "content": "何同学有哪些关联账号？"},
            {"role": "assistant", "content": "我先帮你找相关作者线索。"},
            {"role": "user", "content": "他还有别的号吗？"},
        ]
    )

    assert (
        result["choices"][0]["message"]["content"]
        == "补充找到了何同学的其他关联作者候选。"
    )
    mock_search.related_owners_by_tokens.assert_called_once_with(text="何同学", size=8)
    mock_search.explore.assert_not_called()
    assert mock_llm.chat.call_count == 2

    logger.success(
        "[PASS] relation-only account follow-up skips search_videos promotion"
    )


def test_relation_only_matrix_query_after_capability_chat_skips_search_videos():
    """Capability chatter before a matrix/account query should not trigger video search promotion."""
    logger.note("=" * 60)
    logger.note(
        "[TEST] matrix query after capability chat skips search_videos promotion"
    )

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先找一下影视飓风相关账号。",
            '<related_owners_by_tokens text="影视飓风"/>',
        ),
        make_content_response("找到了影视飓风相关的账号候选。"),
    ]

    mock_search = MagicMock()
    mock_search.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[
            {"role": "user", "content": "你都支持什么功能？"},
            {"role": "assistant", "content": "我支持视频搜索、作者搜索和关系查询。"},
            {"role": "user", "content": "影视飓风账号矩阵有哪些？"},
        ]
    )

    assert (
        result["choices"][0]["message"]["content"] == "找到了影视飓风相关的账号候选。"
    )
    mock_search.related_owners_by_tokens.assert_called_once_with(
        text="影视飓风", size=8
    )
    mock_search.explore.assert_not_called()
    assert mock_llm.chat.call_count == 2

    logger.success(
        "[PASS] matrix query after capability chat skips search_videos promotion"
    )


def test_chat_interruptible_cancels_mid_stream():
    """Test that _chat_interruptible stops consuming chunks when cancelled mid-stream."""
    logger.note("=" * 60)
    logger.note("[TEST] _chat_interruptible cancels mid-stream")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    cancelled = threading.Event()
    chunks_consumed = []

    def slow_stream(**kwargs):
        """Simulate a streaming response that yields chunks one by one."""
        for i, chunk in enumerate(
            [
                {"choices": [{"delta": {"content": "chunk1 "}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "chunk2 "}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "chunk3 "}, "finish_reason": None}]},
                {
                    "choices": [
                        {"delta": {"content": "chunk4"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            ]
        ):
            chunks_consumed.append(i)
            # Cancel after consuming 2 chunks
            if i == 1:
                cancelled.set()
            yield chunk

    mock_llm.chat_stream.side_effect = slow_stream

    response = handler._chat_interruptible(
        messages=[{"role": "user", "content": "test"}],
        temperature=None,
        cancelled=cancelled,
    )

    # Should have consumed chunks 0 and 1, but cancelled is checked BEFORE
    # processing each chunk, so only chunk 0's content is accumulated.
    # (cancelled is set in the generator during chunk 1's yield, so it's
    # detected at the top of the loop before chunk 1 is processed)
    assert response.finish_reason == "cancelled"
    assert response.content is not None
    assert "chunk1" in response.content
    # chunk2, chunk3, chunk4 should NOT be in the content
    assert "chunk2" not in (response.content or "")
    assert "chunk3" not in (response.content or "")
    assert "chunk4" not in (response.content or "")

    logger.success("[PASS] _chat_interruptible cancels mid-stream")


if __name__ == "__main__":
    tests = [
        ("direct_content_response", test_direct_content_response),
        ("single_tool_call", test_single_tool_call),
        ("multi_tool_calls", test_multi_tool_calls),
        ("parallel_tool_calls", test_parallel_tool_calls),
        ("multi_query_search", test_multi_query_search),
        ("cache_token_accumulation", test_cache_token_accumulation),
        ("max_iterations", test_max_iterations),
        ("dsml_sanitization", test_dsml_sanitization),
        ("streaming_response", test_streaming_response),
        ("streaming_with_tools", test_streaming_with_tools),
        ("streaming_reasoning_content", test_streaming_reasoning_content),
        ("system_prompt_included", test_system_prompt_included),
        ("response_format", test_response_format),
        (
            "usage_trace_reports_prompt_and_iterations",
            test_usage_trace_reports_prompt_and_iterations,
        ),
        (
            "duplicate_search_commands_are_suppressed_after_preflight",
            test_duplicate_search_commands_are_suppressed_after_preflight,
        ),
        (
            "duplicate_commands_are_deduped_within_single_iteration",
            test_duplicate_commands_are_deduped_within_single_iteration,
        ),
        ("multi_turn_conversation", test_multi_turn_conversation),
        ("thinking_mode", test_thinking_mode),
        ("thinking_mode_max_iterations", test_thinking_mode_max_iterations),
        ("explicit_max_iterations_override", test_explicit_max_iterations_override),
        ("tool_events_tracking", test_tool_events_tracking),
        ("streaming_with_thinking", test_streaming_with_thinking),
        ("perf_stats_no_overlap_with_usage", test_perf_stats_no_overlap_with_usage),
        ("usage_normalization_gpt_nested", test_usage_normalization_gpt_nested),
        (
            "stream_cancellation_before_iteration",
            test_stream_cancellation_before_iteration,
        ),
        (
            "stream_cancellation_during_tool_loop",
            test_stream_cancellation_during_tool_loop,
        ),
        ("stream_cancellation_during_phase2", test_stream_cancellation_during_phase2),
        (
            "stream_no_cancellation_completes_normally",
            test_stream_no_cancellation_completes_normally,
        ),
        (
            "explicit_video_request_injects_fallback_search_when_model_skips_tools",
            test_explicit_video_request_injects_fallback_search_when_model_skips_tools,
        ),
        (
            "chat_interruptible_cancels_mid_stream",
            test_chat_interruptible_cancels_mid_stream,
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

    # python -m tests.llm.test_handler
