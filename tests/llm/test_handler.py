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
from llms.chat.handler import ChatHandler, _DUPLICATE_TOOL_NUDGE, _FORCE_CONTENT_NUDGE


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


def assistant_content(result: dict) -> str:
    return str(result["choices"][0]["message"].get("content") or "")


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


def test_compact_result_for_context_keeps_default_owner_candidates():
    result = {
        "text": "红警",
        "total_owners": 8,
        "owners": [
            {"name": f"作者{i}", "mid": 1000 + i, "score": 10 - i} for i in range(8)
        ],
    }

    compact = ChatHandler._compact_result_for_context(result)

    assert compact["total_owners"] == 8
    assert len(compact["owners"]) == 8
    assert compact["owners"][0]["name"] == "作者0"
    assert compact["owners"][-1]["name"] == "作者7"


def test_compact_result_for_context_keeps_topic_owner_candidates():
    result = {
        "text": "红警",
        "mode": "topic",
        "total_owners": 20,
        "owners": [
            {"name": f"作者{i}", "mid": 2000 + i, "score": 20 - i} for i in range(20)
        ],
    }

    compact = ChatHandler._compact_result_for_context(result)

    assert compact["total_owners"] == 20
    assert len(compact["owners"]) == 20
    assert compact["owners"][0]["name"] == "作者0"
    assert compact["owners"][-1]["name"] == "作者19"


def test_compact_result_for_context_preserves_google_site_metadata():
    result = {
        "query": "Gemini CLI MCP site:bilibili.com/video",
        "result_count": 3,
        "backend": "mock-google",
        "results": [
            {
                "title": "Gemini CLI MCP 工作流实战",
                "link": "https://www.bilibili.com/video/BV1abc123xyz",
                "domain": "bilibili.com",
                "site_kind": "video",
                "bvid": "BV1abc123xyz",
                "snippet": "MCP 工作流和 Gemini CLI 的实战视频。",
            },
            {
                "title": "MCP 开发者主页",
                "link": "https://space.bilibili.com/12345678",
                "domain": "space.bilibili.com",
                "site_kind": "space",
                "mid": 12345678,
                "snippet": "长期做 MCP / Agent 内容。",
            },
            {
                "title": "Gemini CLI 专栏",
                "link": "https://www.bilibili.com/read/cv24680",
                "domain": "bilibili.com",
                "site_kind": "read",
                "article_id": "cv24680",
                "snippet": "B 站专栏文章。",
            },
        ],
    }

    compact = ChatHandler._compact_result_for_context(result)

    assert compact["query"] == "Gemini CLI MCP site:bilibili.com/video"
    assert compact["result_count"] == 3
    assert compact["backend"] == "mock-google"
    assert len(compact["results"]) == 3
    assert compact["results"][0]["site_kind"] == "video"
    assert compact["results"][0]["bvid"] == "BV1abc123xyz"
    assert compact["results"][1]["site_kind"] == "space"
    assert compact["results"][1]["mid"] == 12345678
    assert compact["results"][2]["site_kind"] == "read"
    assert compact["results"][2]["article_id"] == "cv24680"


def test_compact_result_for_context_preserves_multi_query_video_hits():
    result = {
        "results": [
            {
                "query": ":uid=1629347259 :date<=30d",
                "total_hits": 12,
                "hits": [
                    {
                        "title": "08 最近更新 1",
                        "bvid": "BV108A",
                        "owner": {"name": "红警HBK08"},
                        "stat": {"view": 12345},
                        "pub_to_now_str": "3天前",
                        "duration_str": "12:34",
                        "tags": "红警,月亮3",
                    }
                ],
            },
            {
                "query": ":uid=674510452 :date<=30d",
                "total_hits": 7,
                "hits": [
                    {
                        "title": "月亮3 最近更新 1",
                        "bvid": "BVmoon3",
                        "owner": {"name": "月亮3"},
                        "stat": {"view": 67890},
                        "pub_to_now_str": "1天前",
                        "duration_str": "08:21",
                    }
                ],
            },
        ]
    }

    compact = ChatHandler._compact_result_for_context(result)

    assert len(compact["results"]) == 2
    assert compact["results"][0]["query"] == ":uid=1629347259 :date<=30d"
    assert compact["results"][0]["total_hits"] == 12
    assert compact["results"][0]["hits"][0]["title"] == "08 最近更新 1"
    assert compact["results"][0]["hits"][0]["owner"] == "红警HBK08"
    assert compact["results"][1]["query"] == ":uid=674510452 :date<=30d"
    assert compact["results"][1]["total_hits"] == 7
    assert compact["results"][1]["hits"][0]["title"] == "月亮3 最近更新 1"
    assert compact["results"][1]["hits"][0]["owner"] == "月亮3"


def test_google_keyword_bootstrap_is_prepended_for_title_uncertainty():
    commands = [
        {
            "type": "related_tokens_by_tokens",
            "args": {"text": "OpenAI Agents SDK", "mode": "auto"},
        },
        {"type": "search_videos", "args": {"queries": ["OpenAI Agents SDK q=vwr"]}},
    ]
    messages = [
        {
            "role": "user",
            "content": "我想找 B站上讲 OpenAI Agents SDK 的内容，但我不确定大家会怎么写标题。先帮我摸一下关键词，再给我几条视频。",
        }
    ]

    rewritten = ChatHandler._build_google_keyword_bootstrap_commands(
        commands,
        messages,
        None,
    )

    assert rewritten[0]["type"] == "search_google"
    assert rewritten[0]["args"]["query"] == "OpenAI Agents SDK site:bilibili.com/video"
    assert rewritten[1:] == commands


def test_google_creator_bootstrap_is_prepended_for_creator_uncertainty():
    commands = [
        {"type": "related_owners_by_tokens", "args": {"text": "MCP 工作流", "size": 8}},
    ]
    messages = [
        {
            "role": "user",
            "content": "我想找做 MCP 工作流的 B站UP主，但我不知道作者叫什么。先帮我摸几个作者。",
        }
    ]

    rewritten = ChatHandler._build_google_creator_bootstrap_commands(
        commands,
        messages,
        None,
    )

    assert rewritten[0]["type"] == "search_google"
    assert rewritten[0]["args"]["query"] == "MCP 工作流 site:space.bilibili.com"
    assert rewritten[1:] == commands


def test_google_creator_bootstrap_splits_multi_topic_queries_before_combined_probe():
    commands = [
        {"type": "related_owners_by_tokens", "args": {"text": "MCP 工作流", "size": 8}},
    ]
    messages = [
        {
            "role": "user",
            "content": "我想找做 MCP 工作流和 AI Agent 开发的 B站UP主，但我不知道作者叫什么。先帮我摸几个作者。",
        }
    ]

    rewritten = ChatHandler._build_google_creator_bootstrap_commands(
        commands,
        messages,
        None,
    )

    assert [command["type"] for command in rewritten[:3]] == [
        "search_google",
        "search_google",
        "search_google",
    ]
    assert [command["args"]["query"] for command in rewritten[:3]] == [
        "MCP 工作流 site:space.bilibili.com",
        "AI Agent 开发 site:space.bilibili.com",
        "MCP 工作流 AI Agent 开发 site:space.bilibili.com",
    ]
    assert rewritten[3:] == commands


def test_google_owner_bootstrap_is_prepended_for_unresolved_recent_video_aliases():
    messages = [
        {
            "role": "user",
            "content": "08和月亮3最近都发了哪些视频？",
        }
    ]
    last_tool_results = [
        {"type": "search_owners", "result": {**MOCK_CONTEXTUAL_08_OWNER_RESULT}},
        {"type": "search_owners", "result": {**MOCK_MOON_OWNER_RESULT}},
    ]

    rewritten = ChatHandler._build_google_owner_bootstrap_commands(
        [],
        messages,
        last_tool_results,
    )

    assert [command["type"] for command in rewritten] == [
        "search_google",
        "search_google",
    ]
    assert [command["args"]["query"] for command in rewritten] == [
        "月亮3 site:space.bilibili.com",
        "红警 月亮3 site:space.bilibili.com",
    ]


def test_google_space_creator_followup_prefers_owner_verification():
    messages = [
        {
            "role": "user",
            "content": "我想找做 AI Agent 实战的 B站UP主，但我不知道作者叫什么。先帮我摸几个作者。",
        }
    ]
    last_tool_results = [
        {
            "type": "search_google",
            "result": {
                "query": "AI Agent 实战 site:space.bilibili.com",
                "results": [
                    {
                        "title": "tyk233的个人空间",
                        "link": "https://space.bilibili.com/408202968",
                        "site_kind": "space",
                    },
                    {
                        "title": "慢学AI - 个人空间",
                        "link": "https://space.bilibili.com/28321599",
                        "site_kind": "space",
                    },
                ],
            },
        }
    ]
    commands = [{"type": "search_videos", "args": {"queries": ["AI Agent 实战 q=vwr"]}}]

    rewritten = ChatHandler._build_google_space_owner_followup_commands(
        commands,
        messages,
        last_tool_results,
    )

    assert [command["type"] for command in rewritten] == [
        "search_owners",
        "search_owners",
    ]
    assert rewritten[0]["args"]["text"] == "tyk233"
    assert rewritten[1]["args"]["text"] == "慢学AI"


def test_google_space_creator_followup_falls_back_to_owner_topic_search():
    messages = [
        {
            "role": "user",
            "content": "我想找做 MCP 工作流和 AI Agent 开发的 B站UP主，但我不知道作者叫什么。先帮我摸几个作者。",
        }
    ]
    last_tool_results = [
        {
            "type": "search_google",
            "result": {
                "query": "MCP 工作流 AI Agent 开发 site:space.bilibili.com",
                "results": [
                    {
                        "title": "B站搜索结果页",
                        "link": "https://www.bilibili.com/video/BV1abc123xyz",
                        "site_kind": "video",
                    }
                ],
            },
        }
    ]
    commands = [{"type": "search_videos", "args": {"queries": ["MCP 工作流 q=vwr"]}}]

    rewritten = ChatHandler._build_google_space_owner_followup_commands(
        commands,
        messages,
        last_tool_results,
    )

    assert rewritten[0]["type"] == "search_owners"
    assert rewritten[0]["args"]["mode"] == "topic"
    assert "MCP 工作流" in rewritten[0]["args"]["text"]
    assert "AI Agent 开发" in rewritten[0]["args"]["text"]


MOCK_SEARCH_OWNERS_RESULT = {
    "text": "影视飓风",
    "mode": "relation",
    "owners": [
        {
            "mid": 946974,
            "name": "影视飓风",
            "score": 171.3,
            "sources": ["name", "relation"],
        },
        {
            "mid": 1780480185,
            "name": "飓多多StormCrew",
            "score": 112.5,
            "sources": ["relation"],
        },
    ],
}

MOCK_TIMELINE_OWNER_RESULT = {
    "text": "红警08",
    "mode": "name",
    "owners": [
        {
            "mid": 1629347259,
            "name": "红警HBK08",
            "score": 184.0,
            "sources": ["name", "topic"],
        }
    ],
}

MOCK_HE_TONGXUE_OWNER_RESULT = {
    "text": "何同学",
    "mode": "name",
    "owners": [
        {
            "mid": 163637592,
            "name": "老师好我叫何同学",
            "score": 216.0,
            "sample_view": 4144389,
            "sources": ["name"],
        }
    ],
}

MOCK_MOON_OWNER_RESULT = {
    "text": "月亮3",
    "mode": "name",
    "owners": [
        {
            "mid": 3706946492303759,
            "name": "月亮33222",
            "score": 329.0,
            "sample_view": 186532,
            "sources": ["name", "topic"],
        },
        {
            "mid": 674510452,
            "name": "红警月亮3",
            "score": 325.0,
            "sample_view": 57686,
            "sources": ["name", "topic"],
        },
    ],
}

MOCK_CONTEXTUAL_08_OWNER_RESULT = {
    "text": "红警08",
    "mode": "name",
    "owners": [
        {
            "mid": 1629347259,
            "name": "红警HBK08",
            "score": 184.0,
            "sample_view": 223624,
            "sources": ["name"],
        }
    ],
}

MOCK_CONTEXTUAL_MOON_OWNER_RESULT = {
    "text": "红警月亮3",
    "mode": "name",
    "owners": [
        {
            "mid": 674510452,
            "name": "红警月亮3",
            "score": 250.0,
            "sample_view": 57686,
            "sources": ["name"],
        }
    ],
}

MOCK_RELATED_TOKENS_RESULT = {
    "text": "康夫UI",
    "options": [
        {"text": "ComfyUI", "score": 0.93},
        {"text": "Comfy UI", "score": 0.81},
    ],
}

MOCK_SEMANTIC_TOPIC_TOKENS_RESULT = {
    "text": "口语化标签",
    "options": [
        {"text": "方向一", "score": 0.91},
        {"text": "方向二", "score": 0.88},
        {"text": "方向三", "score": 0.83},
        {"text": "方向四", "score": 0.79},
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
                "supports_owner_search": True,
                "supports_google_search": True,
                "relation_endpoints": ["related_tokens_by_tokens"],
                "available_endpoints": ["/explore", "/search_owners"],
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
        # 2. Model pauses after owner resolution; handler should continue the plan.
        make_content_response("我继续整理影视飓风最近7天的视频。"),
        # 3. Final content
        make_content_response("影视飓风最近7天发布了以下视频..."),
    ]

    mock_search = MagicMock()
    mock_search.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT
    mock_search.search_owners.return_value = {
        "text": "影视飓风",
        "mode": "name",
        "owners": [
            {
                "mid": 946974,
                "name": "影视飓风",
                "score": 264.0,
                "sample_view": 3785326,
                "sources": ["name"],
            }
        ],
    }
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "影视飓风最近有什么新视频？"}]
    )

    content = result["choices"][0]["message"]["content"]
    assert "影视飓风" in content

    # Generic owner resolution inserts an owner-search pass before video search.
    assert mock_llm.chat.call_count == 3

    # Both tools were called, plus owner canonicalization for :user=影视飓风.
    mock_search.related_owners_by_tokens.assert_called_once()
    mock_search.search_owners.assert_called_once_with(
        text="影视飓风",
        mode="name",
        size=8,
    )
    mock_search.explore.assert_called_once_with(query=":uid=946974 :date<=7d")

    # Final usage uses last prompt_tokens + accumulated completion_tokens.
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 20
    assert result["usage"]["total_tokens"] == 30

    logger.success("[PASS] multi tool calls")


def test_preflight_tool_commands_for_ambiguous_author_timeline():
    """The generic planner should not preflight regex-derived timeline commands."""
    logger.note("=" * 60)
    logger.note("[TEST] preflight tool commands for ambiguous author timeline")

    mock_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    commands = handler._preflight_tool_commands(
        [{"role": "user", "content": "红警08最近发了什么视频？"}]
    )

    assert commands == []

    logger.success("[PASS] preflight tool commands for ambiguous author timeline")


def test_ambiguous_author_timeline_is_resolved_before_video_search():
    """Ambiguous author timeline flows should resolve owner candidates before the final video query."""
    logger.note("=" * 60)
    logger.note("[TEST] ambiguous author timeline resolved before video search")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我来搜索红警08最近的视频。",
            "<search_videos queries='[\":user=红警08 :date<=15d\"]'/>",
        ),
        make_content_response("我继续确认一下最近视频。"),
        make_content_response("- [红警全油田一排排](BV1xxx) — 22.3万（3天前）"),
    ]

    mock_search = MagicMock()
    mock_search.search_owners.return_value = MOCK_TIMELINE_OWNER_RESULT
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "红警08最近发了什么视频？"}]
    )

    assert assistant_content(result)
    mock_search.search_owners.assert_called_once_with(
        text="红警08", mode="name", size=8
    )
    mock_search.explore.assert_called_once_with(query=":uid=1629347259 :date<=15d")
    assert result["tool_events"][0]["tools"] == ["search_owners"]
    assert result["tool_events"][1]["tools"] == ["search_videos"]

    logger.success("[PASS] ambiguous author timeline resolved before video search")


def test_multi_owner_recent_videos_continue_after_contextual_owner_resolution():
    """Multi-owner recent-video requests should keep iterating via contextual owner resolution instead of asking the user to clarify."""
    logger.note("=" * 60)
    logger.note(
        "[TEST] multi-owner recent videos continue after contextual owner resolution"
    )

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先确认这两个名字对应的作者。",
            '<search_owners text="08" mode="name"/>\n'
            '<search_owners text="月亮3" mode="name"/>',
        ),
        make_content_response(
            "当前无法确认“08”对应哪个作者；“月亮3”较可能是“红警月亮3”。"
        ),
        make_content_response("我继续整理他们最近的视频。"),
        make_content_response(
            "08和月亮3最近视频：\n- [红警全油田一排排](BV1aaa)\n- [红警海盗争霸一块地](BV1bbb)"
        ),
    ]

    mock_search = MagicMock()
    mock_search.search_owners.side_effect = [
        {
            "text": "08",
            "mode": "name",
            "owners": [
                {
                    "mid": 3706942411246433,
                    "name": "08-v嫖",
                    "score": 224.0,
                    "sample_view": 0,
                    "sources": ["name"],
                }
            ],
        },
        MOCK_MOON_OWNER_RESULT,
        MOCK_CONTEXTUAL_08_OWNER_RESULT,
        MOCK_CONTEXTUAL_MOON_OWNER_RESULT,
    ]
    mock_search.explore.side_effect = [MOCK_EXPLORE_RESULT, MOCK_EXPLORE_RESULT]

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)
    handler.tool_executor.google_client = MagicMock()
    handler.tool_executor.google_client.search.return_value = {
        "backend": "mock-google",
        "result_count": 2,
        "results": [
            {
                "title": "红警HBK08的个人空间",
                "link": "https://space.bilibili.com/1629347259",
                "snippet": "红警作者主页",
            },
            {
                "title": "红警月亮3的个人空间",
                "link": "https://space.bilibili.com/674510452",
                "snippet": "红警作者主页",
            },
        ],
    }
    handler.tool_executor._is_google_available = lambda: True

    result = handler.handle(
        messages=[{"role": "user", "content": "08和月亮3最近都发了哪些视频？"}]
    )

    assert assistant_content(result)
    assert mock_search.search_owners.call_args_list == [
        call(text="08", mode="name", size=8),
        call(text="月亮3", mode="name", size=8),
        call(text="红警08", mode="name", size=8),
        call(text="红警月亮3", mode="name", size=8),
    ]
    handler.tool_executor.google_client.search.assert_not_called()
    assert {
        call_args.kwargs["query"] for call_args in mock_search.explore.call_args_list
    } == {
        ":uid=1629347259 :date<=15d",
        ":uid=674510452 :date<=15d",
    }
    assert result["tool_events"][0]["tools"] == ["search_owners", "search_owners"]
    assert result["tool_events"][1]["tools"] == ["search_owners", "search_owners"]
    assert result["tool_events"][2]["tools"] == ["search_videos"]

    logger.success(
        "[PASS] multi-owner recent videos continue after contextual owner resolution"
    )


def test_owner_only_followup_plan_is_promoted_to_video_search_once_resolved():
    """If the model keeps emitting search_owners after contextual owner resolution, the handler should promote the plan to search_videos."""
    logger.note("=" * 60)
    logger.note("[TEST] owner-only follow-up plan promoted to video search")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先确认这两个名字对应的作者。",
            '<search_owners text="08" mode="name"/>\n'
            '<search_owners text="月亮3" mode="name"/>',
        ),
        make_tool_cmd_response(
            "我继续确认更准确的作者。",
            '<search_owners text="红警08" mode="name"/>\n'
            '<search_owners text="红警月亮3" mode="name"/>',
        ),
        make_tool_cmd_response(
            "我再确认一下作者。",
            '<search_owners text="红警HBK08" mode="name"/>\n'
            '<search_owners text="红警月亮3" mode="name"/>',
        ),
        make_content_response(
            "最近视频：\n- [红警全油田一排排](BV1aaa)\n- [红警海盗争霸一块地](BV1bbb)"
        ),
    ]

    mock_search = MagicMock()
    mock_search.search_owners.side_effect = [
        {
            "text": "08",
            "mode": "name",
            "owners": [
                {
                    "mid": 3706942411246433,
                    "name": "08-v嫖",
                    "score": 224.0,
                    "sample_view": 0,
                    "sources": ["name"],
                }
            ],
        },
        MOCK_MOON_OWNER_RESULT,
        MOCK_CONTEXTUAL_08_OWNER_RESULT,
        MOCK_CONTEXTUAL_MOON_OWNER_RESULT,
    ]
    mock_search.explore.side_effect = [MOCK_EXPLORE_RESULT, MOCK_EXPLORE_RESULT]

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)
    handler.tool_executor.google_client = MagicMock()
    handler.tool_executor.google_client.search.return_value = {
        "backend": "mock-google",
        "result_count": 2,
        "results": [
            {
                "title": "红警HBK08的个人空间",
                "link": "https://space.bilibili.com/1629347259",
                "snippet": "红警作者主页",
            },
            {
                "title": "红警月亮3的个人空间",
                "link": "https://space.bilibili.com/674510452",
                "snippet": "红警作者主页",
            },
        ],
    }
    handler.tool_executor._is_google_available = lambda: True

    result = handler.handle(
        messages=[{"role": "user", "content": "08和月亮3最近都发了哪些视频？"}]
    )

    assert assistant_content(result)
    assert mock_search.search_owners.call_args_list == [
        call(text="08", mode="name", size=8),
        call(text="月亮3", mode="name", size=8),
        call(text="红警08", mode="name", size=8),
        call(text="红警月亮3", mode="name", size=8),
    ]
    handler.tool_executor.google_client.search.assert_not_called()
    assert {
        call_args.kwargs["query"] for call_args in mock_search.explore.call_args_list
    } == {
        ":uid=1629347259 :date<=15d",
        ":uid=674510452 :date<=15d",
    }
    assert result["tool_events"][-1]["tools"] == ["search_videos"]

    logger.success("[PASS] owner-only follow-up plan promoted to video search")


def test_unresolved_owner_aliases_override_generic_topic_drift():
    """If the model drifts into generic topic/video search before owner aliases are resolved, contextual owner resolution should take precedence."""
    logger.note("=" * 60)
    logger.note("[TEST] unresolved owner aliases override generic topic drift")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先确认这两个名字对应的作者。",
            '<search_owners text="08" mode="name"/>\n'
            '<search_owners text="月亮3" mode="name"/>',
        ),
        make_tool_cmd_response(
            "我先看看和他们相关的话题视频。",
            '<related_tokens_by_tokens text="08 月亮3" mode="associate"/>\n'
            "<search_videos queries="
            "'"
            '["08 月亮3 q=vwr"]'
            "'"
            "/>",
        ),
        make_content_response("我继续整理他们最近的视频。"),
        make_content_response(
            "最近视频：\n- [红警全油田一排排](BV1aaa)\n- [红警海盗争霸一块地](BV1bbb)"
        ),
        make_content_response(
            "最近视频：\n- [红警全油田一排排](BV1aaa)\n- [红警海盗争霸一块地](BV1bbb)"
        ),
    ]

    mock_search = MagicMock()
    mock_search.search_owners.side_effect = [
        {
            "text": "08",
            "mode": "name",
            "owners": [
                {
                    "mid": 3706942411246433,
                    "name": "08-v嫖",
                    "score": 224.0,
                    "sample_view": 0,
                    "sources": ["name"],
                }
            ],
        },
        MOCK_MOON_OWNER_RESULT,
        MOCK_CONTEXTUAL_08_OWNER_RESULT,
        MOCK_CONTEXTUAL_MOON_OWNER_RESULT,
    ]
    mock_search.explore.side_effect = [MOCK_EXPLORE_RESULT, MOCK_EXPLORE_RESULT]

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)
    handler.tool_executor.google_client = MagicMock()
    handler.tool_executor.google_client.search.return_value = {
        "backend": "mock-google",
        "result_count": 2,
        "results": [
            {
                "title": "红警HBK08的个人空间",
                "link": "https://space.bilibili.com/1629347259",
                "snippet": "红警作者主页",
            },
            {
                "title": "红警月亮3的个人空间",
                "link": "https://space.bilibili.com/674510452",
                "snippet": "红警作者主页",
            },
        ],
    }
    handler.tool_executor._is_google_available = lambda: True

    result = handler.handle(
        messages=[{"role": "user", "content": "08和月亮3最近都发了哪些视频？"}]
    )

    assert assistant_content(result)
    assert mock_search.search_owners.call_args_list == [
        call(text="08", mode="name", size=8),
        call(text="月亮3", mode="name", size=8),
        call(text="红警08", mode="name", size=8),
        call(text="红警月亮3", mode="name", size=8),
    ]
    handler.tool_executor.google_client.search.assert_not_called()
    mock_search.related_tokens_by_tokens.assert_not_called()
    assert result["tool_events"][1]["tools"] == ["search_owners", "search_owners"]
    assert result["tool_events"][-1]["tools"] == ["search_videos"]

    logger.success("[PASS] unresolved owner aliases override generic topic drift")


def test_author_timeline_final_content_retains_author_name():
    """Author timeline answers should keep the author name even if the LLM omits it."""
    logger.note("=" * 60)
    logger.note("[TEST] author timeline final content retains author name")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我来搜索影视飓风最近的视频。",
            "<search_videos queries=" "'" '["影视飓风 最近视频 q=wv"]' "'" "/>",
        ),
        make_content_response("- [港口都在运些什么](BV1xxx) — 383.7万（6天前）"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "影视飓风最近有什么新视频？"}]
    )

    content = assistant_content(result)
    assert "影视飓风" in content
    assert content.startswith("影视飓风最近视频：")
    mock_search.explore.assert_called_once_with(query="影视飓风 最近视频 q=wv")

    logger.success("[PASS] author timeline final content retains author name")


def test_token_assisted_search_fallback_uses_canonical_entity_query():
    """Token correction should fall through to search_videos with the canonical entity, not the raw typo."""
    logger.note("=" * 60)
    logger.note("[TEST] token-assisted canonical search fallback")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先补一下术语。",
            '<related_tokens_by_tokens text="康夫UI" mode="auto"/>',
        ),
        make_content_response("我继续整理一下搜索词。"),
        make_content_response("这里是 ComfyUI 入门教程。"),
    ]

    mock_search = MagicMock()
    mock_search.related_tokens_by_tokens.return_value = MOCK_RELATED_TOKENS_RESULT
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "康夫UI 有什么入门教程？"}]
    )

    assert assistant_content(result)
    mock_search.related_tokens_by_tokens.assert_called_once_with(
        text="康夫UI", mode="auto", size=8
    )
    mock_search.explore.assert_called_once_with(query="ComfyUI 入门教程 q=vwr")

    logger.success("[PASS] token-assisted canonical search fallback")


def test_token_assisted_search_fallback_expands_implicit_topic_into_multi_queries():
    """Short, implicit topic requests should expand into multiple concrete search hypotheses instead of literalizing the raw slang term."""
    logger.note("=" * 60)
    logger.note("[TEST] token-assisted implicit topic expansion")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先补一下更可搜的主题方向。",
            '<related_tokens_by_tokens text="口语化标签" mode="associate"/>',
        ),
        make_content_response("我继续整理一下更具体的搜索方向。"),
        make_content_response("这里有几条更贴近需求的结果。"),
    ]

    mock_search = MagicMock()
    mock_search.related_tokens_by_tokens.return_value = (
        MOCK_SEMANTIC_TOPIC_TOKENS_RESULT
    )
    mock_search.explore.side_effect = [
        {**MOCK_EXPLORE_RESULT, "query": "方向一 q=vwr"},
        {**MOCK_EXPLORE_RESULT, "query": "方向二 q=vwr"},
        {**MOCK_EXPLORE_RESULT, "query": "方向三 q=vwr"},
    ]

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "来点口语化标签"}])

    assert result["choices"][0]["message"]["content"]
    mock_search.related_tokens_by_tokens.assert_called_once_with(
        text="口语化标签", mode="associate", size=8
    )
    assert mock_search.explore.call_args_list == [
        call(query="方向一 q=vwr"),
        call(query="方向二 q=vwr"),
        call(query="方向三 q=vwr"),
    ]
    assert result["tool_events"][0]["tools"] == ["related_tokens_by_tokens"]
    assert result["tool_events"][1]["tools"] == ["search_videos"]

    logger.success("[PASS] token-assisted implicit topic expansion")


def test_preflight_tool_commands_for_direct_external_request():
    """The generic planner no longer preflights external regex routes."""
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

    assert commands == []

    logger.success("[PASS] preflight direct external request")


def test_preflight_tool_commands_for_followup_external_request():
    """The generic planner no longer preflights follow-up external regex routes."""
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

    assert commands == []

    logger.success("[PASS] preflight follow-up external request")


def test_duplicate_search_commands_are_suppressed_after_preflight():
    """Duplicate commands should still be deduped even without regex preflight routing."""
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

    assert assistant_content(result)
    assert mock_search.explore.call_count >= 1
    assert len(result["tool_events"]) >= 1
    assert result["tool_events"][0]["tools"] == ["search_videos"]

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

    assert assistant_content(result)
    assert mock_search.explore.call_count == 1
    assert result["tool_events"][0]["tools"] == ["search_videos"]

    logger.success("[PASS] duplicate commands deduped within single iteration")


def test_repeated_duplicate_commands_force_answer_after_one_nudge():
    """Repeated duplicate tool commands should trigger forced content instead of burning the full loop budget."""
    logger.note("=" * 60)
    logger.note("[TEST] repeated duplicate commands force answer")

    duplicate_response = make_tool_cmd_response(
        "我再确认一下。",
        "<search_videos queries='[\"ComfyUI 入门教程 q=vwr\"]'/>",
    )
    forced_answer = make_content_response("这里是 ComfyUI 的入门教程整理。")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先搜索教程。",
            "<search_videos queries='[\"ComfyUI 入门教程 q=vwr\"]'/>",
        ),
        duplicate_response,
        duplicate_response,
        forced_answer,
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "康夫UI 有什么入门教程？"}]
    )

    assert assistant_content(result)
    assert mock_llm.chat.call_count == 4
    prompt_snapshots = [
        json.dumps(call.kwargs["messages"], ensure_ascii=False)
        for call in mock_llm.chat.call_args_list
    ]
    assert any(_DUPLICATE_TOOL_NUDGE in snapshot for snapshot in prompt_snapshots)
    assert any(_FORCE_CONTENT_NUDGE in snapshot for snapshot in prompt_snapshots)

    logger.success("[PASS] repeated duplicate commands force answer")


def test_handle_stream_preserves_model_default_reasoning_in_normal_mode():
    """Normal streaming mode should not force-disable provider reasoning."""
    logger.note("=" * 60)
    logger.note("[TEST] handle_stream preserves default reasoning")

    mock_llm = MagicMock(spec=LLMClient)

    def chat_stream_side_effect(*, enable_thinking=None, **_kwargs):
        reasoning = "先分析一下问题。" if enable_thinking is not False else None
        return make_stream_chunks("这是最终答案。", reasoning=reasoning)

    mock_llm.chat_stream.side_effect = chat_stream_side_effect
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = list(
        handler.handle_stream(
            messages=[{"role": "user", "content": "央视新闻最近有什么新视频？"}],
            thinking=False,
        )
    )

    payloads = [json.loads(chunk) for chunk in chunks if chunk != "[DONE]"]
    reasoning_deltas = [
        payload["choices"][0]["delta"].get("reasoning_content", "")
        for payload in payloads
        if payload.get("choices")
    ]

    assert any(reasoning_deltas)
    assert mock_llm.chat_stream.call_args.kwargs["enable_thinking"] is None

    logger.success("[PASS] handle_stream preserves default reasoning")


def test_handle_stream_thinking_mode_still_forces_reasoning():
    """Explicit thinking mode should still force-enable provider reasoning."""
    logger.note("=" * 60)
    logger.note("[TEST] handle_stream explicit thinking mode")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_stream.side_effect = lambda **_kwargs: make_stream_chunks(
        "这是最终答案。", reasoning="先深入分析。"
    )
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    list(
        handler.handle_stream(
            messages=[{"role": "user", "content": "央视新闻最近有什么新视频？"}],
            thinking=True,
        )
    )

    assert mock_llm.chat_stream.call_args.kwargs["enable_thinking"] is True

    logger.success("[PASS] handle_stream explicit thinking mode")


def test_forced_content_retry_does_not_disable_model_default_reasoning():
    """Forced content generation should not send enable_thinking=False in normal mode."""
    logger.note("=" * 60)
    logger.note("[TEST] forced content preserves default reasoning")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先搜索央视新闻最近视频。",
            "<search_videos queries='[\":user=央视新闻 :date<=15d\"]'/>",
        ),
        make_content_response("整理后给你：央视新闻近期视频如下。"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT
    handler = ChatHandler(
        llm_client=mock_llm,
        search_client=mock_search,
        max_iterations=1,
    )

    result = handler.handle(
        messages=[{"role": "user", "content": "央视新闻最近有什么新视频？"}],
        thinking=False,
    )

    assert assistant_content(result)
    assert mock_llm.chat.call_args_list[1].kwargs["enable_thinking"] is None

    logger.success("[PASS] forced content preserves default reasoning")


def test_relation_only_commands_are_promoted_to_search_videos():
    """Relation-only plans are no longer auto-promoted by regex-derived helper logic."""
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

    assert assistant_content(result)
    mock_search.related_owners_by_tokens.assert_called_once_with(
        text="黑神话悟空", size=8
    )
    mock_search.explore.assert_not_called()
    assert result["tool_events"][0]["tools"] == ["related_owners_by_tokens"]

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
    assert assistant_content(result) == "经过深入分析..."

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
    # Simulate max iterations being hit with distinct tool commands.
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "搜索",
            f"<search_videos queries='[\"q{i}\"]'/>",
        )
        for i in range(10)
    ] + [
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
        make_tool_cmd_response(
            "搜索",
            f"<search_videos queries='[\"q{i}\"]'/>",
        )
        for i in range(10)
    ] + [
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
    first_response = make_tool_cmd_response(
        "我来搜索何同学的视频。",
        '<search_owners text="何同学" mode="relation"/>\n'
        "<search_videos queries='[\":user=何同学 :date<=15d\"]'/>",
    )
    mock_llm.chat.side_effect = [
        first_response,
        make_content_response("我继续整理何同学最近的视频。"),
        make_content_response("何同学最近的视频..."),
    ]

    mock_search = MagicMock()
    mock_search.search_owners.side_effect = [
        MOCK_SEARCH_OWNERS_RESULT,
        MOCK_HE_TONGXUE_OWNER_RESULT,
    ]
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "何同学最近的视频"}])

    # tool_events should be returned in the response
    assert "tool_events" in result
    tool_events = result["tool_events"]
    assert len(tool_events) >= 1

    # First iteration: relation lookup + canonical name lookup
    assert tool_events[0]["iteration"] == 1
    assert "search_owners" in tool_events[0]["tools"]
    assert tool_events[0]["tools"] == ["search_owners", "search_owners"]
    assert tool_events[1]["tools"] == ["search_videos"]

    logger.success("[PASS] tool events tracking")


def test_relation_lookup_does_not_block_name_resolution_for_video_search():
    """Relation-mode owner lookups should not count as canonical name resolution for :user video searches."""
    logger.note("=" * 60)
    logger.note("[TEST] relation lookup does not block name resolution")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我先看下账号关系，再找代表作。",
            '<search_owners text="何同学" mode="relation"/>\n'
            "<search_videos queries='[\":user=何同学 代表作 q=vwr\"]'/>",
        ),
        make_content_response("我继续整理这位作者的代表作。"),
        make_content_response("这里是老师好我叫何同学的代表作。"),
    ]

    mock_search = MagicMock()
    mock_search.search_owners.side_effect = [
        MOCK_SEARCH_OWNERS_RESULT,
        MOCK_HE_TONGXUE_OWNER_RESULT,
    ]
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[
            {"role": "user", "content": "何同学有哪些关联账号？那他的代表作有哪些？"}
        ]
    )

    assert assistant_content(result)
    assert mock_search.search_owners.call_args_list == [
        call(text="何同学", mode="relation", size=8),
        call(text="何同学", mode="name", size=8),
    ]
    mock_search.explore.assert_called_once_with(query=":uid=163637592 代表作 q=vwr")
    assert result["tool_events"][0]["tools"] == ["search_owners", "search_owners"]
    assert result["tool_events"][1]["tools"] == ["search_videos"]

    logger.success("[PASS] relation lookup does not block name resolution")


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
        # 2. Model pauses after owner resolution; handler should continue the plan.
        make_content_response("我继续整理这两位作者最近的视频。"),
        # 3. Final content
        make_content_response("这是何同学和影视飓风最近的视频..."),
    ]

    mock_search = MagicMock()
    mock_search.related_owners_by_tokens.return_value = MOCK_RELATED_OWNERS_RESULT
    mock_search.search_owners.side_effect = [
        MOCK_HE_TONGXUE_OWNER_RESULT,
        {
            "text": "影视飓风",
            "mode": "name",
            "owners": [
                {
                    "mid": 946974,
                    "name": "影视飓风",
                    "score": 264.0,
                    "sample_view": 3785326,
                    "sources": ["name"],
                }
            ],
        },
    ]
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "何同学和影视飓风最近有什么新视频？"}]
    )

    content = result["choices"][0]["message"]["content"]
    assert "何同学" in content or "影视飓风" in content

    # Generic owner resolution inserts an extra owner-search pass before video search.
    assert mock_llm.chat.call_count == 3

    # Both related_owners_by_tokens calls should have been made
    assert mock_search.related_owners_by_tokens.call_count == 2

    # Both :user filters should be canonicalized through owner search first.
    assert mock_search.search_owners.call_count == 2

    # Search was called (from the multi-query)
    assert mock_search.explore.call_count >= 1

    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 20
    assert result["usage"]["total_tokens"] == 30

    logger.success("[PASS] parallel tool calls")


def test_multi_author_recent_videos_resolves_short_cjk_alias_before_search():
    """Multi-author recent-video queries should resolve short Chinese aliases like 何同学 before searching videos."""
    logger.note("=" * 60)
    logger.note("[TEST] multi author recent videos resolve short CJK alias")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我来分别搜索两位作者最近 15 天的视频。",
            '<search_videos queries=\'[":user=何同学 :date<=15d", ":user=影视飓风 :date<=15d"]\'/>',
        ),
        make_content_response("我继续整理这两位作者最近的视频。"),
        make_content_response("这是老师好我叫何同学和影视飓风最近的视频..."),
    ]

    mock_search = MagicMock()
    mock_search.search_owners.side_effect = [
        MOCK_HE_TONGXUE_OWNER_RESULT,
        {
            "text": "影视飓风",
            "mode": "name",
            "owners": [
                {
                    "mid": 946974,
                    "name": "影视飓风",
                    "score": 264.0,
                    "sample_view": 3785326,
                    "sources": ["name"],
                }
            ],
        },
    ]
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "何同学和影视飓风最近都发了哪些视频？"}]
    )

    assert assistant_content(result)
    assert mock_search.search_owners.call_args_list == [
        call(text="何同学", mode="name", size=8),
        call(text="影视飓风", mode="name", size=8),
    ]
    assert mock_search.explore.call_args_list == [
        call(query=":uid=163637592 :date<=15d"),
        call(query=":uid=946974 :date<=15d"),
    ]
    assert result["tool_events"][0]["tools"] == ["search_owners", "search_owners"]
    assert result["tool_events"][1]["tools"] == ["search_videos"]
    assert mock_llm.chat.call_count == 3

    logger.success("[PASS] multi author recent videos resolve short CJK alias")


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

    assert assistant_content(result) == "何同学相关账号候选如下。"
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
    """Without regex fallback injection, direct content is returned as-is when the model skips tools."""
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
        == "以下是我为你筛到的黑神话悟空剧情解析视频。"
    )
    mock_search.explore.assert_not_called()
    assert mock_llm.chat.call_count == 1

    logger.success("[PASS] explicit video request fallback search")


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

    assert assistant_content(result) == "找到了何同学相关的作者候选。"
    mock_search.related_owners_by_tokens.assert_called_once_with(text="何同学", size=8)
    mock_search.explore.assert_not_called()
    assert mock_llm.chat.call_count == 2

    logger.success("[PASS] relation-only account query skips search_videos promotion")


def test_direct_account_query_injects_relation_fallback_when_model_skips_tools():
    """Without regex fallback injection, direct account queries stay model-driven when no tools are emitted."""
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
    mock_search.search_owners.return_value = MOCK_SEARCH_OWNERS_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[
            {"role": "user", "content": "你有什么功能"},
            {"role": "assistant", "content": "我可以帮你搜索 B 站内容。"},
            {"role": "user", "content": "何同学有哪些关联账号？"},
        ]
    )

    assert assistant_content(result) == "我来帮你查一下。"
    mock_search.search_owners.assert_not_called()
    mock_search.explore.assert_not_called()
    assert mock_llm.chat.call_count == 1

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

    assert assistant_content(result) == "补充找到了何同学的其他关联作者候选。"
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

    assert assistant_content(result) == "找到了影视飓风相关的账号候选。"
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
