"""Tests for llms.chat.handler — chat handler with tool-calling loop.

Uses mocked LLM client and search client for unit tests.

Run:
    python -m tests.llm.test_handler
"""

import json
from unittest.mock import MagicMock, patch, call
from tclogger import logger

from llms.llm_client import LLMClient, ChatResponse, ToolCall
from llms.chat.handler import ChatHandler


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


def make_tool_call_response(
    name: str, arguments: dict, call_id: str = "call_1", extra_usage: dict = None
) -> ChatResponse:
    """Build a ChatResponse with a single tool call."""
    usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    if extra_usage:
        usage.update(extra_usage)
    return ChatResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            )
        ],
        finish_reason="tool_calls",
        usage=usage,
    )


def make_multi_tool_call_response(
    calls: list[tuple[str, dict, str]], extra_usage: dict = None
) -> ChatResponse:
    """Build a ChatResponse with multiple parallel tool calls.

    Args:
        calls: List of (name, arguments, call_id) tuples.
    """
    usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    if extra_usage:
        usage.update(extra_usage)
    return ChatResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=json.dumps(args, ensure_ascii=False),
            )
            for name, args, call_id in calls
        ],
        finish_reason="tool_calls",
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

MOCK_SUGGEST_RESULT = {
    "query": "影视飓风",
    "total_hits": 25,
    "hits": [
        {
            "bvid": "BV1x",
            "title": "影视飓风测评",
            "owner": {"mid": 946974, "name": "影视飓风"},
            "highlights": {"merged": ["<em>影视飓风</em>测评"]},
        },
        {
            "bvid": "BV1y",
            "title": "影视飓风年度",
            "owner": {"mid": 946974, "name": "影视飓风"},
            "highlights": {"merged": ["<em>影视飓风</em>年度"]},
        },
    ],
}


# --- Streaming mock helpers ---


def make_content_stream_chunks(content: str, extra_usage: dict = None) -> list[dict]:
    """Build a sequence of stream chunks for content response."""
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    if extra_usage:
        usage.update(extra_usage)

    chunks = []
    # Stream content 2 chars at a time
    for i in range(0, len(content), 2):
        text = content[i : i + 2]
        chunks.append(
            {
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ]
            }
        )
    # Final chunk with finish_reason and usage
    chunks.append(
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": usage,
        }
    )
    return chunks


def make_tool_call_stream_chunks(
    name: str,
    arguments: dict,
    call_id: str = "call_1",
    extra_usage: dict = None,
) -> list[dict]:
    """Build stream chunks for a single tool call response."""
    usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    if extra_usage:
        usage.update(extra_usage)
    args_str = json.dumps(arguments, ensure_ascii=False)
    chunks = [
        # Tool call start: id + function name
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "function": {"name": name, "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        # Tool call arguments
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": args_str}}
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        # Final chunk with finish_reason and usage
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": usage,
        },
    ]
    return chunks


def make_reasoning_stream_chunks(
    reasoning: str, content: str, extra_usage: dict = None
) -> list[dict]:
    """Build stream chunks with reasoning_content followed by content."""
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    if extra_usage:
        usage.update(extra_usage)

    chunks = []
    # Reasoning chunks
    for i in range(0, len(reasoning), 2):
        text = reasoning[i : i + 2]
        chunks.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": text},
                        "finish_reason": None,
                    }
                ]
            }
        )
    # Content chunks
    for i in range(0, len(content), 2):
        text = content[i : i + 2]
        chunks.append(
            {
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ]
            }
        )
    # Final chunk
    chunks.append(
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": usage,
        }
    )
    return chunks


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


def test_single_tool_call():
    """Test handler with one tool call → content response."""
    logger.note("=" * 60)
    logger.note("[TEST] single tool call")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        # First call: tool call
        make_tool_call_response("search_videos", {"queries": ["黑神话"]}),
        # Second call: content response
        make_content_response("找到了10个黑神话相关视频。"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "搜索黑神话"}])

    assert result["choices"][0]["message"]["content"] == "找到了10个黑神话相关视频。"

    # LLM called twice (tool call + content)
    assert mock_llm.chat.call_count == 2

    # Search client was called
    mock_search.explore.assert_called_once_with(query="黑神话")

    # Verify tool result was fed back to LLM
    second_call_messages = (
        mock_llm.chat.call_args_list[1].kwargs.get("messages")
        or mock_llm.chat.call_args_list[1][1].get("messages")
        if len(mock_llm.chat.call_args_list[1]) > 1
        else mock_llm.chat.call_args_list[1][0][0]
    )
    # Find tool message in the conversation
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"

    logger.success("[PASS] single tool call")


def test_multi_tool_calls():
    """Test handler with check_author → search_videos → content."""
    logger.note("=" * 60)
    logger.note("[TEST] multi tool calls")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        # 1. Check author
        make_tool_call_response("check_author", {"name": "影视飓风"}, "call_1"),
        # 2. Search videos
        make_tool_call_response(
            "search_videos", {"queries": [":user=影视飓风 :date<=7d"]}, "call_2"
        ),
        # 3. Final content
        make_content_response("影视飓风最近7天发布了以下视频..."),
    ]

    mock_search = MagicMock()
    mock_search.suggest.return_value = MOCK_SUGGEST_RESULT
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "影视飓风最近有什么新视频？"}]
    )

    content = result["choices"][0]["message"]["content"]
    assert "影视飓风" in content

    # LLM called 3 times
    assert mock_llm.chat.call_count == 3

    # Both tools were called
    mock_search.suggest.assert_called_once()
    mock_search.explore.assert_called_once()

    # Verify usage accumulation
    assert result["usage"]["total_tokens"] == 15 + 30 + 30  # content + 2 tool calls

    logger.success("[PASS] multi tool calls")


def test_cache_token_accumulation():
    """Test that cache hit/miss tokens are accumulated in usage."""
    logger.note("=" * 60)
    logger.note("[TEST] cache token accumulation")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_call_response(
            "search_videos",
            {"queries": ["test"]},
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
    assert usage["prompt_cache_hit_tokens"] == 300  # 100 + 200
    assert usage["prompt_cache_miss_tokens"] == 80  # 50 + 30
    assert usage["prompt_tokens"] == 30  # 20 + 10
    assert usage["completion_tokens"] == 15  # 10 + 5

    logger.success(f"  Cache hit: {usage['prompt_cache_hit_tokens']}")
    logger.success(f"  Cache miss: {usage['prompt_cache_miss_tokens']}")
    logger.success("[PASS] cache token accumulation")


def test_max_iterations():
    """Test that handler stops at max iterations and forces content."""
    logger.note("=" * 60)
    logger.note("[TEST] max iterations")

    mock_llm = MagicMock(spec=LLMClient)
    # First 3 calls: always tool calls (simulates infinite loop)
    # 4th call: forced content response (tools=None)
    mock_llm.chat.side_effect = [
        make_tool_call_response("search_videos", {"queries": ["test"]}),
        make_tool_call_response("search_videos", {"queries": ["test"]}),
        make_tool_call_response("search_videos", {"queries": ["test"]}),
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
    # The forced call should have tools=None
    last_call = mock_llm.chat.call_args_list[-1]
    assert last_call.kwargs.get("tools") is None
    # Should have real content, not a timeout
    content = result["choices"][0]["message"]["content"]
    assert "搜索结果" in content

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
        make_tool_call_response("search_videos", {"queries": ["test"]}),
        make_tool_call_response("search_videos", {"queries": ["test"]}),
        make_tool_call_response("search_videos", {"queries": ["test"]}),
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
    mock_llm.chat.return_value = make_content_response("你好世界！")

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
    # perf_stats should NOT contain token counts (they're in usage)
    assert "completion_tokens" not in last_data["perf_stats"]
    assert "total_tokens" not in last_data["perf_stats"]
    # tokens_per_second should be int
    assert isinstance(last_data["perf_stats"]["tokens_per_second"], int)

    # Reconstruct content from chunks
    content = ""
    for chunk_str in chunks[:-1]:  # Skip [DONE]
        chunk = json.loads(chunk_str)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            content += delta["content"]
    assert content == "你好世界！"

    # chat should be called (not chat_stream)
    assert mock_llm.chat.call_count == 1
    assert mock_llm.chat_stream.call_count == 0

    logger.success("[PASS] streaming response")


def test_streaming_with_tools():
    """Test streaming mode with tool calls using chat()."""
    logger.note("=" * 60)
    logger.note("[TEST] streaming with tools")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_call_response("search_videos", {"queries": ["test"]}),
        make_content_response("找到了结果。"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = list(handler.handle_stream(messages=[{"role": "user", "content": "test"}]))

    # Content should be in the chunks
    content = ""
    tool_event_found = False
    for chunk_str in chunks[:-1]:
        chunk = json.loads(chunk_str)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            content += delta["content"]
        if chunk.get("tool_events"):
            tool_event_found = True
            assert chunk["tool_events"][0]["tools"] == ["search_videos"]
    assert content == "找到了结果。"
    assert tool_event_found, "Tool event chunk should be present"

    # chat called twice (tool iteration + content iteration)
    assert mock_llm.chat.call_count == 2

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

    logger.success("[PASS] response format")


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
    # Simulate max iterations being hit (all tool calls, never content)
    mock_llm.chat.side_effect = [
        make_tool_call_response("search_videos", {"queries": ["q"]}),
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
        make_tool_call_response("search_videos", {"queries": ["q"]}),
    ] * 10 + [
        make_content_response("思考后的最终结果"),
    ]

    # Thinking: max 10 iterations (default thinking max)
    result_thinking = handler.handle(
        messages=[{"role": "user", "content": "test"}],
        thinking=True,
    )
    thinking_calls = mock_llm.chat.call_count
    assert thinking_calls == 11  # 10 tool iterations + 1 forced

    logger.success("[PASS] thinking mode max iterations")


def test_explicit_max_iterations_override():
    """Test per-request max_iterations override."""
    logger.note("=" * 60)
    logger.note("[TEST] explicit max iterations override")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_call_response("search_videos", {"queries": ["q"]}),
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
        # Iteration 1: parallel tool calls
        make_multi_tool_call_response(
            [
                ("check_author", {"name": "何同学"}, "call_1"),
                ("search_videos", {"queries": ["何同学"]}, "call_2"),
            ]
        ),
        # Iteration 2: another search
        make_tool_call_response(
            "search_videos", {"queries": ["何同学 :date<=7d"]}, "call_3"
        ),
        # Final content
        make_content_response("何同学最近的视频..."),
    ]

    mock_search = MagicMock()
    mock_search.suggest.return_value = MOCK_SUGGEST_RESULT
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "何同学最近的视频"}])

    # tool_events should be returned in the response
    assert "tool_events" in result
    tool_events = result["tool_events"]
    assert len(tool_events) == 2

    # First iteration: parallel calls
    assert tool_events[0]["iteration"] == 1
    assert "check_author" in tool_events[0]["tools"]
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
    mock_llm.chat.side_effect = [
        make_tool_call_response("search_videos", {"queries": ["test"]}),
        make_content_response("思考后回答"),
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

    # Find tool event chunk (separate from first chunk now)
    tool_event_chunks = [json.loads(c) for c in chunks[:-1] if "tool_events" in c]
    assert len(tool_event_chunks) == 1
    assert len(tool_event_chunks[0]["tool_events"]) == 1

    # Last real chunk should have finish_reason and stats
    done_chunk = json.loads(chunks[-2])  # -1 is [DONE]
    assert done_chunk["choices"][0]["finish_reason"] == "stop"
    assert "perf_stats" in done_chunk
    assert "usage" in done_chunk

    # Last element is [DONE]
    assert chunks[-1] == "[DONE]"

    logger.success("[PASS] streaming with thinking")


def test_parallel_tool_calls():
    """Test handler with parallel tool calls (multiple tools in one response)."""
    logger.note("=" * 60)
    logger.note("[TEST] parallel tool calls")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        # 1. LLM calls both check_author in parallel
        make_multi_tool_call_response(
            [
                ("check_author", {"name": "何同学"}, "call_p1"),
                ("check_author", {"name": "影视飓风"}, "call_p2"),
            ]
        ),
        # 2. Search with multi-query
        make_tool_call_response(
            "search_videos",
            {"queries": [":user=何同学 :date<=15d", ":user=影视飓风 :date<=15d"]},
            "call_p3",
        ),
        # 3. Final content
        make_content_response("这是何同学和影视飓风最近的视频..."),
    ]

    mock_search = MagicMock()
    mock_search.suggest.return_value = MOCK_SUGGEST_RESULT
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "何同学和影视飓风最近有什么新视频？"}]
    )

    content = result["choices"][0]["message"]["content"]
    assert "何同学" in content or "影视飓风" in content

    # LLM called 3 times
    assert mock_llm.chat.call_count == 3

    # Both check_author calls should have been made (parallel)
    assert mock_search.suggest.call_count == 2

    # Search was called (from the multi-query call)
    assert mock_search.explore.call_count >= 1

    # Verify usage accumulated from all 3 calls (30 + 30 + 15 = 75)
    assert result["usage"]["total_tokens"] == 75

    logger.success("[PASS] parallel tool calls")


def test_multi_query_search():
    """Test handler with multi-query search_videos (queries array)."""
    logger.note("=" * 60)
    logger.note("[TEST] multi-query search")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        # Multi-query search
        make_tool_call_response(
            "search_videos",
            {"queries": ["黑神话 :view>=1w", "原神 :view>=1w"]},
            "call_mq1",
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


def test_streaming_reasoning_content():
    """Test streaming with reasoning_content (DeepSeek chain-of-thought)."""
    logger.note("=" * 60)
    logger.note("[TEST] streaming reasoning content")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = ChatResponse(
        content="答案是42。",
        reasoning_content="让我思考一下这个问题...",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
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
        ("multi_turn_conversation", test_multi_turn_conversation),
        ("thinking_mode", test_thinking_mode),
        ("thinking_mode_max_iterations", test_thinking_mode_max_iterations),
        ("explicit_max_iterations_override", test_explicit_max_iterations_override),
        ("tool_events_tracking", test_tool_events_tracking),
        ("streaming_with_thinking", test_streaming_with_thinking),
        ("perf_stats_no_overlap_with_usage", test_perf_stats_no_overlap_with_usage),
        ("usage_normalization_gpt_nested", test_usage_normalization_gpt_nested),
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
