"""Tests for llms.chat.handler — chat handler with tool-calling loop.

Uses mocked LLM client and search client for unit tests.

Run:
    python -m tests.llm.test_handler
"""

import json
from unittest.mock import MagicMock, patch, call
from tclogger import logger

from llms.llm_client import LLMClient, ChatResponse, ToolCall
from llms.search_service import SearchServiceClient
from llms.chat.handler import ChatHandler


# ============================================================
# Mock helpers
# ============================================================


def make_content_response(content: str) -> ChatResponse:
    """Build a ChatResponse with content."""
    return ChatResponse(
        content=content,
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def make_tool_call_response(
    name: str, arguments: dict, call_id: str = "call_1"
) -> ChatResponse:
    """Build a ChatResponse with a tool call."""
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
        usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
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


# ============================================================
# Tests
# ============================================================


def test_direct_content_response():
    """Test handler when LLM returns content directly (no tool calls)."""
    logger.note("=" * 60)
    logger.note("[TEST] direct content response")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response("你好！我是AI助手。")

    mock_search = MagicMock(spec=SearchServiceClient)

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
        make_tool_call_response("search_videos", {"query": "黑神话"}),
        # Second call: content response
        make_content_response("找到了10个黑神话相关视频。"),
    ]

    mock_search = MagicMock(spec=SearchServiceClient)
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
            "search_videos", {"query": ":user=影视飓风 :date<=7d"}, "call_2"
        ),
        # 3. Final content
        make_content_response("影视飓风最近7天发布了以下视频..."),
    ]

    mock_search = MagicMock(spec=SearchServiceClient)
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


def test_max_iterations():
    """Test that handler stops at max iterations."""
    logger.note("=" * 60)
    logger.note("[TEST] max iterations")

    mock_llm = MagicMock(spec=LLMClient)
    # Always return tool calls (infinite loop)
    mock_llm.chat.return_value = make_tool_call_response(
        "search_videos", {"query": "test"}
    )

    mock_search = MagicMock(spec=SearchServiceClient)
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(
        llm_client=mock_llm, search_client=mock_search, max_iterations=3
    )

    result = handler.handle(messages=[{"role": "user", "content": "test"}])

    # Should have stopped after max_iterations
    assert mock_llm.chat.call_count == 3
    # Should return a timeout message
    content = result["choices"][0]["message"]["content"]
    assert "超时" in content or "重试" in content

    logger.success("[PASS] max iterations")


def test_streaming_response():
    """Test streaming mode returns proper SSE chunks."""
    logger.note("=" * 60)
    logger.note("[TEST] streaming response")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response("你好世界！")

    mock_search = MagicMock(spec=SearchServiceClient)

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = list(handler.handle_stream(messages=[{"role": "user", "content": "你好"}]))

    # Should have multiple chunks
    assert len(chunks) > 2  # At least: role chunk + content chunks + done

    # First chunk should have role
    first = json.loads(chunks[0])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"

    # Last chunk should be [DONE]
    assert chunks[-1] == "[DONE]"

    # Second-to-last should have finish_reason=stop
    last_data = json.loads(chunks[-2])
    assert last_data["choices"][0]["finish_reason"] == "stop"

    # Reconstruct content from chunks
    content = ""
    for chunk_str in chunks[:-1]:  # Skip [DONE]
        chunk = json.loads(chunk_str)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            content += delta["content"]
    assert content == "你好世界！"

    logger.success("[PASS] streaming response")


def test_streaming_with_tools():
    """Test streaming mode with tool calls."""
    logger.note("=" * 60)
    logger.note("[TEST] streaming with tools")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_call_response("search_videos", {"query": "test"}),
        make_content_response("找到了结果。"),
    ]

    mock_search = MagicMock(spec=SearchServiceClient)
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = list(handler.handle_stream(messages=[{"role": "user", "content": "test"}]))

    # Content should be in the chunks
    content = ""
    for chunk_str in chunks[:-1]:
        chunk = json.loads(chunk_str)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            content += delta["content"]
    assert content == "找到了结果。"

    logger.success("[PASS] streaming with tools")


def test_system_prompt_included():
    """Test that system prompt is prepended to messages."""
    logger.note("=" * 60)
    logger.note("[TEST] system prompt included")

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response("OK")

    mock_search = MagicMock(spec=SearchServiceClient)

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

    mock_search = MagicMock(spec=SearchServiceClient)

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

    mock_search = MagicMock(spec=SearchServiceClient)

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


if __name__ == "__main__":
    tests = [
        ("direct_content_response", test_direct_content_response),
        ("single_tool_call", test_single_tool_call),
        ("multi_tool_calls", test_multi_tool_calls),
        ("max_iterations", test_max_iterations),
        ("streaming_response", test_streaming_response),
        ("streaming_with_tools", test_streaming_with_tools),
        ("system_prompt_included", test_system_prompt_included),
        ("response_format", test_response_format),
        ("multi_turn_conversation", test_multi_turn_conversation),
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
