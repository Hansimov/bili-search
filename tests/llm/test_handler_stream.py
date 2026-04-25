"""Streaming tests for the ChatHandler orchestration layer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from llms.chat.handler import ChatHandler
from llms.models import LLMClient
from handler_test_utils import MOCK_EXPLORE_RESULT, make_stream_chunk


def test_handle_stream_emits_tool_events_content_and_done():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_stream.side_effect = [
        iter(
            [
                make_stream_chunk(
                    delta={"reasoning_content": "我来搜索黑神话相关视频。"}
                ),
                make_stream_chunk(
                    delta={"content": "<search_videos queries='[\"黑神话\"]'/>"},
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                ),
            ]
        ),
        iter(
            [
                make_stream_chunk(delta={"reasoning_content": "根据结果整理答案。"}),
                make_stream_chunk(
                    delta={"content": "找到了黑神话相关视频。"},
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                ),
            ]
        ),
    ]
    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = list(handler.handle_stream(messages=[{"role": "user", "content": "test"}]))

    assert chunks[0] != "[DONE]"
    assert chunks[-1] == "[DONE]"
    parsed_chunks = [json.loads(chunk) for chunk in chunks[:-1]]
    assert parsed_chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    tool_event_chunks = [chunk for chunk in parsed_chunks if chunk.get("tool_events")]
    assert len(tool_event_chunks) == 2
    assert tool_event_chunks[0]["tool_events"][0]["calls"][0]["status"] == "pending"
    assert tool_event_chunks[1]["tool_events"][0]["calls"][0]["status"] == "completed"
    assert any(
        chunk["choices"][0]["delta"].get("reasoning_content") for chunk in parsed_chunks
    )
    streamed_content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in parsed_chunks
    )
    assert "<search_videos" not in streamed_content
    assert "找到了黑神话相关视频。" in streamed_content
    assert parsed_chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_handle_stream_emits_streaming_internal_small_tool_updates():
    mock_large_llm = MagicMock(spec=LLMClient)
    mock_large_llm.chat_stream.side_effect = [
        iter(
            [
                make_stream_chunk(
                    delta={
                        "content": "<run_small_llm_task task='把候选结果压成 2 条要点' context='候选结果很多'/>"
                    },
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 12,
                        "completion_tokens": 6,
                        "total_tokens": 18,
                    },
                ),
            ]
        ),
        iter(
            [
                make_stream_chunk(
                    delta={"content": "最终回答如下。"},
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 8,
                        "completion_tokens": 4,
                        "total_tokens": 12,
                    },
                )
            ]
        ),
    ]
    mock_small_llm = MagicMock(spec=LLMClient)
    mock_small_llm.chat_stream.return_value = iter(
        [
            make_stream_chunk(delta={"content": "- 要点1"}),
            make_stream_chunk(
                delta={"content": "\n- 要点2"},
                finish_reason="stop",
            ),
        ]
    )
    mock_search = MagicMock()

    handler = ChatHandler(
        llm_client=mock_large_llm,
        small_llm_client=mock_small_llm,
        search_client=mock_search,
    )

    chunks = list(
        handler.handle_stream(
            messages=[{"role": "user", "content": "请总结一下"}],
            thinking=True,
        )
    )
    parsed_chunks = [json.loads(chunk) for chunk in chunks[:-1]]

    tool_event_chunks = [chunk for chunk in parsed_chunks if chunk.get("tool_events")]
    statuses = [
        chunk["tool_events"][0]["calls"][0]["status"] for chunk in tool_event_chunks
    ]

    assert statuses == [
        "pending",
        "streaming",
        "streaming",
        "streaming",
        "completed",
    ]
    assert tool_event_chunks[1]["tool_events"][0]["calls"][0]["result"]["result"] == ""
    placeholder_model_name = tool_event_chunks[1]["tool_events"][0]["calls"][0][
        "result"
    ]["model_name"]
    assert isinstance(placeholder_model_name, str)
    assert placeholder_model_name
    assert (
        placeholder_model_name
        == tool_event_chunks[-1]["tool_events"][0]["calls"][0]["result"]["model_name"]
    )
    assert (
        tool_event_chunks[2]["tool_events"][0]["calls"][0]["result"]["result"]
        == "- 要点1"
    )
    assert (
        tool_event_chunks[-1]["tool_events"][0]["calls"][0]["result"]["result"]
        == "- 要点1\n- 要点2"
    )
    assert (
        tool_event_chunks[-1]["tool_events"][0]["calls"][0]["visibility"] == "internal"
    )
    assert mock_small_llm.chat_stream.call_count == 1
    assert mock_small_llm.chat_stream.call_args.kwargs["enable_thinking"] is False


def test_handle_stream_retracts_planning_content_into_thinking():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_stream.side_effect = [
        iter(
            [
                make_stream_chunk(delta={"content": "我先查一下黑神话相关结果。"}),
                make_stream_chunk(
                    delta={"content": "<search_videos queries='[\"黑神话\"]'/>"},
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                ),
            ]
        ),
        iter(
            [
                make_stream_chunk(
                    delta={"content": "找到了相关视频，可以先看 BV1abc。"},
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                )
            ]
        ),
    ]
    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = list(handler.handle_stream(messages=[{"role": "user", "content": "test"}]))
    parsed_chunks = [json.loads(chunk) for chunk in chunks[:-1]]

    assert any(
        chunk["choices"][0]["delta"].get("retract_content") for chunk in parsed_chunks
    )
    streamed_reasoning = "".join(
        chunk["choices"][0]["delta"].get("reasoning_content", "")
        for chunk in parsed_chunks
    )
    assert "我先查一下黑神话相关结果。" in streamed_reasoning

    effective_content = ""
    for chunk in parsed_chunks:
        delta = chunk["choices"][0]["delta"]
        if delta.get("retract_content"):
            effective_content = ""
            continue
        effective_content += delta.get("content", "")
    assert "我先查一下黑神话相关结果。" not in effective_content
    assert effective_content.endswith("找到了相关视频，可以先看 BV1abc。")


def test_handle_stream_replays_postprocessed_final_content_when_it_changes():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_stream.return_value = iter(
        [
            make_stream_chunk(
                delta={"content": "这是正文。"},
                finish_reason="stop",
                usage={
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                },
            )
        ]
    )
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    with patch.object(
        ChatHandler,
        "_ensure_response_context",
        return_value="主题：\n这是正文。",
    ):
        chunks = list(
            handler.handle_stream(messages=[{"role": "user", "content": "test"}])
        )

    parsed_chunks = [json.loads(chunk) for chunk in chunks[:-1]]
    assert any(
        chunk["choices"][0]["delta"].get("retract_content") for chunk in parsed_chunks
    )

    effective_content = ""
    for chunk in parsed_chunks:
        delta = chunk["choices"][0]["delta"]
        if delta.get("retract_content"):
            effective_content = ""
            continue
        effective_content += delta.get("content", "")

    assert effective_content.endswith("主题：\n这是正文。")


def test_handle_stream_emits_reasoning_reset_between_orchestration_phases():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_stream.side_effect = [
        iter(
            [
                make_stream_chunk(delta={"reasoning_content": "先读取转写。"}),
                make_stream_chunk(
                    delta={
                        "content": "<get_video_transcript video_id='BV1R2XZBQEio'/>"
                    },
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                ),
            ]
        ),
        iter(
            [
                make_stream_chunk(
                    delta={"reasoning_content": "根据压缩结果整理回答。"}
                ),
                make_stream_chunk(
                    delta={"content": "这是最终回答。"},
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 12,
                        "completion_tokens": 6,
                        "total_tokens": 18,
                    },
                ),
            ]
        ),
    ]
    mock_search = MagicMock()
    mock_search.capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": False,
        "supports_google_search": False,
        "supports_transcript_lookup": True,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }
    mock_search.get_video_transcript.return_value = {
        "bvid": "BV1R2XZBQEio",
        "title": "示例视频",
        "selection": {
            "selected_text_length": 24,
            "full_text_length": 24,
        },
        "transcript": {
            "text": "这是完整转写。",
            "text_length": 24,
            "segment_count": 1,
        },
    }
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    chunks = list(
        handler.handle_stream(
            messages=[{"role": "user", "content": "BV1R2XZBQEio 这期视频讲了什么"}]
        )
    )
    parsed_chunks = [json.loads(chunk) for chunk in chunks[:-1]]

    reset_events = [
        chunk["choices"][0]["delta"]
        for chunk in parsed_chunks
        if chunk["choices"][0]["delta"].get("reset_reasoning")
    ]

    assert [
        (event["reasoning_phase"], event["reasoning_iteration"])
        for event in reset_events
    ] == [
        ("planner", 1),
        ("planner", 2),
    ]
