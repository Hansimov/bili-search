"""Transcript-focused ChatHandler orchestration tests."""

from unittest.mock import MagicMock

from llms.chat.handler import ChatHandler
from llms.contracts import IntentProfile, ToolCallRequest, ToolExecutionRecord
from llms.models import LLMClient, ToolCall
from llms.orchestration.result_store import ResultStore, summarize_result
from handler_test_utils import (
    assistant_content,
    make_content_response,
    make_function_call_response,
)


def test_normalize_request_canonicalizes_verbose_transcript_small_task():
    handler = ChatHandler(
        llm_client=MagicMock(spec=LLMClient),
        small_llm_client=MagicMock(spec=LLMClient),
        search_client=MagicMock(),
    )

    request = ToolCallRequest(
        id="call_small_tx_1",
        name="run_small_llm_task",
        arguments={
            "task": "把BV1uPDTBhEHX的视频转写整理成主题概括和覆盖全片的中文要点总结，要求结构清晰，覆盖全部核心内容",
            "context": "视频转写的chars=12000/39901, segments=37，开头提到1999年北约轰炸中国驻南联盟大使馆，团队自驾到塞尔维亚探寻事件相关故事，包含当地历史、饮食、现状等内容",
            "result_ids": ["R1"],
            "output_format": "主题概括+分点中文要点",
        },
        visibility="internal",
        source="xml",
    )

    normalized = handler.orchestrator._normalize_request(
        request,
        IntentProfile(
            raw_query="BV1uPDTBhEHX 请读取转写后做主题概括和中文要点总结。",
            normalized_query="BV1uPDTBhEHX 请读取转写后做主题概括和中文要点总结。",
            final_target="videos",
            task_mode="known_item",
        ),
        {"supports_transcript_lookup": True},
    )

    assert normalized.visibility == "internal"
    assert normalized.arguments == {
        "task": "整理转写：主题概括 + 覆盖全片要点",
        "result_ids": ["R1"],
        "output_format": "主题概括+中文要点",
    }


def test_handle_transcript_summary_flow_canonicalizes_small_task_request_and_finalizes_answer():
    mock_large_llm = MagicMock(spec=LLMClient)
    mock_large_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_tx_1",
                name="get_video_transcript",
                arguments={
                    "video_id": "BV1uPDTBhEHX",
                    "head_chars": 12000,
                    "include_segments": False,
                },
            )
        ),
        make_function_call_response(
            ToolCall(
                id="call_small_tx_1",
                name="run_small_llm_task",
                arguments={
                    "task": "把BV1uPDTBhEHX的视频转写整理成主题概括和覆盖全片的中文要点总结，要求结构清晰，覆盖全部核心内容",
                    "context": "视频转写的chars=12000/39901, segments=37，开头提到1999年北约轰炸中国驻南联盟大使馆，团队自驾到塞尔维亚探寻事件相关故事，包含当地历史、饮食、现状等内容",
                    "result_ids": ["R1"],
                    "output_format": "主题概括+分点中文要点",
                },
            )
        ),
        make_content_response("这是最终回答。"),
    ]
    mock_small_llm = MagicMock(spec=LLMClient)
    mock_small_llm.chat.return_value = make_content_response(
        "主题概括\n- 要点1\n- 要点2"
    )
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
        "bvid": "BV1uPDTBhEHX",
        "title": "美国轰炸伊朗的剧本，27年前就写好了",
        "selection": {
            "selected_text_length": 12000,
            "full_text_length": 39901,
        },
        "transcript": {
            "text": "当地时间一九九九年五月七日夜，北约轰炸了中国驻南联盟大使馆。",
            "text_length": 12000,
            "segment_count": 37,
        },
    }
    mock_search.lookup_videos.return_value = {
        "lookup_by": "bvids",
        "hits": [
            {
                "bvid": "BV1uPDTBhEHX",
                "title": "美国轰炸伊朗的剧本，27年前就写好了",
                "desc": "食贫道团队自驾到塞尔维亚，追溯 1999 年使馆被炸事件，并结合当地历史和现状展开探访。",
                "tags": "历史,塞尔维亚,食贫道",
                "pubdate": 1708700000,
                "owner": {"mid": 642389251, "name": "食贫道"},
            }
        ],
        "total_hits": 1,
    }

    handler = ChatHandler(
        llm_client=mock_large_llm,
        small_llm_client=mock_small_llm,
        search_client=mock_search,
    )

    result = handler.handle(
        messages=[
            {
                "role": "user",
                "content": "BV1uPDTBhEHX 请读取转写后做主题概括和中文要点总结。",
            }
        ],
        thinking=True,
    )

    assert assistant_content(result) == "这是最终回答。"
    assert len(result["tool_events"]) == 2

    transcript_call = result["tool_events"][0]["calls"][0]
    assert transcript_call["type"] == "get_video_transcript"

    small_task_call = result["tool_events"][1]["calls"][0]
    assert small_task_call["type"] == "run_small_llm_task"
    assert small_task_call["visibility"] == "internal"
    assert small_task_call["args"] == {
        "task": "整理转写：主题概括 + 覆盖全片要点",
        "result_ids": ["R1"],
        "output_format": "主题概括+中文要点",
    }
    assert small_task_call["result"]["task"] == "整理转写：主题概括 + 覆盖全片要点"

    small_task_prompt = "\n".join(
        str(message["content"])
        for message in mock_small_llm.chat.call_args.kwargs["messages"]
    )
    assert "标题: 美国轰炸伊朗的剧本，27年前就写好了" in small_task_prompt
    assert "作者: 食贫道" in small_task_prompt
    assert "标签: 历史、塞尔维亚、食贫道" in small_task_prompt
    assert (
        "简介: 食贫道团队自驾到塞尔维亚，追溯 1999 年使馆被炸事件" in small_task_prompt
    )
    assert "输出格式: 主题概括+中文要点" in small_task_prompt


def test_small_task_messages_enrich_transcript_context_with_video_metadata():
    mock_large_llm = MagicMock(spec=LLMClient)
    mock_small_llm = MagicMock(spec=LLMClient)
    mock_search = MagicMock()
    mock_search.lookup_videos.return_value = {
        "lookup_by": "bvids",
        "hits": [
            {
                "bvid": "BV1R2XZBQEio",
                "title": "lookup 标题",
                "desc": "这是一段更完整的视频简介，包含背景和环节。",
                "tags": "数码,AI,实测",
                "pubdate": 1708700000,
                "owner": {"mid": 946974, "name": "影视飓风"},
            }
        ],
        "total_hits": 1,
    }

    handler = ChatHandler(
        llm_client=mock_large_llm,
        small_llm_client=mock_small_llm,
        search_client=mock_search,
    )

    transcript_result = {
        "bvid": "BV1R2XZBQEio",
        "title": "转写返回标题",
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
    request = ToolCallRequest(
        id="call_tx_1",
        name="get_video_transcript",
        arguments={"video_id": "BV1R2XZBQEio", "head_chars": 6000},
    )
    record = ToolExecutionRecord(
        result_id="R1",
        request=request,
        result=transcript_result,
        summary=summarize_result("R1", request.name, transcript_result),
    )
    result_store = ResultStore()
    result_store.add(record)

    _, _, messages = handler.orchestrator._build_small_task_messages(
        result_store,
        {"task": "整理转写", "result_ids": ["R1"]},
        IntentProfile(
            raw_query="BV1R2XZBQEio 这期视频讲了什么",
            normalized_query="BV1R2XZBQEio 这期视频讲了什么",
            final_target="videos",
            task_mode="known_item",
        ),
    )

    prompt_text = "\n".join(str(message["content"]) for message in messages)

    assert "标题: 转写返回标题" in prompt_text
    assert "作者: 影视飓风" in prompt_text
    assert "发布时间:" in prompt_text
    assert "标签: 数码、AI、实测" in prompt_text
    assert "简介: 这是一段更完整的视频简介，包含背景和环节。" in prompt_text
    assert "标题、标签、简介、作者、发布时间" in prompt_text
    assert "4 到 8 条" not in prompt_text
    mock_search.lookup_videos.assert_called_once_with(
        bvids=["BV1R2XZBQEio"],
        limit=1,
        verbose=False,
    )


def test_transcript_post_execution_nudge_uses_unbounded_summary_instruction():
    handler = ChatHandler(
        llm_client=MagicMock(spec=LLMClient),
        small_llm_client=MagicMock(spec=LLMClient),
        search_client=MagicMock(),
    )
    transcript_result = {
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
    result_store = ResultStore()
    result_store.add(
        ToolExecutionRecord(
            result_id="R1",
            request=ToolCallRequest(
                id="call_tx_1",
                name="get_video_transcript",
                arguments={"video_id": "BV1R2XZBQEio"},
            ),
            result=transcript_result,
            summary=summarize_result("R1", "get_video_transcript", transcript_result),
        )
    )

    rule = handler.orchestrator._select_transcript_post_execution_nudge(
        result_store,
        prefer_transcript_lookup=True,
        prompted_nudges=set(),
    )

    assert rule is not None
    assert rule[0] == "transcript_result_should_be_compressed"
    assert "主题概括和覆盖全片的中文要点" in rule[1]
    assert "4 到 8 条" not in rule[1]
