"""Tests for the rewritten llms.chat.handler orchestration layer."""

from __future__ import annotations

from types import SimpleNamespace

from unittest.mock import MagicMock, patch

from llms.chat.handler import ChatHandler
from llms.contracts import IntentProfile, ToolCallRequest, ToolExecutionRecord
from llms.models import ChatResponse, LLMClient, ToolCall
from llms.orchestration.result_store import ResultStore, summarize_result
from handler_test_utils import (
    MOCK_EXPLORE_RESULT,
    assistant_content,
    make_content_response,
    make_function_call_response,
    make_tool_cmd_response,
)


def test_build_messages_includes_new_capability_sections():
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
        llm_client=mock_llm, search_client=SearchClientWithCapabilities()
    )
    messages = handler._build_messages([{"role": "user", "content": "test"}])

    system_prompt = messages[0]["content"]
    assert "[TOOL_OVERVIEW]" in system_prompt
    assert "search_owners" in system_prompt
    assert "search_google" in system_prompt
    assert "read_prompt_assets" in system_prompt
    assert "统一使用 XML 工具协议" in system_prompt
    assert "<search_videos" in system_prompt


def test_compact_result_for_context_preserves_multi_query_hits():
    result = {
        "results": [
            {
                "query": ":uid=1 :date<=30d",
                "total_hits": 12,
                "hits": [
                    {
                        "title": "08 最近更新 1",
                        "bvid": "BV108A",
                        "owner": {"name": "红警HBK08"},
                        "stat": {"view": 12345},
                        "pub_to_now_str": "3天前",
                    }
                ],
            },
            {
                "query": ":uid=2 :date<=30d",
                "total_hits": 7,
                "hits": [
                    {
                        "title": "月亮3 最近更新 1",
                        "bvid": "BVmoon3",
                        "owner": {"name": "月亮3"},
                        "stat": {"view": 67890},
                        "pub_to_now_str": "1天前",
                    }
                ],
            },
        ]
    }

    compact = ChatHandler._compact_result_for_context(result)

    assert len(compact["results"]) == 2
    assert compact["results"][0]["query"] == ":uid=1 :date<=30d"
    assert compact["results"][0]["hits"][0]["title"] == "08 最近更新 1"
    assert compact["results"][1]["hits"][0]["owner"] == "月亮3"


def test_handle_xml_tool_flow_uses_observation_context_instead_of_raw_results():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response(
            "我来搜索黑神话相关视频。",
            "<search_videos queries='[\"黑神话\"]'/>",
        ),
        make_content_response("找到了黑神话相关视频。"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "搜索黑神话"}])

    assert assistant_content(result) == "找到了黑神话相关视频。"
    assert mock_llm.chat.call_count == 2
    second_call_messages = mock_llm.chat.call_args_list[1].kwargs["messages"]
    observation_messages = [
        message
        for message in second_call_messages
        if message.get("role") == "user"
        and "[TOOL_OBSERVATIONS]" in str(message.get("content") or "")
    ]
    assert len(observation_messages) == 1
    assert "[搜索结果]" not in observation_messages[0]["content"]


def test_handle_xml_tool_flow_injects_observation_messages():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_search_1",
                name="search_videos",
                arguments={"queries": ["黑神话 q=vwr"]},
            )
        ),
        make_content_response("这里是黑神话的高相关视频。"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "黑神话教程"}])

    assert "黑神话" in assistant_content(result)
    second_call_messages = mock_llm.chat.call_args_list[1].kwargs["messages"]
    observation_messages = [
        message
        for message in second_call_messages
        if message.get("role") == "user"
        and "[TOOL_OBSERVATIONS]" in str(message.get("content") or "")
    ]
    assert observation_messages
    assert "BV1abc" in observation_messages[0]["content"]
    assert "search_videos" in observation_messages[0]["content"]


def test_normalize_search_video_commands_coerces_explicit_bv_lookup():
    normalized = ChatHandler._normalize_search_video_commands(
        [
            {
                "type": "search_videos",
                "args": {"queries": ["BV1e9cfz5EKj"]},
            }
        ]
    )

    assert normalized == [
        {
            "type": "search_videos",
            "args": {"mode": "lookup", "bv": "BV1e9cfz5EKj"},
        }
    ]


def test_plan_tool_commands_continues_recent_video_lookup_after_explicit_bv_hit():
    planned = ChatHandler._plan_tool_commands(
        commands=[],
        messages=[
            {
                "role": "user",
                "content": "BV1e9cfz5EKj 这期视频的作者是谁。他最近还发了哪些视频。",
            }
        ],
        last_tool_results=[
            {
                "type": "search_videos",
                "args": {"mode": "lookup", "bv": "BV1e9cfz5EKj"},
                "result": {
                    "mode": "lookup",
                    "lookup_by": "bvids",
                    "bvids": ["BV1e9cfz5EKj"],
                    "hits": [
                        {
                            "bvid": "BV1e9cfz5EKj",
                            "title": "人在柬埔寨，刚下飞机，现在跑还来得及吗？",
                            "owner": {"mid": 39627524, "name": "食贫道"},
                        }
                    ],
                },
            }
        ],
        owner_result_scope=None,
    )

    assert planned == [
        {
            "type": "search_videos",
            "args": {
                "mode": "lookup",
                "date_window": "30d",
                "limit": 10,
                "exclude_bvids": ["BV1e9cfz5EKj"],
                "mid": "39627524",
            },
        }
    ]


def test_plan_tool_commands_drops_same_round_unresolved_user_scoped_video_search():
    planned = ChatHandler._plan_tool_commands(
        commands=[
            {
                "type": "search_owners",
                "args": {"text": "红警08", "mode": "name"},
            },
            {
                "type": "search_videos",
                "args": {"queries": [":user=红警08 :date<=30d"]},
            },
        ],
        messages=[
            {
                "role": "user",
                "content": "红警08最近发了哪些视频",
            }
        ],
        last_tool_results=None,
        owner_result_scope=None,
    )

    assert all(command["type"] != "search_videos" for command in planned)
    assert {
        "type": "search_owners",
        "args": {"text": "红警08", "mode": "name"},
    } in planned


def test_plan_tool_commands_defers_uid_video_search_during_owner_resolution():
    planned = ChatHandler._plan_tool_commands(
        commands=[
            {
                "type": "search_owners",
                "args": {"text": "月亮3", "mode": "name"},
            },
            {
                "type": "search_videos",
                "args": {
                    "queries": [
                        ":uid=674510452 最近3期视频内容",
                        "月亮3最近3期视频内容",
                    ]
                },
            },
        ],
        messages=[
            {
                "role": "user",
                "content": "月亮3最近3期视频内容",
            }
        ],
        last_tool_results=None,
        owner_result_scope=None,
    )

    assert planned == [
        {
            "type": "search_owners",
            "args": {"text": "月亮3", "mode": "name"},
        }
    ]


def test_tool_events_include_visibility_summary_and_result_ids():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_prompt_1",
                name="read_prompt_assets",
                arguments={"tool_names": ["search_videos"], "levels": ["examples"]},
            ),
            ToolCall(
                id="call_search_1",
                name="search_videos",
                arguments={"queries": ["黑神话 q=vwr"]},
            ),
        ),
        make_content_response("这里是黑神话的搜索结果。"),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(messages=[{"role": "user", "content": "黑神话教程"}])
    tool_events = result["tool_events"]

    assert len(tool_events) == 1
    calls = tool_events[0]["calls"]
    assert calls[0]["visibility"] == "internal"
    assert calls[0]["result_id"].startswith("R")
    assert calls[1]["visibility"] == "user"
    assert "summary" in calls[1]
    assert calls[1]["summary"]["top_hits"][0]["bvid"] == "BV1abc"
    assert (
        calls[1]["summary"]["top_hits"][0]["url"]
        == "https://www.bilibili.com/video/BV1abc"
    )


def test_google_tool_request_is_blocked_when_capability_unavailable():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_google_1",
                name="search_google",
                arguments={"query": "Gemini site:bilibili.com/video"},
            )
        ),
        make_content_response("只能基于站内工具回答。"),
    ]

    mock_search = MagicMock()
    mock_search.capabilities.return_value = {
        "service_name": "runtime-search",
        "service_type": "remote",
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": False,
        "relation_endpoints": ["related_tokens_by_tokens"],
        "available_endpoints": ["/explore", "/search_owners"],
        "docs": ["search_syntax"],
    }

    with patch.dict("os.environ", {"BILI_GOOGLE_HUB_DISABLED": "1"}, clear=False):
        handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)
        result = handler.handle(
            messages=[{"role": "user", "content": "查站内 Gemini 视频"}]
        )

    assert assistant_content(result).endswith("只能基于站内工具回答。")
    assert result.get("tool_events", []) == []
    second_call_messages = mock_llm.chat.call_args_list[1].kwargs["messages"]
    assert any(
        message.get("role") == "user"
        and "当前环境没有 search_google 工具" in message.get("content", "")
        for message in second_call_messages
    )


def test_thinking_mode_prefixes_prompt_and_sets_flag():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response(
        "经过深入分析...",
        usage={"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
    )
    mock_search = MagicMock()

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)
    result = handler.handle(
        messages=[{"role": "user", "content": "分析黑神话的视频趋势"}],
        thinking=True,
    )

    assert result.get("thinking") is True
    assert assistant_content(result) == "经过深入分析..."
    sent_messages = mock_llm.chat.call_args.kwargs["messages"]
    assert "[思考模式]" in sent_messages[0]["content"]


def test_pre_execution_nudge_replans_owner_discovery_instead_of_exiting_loop():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_video_wrong_1",
                name="search_videos",
                arguments={"queries": ["黑神话悟空 q=vwr"]},
            )
        ),
        make_function_call_response(
            ToolCall(
                id="call_owner_1",
                name="search_owners",
                arguments={"text": "黑神话悟空", "mode": "topic"},
            )
        ),
        make_content_response("这里是黑神话悟空相关 UP 主。"),
    ]

    mock_search = MagicMock()
    mock_search.capabilities.return_value = {
        "service_name": "runtime-search",
        "service_type": "remote",
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": False,
        "relation_endpoints": ["related_owners_by_tokens"],
        "available_endpoints": ["/explore", "/search_owners"],
        "docs": ["search_syntax"],
    }
    mock_search.related_owners_by_tokens.return_value = {
        "text": "黑神话悟空",
        "owners": [{"mid": 1, "name": "作者1", "score": 100}],
    }

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "推荐几个做黑神话悟空内容的UP主"}]
    )

    assert assistant_content(result) == "这里是黑神话悟空相关 UP 主。"
    assert mock_search.explore.call_count == 0
    mock_search.related_owners_by_tokens.assert_called_once_with(
        text="黑神话悟空", size=8
    )
    assert mock_llm.chat.call_count == 3


def test_owner_request_without_text_uses_intent_topic_seed():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_owner_implicit_1",
                name="search_owners",
                arguments={"mode": "topic"},
            )
        ),
        make_content_response("这里是黑神话悟空相关 UP 主。"),
    ]

    mock_search = MagicMock()
    mock_search.capabilities.return_value = {
        "service_name": "runtime-search",
        "service_type": "remote",
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": False,
        "relation_endpoints": ["related_owners_by_tokens"],
        "available_endpoints": ["/explore", "/search_owners"],
        "docs": ["search_syntax"],
    }
    mock_search.related_owners_by_tokens.return_value = {
        "text": "黑神话悟空",
        "owners": [{"mid": 1, "name": "作者1", "score": 100}],
    }

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "推荐几个做黑神话悟空内容的UP主"}]
    )

    assert assistant_content(result) == "这里是黑神话悟空相关 UP 主。"
    mock_search.related_owners_by_tokens.assert_called_once_with(
        text="黑神话悟空", size=8
    )


def test_owner_request_with_query_alias_rewrites_to_relation_lookup():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_owner_query_alias_1",
                name="search_owners",
                arguments={"query": "黑神话悟空", "mode": "topic", "num": 5},
            )
        ),
        make_content_response("这里是黑神话悟空相关 UP 主。"),
    ]

    mock_search = MagicMock()
    mock_search.capabilities.return_value = {
        "service_name": "runtime-search",
        "service_type": "remote",
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": False,
        "relation_endpoints": ["related_owners_by_tokens"],
        "available_endpoints": ["/explore", "/search_owners"],
        "docs": ["search_syntax"],
    }
    mock_search.related_owners_by_tokens.return_value = {
        "text": "黑神话悟空",
        "owners": [{"mid": 1, "name": "作者1", "score": 100}],
    }

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "推荐几个做黑神话悟空内容的UP主"}]
    )

    assert assistant_content(result) == "这里是黑神话悟空相关 UP 主。"
    mock_search.related_owners_by_tokens.assert_called_once_with(
        text="黑神话悟空", size=5
    )


def test_alias_like_video_query_rewrites_to_known_canonical_term():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_alias_tokens_1",
                name="expand_query",
                arguments={"text": "康夫UI", "mode": "correction"},
            )
        ),
        make_function_call_response(
            ToolCall(
                id="call_alias_search_1",
                name="search_videos",
                arguments={"queries": ["ComfyUI 入门教程"]},
            )
        ),
        make_content_response("ComfyUI 入门可看 https://www.bilibili.com/video/BV1abc"),
    ]

    mock_search = MagicMock()
    mock_search.capabilities.return_value = {
        "service_name": "runtime-search",
        "service_type": "remote",
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": False,
        "relation_endpoints": ["related_tokens_by_tokens"],
        "available_endpoints": ["/explore", "/search_owners"],
        "docs": ["search_syntax"],
    }
    mock_search.related_tokens_by_tokens.return_value = {
        "text": "康夫UI",
        "options": [{"text": "ComfyUI", "score": 1.0}],
    }
    mock_search.explore.return_value = {
        "query": "ComfyUI 教程 :date>=2024",
        "status": "finished",
        "data": [
            {
                "step": 0,
                "name": "most_relevant_search",
                "output": {
                    "hits": [
                        {
                            "bvid": "BV1abc",
                            "title": "ComfyUI 入门教程",
                            "owner": {"mid": 100, "name": "教程UP主"},
                            "pubdate": 1708700000,
                            "stat": {"view": 500000},
                        }
                    ],
                    "total_hits": 1,
                },
            }
        ],
    }

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "康夫UI 有什么入门教程？"}]
    )

    assert (
        assistant_content(result)
        == "ComfyUI 入门可看 https://www.bilibili.com/video/BV1abc"
    )
    mock_search.explore.assert_called_once_with(query="ComfyUI 入门教程")


def test_ensure_primary_subject_context_prefixes_missing_external_subject():
    content = ChatHandler._ensure_primary_subject_context(
        [{"role": "user", "content": "Gemini 2.5 最近有哪些官方更新？"}],
        "最近有几项官方更新。",
    )

    assert content.startswith("Gemini 2.5：\n")


def test_ensure_primary_subject_context_prefers_rewritten_alias_when_missing():
    content = ChatHandler._ensure_primary_subject_context(
        [{"role": "user", "content": "康夫UI 有什么入门教程？"}],
        "入门可看 https://www.bilibili.com/video/BV1abc",
    )

    assert content.startswith("ComfyUI：\n")


def test_ensure_primary_subject_context_keeps_compact_subject_instead_of_question():
    content = ChatHandler._ensure_primary_subject_context(
        [{"role": "user", "content": "红警08是谁"}],
        "他是红警区创作者。",
        intent=SimpleNamespace(
            final_target="external",
            needs_term_normalization=False,
            explicit_entities=["红警08"],
            explicit_topics=[],
        ),
    )

    assert content.startswith("红警08：\n")
    assert not content.startswith("红警08是谁")


def test_ensure_author_timeline_context_prefers_compact_subject_for_recent_query():
    content = ChatHandler._ensure_author_timeline_context(
        [{"role": "user", "content": "红警08最近发了什么视频"}],
        "找到了最近视频。",
        intent=SimpleNamespace(
            final_target="videos",
            task_mode="repeat",
            explicit_entities=["红警08最近发了什么视频"],
            explicit_topics=["红警08最近发了什么视频"],
        ),
    )

    assert content.startswith("红警08最近视频：\n")
    assert not content.startswith("红警08最近发了什么视频最近视频")


def test_ensure_response_context_ignores_previous_turn_subject_for_new_identity_query():
    content = ChatHandler._ensure_response_context(
        [
            {"role": "user", "content": "红警08是谁"},
            {"role": "assistant", "content": "红警08是红警区作者。"},
            {"role": "user", "content": "月亮三是谁"},
        ],
        "红警月亮3是红警区创作者。",
    )

    assert content == "红警月亮3是红警区创作者。"


def test_handle_falls_back_to_small_model_when_final_response_errors():
    mock_large_llm = MagicMock(spec=LLMClient)
    mock_small_llm = MagicMock(spec=LLMClient)
    mock_large_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_search_1",
                name="search_videos",
                arguments={"queries": ["黑神话 q=vwr"]},
            )
        ),
        ChatResponse(content="[Error: 400 Client Error]", finish_reason="error"),
    ]
    mock_small_llm.chat.return_value = make_content_response(
        "这里是黑神话的最终整理结果。"
    )

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(
        llm_client=mock_large_llm,
        small_llm_client=mock_small_llm,
        search_client=mock_search,
    )

    result = handler.handle(messages=[{"role": "user", "content": "黑神话教程"}])

    assert assistant_content(result) == "这里是黑神话的最终整理结果。"
    assert mock_small_llm.chat.call_count == 1


def test_handle_respects_max_iterations_before_final_answer_nudge():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_tool_cmd_response("搜", f"<search_videos queries='[\"q{idx}\"]'/>")
        for idx in range(3)
    ] + [make_content_response("最终结果")]

    mock_search = MagicMock()
    mock_search.explore.return_value = {
        "query": "q0",
        "status": "finished",
        "data": [
            {
                "step": 0,
                "name": "most_relevant_search",
                "output": {"hits": [], "total_hits": 0},
            }
        ],
    }
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "test"}],
        max_iterations=3,
    )

    assert assistant_content(result) == "最终结果"
    assert mock_llm.chat.call_count == 4


def test_small_model_delegate_internal_tool_uses_small_client():
    mock_large_llm = MagicMock(spec=LLMClient)
    mock_large_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_small_1",
                name="run_small_llm_task",
                arguments={
                    "task": "把候选结果压成 2 条要点",
                    "context": "候选结果很多",
                },
            )
        ),
        make_content_response("这是最终回答。"),
    ]
    mock_small_llm = MagicMock(spec=LLMClient)
    mock_small_llm.chat.return_value = make_content_response("- 要点1\n- 要点2")

    mock_search = MagicMock()
    handler = ChatHandler(
        llm_client=mock_large_llm,
        small_llm_client=mock_small_llm,
        search_client=mock_search,
    )

    result = handler.handle(
        messages=[{"role": "user", "content": "请总结一下"}],
        thinking=True,
    )

    assert assistant_content(result) == "这是最终回答。"
    mock_small_llm.chat.assert_called_once()
    internal_call = result["tool_events"][0]["calls"][0]
    assert internal_call["type"] == "run_small_llm_task"
    assert internal_call["visibility"] == "internal"


def test_mixed_queries_stop_repeating_search_google_and_search_videos_after_one_round():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_google_1",
                name="search_google",
                arguments={"query": "Gemini 2.5 API updates"},
            ),
            ToolCall(
                id="call_video_1",
                name="search_videos",
                arguments={"queries": ["Gemini 2.5 API 解读 q=vwr"]},
            ),
        ),
        make_content_response(
            "Gemini 2.5 官方更新与 B 站解读如下：\n- [黑神话评测](https://www.bilibili.com/video/BV1abc)"
        ),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = MOCK_EXPLORE_RESULT

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)
    handler.tool_executor.google_client = MagicMock()
    handler.tool_executor.google_client.search.return_value = {
        "backend": "mock-google",
        "result_count": 1,
        "results": [
            {
                "title": "Gemini 2.5 API 更新",
                "link": "https://example.com/gemini-api-updates",
                "domain": "example.com",
            }
        ],
    }
    handler.tool_executor._is_google_available = lambda: True

    result = handler.handle(
        messages=[
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            }
        ]
    )

    assert "https://www.bilibili.com/video/BV1abc" in assistant_content(result)
    assert mock_search.explore.call_count == 1
    assert handler.tool_executor.google_client.search.call_count == 1
    assert mock_llm.chat.call_count == 2


def test_mixed_queries_stop_when_google_already_covers_official_and_bili_video():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [
        make_function_call_response(
            ToolCall(
                id="call_google_1",
                name="search_google",
                arguments={"query": "Gemini 2.5 官方更新"},
            ),
            ToolCall(
                id="call_video_1",
                name="search_videos",
                arguments={"queries": ["Gemini 2.5 解读 :date<=30d"]},
            ),
        ),
        make_function_call_response(
            ToolCall(
                id="call_google_2",
                name="search_google",
                arguments={"query": "Gemini 更新 site:bilibili.com/video"},
            ),
            ToolCall(
                id="call_video_2",
                name="search_videos",
                arguments={"queries": ["Gemini 最新 解读 :date<=60d"]},
            ),
        ),
        make_content_response(
            "官方更新见 changelog，B站解读可看 https://www.bilibili.com/video/BV1xyz"
        ),
    ]

    mock_search = MagicMock()
    mock_search.explore.return_value = {
        "query": "Gemini 2.5 解读",
        "status": "finished",
        "data": [
            {
                "step": 0,
                "name": "most_relevant_search",
                "output": {"hits": [], "total_hits": 0},
            }
        ],
    }

    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)
    handler.tool_executor.google_client = MagicMock()
    handler.tool_executor.google_client.search.side_effect = [
        {
            "backend": "mock-google",
            "result_count": 1,
            "results": [
                {
                    "title": "Gemini API Changelog",
                    "link": "https://ai.google.dev/gemini-api/docs/changelog",
                    "domain": "ai.google.dev",
                }
            ],
        },
        {
            "backend": "mock-google",
            "result_count": 1,
            "results": [
                {
                    "title": "Gemini 重大更新解读",
                    "link": "https://www.bilibili.com/video/BV1xyz",
                    "domain": "bilibili.com",
                    "site_kind": "video",
                }
            ],
        },
    ]
    handler.tool_executor._is_google_available = lambda: True

    result = handler.handle(
        messages=[
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            }
        ]
    )

    assert "https://www.bilibili.com/video/BV1xyz" in assistant_content(result)
    assert handler.tool_executor.google_client.search.call_count == 2
    assert mock_llm.chat.call_count == 3


def test_usage_trace_reports_intent_models_and_summary():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = make_content_response("直接回答")
    mock_search = MagicMock()
    handler = ChatHandler(llm_client=mock_llm, search_client=mock_search)

    result = handler.handle(
        messages=[{"role": "user", "content": "来点让我开心的视频"}]
    )
    usage_trace = result["usage_trace"]

    assert usage_trace["intent"]["final_target"] == "videos"
    assert "planner" in usage_trace["models"]
    assert usage_trace["summary"]["llm_calls"] == 1
    assert usage_trace["summary"]["peak_prompt_tokens"] >= 0
