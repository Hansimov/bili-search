"""Run/recovery tests for ChatOrchestrator policies."""

from unittest.mock import MagicMock

from llms.models import ChatResponse, ModelRegistry
from llms.orchestration.engine import ChatOrchestrator
from test_orchestration_policies import FakeResultStore, _intent


def test_run_blocks_search_detour_when_transcript_is_unavailable():
    llm = MagicMock()
    llm.chat.side_effect = [
        ChatResponse(
            content="<search_videos bvids='[\"BV1R2XZBQEio\"]' size='1'/>",
            finish_reason="stop",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
            },
        ),
        ChatResponse(
            content="当前环境未提供 get_video_transcript 工具，无法读取该视频的音频转写文本。",
            finish_reason="stop",
            usage={
                "prompt_tokens": 18,
                "completion_tokens": 12,
                "total_tokens": 30,
            },
        ),
    ]
    tool_executor = MagicMock()
    tool_executor.get_search_capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": False,
        "supports_google_search": True,
        "supports_transcript_lookup": False,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }

    orchestrator = ChatOrchestrator(
        llm_client=llm,
        small_llm_client=llm,
        tool_executor=tool_executor,
        model_registry=ModelRegistry.from_envs(),
    )

    result = orchestrator.run(
        messages=[
            {
                "role": "user",
                "content": "请获取 BV1R2XZBQEio 的音频转写文本，然后阅读后再回答。",
            }
        ],
        thinking=False,
        max_iterations=2,
    )

    assert "无法读取该视频的音频转写文本" in result.content
    tool_executor.execute_request.assert_not_called()


def test_run_auto_follows_explicit_bv_lookup_with_recent_mid_lookup():
    llm = MagicMock()
    llm.chat.side_effect = [
        ChatResponse(
            content="<search_videos bv='BV1e9cfz5EKj' mode='lookup' />",
            finish_reason="stop",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        ),
        ChatResponse(
            content="作者是食贫道，近期暂无更多结果。",
            finish_reason="stop",
            usage={
                "prompt_tokens": 18,
                "completion_tokens": 12,
                "total_tokens": 30,
            },
        ),
    ]
    tool_executor = MagicMock()
    tool_executor.get_search_capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": True,
        "supports_transcript_lookup": False,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }
    tool_executor.execute_request.side_effect = [
        {
            "mode": "lookup",
            "lookup_by": "bvids",
            "total_hits": 1,
            "bvids": ["BV1e9cfz5EKj"],
            "hits": [
                {
                    "bvid": "BV1e9cfz5EKj",
                    "title": "人在柬埔寨，刚下飞机，现在跑还来得及吗？",
                    "owner": {"mid": 39627524, "name": "食贫道"},
                }
            ],
            "source_counts": {"mongo": 1, "es": 1},
        },
        {
            "mode": "lookup",
            "lookup_by": "mids",
            "total_hits": 0,
            "mids": ["39627524"],
            "date_window": "30d",
            "hits": [],
            "source_counts": {"mongo": 0, "es": 0},
        },
    ]

    orchestrator = ChatOrchestrator(
        llm_client=llm,
        small_llm_client=llm,
        tool_executor=tool_executor,
        model_registry=ModelRegistry.from_envs(),
    )

    result = orchestrator.run(
        messages=[
            {
                "role": "user",
                "content": "BV1e9cfz5EKj 这期视频的作者是谁。他最近还发了哪些视频。",
            }
        ],
        thinking=False,
        max_iterations=2,
    )

    assert "作者是 食贫道" in result.content
    assert "当前 30 天时间窗内未检索到该作者的其他公开视频。" in result.content
    assert llm.chat.call_count == 1
    assert tool_executor.execute_request.call_count == 2

    first_request = tool_executor.execute_request.call_args_list[0].args[0]
    second_request = tool_executor.execute_request.call_args_list[1].args[0]
    assert first_request.name == "search_videos"
    assert first_request.arguments == {"bv": "BV1e9cfz5EKj", "mode": "lookup"}
    assert second_request.name == "search_videos"
    assert second_request.arguments == {
        "mode": "lookup",
        "date_window": "30d",
        "limit": 10,
        "exclude_bvids": ["BV1e9cfz5EKj"],
        "mid": "39627524",
    }

    calls = result.tool_events[0]["calls"]
    assert len(calls) == 2
    assert calls[1]["type"] == "search_videos"
    assert calls[1]["args"]["mid"] == "39627524"


def test_run_auto_follows_owner_search_with_recent_mid_lookup():
    llm = MagicMock()
    llm.chat.side_effect = [
        ChatResponse(
            content="<search_owners text='红警08' mode='name' />",
            finish_reason="stop",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        ),
    ]
    tool_executor = MagicMock()
    tool_executor.get_search_capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": True,
        "supports_transcript_lookup": False,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }
    tool_executor.execute_request.side_effect = [
        {
            "text": "红警08",
            "mode": "name",
            "total_owners": 2,
            "owners": [
                {
                    "mid": 1629347259,
                    "name": "红警HBK08",
                    "score": 184,
                },
                {
                    "mid": 174335400,
                    "name": "红警HBK08老公",
                    "score": 177,
                },
            ],
        },
        {
            "mode": "lookup",
            "lookup_by": "mid",
            "total_hits": 2,
            "mid": "1629347259",
            "date_window": "30d",
            "hits": [
                {
                    "bvid": "BV1f1d5BVEWB",
                    "title": "红警围攻之都团战！苏军挡在前，掩护盟军后方输出！",
                    "owner": {"mid": 1629347259, "name": "红警HBK08"},
                },
                {
                    "bvid": "BV1Z2d5BkEKr",
                    "title": "红警缆桩脊团战！铁幕和队友接力，冲入敌营！",
                    "owner": {"mid": 1629347259, "name": "红警HBK08"},
                },
            ],
        },
    ]

    orchestrator = ChatOrchestrator(
        llm_client=llm,
        small_llm_client=llm,
        tool_executor=tool_executor,
        model_registry=ModelRegistry.from_envs(),
    )

    result = orchestrator.run(
        messages=[
            {
                "role": "user",
                "content": "红警08最近发了什么视频",
            }
        ],
        thinking=False,
        max_iterations=2,
    )

    assert "红警HBK08" in result.content
    assert "BV1f1d5BVEWB" in result.content
    assert llm.chat.call_count == 1
    assert tool_executor.execute_request.call_count == 2

    first_request = tool_executor.execute_request.call_args_list[0].args[0]
    second_request = tool_executor.execute_request.call_args_list[1].args[0]
    assert first_request.name == "search_owners"
    assert first_request.arguments == {"text": "红警08", "mode": "name"}
    assert second_request.name == "search_videos"
    assert second_request.arguments == {
        "mode": "lookup",
        "mid": "1629347259",
        "date_window": "30d",
        "limit": 10,
    }

    calls = result.tool_events[0]["calls"]
    assert len(calls) == 2
    assert calls[0]["type"] == "search_owners"


def test_run_defers_unscoped_video_query_until_owner_resolution_has_mid():
    llm = MagicMock()
    llm.chat.side_effect = [
        ChatResponse(
            content=(
                "<expand_query text='红色警戒08' mode='auto' size='5'/>"
                "<search_owners text='红色警戒08' size='5'/>"
                "<search_videos queries='[\"红色警戒08 近期投稿视频\"]'/>"
            ),
            finish_reason="stop",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 14,
                "total_tokens": 34,
            },
        ),
    ]
    tool_executor = MagicMock()
    tool_executor.get_search_capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": True,
        "supports_transcript_lookup": False,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }
    tool_executor.execute_request.side_effect = [
        {
            "text": "红色警戒08",
            "mode": "aggregate",
            "total_owners": 2,
            "owners": [
                {
                    "mid": 1629347259,
                    "name": "红警HBK08",
                    "score": 3.1,
                    "sources": [
                        "name",
                        "topic",
                        "related_tokens",
                        "google_space",
                    ],
                    "sample_title": "红警新版本越狱一块地3！",
                    "sample_bvid": "BV1FpQpBtEoc",
                    "sample_view": 313139,
                },
                {
                    "mid": 884798,
                    "name": "蓝天上的流云",
                    "score": 2.1,
                    "sources": ["topic", "related_tokens"],
                    "sample_title": "【红色警戒】当巨炮拥有了谭雅的攻速",
                    "sample_bvid": "BV1aCd7BGE74",
                    "sample_view": 71309,
                },
            ],
        },
        {
            "mode": "lookup",
            "lookup_by": "mid",
            "total_hits": 1,
            "mid": "1629347259",
            "date_window": "30d",
            "hits": [
                {
                    "bvid": "BV1f1d5BVEWB",
                    "title": "红警围攻之都团战！苏军挡在前，掩护盟军后方输出！",
                    "owner": {"mid": 1629347259, "name": "红警HBK08"},
                }
            ],
        },
    ]

    orchestrator = ChatOrchestrator(
        llm_client=llm,
        small_llm_client=llm,
        tool_executor=tool_executor,
        model_registry=ModelRegistry.from_envs(),
    )

    result = orchestrator.run(
        messages=[
            {
                "role": "user",
                "content": "红色警戒08是谁？最近发了哪些视频？",
            }
        ],
        thinking=False,
        max_iterations=2,
    )

    assert "红警HBK08" in result.content
    assert "BV1f1d5BVEWB" in result.content
    assert llm.chat.call_count == 1
    assert tool_executor.execute_request.call_count == 2

    first_request = tool_executor.execute_request.call_args_list[0].args[0]
    second_request = tool_executor.execute_request.call_args_list[1].args[0]
    assert first_request.name == "search_owners"
    assert first_request.arguments == {"text": "红色警戒08", "size": 5}
    assert second_request.name == "search_videos"
    assert second_request.arguments == {
        "mode": "lookup",
        "mid": "1629347259",
        "date_window": "30d",
        "limit": 10,
    }

    executed_queries = [
        str(request.args[0].arguments)
        for request in tool_executor.execute_request.call_args_list
    ]
    executed_tools = [
        request.args[0].name for request in tool_executor.execute_request.call_args_list
    ]
    assert "expand_query" not in executed_tools
    assert not any("红色警戒08 近期投稿视频" in query for query in executed_queries)
    assert not any("红色警戒08是谁" in query for query in executed_queries)


def test_run_defers_uid_video_query_during_same_round_owner_resolution():
    llm = MagicMock()
    llm.chat.side_effect = [
        ChatResponse(
            content=(
                "<search_owners text='月亮3' size='5'/>"
                "<search_videos queries='[\":uid=674510452 最近3期视频内容\", \"月亮3最近3期视频内容\"]'/>"
            ),
            finish_reason="stop",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 12,
                "total_tokens": 32,
            },
        ),
        ChatResponse(
            content="红警月亮3 最近的视频包括 BV12Ed9BNEqJ。",
            finish_reason="stop",
            usage={
                "prompt_tokens": 18,
                "completion_tokens": 12,
                "total_tokens": 30,
            },
        ),
    ]
    tool_executor = MagicMock()
    tool_executor.get_search_capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": True,
        "supports_transcript_lookup": False,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }
    tool_executor.execute_request.side_effect = [
        {
            "text": "月亮3",
            "mode": "aggregate",
            "total_owners": 1,
            "owners": [
                {
                    "mid": 674510452,
                    "name": "红警月亮3",
                    "score": 4.0,
                    "sources": [
                        "name",
                        "topic",
                        "relation",
                        "related_tokens",
                        "google_space",
                    ],
                    "sample_title": "红警占领潘多拉魔盒！",
                    "sample_bvid": "BV1TwdeBLEYL",
                    "sample_view": 60367,
                }
            ],
        },
        {
            "mode": "lookup",
            "lookup_by": "mid",
            "total_hits": 1,
            "mid": "674510452",
            "date_window": "30d",
            "hits": [
                {
                    "bvid": "BV12Ed9BNEqJ",
                    "title": "红警谁是幸运儿！开局在风水宝地资源多，发展起来直接横扫一圈！",
                    "owner": {"mid": 674510452, "name": "红警月亮3"},
                }
            ],
        },
    ]

    orchestrator = ChatOrchestrator(
        llm_client=llm,
        small_llm_client=llm,
        tool_executor=tool_executor,
        model_registry=ModelRegistry.from_envs(),
    )

    result = orchestrator.run(
        messages=[
            {
                "role": "user",
                "content": "月亮3最近3期视频内容",
            }
        ],
        thinking=False,
        max_iterations=2,
    )

    assert "红警月亮3" in result.content
    assert "BV12Ed9BNEqJ" in result.content
    assert llm.chat.call_count == 2
    assert tool_executor.execute_request.call_count == 2

    first_request = tool_executor.execute_request.call_args_list[0].args[0]
    second_request = tool_executor.execute_request.call_args_list[1].args[0]
    assert first_request.name == "search_owners"
    assert first_request.arguments == {"text": "月亮3", "size": 5}
    assert second_request.name == "search_videos"
    assert second_request.arguments == {
        "mode": "lookup",
        "mid": "674510452",
        "limit": 10,
    }

    executed_queries = [
        str(request.args[0].arguments)
        for request in tool_executor.execute_request.call_args_list
    ]
    assert not any("最近3期视频内容" in query for query in executed_queries)


def test_run_replaces_recent_user_scoped_video_search_with_owner_resolution_first():
    llm = MagicMock()
    llm.chat.side_effect = [
        ChatResponse(
            content=(
                "<expand_query text='红色警戒08' mode='auto' size='8'/>"
                "<search_videos queries='[\":user=红色警戒08 :date<=30d\"]'/>"
            ),
            finish_reason="stop",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 12,
                "total_tokens": 32,
            },
        ),
    ]
    tool_executor = MagicMock()
    tool_executor.get_search_capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": True,
        "supports_transcript_lookup": False,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }
    tool_executor.execute_request.side_effect = [
        {
            "text": "红色警戒08",
            "mode": "aggregate",
            "total_owners": 2,
            "owners": [
                {
                    "mid": 1629347259,
                    "name": "红警HBK08",
                    "score": 3.1,
                    "sources": [
                        "name",
                        "topic",
                        "related_tokens",
                        "google_space",
                    ],
                    "sample_title": "红警新版本越狱一块地3！",
                    "sample_view": 313139,
                },
                {
                    "mid": 884798,
                    "name": "蓝天上的流云",
                    "score": 2.1,
                    "sources": ["topic", "related_tokens"],
                    "sample_title": "【红色警戒】当巨炮拥有了谭雅的攻速",
                    "sample_view": 71309,
                },
            ],
        },
        {
            "mode": "lookup",
            "lookup_by": "mid",
            "total_hits": 1,
            "mid": "1629347259",
            "date_window": "30d",
            "hits": [
                {
                    "bvid": "BV1xyo7BvEej",
                    "title": "红警每人一个神车！以光速移动，神偷小车子！",
                    "owner": {"mid": 1629347259, "name": "红警HBK08"},
                }
            ],
        },
    ]

    orchestrator = ChatOrchestrator(
        llm_client=llm,
        small_llm_client=llm,
        tool_executor=tool_executor,
        model_registry=ModelRegistry.from_envs(),
    )

    result = orchestrator.run(
        messages=[
            {
                "role": "user",
                "content": "红色警戒08是谁？最近发了哪些视频？",
            }
        ],
        thinking=False,
        max_iterations=2,
    )

    assert "红警HBK08" in result.content
    assert "BV1xyo7BvEej" in result.content
    assert llm.chat.call_count == 1
    assert tool_executor.execute_request.call_count == 2

    first_request = tool_executor.execute_request.call_args_list[0].args[0]
    second_request = tool_executor.execute_request.call_args_list[1].args[0]
    assert first_request.name == "search_owners"
    assert first_request.arguments == {"text": "红色警戒08", "size": 5}
    assert second_request.name == "search_videos"
    assert second_request.arguments["mid"] == "1629347259"

    executed_tools = [
        request.args[0].name for request in tool_executor.execute_request.call_args_list
    ]
    executed_args = [
        str(request.args[0].arguments)
        for request in tool_executor.execute_request.call_args_list
    ]
    assert "expand_query" not in executed_tools
    assert not any(":user=红色警戒08" in args for args in executed_args)


def test_recent_timeline_answer_prefers_later_mid_lookup_over_earlier_zero_hit_user_query():
    orchestrator = ChatOrchestrator(
        llm_client=MagicMock(),
        small_llm_client=MagicMock(),
        tool_executor=MagicMock(),
        model_registry=ModelRegistry.from_envs(),
    )
    orchestrator.result_store = FakeResultStore()
    orchestrator.result_store.add(
        "search_videos",
        {
            "results": [
                {
                    "query": ":user=红警08 :date<=30d",
                    "total_hits": 0,
                    "hits": [],
                }
            ]
        },
        arguments={"queries": [":user=红警08 :date<=30d"]},
    )
    orchestrator.result_store.add(
        "search_owners",
        {
            "text": "红警08",
            "mode": "name",
            "total_owners": 2,
            "owners": [
                {
                    "mid": 1629347259,
                    "name": "红警HBK08",
                    "score": 184,
                },
                {
                    "mid": 174335400,
                    "name": "红警HBK08老公",
                    "score": 177,
                },
            ],
        },
        arguments={"text": "红警08", "mode": "name"},
    )
    orchestrator.result_store.add(
        "search_videos",
        {
            "mode": "lookup",
            "lookup_by": "mids",
            "total_hits": 2,
            "mids": ["1629347259"],
            "date_window": "30d",
            "hits": [
                {
                    "bvid": "BV1f1d5BVEWB",
                    "title": "红警围攻之都团战！苏军挡在前，掩护盟军后方输出！",
                    "owner": {"mid": 1629347259, "name": "红警HBK08"},
                },
                {
                    "bvid": "BV1Z2d5BkEKr",
                    "title": "红警缆桩脊团战！铁幕和队友接力，冲入敌营！",
                    "owner": {"mid": 1629347259, "name": "红警HBK08"},
                },
            ],
        },
        arguments={
            "mode": "lookup",
            "mid": "1629347259",
            "date_window": "30d",
            "limit": 10,
        },
    )

    answer = orchestrator._build_owner_recent_timeline_answer(
        _intent(
            raw_query="红警08最近发了哪些视频",
            normalized_query="红警08最近发了哪些视频",
            final_target="videos",
            task_mode="repeat",
            explicit_topics=["红警08最近发了哪些视频"],
        ),
        [{"role": "user", "content": "红警08最近发了哪些视频"}],
    )

    assert answer is not None
    assert "红警HBK08" in answer
    assert "BV1f1d5BVEWB" in answer
    assert "未检索到" not in answer


def test_run_recovers_owner_recent_query_after_planner_rate_limit():
    llm = MagicMock()
    llm.chat.side_effect = [
        ChatResponse(
            content="[Error: 429 Client Error: Too Many Requests for url: https://ark.cn-beijing.volces.com/api/v3/chat/completions]",
            finish_reason="error",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "total_tokens": 24,
            },
        ),
    ]
    tool_executor = MagicMock()
    tool_executor.get_search_capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": True,
        "supports_transcript_lookup": False,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }
    tool_executor.execute_request.side_effect = [
        {
            "text": "红警08",
            "mode": "name",
            "total_owners": 1,
            "owners": [
                {
                    "mid": 1629347259,
                    "name": "红警HBK08",
                    "score": 184,
                }
            ],
        },
        {
            "mode": "lookup",
            "lookup_by": "mid",
            "total_hits": 1,
            "mid": "1629347259",
            "date_window": "30d",
            "hits": [
                {
                    "bvid": "BV1f1d5BVEWB",
                    "title": "红警围攻之都团战！苏军挡在前，掩护盟军后方输出！",
                    "owner": {"mid": 1629347259, "name": "红警HBK08"},
                }
            ],
        },
    ]

    orchestrator = ChatOrchestrator(
        llm_client=llm,
        small_llm_client=llm,
        tool_executor=tool_executor,
        model_registry=ModelRegistry.from_envs(),
    )

    result = orchestrator.run(
        messages=[
            {
                "role": "user",
                "content": "红警08最近发了什么视频",
            }
        ],
        thinking=False,
        max_iterations=2,
    )

    assert "红警HBK08" in result.content
    assert "BV1f1d5BVEWB" in result.content
    assert llm.chat.call_count == 1
    assert tool_executor.execute_request.call_count == 2
    assert result.tool_events[0]["calls"][0]["type"] == "search_owners"
    assert result.tool_events[0]["calls"][1]["args"]["mid"] == "1629347259"


def test_run_recovers_explicit_bv_query_after_planner_rate_limit():
    llm = MagicMock()
    llm.chat.side_effect = [
        ChatResponse(
            content="[Error: 429 Client Error: Too Many Requests for url: https://ark.cn-beijing.volces.com/api/v3/chat/completions]",
            finish_reason="error",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "total_tokens": 24,
            },
        ),
    ]
    tool_executor = MagicMock()
    tool_executor.get_search_capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": True,
        "supports_transcript_lookup": False,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }
    tool_executor.execute_request.side_effect = [
        {
            "mode": "lookup",
            "lookup_by": "bvids",
            "total_hits": 1,
            "bvids": ["BV1e9cfz5EKj"],
            "hits": [
                {
                    "bvid": "BV1e9cfz5EKj",
                    "title": "人在柬埔寨，刚下飞机，现在跑还来得及吗？",
                    "owner": {"mid": 39627524, "name": "食贫道"},
                }
            ],
        },
        {
            "mode": "lookup",
            "lookup_by": "mids",
            "total_hits": 0,
            "mids": ["39627524"],
            "date_window": "30d",
            "hits": [],
        },
    ]

    orchestrator = ChatOrchestrator(
        llm_client=llm,
        small_llm_client=llm,
        tool_executor=tool_executor,
        model_registry=ModelRegistry.from_envs(),
    )

    result = orchestrator.run(
        messages=[
            {
                "role": "user",
                "content": "BV1e9cfz5EKj 这期视频的作者是谁。他最近还发了哪些视频。",
            }
        ],
        thinking=False,
        max_iterations=2,
    )

    assert "作者是 食贫道" in result.content
    assert "当前 30 天时间窗内未检索到该作者的其他公开视频。" in result.content
    assert llm.chat.call_count == 1
    assert tool_executor.execute_request.call_count == 2
    assert result.tool_events[0]["calls"][0]["args"] == {
        "bv": "BV1e9cfz5EKj",
        "mode": "lookup",
    }
    assert result.tool_events[0]["calls"][1]["args"]["mid"] == "39627524"


def test_build_deterministic_followup_adds_owner_scoped_video_query_for_owner_topic():
    orchestrator = ChatOrchestrator(
        llm_client=MagicMock(),
        small_llm_client=MagicMock(),
        tool_executor=MagicMock(),
        model_registry=ModelRegistry.from_envs(),
    )
    store = FakeResultStore()
    store.add(
        "search_owners",
        {
            "text": "月栖乐序",
            "mode": "name",
            "owners": [
                {
                    "mid": 31572735,
                    "name": "月栖乐序",
                    "score": 280.0,
                    "sources": ["name", "topic"],
                }
            ],
        },
        arguments={"text": "月栖乐序", "mode": "name"},
    )

    requests = orchestrator._build_deterministic_followup_requests(
        store,
        _intent(
            raw_query="月栖乐序 关于 高能音乐挑战赛 有哪些值得看的视频？",
            normalized_query="月栖乐序 关于 高能音乐挑战赛 有哪些值得看的视频",
            final_target="videos",
            explicit_entities=["月栖乐序"],
            explicit_topics=["月栖乐序", "高能音乐挑战赛"],
        ),
        messages=[
            {
                "role": "user",
                "content": "月栖乐序 关于 高能音乐挑战赛 有哪些值得看的视频？",
            }
        ],
    )

    assert len(requests) == 1
    assert requests[0].name == "search_videos"
    assert requests[0].arguments["queries"][0] == ":uid=31572735 高能音乐挑战赛"
    assert requests[0].arguments["queries"][1] == "月栖乐序 高能音乐挑战赛"


def test_build_deterministic_followup_uses_focus_query_when_owner_results_are_not_confident():
    orchestrator = ChatOrchestrator(
        llm_client=MagicMock(),
        small_llm_client=MagicMock(),
        tool_executor=MagicMock(),
        model_registry=ModelRegistry.from_envs(),
    )
    store = FakeResultStore()
    store.add(
        "search_owners",
        {
            "text": "大毛-小厨",
            "mode": "name",
            "owners": [
                {
                    "mid": 437053628,
                    "name": "大毛猴猴",
                    "score": 245.0,
                    "sources": ["name", "topic"],
                }
            ],
        },
        arguments={"text": "大毛-小厨", "mode": "name"},
    )

    requests = orchestrator._build_deterministic_followup_requests(
        store,
        _intent(
            raw_query="我可能打错了字，想找 【大毛-小厨】Up主探索中，欢迎收看求三连！ 相关的视频。",
            normalized_query="我可能打错了字 想找 大毛-小厨 up主探索中 欢迎收看求三连 相关的视频",
            final_target="videos",
            explicit_topics=["大毛-小厨", "Up主探索中", "欢迎收看求三连"],
        ),
        messages=[
            {
                "role": "user",
                "content": "我可能打错了字，想找 【大毛-小厨】Up主探索中，欢迎收看求三连！ 相关的视频。",
            }
        ],
    )

    assert len(requests) == 1
    assert requests[0].name == "search_videos"
    assert requests[0].arguments == {"queries": ["大毛-小厨 Up主探索中 欢迎收看求三连"]}


def _drain_orchestration_stream(stream):
    chunks = []
    while True:
        try:
            chunks.append(next(stream))
        except StopIteration as stop:
            return chunks, stop.value


def test_run_stream_recovers_owner_recent_query_after_planner_stream_error():
    llm = MagicMock()
    llm.chat_stream.side_effect = [
        iter(
            [
                {
                    "id": "chunk-1",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error",
                        }
                    ],
                }
            ]
        )
    ]
    tool_executor = MagicMock()
    tool_executor.get_search_capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": True,
        "supports_transcript_lookup": False,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }
    tool_executor.execute_request.side_effect = [
        {
            "text": "红警08",
            "mode": "name",
            "total_owners": 1,
            "owners": [
                {
                    "mid": 1629347259,
                    "name": "红警HBK08",
                    "score": 184,
                }
            ],
        },
        {
            "mode": "lookup",
            "lookup_by": "mid",
            "total_hits": 1,
            "mid": "1629347259",
            "date_window": "30d",
            "hits": [
                {
                    "bvid": "BV1f1d5BVEWB",
                    "title": "红警围攻之都团战！苏军挡在前，掩护盟军后方输出！",
                    "owner": {"mid": 1629347259, "name": "红警HBK08"},
                }
            ],
        },
    ]

    orchestrator = ChatOrchestrator(
        llm_client=llm,
        small_llm_client=llm,
        tool_executor=tool_executor,
        model_registry=ModelRegistry.from_envs(),
    )

    chunks, result = _drain_orchestration_stream(
        orchestrator.run_stream(
            messages=[{"role": "user", "content": "红警08最近发了什么视频"}],
            thinking=False,
            max_iterations=2,
        )
    )

    assert "红警HBK08" in result.content
    assert "BV1f1d5BVEWB" in result.content
    assert tool_executor.execute_request.call_count == 2
    assert result.tool_events[0]["calls"][0]["type"] == "search_owners"
    assert result.tool_events[0]["calls"][1]["args"]["mid"] == "1629347259"
    assert any(chunk.get("tool_events") for chunk in chunks)
    assert any(
        chunk.get("delta", {}).get("content")
        and "红警HBK08" in chunk.get("delta", {}).get("content", "")
        for chunk in chunks
    )


def test_run_stream_recovers_explicit_bv_query_after_planner_stream_error():
    llm = MagicMock()
    llm.chat_stream.side_effect = [
        iter(
            [
                {
                    "id": "chunk-1",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error",
                        }
                    ],
                }
            ]
        )
    ]
    tool_executor = MagicMock()
    tool_executor.get_search_capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": True,
        "supports_transcript_lookup": False,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }
    tool_executor.execute_request.side_effect = [
        {
            "mode": "lookup",
            "lookup_by": "bvids",
            "total_hits": 1,
            "bvids": ["BV1PSdpBCE3H"],
            "hits": [
                {
                    "bvid": "BV1PSdpBCE3H",
                    "title": "当高数值遇上好战术！绿龙完美假打拿下双人五杀！不得不说一句陌生~",
                    "owner": {"mid": 203680252, "name": "AYCS2"},
                }
            ],
        },
        {
            "mode": "lookup",
            "lookup_by": "mids",
            "total_hits": 1,
            "mids": ["203680252"],
            "date_window": "30d",
            "hits": [
                {
                    "bvid": "BV1F2djBHETQ",
                    "title": "别炸我职业哥了！Zywoo面对NAVI打出30-7逆天数据，豆豆转基因直接化身第二个大番薯，小蜜蜂挺进四强！",
                    "owner": {"mid": 203680252, "name": "AYCS2"},
                }
            ],
        },
    ]

    orchestrator = ChatOrchestrator(
        llm_client=llm,
        small_llm_client=llm,
        tool_executor=tool_executor,
        model_registry=ModelRegistry.from_envs(),
    )

    chunks, result = _drain_orchestration_stream(
        orchestrator.run_stream(
            messages=[
                {
                    "role": "user",
                    "content": "BV1PSdpBCE3H 这期视频作者是谁？他还发了哪些视频？",
                }
            ],
            thinking=False,
            max_iterations=2,
        )
    )

    assert "作者是 AYCS2" in result.content
    assert "BV1F2djBHETQ" in result.content
    assert tool_executor.execute_request.call_count == 2
    assert result.tool_events[0]["calls"][0]["args"] == {
        "bv": "BV1PSdpBCE3H",
        "mode": "lookup",
    }
    assert result.tool_events[0]["calls"][1]["args"]["mid"] == "203680252"
    assert any(chunk.get("tool_events") for chunk in chunks)
