from unittest.mock import MagicMock

from llms.models import ChatResponse, DEFAULT_SMALL_MODEL_CONFIG, ModelRegistry
from llms.orchestration.engine import ChatOrchestrator
from llms.orchestration.policies import has_target_coverage
from llms.orchestration.policies import select_post_execution_nudge
from llms.orchestration.policies import select_pre_execution_nudge
from llms.contracts import IntentProfile, ToolCallRequest, ToolExecutionRecord


class FakeResultStore:
    def __init__(self):
        self.records = {}
        self.order = []

    def add(self, tool_name: str, result: dict, arguments: dict | None = None):
        result_id = f"R{len(self.order) + 1}"
        record = ToolExecutionRecord(
            result_id=result_id,
            request=ToolCallRequest(
                id=result_id,
                name=tool_name,
                arguments=arguments or {},
            ),
            result=result,
            summary={},
        )
        self.records[result_id] = record
        self.order.append(result_id)
        return record

    def get(self, result_id: str):
        return self.records.get(result_id)


def _intent(**kwargs) -> IntentProfile:
    return IntentProfile(
        raw_query=kwargs.pop("raw_query", "test"),
        normalized_query=kwargs.pop("normalized_query", "test"),
        **kwargs,
    )


def test_has_target_coverage_for_mixed_route_requires_external_and_video_signals():
    store = FakeResultStore()
    store.add(
        "search_google",
        {
            "results": [
                {
                    "title": "Gemini 2.5 release notes",
                    "link": "https://ai.google.dev/release-notes",
                    "domain": "ai.google.dev",
                    "site_kind": "docs",
                }
            ]
        },
    )
    store.add(
        "search_videos",
        {
            "results": [
                {
                    "query": "Gemini 2.5 解读",
                    "total_hits": 1,
                    "hits": [{"title": "Gemini 2.5 更新解读", "bvid": "BV1abc"}],
                }
            ]
        },
    )

    assert has_target_coverage(store, _intent(final_target="mixed")) is True


def test_has_target_coverage_for_videos_accepts_internal_small_task_results():
    store = FakeResultStore()
    store.add(
        "run_small_llm_task",
        {
            "task": "把视频转写压成 5 条要点",
            "result": "- 主题\n- 要点1\n- 要点2",
        },
    )

    assert has_target_coverage(store, _intent(final_target="videos")) is True


def test_has_target_coverage_requires_recent_followup_after_explicit_bv_lookup():
    store = FakeResultStore()
    store.add(
        "search_videos",
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
        arguments={"mode": "lookup", "bv": "BV1e9cfz5EKj"},
    )

    assert (
        has_target_coverage(
            store,
            _intent(
                raw_query="BV1e9cfz5EKj 这期视频的作者是谁。他最近还发了哪些视频。",
                normalized_query="bv1e9cfz5ekj 这期视频的作者是谁 他最近还发了哪些视频",
                final_target="videos",
                needs_owner_resolution=True,
                explicit_entities=["BV1e9cfz5EKj"],
                explicit_topics=["BV1e9cfz5EKj", "视频的作者"],
            ),
        )
        is False
    )


def test_has_target_coverage_accepts_recent_followup_after_mid_lookup():
    store = FakeResultStore()
    store.add(
        "search_videos",
        {
            "mode": "lookup",
            "lookup_by": "bvids",
            "total_hits": 1,
            "bvids": ["BV1e9cfz5EKj"],
            "hits": [
                {
                    "bvid": "BV1e9cfz5EKj",
                    "owner": {"mid": 39627524, "name": "食贫道"},
                }
            ],
        },
        arguments={"mode": "lookup", "bv": "BV1e9cfz5EKj"},
    )
    store.add(
        "search_videos",
        {
            "mode": "lookup",
            "lookup_by": "mids",
            "total_hits": 0,
            "mids": ["39627524"],
            "date_window": "30d",
            "hits": [],
        },
        arguments={
            "mode": "lookup",
            "mid": "39627524",
            "date_window": "30d",
            "exclude_bvids": ["BV1e9cfz5EKj"],
        },
    )

    assert (
        has_target_coverage(
            store,
            _intent(
                raw_query="BV1e9cfz5EKj 这期视频的作者是谁。他最近还发了哪些视频。",
                normalized_query="bv1e9cfz5ekj 这期视频的作者是谁 他最近还发了哪些视频",
                final_target="videos",
                needs_owner_resolution=True,
                explicit_entities=["BV1e9cfz5EKj"],
                explicit_topics=["BV1e9cfz5EKj", "视频的作者"],
            ),
        )
        is True
    )


def test_extract_recent_window_ignores_circled_index_in_title_query():
    assert (
        ChatOrchestrator._extract_recent_window(
            "2026.04.21 T.赵俊杰个播录屏④（22：50—23：42）"
        )
        == "30d"
    )


def test_extract_recent_window_supports_unicode_digit_day_window():
    assert ChatOrchestrator._extract_recent_window("最近④天赵俊杰视频") == "4d"


def test_select_pre_execution_nudge_blocks_repeating_mixed_searches():
    store = FakeResultStore()
    store.add(
        "search_google",
        {
            "results": [
                {
                    "title": "Gemini 2.5 release notes",
                    "link": "https://ai.google.dev/release-notes",
                    "domain": "ai.google.dev",
                    "site_kind": "docs",
                }
            ]
        },
    )
    store.add(
        "search_videos",
        {
            "results": [
                {
                    "query": "Gemini 2.5 解读",
                    "total_hits": 1,
                    "hits": [{"title": "Gemini 2.5 更新解读", "bvid": "BV1abc"}],
                }
            ]
        },
    )

    rule = select_pre_execution_nudge(
        store,
        _intent(final_target="mixed"),
        ["search_google"],
        set(),
    )

    assert rule is not None
    assert rule[0] == "mixed_results_already_sufficient"


def test_select_pre_execution_nudge_prefers_term_normalization_before_video_search():
    store = FakeResultStore()

    rule = select_pre_execution_nudge(
        store,
        _intent(
            final_target="videos",
            needs_keyword_expansion=True,
            needs_term_normalization=True,
        ),
        ["search_videos"],
        set(),
    )

    assert rule is not None
    assert rule[0] == "prefer_term_normalization_before_video_search"
    assert "默认直接用 semantic" in rule[1]
    assert "associate" not in rule[1]


def test_select_pre_execution_nudge_prefers_video_search_after_token_expansion():
    store = FakeResultStore()
    store.add(
        "expand_query",
        {
            "text": "康夫UI",
            "options": [{"text": "ComfyUI", "score": 0.92}],
        },
    )

    rule = select_pre_execution_nudge(
        store,
        _intent(
            final_target="videos",
            needs_term_normalization=True,
        ),
        ["search_google"],
        set(),
    )

    assert rule is not None
    assert rule[0] == "prefer_video_search_after_token_expansion"


def test_select_pre_execution_nudge_prefers_owner_discovery_before_video_search():
    store = FakeResultStore()

    rule = select_pre_execution_nudge(
        store,
        _intent(final_target="owners"),
        ["search_videos"],
        set(),
    )

    assert rule is not None
    assert rule[0] == "prefer_owner_discovery_before_video_search"


def test_select_pre_execution_nudge_prefers_owner_resolution_before_external_detour():
    store = FakeResultStore()

    rule = select_pre_execution_nudge(
        store,
        _intent(final_target="videos", needs_owner_resolution=True),
        ["search_google"],
        set(),
    )

    assert rule is not None
    assert rule[0] == "prefer_owner_resolution_before_external_detour"


def test_select_pre_execution_nudge_prefers_scoped_video_search_after_owner_resolution():
    store = FakeResultStore()
    store.add(
        "search_owners",
        {
            "text": "何同学",
            "owners": [{"name": "何同学", "mid": 123, "score": 1.0}],
        },
    )

    rule = select_pre_execution_nudge(
        store,
        _intent(
            final_target="videos",
            needs_owner_resolution=True,
        ),
        ["search_google"],
        set(),
    )

    assert rule is not None
    assert rule[0] == "prefer_owner_scoped_video_search_after_resolution"


def test_select_post_execution_nudge_sends_zero_hit_video_fallback():
    store = FakeResultStore()
    store.add(
        "search_videos",
        {
            "results": [
                {
                    "query": "Gemini 2.5 API 解读",
                    "total_hits": 0,
                    "hits": [],
                }
            ]
        },
    )

    rule = select_post_execution_nudge(
        store,
        _intent(
            final_target="videos",
            explicit_entities=["Gemini 2.5"],
        ),
        "Gemini 2.5 API 解读视频",
        set(),
    )

    assert rule is not None
    assert rule[0] == "video_zero_hit_google_fallback"


def test_select_post_execution_nudge_flags_weak_interview_video_evidence():
    store = FakeResultStore()
    store.add(
        "expand_query",
        {
            "text": "袁启 采访",
            "options": [
                {"text": "袁启 专访", "score": 0.95},
                {"text": "袁启 访谈", "score": 0.9},
            ],
        },
    )
    store.add(
        "search_videos",
        {
            "results": [
                {
                    "query": "袁启 专访",
                    "total_hits": 2,
                    "hits": [
                        {"title": "雪王看袁启聪《看了就知道，老头乐到底有多么危险》"},
                        {"title": "用《喜鹊谋杀案》的方式介绍袁启聪（子）"},
                    ],
                }
            ]
        },
    )

    rule = select_post_execution_nudge(
        store,
        _intent(final_target="videos"),
        "袁启 采访",
        set(),
    )

    assert rule is not None
    assert rule[0] == "weak_interview_video_evidence"
    assert "不要把这些结果包装成采访命中" in rule[1]


def test_select_post_execution_nudge_skips_when_interview_anchor_exists_in_hits():
    store = FakeResultStore()
    store.add(
        "search_videos",
        {
            "results": [
                {
                    "query": "袁启 专访",
                    "total_hits": 1,
                    "hits": [
                        {"title": "袁启聪专访：聊聊汽车媒体行业"},
                    ],
                }
            ]
        },
    )

    rule = select_post_execution_nudge(
        store,
        _intent(final_target="videos"),
        "袁启 采访",
        set(),
    )

    assert rule is None


def test_select_post_execution_nudge_requests_recent_followup_after_explicit_bv_lookup():
    store = FakeResultStore()
    store.add(
        "search_videos",
        {
            "mode": "lookup",
            "lookup_by": "bvids",
            "total_hits": 1,
            "bvids": ["BV1e9cfz5EKj"],
            "hits": [
                {
                    "bvid": "BV1e9cfz5EKj",
                    "owner": {"mid": 39627524, "name": "食贫道"},
                }
            ],
        },
        arguments={"mode": "lookup", "bv": "BV1e9cfz5EKj"},
    )

    rule = select_post_execution_nudge(
        store,
        _intent(
            raw_query="BV1e9cfz5EKj 这期视频的作者是谁。他最近还发了哪些视频。",
            normalized_query="bv1e9cfz5ekj 这期视频的作者是谁 他最近还发了哪些视频",
            final_target="videos",
            needs_owner_resolution=True,
            explicit_entities=["BV1e9cfz5EKj"],
            explicit_topics=["BV1e9cfz5EKj", "视频的作者"],
        ),
        "BV1e9cfz5EKj 这期视频的作者是谁。他最近还发了哪些视频。",
        set(),
    )

    assert rule is not None
    assert rule[0] == "explicit_video_lookup_recent_followup"


def test_select_post_execution_nudge_prefers_token_retry_for_alias_expansion():
    store = FakeResultStore()
    store.add(
        "expand_query",
        {
            "text": "康夫UI",
            "options": [{"text": "ComfyUI", "score": 0.92}],
        },
    )
    store.add(
        "search_videos",
        {
            "results": [
                {
                    "query": "康夫UI 教程",
                    "total_hits": 0,
                    "hits": [],
                }
            ]
        },
    )

    rule = select_post_execution_nudge(
        store,
        _intent(
            final_target="videos",
            explicit_entities=["康夫UI"],
            needs_keyword_expansion=True,
        ),
        "康夫UI 有什么入门教程？",
        set(),
    )

    assert rule is not None
    assert rule[0] == "token_expansion_retry"


def test_select_model_prefers_small_for_known_video_transcript():
    orchestrator = ChatOrchestrator(
        llm_client=MagicMock(),
        small_llm_client=MagicMock(),
        tool_executor=MagicMock(),
        model_registry=ModelRegistry.from_envs(),
    )
    intent = _intent(
        final_target="videos",
        task_mode="known_item",
        ambiguity=0.12,
        complexity_score=0.28,
    )
    messages = [{"role": "user", "content": "BV1YXZPB1Erc 这个视频主要讲了什么"}]

    prefer_small = orchestrator._wants_transcript_lookup(
        messages,
        intent,
        {"supports_transcript_lookup": True},
    )
    decision = orchestrator._select_model(
        intent,
        stage="planner",
        thinking=False,
        prefer_small=prefer_small,
    )

    assert prefer_small is True
    assert decision.spec.config_name == DEFAULT_SMALL_MODEL_CONFIG
    assert "转写" in decision.reason


def test_normalize_request_rewrites_known_bv_video_search_to_transcript():
    orchestrator = ChatOrchestrator(
        llm_client=MagicMock(),
        small_llm_client=MagicMock(),
        tool_executor=MagicMock(),
        model_registry=ModelRegistry.from_envs(),
    )
    request = ToolCallRequest(
        id="call_1",
        name="search_videos",
        arguments={"bvids": ["BV1R2XZBQEio"]},
    )
    normalized = orchestrator._normalize_request(
        request,
        _intent(
            final_target="videos",
            task_mode="known_item",
            explicit_entities=["BV1R2XZBQEio"],
            explicit_topics=["BV1R2XZBQEio", "讲了什么"],
        ),
        {"supports_transcript_lookup": True},
        prefer_transcript_lookup=True,
    )

    assert normalized.name == "get_video_transcript"
    assert normalized.arguments == {"video_id": "BV1R2XZBQEio"}


def test_normalize_request_rewrites_explicit_bv_query_to_lookup():
    orchestrator = ChatOrchestrator(
        llm_client=MagicMock(),
        small_llm_client=MagicMock(),
        tool_executor=MagicMock(),
        model_registry=ModelRegistry.from_envs(),
    )
    request = ToolCallRequest(
        id="call_lookup_1",
        name="search_videos",
        arguments={"queries": ["BV1e9cfz5EKj"]},
    )
    normalized = orchestrator._normalize_request(
        request,
        _intent(
            final_target="videos",
            task_mode="exploration",
            explicit_entities=["BV1e9cfz5EKj"],
            explicit_topics=["BV1e9cfz5EKj"],
        ),
        {"supports_transcript_lookup": True},
        prefer_transcript_lookup=False,
    )

    assert normalized.name == "search_videos"
    assert normalized.arguments == {
        "mode": "lookup",
        "bv": "BV1e9cfz5EKj",
    }


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
