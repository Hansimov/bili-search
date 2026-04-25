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


def test_orchestrator_normalize_request_preserves_explicit_q_vr_dsl():
    orchestrator = ChatOrchestrator(
        llm_client=MagicMock(),
        tool_executor=MagicMock(),
    )
    request = ToolCallRequest(
        id="call_1",
        name="search_videos",
        arguments={"queries": ["h20 英伟达 vr 相关视频 :view>=500"]},
        visibility="user",
    )

    normalized = orchestrator._normalize_request(
        request,
        _intent(
            raw_query="帮我找 h20 显卡 q=vr 的相关视频，列出最相关的几个",
            normalized_query="帮我找 h20 显卡 q=vr 的相关视频 列出最相关的几个",
            final_target="videos",
        ),
        {"default_query_mode": "wv", "rerank_query_mode": "vwr"},
    )

    assert normalized.arguments == {"queries": ["h20 显卡 q=vr"]}


def test_orchestrator_run_shortcuts_explicit_q_vr_search_without_planner():
    llm_client = MagicMock()
    llm_client.chat.side_effect = AssertionError("planner should be skipped")
    tool_executor = MagicMock()
    tool_executor.get_search_capabilities.return_value = {
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
    }
    tool_executor.execute_request.return_value = {
        "query": "h20 显卡 q=vr",
        "total_hits": 3,
        "hits": [
            {
                "title": "华为昇腾到底行不行？",
                "bvid": "BV1Ced5BQEWG",
                "owner": {"name": "科技龙门阵TechTalk"},
            }
        ],
    }
    orchestrator = ChatOrchestrator(
        llm_client=llm_client,
        tool_executor=tool_executor,
    )

    result = orchestrator.run(
        messages=[
            {
                "role": "user",
                "content": "帮我找 h20 显卡 q=vr 的相关视频，列出最相关的几个",
            }
        ],
    )

    tool_executor.execute_request.assert_called_once()
    request = tool_executor.execute_request.call_args.args[0]
    assert request.name == "search_videos"
    assert request.arguments == {"queries": ["h20 显卡 q=vr"]}
    assert result.reasoning_content == ""
    assert result.usage_trace["summary"]["llm_calls"] == 0
    assert "按 `h20 显卡 q=vr` 找到这些相关视频" in result.content
    assert "BV1Ced5BQEWG" in result.content


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


def test_normalize_request_rewrites_title_like_search_video_queries_from_raw_user_text():
    orchestrator = ChatOrchestrator(
        llm_client=MagicMock(),
        small_llm_client=MagicMock(),
        tool_executor=MagicMock(),
        model_registry=ModelRegistry.from_envs(),
    )
    request = ToolCallRequest(
        id="call_title_query_1",
        name="search_videos",
        arguments={
            "queries": [
                "可能打错了字，想找 【大毛-小厨】Up主探索中，欢迎收看求三连！ 相关 找 【大毛-小厨】Up主探索中，欢迎收看求三连！ 相关的视频"
            ]
        },
    )

    normalized = orchestrator._normalize_request(
        request,
        _intent(
            raw_query="我可能打错了字，想找 【大毛-小厨】Up主探索中，欢迎收看求三连！ 相关的视频。",
            normalized_query="我可能打错了字 想找 大毛-小厨 up主探索中 欢迎收看求三连 相关的视频",
            final_target="videos",
        ),
        {"supports_transcript_lookup": True},
        prefer_transcript_lookup=False,
    )

    assert normalized.name == "search_videos"
    assert normalized.arguments == {"queries": ["大毛-小厨 Up主探索中 欢迎收看求三连"]}


def test_normalize_request_rewrites_video_tool_even_without_video_final_target():
    orchestrator = ChatOrchestrator(
        llm_client=MagicMock(),
        small_llm_client=MagicMock(),
        tool_executor=MagicMock(),
        model_registry=ModelRegistry.from_envs(),
    )
    request = ToolCallRequest(
        id="call_tag_query_1",
        name="search_videos",
        arguments={"queries": ["Samara Cyn :view>=1w :date<=30d"]},
    )

    normalized = orchestrator._normalize_request(
        request,
        _intent(
            raw_query="请在B站站内搜索，最近有哪些关于 Samara Cyn 的热门视频？",
            normalized_query="请在b站站内搜索 最近有哪些关于 samara cyn 的热门视频",
            final_target="answer",
        ),
        {"supports_transcript_lookup": True},
        prefer_transcript_lookup=False,
    )

    assert normalized.name == "search_videos"
    assert normalized.arguments == {"queries": ["Samara Cyn"]}


def test_owner_recent_followup_falls_back_to_user_query_when_owner_resolution_drifts():
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
            "text": "贩卖可爱の喵呜",
            "owners": [
                {
                    "name": "不相关作者",
                    "mid": 1001,
                    "score": 1.0,
                }
            ],
        },
        arguments={"text": "贩卖可爱の喵呜"},
    )

    followups = orchestrator._build_deterministic_followup_requests(
        store,
        _intent(
            raw_query="贩卖可爱の喵呜 最近发了什么值得看的视频？",
            normalized_query="贩卖可爱の喵呜 最近发了什么值得看的视频",
            final_target="videos",
        ),
        messages=[
            {
                "role": "user",
                "content": "贩卖可爱の喵呜 最近发了什么值得看的视频？",
            }
        ],
    )

    assert followups
    assert followups[0].name == "search_videos"
    assert followups[0].arguments == {"queries": [":user=贩卖可爱の喵呜 :date<=30d"]}


def test_normalize_request_converts_title_like_owner_probe_to_video_search():
    orchestrator = ChatOrchestrator(
        llm_client=MagicMock(),
        small_llm_client=MagicMock(),
        tool_executor=MagicMock(),
        model_registry=ModelRegistry.from_envs(),
    )
    request = ToolCallRequest(
        id="call_title_owner_1",
        name="search_owners",
        arguments={"text": "一只小雪莉ovo", "size": 5},
    )

    normalized = orchestrator._normalize_request(
        request,
        _intent(
            raw_query="忽略口播和套话，帮我找和 【一只小雪莉ovo】寄明月~ 真正相关的视频。",
            normalized_query="忽略口播和套话 帮我找和 一只小雪莉ovo 寄明月 真正相关的视频",
            final_target="videos",
        ),
        {"supports_transcript_lookup": True},
        prefer_transcript_lookup=False,
    )

    assert normalized.name == "search_videos"
    assert normalized.arguments == {"queries": ["一只小雪莉ovo 寄明月"]}
