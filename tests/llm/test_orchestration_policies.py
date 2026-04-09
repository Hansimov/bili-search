from llms.orchestration.policies import has_target_coverage
from llms.orchestration.policies import select_post_execution_nudge
from llms.orchestration.policies import select_pre_execution_nudge
from llms.contracts import IntentProfile, ToolCallRequest, ToolExecutionRecord


class FakeResultStore:
    def __init__(self):
        self.records = {}
        self.order = []

    def add(self, tool_name: str, result: dict):
        result_id = f"R{len(self.order) + 1}"
        record = ToolExecutionRecord(
            result_id=result_id,
            request=ToolCallRequest(
                id=result_id,
                name=tool_name,
                arguments={},
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
