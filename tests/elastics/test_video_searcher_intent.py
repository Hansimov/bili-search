from dsl.fields.word import get_auto_require_short_han_exact_mode
from dsl.rewrite import DslExprRewriter
from dsl.elastic import DslExprToElasticConverter
from dsl.filter import QueryDslDictFilterMerger
from elastics.videos.explorer import VideoExplorer
from elastics.videos.constants import SEARCH_BOOSTED_FIELDS
from elastics.videos.results.reranking import rerank_focused_title_hits
from elastics.videos.searcher_v2 import VideoSearcherV2
from recalls.base import RecallPool


class _DummyRanker:
    @staticmethod
    def heads(result, top_k):
        return result


class _DummyParser:
    @staticmethod
    def parse(*args, **kwargs):
        return {
            "query_info": {},
            "rewrite_info": {},
            "hits": [],
            "total_hits": 0,
            "return_hits": 0,
        }


def make_searcher(owner_result: dict) -> tuple[VideoSearcherV2, dict]:
    searcher = VideoSearcherV2.__new__(VideoSearcherV2)
    captured: dict = {}

    class _StubOwnerSearcher:
        @staticmethod
        def search(query, mode="name", size=5):
            return owner_result

    class _StubRelationsClient:
        @staticmethod
        def related_tokens_by_tokens(**kwargs):
            captured["relation_kwargs"] = kwargs
            return {"mode": kwargs.get("mode", "semantic"), "options": []}

    searcher._owner_searcher = _StubOwnerSearcher()
    searcher._relations_client = _StubRelationsClient()
    searcher.query_rewriter = DslExprRewriter()
    searcher.has_search_keywords = lambda query: True
    searcher.get_info_of_query_rewrite_dsl = lambda **kwargs: (
        captured.setdefault("query_info", {}),
        captured.setdefault("rewrite_info", {}),
        captured.setdefault("query_dsl_dict", {"match_all": {}}),
    )
    searcher.construct_search_body = lambda **kwargs: {
        "query": kwargs["query_dsl_dict"]
    }
    searcher.submit_to_es = lambda body, context=None: {
        "hits": {"hits": [], "total": {"value": 0, "relation": "eq"}},
        "timed_out": False,
    }
    searcher.hit_parser = _DummyParser()
    searcher.hit_ranker = _DummyRanker()
    searcher.rewrite_by_suggest = lambda *args, **kwargs: kwargs["return_res"]
    searcher.post_process_return_res = lambda result: result
    searcher.sanitize_search_body_for_client = lambda body: body
    return searcher, captured


def test_get_info_of_query_rewrite_dsl_uses_primary_rewrite_expr_tree():
    searcher = VideoSearcherV2.__new__(VideoSearcherV2)
    searcher.query_rewriter = DslExprRewriter()
    searcher.elastic_converter = DslExprToElasticConverter()
    searcher.filter_merger = QueryDslDictFilterMerger()

    _, rewrite_info, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
        query="康夫ui 教程",
        suggest_info={
            "group_replaces_count": [
                [["康夫ui", "ComfyUI"], 10],
            ]
        },
    )

    assert rewrite_info["rewrited"] is True
    assert "ComfyUI" in str(query_dsl_dict)
    assert "康夫ui" not in str(query_dsl_dict)


def test_search_does_not_apply_case_level_alias_without_data_rule():
    searcher, captured = make_searcher({"owners": []})

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["suggest_info"] = kwargs["suggest_info"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("康夫ui 教程", limit=5)

    assert captured["dsl_query"] == "康夫ui 教程"
    assert captured["suggest_info"] == {}
    assert result["semantic_rewrite_info"]["alias_rewritten"] is False
    assert result["semantic_rewrite_info"]["applied_query"] == "康夫ui 教程"


def test_search_builds_semantic_suggest_info_for_mixed_script_relation_rewrite():
    searcher, captured = make_searcher({"owners": []})

    class _StubRelationsClient:
        @staticmethod
        def related_tokens_by_tokens(**kwargs):
            captured["relation_kwargs"] = kwargs
            return {
                "mode": kwargs.get("mode", "semantic"),
                "options": [
                    {"text": "ComfyUI 教学", "score": 920.0},
                    {"text": "__bad__|候选", "score": 910.0},
                ],
            }

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["suggest_info"] = kwargs["suggest_info"]
        return (
            captured.setdefault("query_info", {}),
            {"rewrited_word_exprs": ["ComfyUI 教学"]},
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher._relations_client = _StubRelationsClient()
    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("康夫ui 教程", limit=5)

    assert captured["relation_kwargs"]["text"] == "康夫ui 教程"
    assert captured["dsl_query"] == "康夫ui 教程"
    assert captured["suggest_info"] == {
        "group_replaces_count": [
            [["康夫ui", "ComfyUI", "教程", "教学"], 920],
        ]
    }
    assert result["semantic_rewrite_info"]["relation_rewritten"] is True
    assert result["semantic_rewrite_info"]["applied_query"] == "ComfyUI 教学"


def test_search_no_longer_masks_spaced_owner_candidates_with_case_alias():
    owner_result = {
        "owners": [
            {
                "mid": 14813517,
                "name": "康夫太太",
                "score": 374.0,
                "sample_view": 447,
                "sources": ["name", "topic"],
            },
            {
                "mid": 3494377187969081,
                "name": "远康夫妇",
                "score": 334.0,
                "sample_view": 302857,
                "sources": ["name", "topic"],
            },
        ]
    }
    searcher, captured = make_searcher(owner_result)

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["extra_filters"] = kwargs["extra_filters"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("康夫 UI 工作流", limit=5)

    assert captured["dsl_query"] == "康夫 UI 工作流"
    assert captured["extra_filters"] == [
        {"terms": {"owner.mid": [14813517, 3494377187969081]}}
    ]
    assert "owners" in result["intent_info"]
    assert "owner_filter" not in result["intent_info"]
    assert result["semantic_rewrite_info"]["alias_rewritten"] is False


def test_search_suppresses_owner_filter_for_title_like_query_with_partial_overlap():
    owner_result = {
        "owners": [
            {
                "mid": 535744537,
                "name": "带你去有风的地方吧",
                "score": 241.5,
                "sample_view": 48,
                "sources": ["name", "topic"],
            }
        ]
    }
    searcher, captured = make_searcher(owner_result)

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["extra_filters"] = kwargs["extra_filters"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("女生吉他弹唱《去有风的地方》", limit=5)

    assert captured["dsl_query"] == "女生吉他弹唱《去有风的地方》"
    assert captured["extra_filters"] == []
    assert result["intent_info"]["owner"]["mid"] == 535744537
    assert "owner_filter" not in result["intent_info"]
    assert result["intent_info"]["filter_suppressed_reason"] == "title_like_query"


def test_search_suppresses_topic_only_owner_filter_for_bracketed_mixed_query():
    owner_result = {
        "owners": [
            {
                "mid": 294911874,
                "name": "千早爱音要玩终末地",
                "score": 190.0,
                "sample_view": 4271,
                "sources": ["topic"],
            }
        ]
    }
    searcher, captured = make_searcher(owner_result)

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["extra_filters"] = kwargs["extra_filters"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("【AI爱音】Pieces", limit=5)

    assert captured["dsl_query"] == "AI爱音 Pieces"
    assert captured["extra_filters"] == []
    assert result["intent_info"] == {}


def test_search_suppresses_owner_filter_for_owner_name_plus_title_query():
    owner_result = {
        "owners": [
            {
                "mid": 138886859,
                "name": "一只小雪莉ovo",
                "score": 305.0,
                "sample_view": 857,
                "sources": ["name", "topic"],
            }
        ]
    }
    searcher, captured = make_searcher(owner_result)

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["extra_filters"] = kwargs["extra_filters"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("【一只小雪莉ovo】寄明月~", limit=5)

    assert captured["dsl_query"] == "一只小雪莉ovo 寄明月"
    assert captured["extra_filters"] == []
    assert result["intent_info"] == {}


def test_search_suppresses_owner_filter_for_broad_short_query_with_long_owner_name():
    owner_result = {
        "owners": [
            {
                "mid": 92929,
                "name": "直播回放哟",
                "score": 245.0,
                "sample_view": 1203,
                "sources": ["name", "topic"],
            }
        ]
    }
    searcher, captured = make_searcher(owner_result)

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["extra_filters"] = kwargs["extra_filters"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("直播", limit=5)

    assert captured["dsl_query"] == "直播"
    assert captured["extra_filters"] == []
    assert result["intent_info"]["owner"]["mid"] == 92929
    assert "owner_filter" not in result["intent_info"]
    assert result["intent_info"]["filter_suppressed_reason"] == "broad_short_query"


def test_search_suppresses_owner_filter_for_broad_short_query_without_name_source():
    owner_result = {
        "owners": [
            {
                "mid": 88888,
                "name": "乄Marquis",
                "score": 231.0,
                "sample_view": 512,
                "sources": ["topic"],
            }
        ]
    }
    searcher, captured = make_searcher(owner_result)

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["extra_filters"] = kwargs["extra_filters"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("ai奶绿", limit=5)

    assert captured["dsl_query"] == "ai奶绿"
    assert captured["extra_filters"] == []
    assert result["intent_info"]["owner"]["mid"] == 88888
    assert "owner_filter" not in result["intent_info"]
    assert result["intent_info"]["filter_suppressed_reason"] == "broad_short_query"


def test_search_keeps_owner_filter_for_exact_short_owner_name_query():
    owner_result = {
        "owners": [
            {
                "mid": 77777,
                "name": "月栖乐序",
                "score": 280.0,
                "sample_view": 643,
                "sources": ["name", "topic"],
            }
        ]
    }
    searcher, captured = make_searcher(owner_result)

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["extra_filters"] = kwargs["extra_filters"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("月栖乐序", limit=5)

    assert captured["dsl_query"] == "月栖乐序"
    assert captured["extra_filters"] == [{"term": {"owner.mid": 77777}}]
    assert result["intent_info"]["owner_filter"] == [{"term": {"owner.mid": 77777}}]


def test_search_focus_rewrites_bracketed_owner_title_query_before_dsl():
    searcher, captured = make_searcher({"owners": []})

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("【一只小雪莉ovo】寄明月~", limit=5)

    assert captured["dsl_query"] == "一只小雪莉ovo 寄明月"
    assert result["query_focus_info"]["applied"] is True
    assert result["query_focus_info"]["applied_query"] == "一只小雪莉ovo 寄明月"


def test_search_focus_strips_boilerplate_suffix_before_dsl():
    searcher, captured = make_searcher({"owners": []})

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search(
        "【一只小雪莉ovo】寄明月~ 点点关注不错过 持续更新系列中",
        limit=5,
    )

    assert captured["dsl_query"] == "一只小雪莉ovo 寄明月"
    assert result["query_focus_info"]["applied"] is True
    assert result["query_focus_info"]["applied_query"] == "一只小雪莉ovo 寄明月"
    assert "relation_kwargs" not in captured
    assert result["semantic_rewrite_info"]["relation_rewritten"] is False


def test_search_focus_keeps_bracket_prefix_when_body_is_creator_boilerplate():
    searcher, captured = make_searcher({"owners": []})

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("【大毛-小厨】Up主探索中，欢迎收看求三连！", limit=5)

    assert captured["dsl_query"] == "大毛-小厨 Up主探索中，欢迎收看求三连"
    assert result["query_focus_info"]["applied"] is True
    assert (
        result["query_focus_info"]["applied_query"]
        == "大毛-小厨 Up主探索中，欢迎收看求三连"
    )
    assert "relation_kwargs" not in captured
    assert result["semantic_rewrite_info"]["relation_rewritten"] is False


def test_focused_title_reranker_promotes_near_exact_typo_match():
    hits = [
        {"title": "【大毛小样儿】Up主探索中，欢迎收看求三连！", "score": 248.47},
        {"title": "【大毛毛球儿】Up主探索中，欢迎收看求三连！", "score": 246.67},
        {"title": "【大猫-小厨】Up主探索中，欢迎收看求三连！", "score": 246.05},
    ]

    reranked, info = rerank_focused_title_hits(
        hits,
        query="大毛-小厨 Up主探索中，欢迎收看求三连",
        focus_applied=True,
        relation_rewritten=False,
    )

    assert reranked[0]["title"] == "【大猫-小厨】Up主探索中，欢迎收看求三连！"
    assert info.applied is True


def test_focused_title_reranker_skips_relation_rewritten_queries():
    hits = [
        {"title": "A", "score": 10.0},
        {"title": "B", "score": 9.0},
    ]

    reranked, info = rerank_focused_title_hits(
        hits,
        query="一只小雪莉ovo 寄明月",
        focus_applied=True,
        relation_rewritten=True,
    )

    assert reranked == hits
    assert info.applied is False


def test_search_focus_rewrites_date_recording_query_before_dsl():
    searcher, captured = make_searcher({"owners": []})

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search(
        "2026.04.21 T.赵俊杰个播录屏④（22：50—23：42）",
        limit=5,
    )

    assert captured["dsl_query"] == "赵俊杰个播录屏④"
    assert result["query_focus_info"]["applied"] is True
    assert result["query_focus_info"]["applied_query"] == "赵俊杰个播录屏④"


def test_search_focus_rewrites_segmented_long_title_before_dsl():
    searcher, captured = make_searcher({"owners": []})

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search(
        "灵疾｜先天剧本影响有多大，8个字原始代码理隐秘的前世今生",
        limit=5,
    )

    assert captured["dsl_query"] == "先天剧本影响有多大 8个字原始代码理隐秘的前世今生"
    assert result["query_focus_info"]["applied"] is True
    assert (
        result["query_focus_info"]["applied_query"]
        == "先天剧本影响有多大 8个字原始代码理隐秘的前世今生"
    )


def make_explorer(
    owner_result: dict, qmod: list[str] | None = None
) -> tuple[VideoExplorer, dict]:
    explorer = VideoExplorer.__new__(VideoExplorer)
    captured: dict = {}

    class _StubOwnerSearcher:
        @staticmethod
        def search(query, mode="name", size=5):
            return owner_result

    def _capture_call(path: str, **kwargs):
        captured["path"] = path
        captured["query"] = kwargs["query"]
        captured["extra_filters"] = kwargs["extra_filters"]
        captured["constraint_filter"] = kwargs.get("constraint_filter")
        captured["owner_intent_info"] = kwargs.get("owner_intent_info")
        captured["enable_rerank"] = kwargs.get("enable_rerank")
        return {
            "query": kwargs["query"],
            "status": "finished",
            "data": [{"output": {"hits": [], "total_hits": 0}}],
        }

    class _StubEmbedClient:
        @staticmethod
        def is_available():
            return True

    explorer._owner_searcher = _StubOwnerSearcher()
    explorer.query_rewriter = DslExprRewriter()
    explorer._embed_client = _StubEmbedClient()
    explorer.get_qmod_from_query = lambda query: qmod or ["word"]
    explorer.explore_v2 = lambda **kwargs: _capture_call("word", **kwargs)
    explorer.knn_explore_v2 = lambda **kwargs: _capture_call("vector", **kwargs)
    explorer.hybrid_explore_v2 = lambda **kwargs: _capture_call("hybrid", **kwargs)
    return explorer, captured


def test_search_adds_owner_mid_filter_for_confident_owner_intent_query():
    owner_result = {
        "owners": [
            {
                "mid": 1935882,
                "name": "指法芬芳张大仙",
                "score": 190.0,
                "sources": ["topic"],
            },
            {
                "mid": 385089497,
                "name": "王者荣耀-大仙",
                "score": 150.0,
                "sources": ["name"],
            },
        ]
    }
    searcher, captured = make_searcher(owner_result)

    result = searcher.search("王者荣耀张大仙", limit=5)

    assert result["intent_info"]["owner"]["mid"] == 1935882
    assert result["intent_info"]["owner_filter"] == [{"term": {"owner.mid": 1935882}}]


def test_search_skips_owner_mid_filter_for_ambiguous_owner_intent_query():
    owner_result = {
        "owners": [
            {
                "mid": 1,
                "name": "红色警戒21-",
                "score": 150.0,
                "sources": ["name"],
            },
            {
                "mid": 2,
                "name": "红色警戒7",
                "score": 146.5,
                "sources": ["name"],
            },
        ]
    }
    searcher, captured = make_searcher(owner_result)

    result = searcher.search("红色警戒08", limit=5)

    assert result["intent_info"] == {}


def test_search_uses_spaced_owner_intent_filter_and_context_query():
    owner_result = {
        "owners": [
            {
                "mid": 502925577,
                "name": "袁启聪",
                "score": 220.5,
                "sample_view": 111411,
                "sources": ["name"],
            }
        ]
    }
    searcher, captured = make_searcher(owner_result)

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["extra_filters"] = kwargs["extra_filters"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    class _NonZeroParser:
        @staticmethod
        def parse(*args, **kwargs):
            return {
                "query_info": {},
                "rewrite_info": {},
                "hits": [{"title": "owner hit"}],
                "total_hits": 1,
                "return_hits": 1,
            }

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info
    searcher.hit_parser = _NonZeroParser()

    result = searcher.search("袁启 采访", limit=5)

    assert captured["dsl_query"] == "采访"
    assert captured["extra_filters"] == [{"term": {"owner.mid": 502925577}}]
    assert result["intent_info"]["query"] == "袁启"
    assert result["intent_info"]["source_query"] == "袁启 采访"
    assert result["intent_info"]["owner"]["mid"] == 502925577


def test_search_uses_spaced_owner_candidate_terms_filter_when_owner_is_ambiguous():
    owner_result = {
        "owners": [
            {
                "mid": 3546588246968975,
                "name": "袁启豪",
                "score": 224.0,
                "sample_view": 0,
                "sources": ["name"],
            },
            {
                "mid": 502925577,
                "name": "袁启聪",
                "score": 220.5,
                "sample_view": 111411,
                "sources": ["name"],
            },
            {
                "mid": 374010007,
                "name": "袁启俊H2O",
                "score": 217.0,
                "sample_view": 4843,
                "sources": ["name"],
            },
        ]
    }
    searcher, captured = make_searcher(owner_result)

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["extra_filters"] = kwargs["extra_filters"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    class _NonZeroParser:
        @staticmethod
        def parse(*args, **kwargs):
            return {
                "query_info": {},
                "rewrite_info": {},
                "hits": [{"title": "owner hit"}],
                "total_hits": 1,
                "return_hits": 1,
            }

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info
    searcher.hit_parser = _NonZeroParser()

    result = searcher.search("袁启 采访", limit=5)

    assert captured["dsl_query"] == "采访"
    assert captured["extra_filters"] == [
        {"terms": {"owner.mid": [502925577, 374010007, 3546588246968975]}}
    ]
    assert result["intent_info"]["query"] == "袁启"
    assert result["intent_info"]["source_query"] == "袁启 采访"
    assert "owner_filter" not in result["intent_info"]


def test_search_retries_spaced_owner_context_with_original_query_on_zero_hits():
    owner_result = {
        "owners": [
            {
                "mid": 3546588246968975,
                "name": "袁启豪",
                "score": 224.0,
                "sample_view": 0,
                "sources": ["name"],
            },
            {
                "mid": 502925577,
                "name": "袁启聪",
                "score": 220.5,
                "sample_view": 111411,
                "sources": ["name"],
            },
            {
                "mid": 374010007,
                "name": "袁启俊H2O",
                "score": 217.0,
                "sample_view": 4843,
                "sources": ["name"],
            },
        ]
    }
    searcher, captured = make_searcher(owner_result)
    captured["queries"] = []
    captured["filters"] = []

    def _capture_get_info(**kwargs):
        captured["queries"].append(kwargs["query"])
        captured["filters"].append(kwargs["extra_filters"])
        captured["current_query"] = kwargs["query"]
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    class _RetryParser:
        @staticmethod
        def parse(*args, **kwargs):
            if captured["current_query"] == "采访":
                return {
                    "query_info": {},
                    "rewrite_info": {},
                    "hits": [],
                    "total_hits": 0,
                    "return_hits": 0,
                }
            return {
                "query_info": {},
                "rewrite_info": {},
                "hits": [{"title": "fallback hit"}],
                "total_hits": 1,
                "return_hits": 1,
            }

    searcher.get_info_of_query_rewrite_dsl = _capture_get_info
    searcher.hit_parser = _RetryParser()

    result = searcher.search("袁启 采访", limit=5)

    assert captured["queries"] == ["采访", "袁启 采访"]
    assert captured["filters"] == [
        [{"terms": {"owner.mid": [502925577, 374010007, 3546588246968975]}}],
        [{"terms": {"owner.mid": [502925577, 374010007, 3546588246968975]}}],
    ]
    assert result["total_hits"] == 1
    assert result["retry_info"]["relaxed_spaced_owner_context"] is True


def test_search_keeps_multi_owner_candidates_without_hard_filter_for_broad_query():
    owner_result = {
        "owners": [
            {
                "mid": 17528467,
                "name": "红警360",
                "score": 333.5,
                "sources": ["name", "topic"],
            },
            {
                "mid": 1629347259,
                "name": "红警HBK08",
                "score": 290.0,
                "sources": ["name", "topic"],
            },
            {
                "mid": 31961422,
                "name": "红警土豆_",
                "score": 285.5,
                "sources": ["name", "topic"],
            },
        ]
    }
    searcher, captured = make_searcher(owner_result)

    result = searcher.search("红警", limit=5)

    assert "owner_filter" not in result["intent_info"]
    assert [owner["name"] for owner in result["intent_info"]["owners"]] == [
        "红警360",
        "红警HBK08",
        "红警土豆_",
    ]


def test_search_skips_owner_intent_for_model_like_ascii_query():
    owner_result = {
        "owners": [
            {
                "mid": 675107370,
                "name": "b2002410",
                "score": 329.9559,
                "sources": ["name", "topic"],
            }
        ]
    }
    searcher, _ = make_searcher(owner_result)

    result = searcher.search("b200", limit=5)

    assert result["intent_info"] == {}


def test_search_retries_model_code_attribute_query_on_low_recall():
    searcher, captured = make_searcher({"owners": []})
    captured["dsl_queries"] = []

    def _get_info(**kwargs):
        captured["dsl_queries"].append(kwargs["query"])
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    def _parse(*args, **kwargs):
        if len(captured["dsl_queries"]) == 1:
            return {
                "query_info": {},
                "rewrite_info": {},
                "hits": [{"title": "only one strict hit"}],
                "total_hits": 1,
                "return_hits": 1,
            }
        return {
            "query_info": {},
            "rewrite_info": {},
            "hits": [{"title": "relaxed hit"} for _ in range(5)],
            "total_hits": 5,
            "return_hits": 5,
        }

    searcher.get_info_of_query_rewrite_dsl = _get_info
    searcher.hit_parser.parse = _parse

    result = searcher.search("b200 价格", limit=5)

    assert len(captured["dsl_queries"]) == 2
    assert result["total_hits"] == 5
    assert result["retry_info"]["relaxed_auto_exact_segments"] is True


def test_search_retries_without_short_han_exact_on_zero_hits():
    searcher, captured = make_searcher({"owners": []})
    captured["short_han_exact_flags"] = []
    captured["match_fields"] = []

    def _get_info(**kwargs):
        captured["short_han_exact_flags"].append(
            get_auto_require_short_han_exact_mode()
        )
        return (
            captured.setdefault("query_info", {}),
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher.get_info_of_query_rewrite_dsl = _get_info
    searcher.construct_search_body = lambda **kwargs: captured["match_fields"].append(
        kwargs["match_fields"]
    ) or {"query": kwargs["query_dsl_dict"]}

    result = searcher.search("袁启 采访", limit=5)

    assert captured["short_han_exact_flags"] == ["all", "first"]
    assert "title.words^3" in captured["match_fields"][0]
    assert "title.words^5.0" in captured["match_fields"][1]
    assert "owner.name.words^4.0" in captured["match_fields"][1]
    assert (
        f"desc.words^{SEARCH_BOOSTED_FIELDS['desc.words']}"
        in captured["match_fields"][0]
    )
    assert "desc.words^0.02" in captured["match_fields"][1]
    assert result["retry_info"]["relaxed_short_han_exact"] is True


def test_resolve_vector_auto_constraint_query_prefers_short_han_owner_anchor():
    owner_result = {
        "owners": [
            {
                "mid": 502925577,
                "name": "袁启聪",
                "score": 220.5,
                "sources": ["name"],
            },
            {
                "mid": 374010007,
                "name": "袁启俊H2O",
                "score": 217.0,
                "sources": ["name"],
            },
        ]
    }
    searcher, _ = make_searcher(owner_result)

    assert searcher._resolve_vector_auto_constraint_query("袁启 采访") == "袁启"


def test_resolve_vector_auto_constraint_query_keeps_topic_query_without_compact_prefix():
    owner_result = {
        "owners": [
            {
                "mid": 3546742693825222,
                "name": "宇哥鬼畜剧场",
                "score": 338.0,
                "sources": ["name", "topic"],
            },
            {
                "mid": 16054375,
                "name": "自动鬼畜中的WZ",
                "score": 242.5,
                "sources": ["name", "topic"],
            },
            {
                "mid": 3546953346452353,
                "name": "鬼畜天线宝宝糕手",
                "score": 220.5,
                "sources": ["name"],
            },
        ]
    }
    searcher, _ = make_searcher(owner_result)

    assert searcher._resolve_vector_auto_constraint_query("鬼畜 教程") == "鬼畜 教程"


def test_resolve_spaced_owner_intent_info_uses_owner_anchor_query():
    owner_result = {
        "owners": [
            {
                "mid": 502925577,
                "name": "袁启聪",
                "score": 220.5,
                "sources": ["name"],
            }
        ]
    }
    searcher, _ = make_searcher(owner_result)

    result = searcher._resolve_spaced_owner_intent_info("袁启 采访")

    assert result["query"] == "袁启"
    assert result["source_query"] == "袁启 采访"
    assert result["owner"]["mid"] == 502925577
    assert result["owner_filter"] == [{"term": {"owner.mid": 502925577}}]


def test_resolve_spaced_owner_intent_info_reranks_near_tied_candidates_by_sample_view():
    owner_result = {
        "owners": [
            {
                "mid": 3546588246968975,
                "name": "袁启豪",
                "score": 224.0,
                "sample_view": 0,
                "sources": ["name"],
            },
            {
                "mid": 502925577,
                "name": "袁启聪",
                "score": 220.5,
                "sample_view": 111411,
                "sources": ["name"],
            },
            {
                "mid": 374010007,
                "name": "袁启俊H2O",
                "score": 217.0,
                "sample_view": 4843,
                "sources": ["name"],
            },
        ]
    }
    searcher, _ = make_searcher(owner_result)

    result = searcher._resolve_spaced_owner_intent_info("袁启 采访")

    assert [owner["name"] for owner in result["owners"][:3]] == [
        "袁启聪",
        "袁启俊H2O",
        "袁启豪",
    ]


def test_unified_explore_builds_auto_constraint_from_owner_anchor(monkeypatch):
    owner_result = {
        "owners": [
            {
                "mid": 502925577,
                "name": "袁启聪",
                "score": 220.5,
                "sources": ["name"],
            }
        ]
    }
    explorer, captured = make_explorer(owner_result, qmod=["vector"])
    explorer.index_name = "test_index"
    explorer.es = type("_StubES", (), {"client": object()})()

    def _stub_build_auto_constraint_filter(*args, **kwargs):
        captured["constraint_query"] = kwargs["query"]
        return {
            "es_tok_constraints": {"constraints": [{"have_token": [kwargs["query"]]}]}
        }

    monkeypatch.setattr(
        "elastics.videos.explorer.build_auto_constraint_filter",
        _stub_build_auto_constraint_filter,
    )

    explorer.unified_explore("袁启 采访 q=v", _allow_short_han_retry=False)

    assert captured["constraint_query"] == "袁启"
    assert captured["constraint_filter"] == {
        "es_tok_constraints": {"constraints": [{"have_token": ["袁启"]}]}
    }


def test_unified_explore_uses_spaced_owner_intent_anchor():
    owner_result = {
        "owners": [
            {
                "mid": 502925577,
                "name": "袁启聪",
                "score": 220.5,
                "sources": ["name"],
            }
        ]
    }
    explorer, captured = make_explorer(owner_result, qmod=["word", "vector"])

    result = explorer.unified_explore("袁启 采访", _allow_short_han_retry=False)

    assert captured["owner_intent_info"]["query"] == "袁启"
    assert captured["owner_intent_info"]["source_query"] == "袁启 采访"
    assert captured["owner_intent_info"]["owner"]["mid"] == 502925577
    assert result["intent_info"]["owner"]["mid"] == 502925577


def test_unified_explore_uses_spaced_owner_intent_anchor_with_qmod_marker():
    owner_result = {
        "owners": [
            {
                "mid": 502925577,
                "name": "袁启聪",
                "score": 220.5,
                "sources": ["name"],
            }
        ]
    }
    explorer, captured = make_explorer(owner_result, qmod=["vector"])

    result = explorer.unified_explore("袁启 采访 q=v", _allow_short_han_retry=False)

    assert captured["owner_intent_info"]["query"] == "袁启"
    assert captured["owner_intent_info"]["source_query"] == "袁启 采访"
    assert captured["owner_intent_info"]["owner"]["mid"] == 502925577
    assert result["intent_info"]["owner"]["mid"] == 502925577


def test_owner_intent_supplement_filters_use_multi_owner_terms_for_spaced_query():
    owner_intent_info = {
        "query": "袁启",
        "source_query": "袁启 采访",
        "owners": [
            {"mid": 3546588246968975, "name": "袁启豪", "score": 224.0},
            {"mid": 502925577, "name": "袁启聪", "score": 220.5},
            {"mid": 374010007, "name": "袁启俊H2O", "score": 217.0},
        ],
    }

    filters = VideoExplorer._resolve_owner_intent_supplement_filters(owner_intent_info)

    assert filters == [
        {"terms": {"owner.mid": [3546588246968975, 502925577, 374010007]}}
    ]


def test_owner_intent_supplement_filters_preserve_single_owner_filter():
    owner_intent_info = {
        "query": "张大仙",
        "owner": {"mid": 1935882, "name": "指法芬芳张大仙", "score": 190.0},
        "owner_filter": [{"term": {"owner.mid": 1935882}}],
        "owners": [{"mid": 1935882, "name": "指法芬芳张大仙", "score": 190.0}],
    }

    filters = VideoExplorer._resolve_owner_intent_supplement_filters(owner_intent_info)

    assert filters == [{"term": {"owner.mid": 1935882}}]


def test_owner_intent_supplement_uses_anchor_query_for_spaced_fallback():
    explorer, captured = make_explorer({"owners": []}, qmod=["vector"])

    def _search(**kwargs):
        captured["supplement_query"] = kwargs["query"]
        captured["supplement_filters"] = kwargs["extra_filters"]
        return {
            "hits": [
                {
                    "bvid": "BV1owner",
                    "title": "袁启聪采访实录",
                    "owner": {"mid": 502925577, "name": "袁启聪"},
                    "score": 3.0,
                }
            ],
            "total_hits": 1,
            "timed_out": False,
        }

    explorer.search = _search
    pool = RecallPool()
    owner_intent_info = {
        "query": "袁启",
        "source_query": "袁启 采访",
        "owners": [
            {"mid": 502925577, "name": "袁启聪", "score": 220.5},
            {"mid": 374010007, "name": "袁启俊H2O", "score": 217.0},
        ],
    }

    supplemented = explorer._supplement_with_owner_intent_hits(
        pool=pool,
        query="袁启 采访 q=vwr",
        owner_intent_info=owner_intent_info,
        verbose=False,
    )

    assert captured["supplement_query"] == "袁启"
    assert captured["supplement_filters"] == [
        {"terms": {"owner.mid": [502925577, 374010007]}}
    ]
    assert supplemented.hits[0]["owner"]["name"] == "袁启聪"


def test_spaced_owner_context_terms_extract_tail_terms():
    owner_intent_info = {
        "query": "袁启",
        "source_query": "袁启 采访 视频",
    }

    assert VideoExplorer._get_spaced_owner_context_terms(owner_intent_info) == [
        "采访",
        "视频",
    ]


def test_owner_intent_supplement_filters_context_blind_hits_for_spaced_query():
    explorer = VideoExplorer.__new__(VideoExplorer)

    def _search(**kwargs):
        return {
            "hits": [
                {
                    "bvid": "BV-context",
                    "title": "袁启聪采访录：聊聊老头乐",
                    "owner": {"mid": 502925577, "name": "袁启聪"},
                    "score": 2.0,
                },
                {
                    "bvid": "BV-no-context",
                    "title": "老车玩家狂喜，这里才是真正宝藏店！",
                    "owner": {"mid": 502925577, "name": "袁启聪"},
                    "score": 3.0,
                },
            ],
            "total_hits": 2,
            "timed_out": False,
        }

    explorer.search = _search
    pool = RecallPool()
    owner_intent_info = {
        "query": "袁启",
        "source_query": "袁启 采访",
        "owners": [{"mid": 502925577, "name": "袁启聪", "score": 220.5}],
    }

    supplemented = explorer._supplement_with_owner_intent_hits(
        pool=pool,
        query="袁启 采访 q=vwr",
        owner_intent_info=owner_intent_info,
        verbose=False,
    )

    assert [hit["bvid"] for hit in supplemented.hits] == ["BV-context"]
    assert supplemented.lanes_info["owner_intent"]["hit_count"] == 1


def test_build_group_step_injects_missing_owner_candidates_without_hits():
    explorer = VideoExplorer.__new__(VideoExplorer)
    explorer.get_user_docs = lambda mids: {
        502925577: {"mid": 502925577, "face": "face://yuan"}
    }
    search_res = {
        "hits": [
            {
                "bvid": "BV-other",
                "owner": {"mid": 18385164, "name": "折腿的老猫"},
                "rank_score": 1.0,
                "pubdate": 1,
                "stat": {"view": 100},
            }
        ]
    }
    owner_intent_info = {
        "query": "袁启",
        "source_query": "袁启 采访",
        "owners": [{"mid": 502925577, "name": "袁启聪", "score": 220.5}],
    }

    authors = explorer._build_group_step(
        search_res=search_res,
        group_owner_limit=5,
        owner_intent_info=owner_intent_info,
    )

    assert authors[0]["name"] == "袁启聪"
    assert authors[0]["intent_match"] is True
    assert authors[0]["face"] == "face://yuan"
    assert authors[1]["name"] == "折腿的老猫"


def test_build_group_step_prefers_reranked_spaced_owner_order_for_seeded_authors():
    explorer = VideoExplorer.__new__(VideoExplorer)
    explorer.get_user_docs = lambda mids: {
        502925577: {"mid": 502925577, "face": "face://yuan"},
        374010007: {"mid": 374010007, "face": "face://jun"},
        3546588246968975: {"mid": 3546588246968975, "face": "face://hao"},
    }
    search_res = {
        "hits": [
            {
                "bvid": "BV-other",
                "owner": {"mid": 18385164, "name": "折腿的老猫"},
                "rank_score": 1.0,
                "pubdate": 1,
                "stat": {"view": 100},
            }
        ]
    }
    owner_intent_info = {
        "query": "袁启",
        "source_query": "袁启 采访",
        "owners": [
            {"mid": 502925577, "name": "袁启聪", "score": 220.5},
            {"mid": 374010007, "name": "袁启俊H2O", "score": 217.0},
            {"mid": 3546588246968975, "name": "袁启豪", "score": 224.0},
        ],
    }

    authors = explorer._build_group_step(
        search_res=search_res,
        group_owner_limit=6,
        owner_intent_info=owner_intent_info,
    )

    assert [author["name"] for author in authors[:4]] == [
        "袁启聪",
        "袁启俊H2O",
        "袁启豪",
        "折腿的老猫",
    ]


def test_unified_explore_keeps_owner_intent_info_without_hard_filtering_all_recalls():
    owner_result = {
        "owners": [
            {
                "mid": 1629347259,
                "name": "红警HBK08",
                "score": 190.0,
                "sources": ["topic"],
            }
        ]
    }
    explorer, captured = make_explorer(owner_result, qmod=["word"])

    result = explorer.unified_explore("红色警戒08")

    assert captured["extra_filters"] == []
    assert captured["owner_intent_info"]["owner"]["mid"] == 1629347259
    assert result["intent_info"]["owner"]["mid"] == 1629347259
    assert result["intent_info"]["owner_filter"] == [
        {"term": {"owner.mid": 1629347259}}
    ]


def test_unified_explore_scope_forces_word_mode_over_hybrid_qmod():
    explorer, captured = make_explorer({"owners": []}, qmod=["word", "vector"])

    result = explorer.unified_explore("红警 :scope=n")

    assert captured["path"] == "word"
    assert captured["enable_rerank"] is False
    assert result["data"][0]["output"]["qmod"] == ["word"]


def test_unified_explore_retries_without_short_han_exact_on_zero_hits():
    explorer, captured = make_explorer({"owners": []}, qmod=["word", "vector"])
    captured["short_han_exact_flags"] = []

    def _capture_hybrid(**kwargs):
        captured["path"] = "hybrid"
        captured["short_han_exact_flags"].append(
            get_auto_require_short_han_exact_mode()
        )
        return {
            "query": kwargs["query"],
            "status": "finished",
            "data": [{"output": {"hits": [], "total_hits": 0}}],
        }

    explorer.hybrid_explore_v2 = _capture_hybrid

    result = explorer.unified_explore("袁启 采访")

    assert captured["path"] == "hybrid"
    assert captured["short_han_exact_flags"] == ["all", "first"]
    assert result["retry_info"]["relaxed_short_han_exact"] is True


def test_owner_intent_supplement_adds_hits_without_replacing_existing_pool():
    explorer = VideoExplorer.__new__(VideoExplorer)
    captured: dict = {}

    def _stub_search(**kwargs):
        captured["extra_filters"] = kwargs["extra_filters"]
        return {
            "hits": [
                {"bvid": "owner-1", "score": 12.0},
                {"bvid": "base-1", "score": 11.0},
            ],
            "total_hits": 2,
            "timed_out": False,
        }

    explorer.search = _stub_search
    base_pool = RecallPool(
        hits=[
            {"bvid": "base-1", "score": 8.0},
            {"bvid": "base-2", "score": 7.0},
        ],
        lanes_info={"word": {"hit_count": 2}},
        total_hits=2,
        lane_tags={"base-1": {"word"}, "base-2": {"word"}},
    )

    result_pool = explorer._supplement_with_owner_intent_hits(
        pool=base_pool,
        query="红色警戒08",
        owner_intent_info={
            "owner": {"mid": 1629347259, "name": "红警HBK08"},
            "owner_filter": [{"term": {"owner.mid": 1629347259}}],
        },
        extra_filters=[],
        rank_top_k=400,
    )

    assert captured["extra_filters"] == [{"term": {"owner.mid": 1629347259}}]
    assert [hit["bvid"] for hit in result_pool.hits] == [
        "base-1",
        "base-2",
        "owner-1",
    ]
    assert result_pool.lane_tags["base-1"] == {"word", "owner_intent"}
    assert result_pool.lane_tags["owner-1"] == {"owner_intent"}
    assert result_pool.lanes_info["owner_intent"]["owner_name"] == "红警HBK08"


def test_owner_intent_blend_surfaces_late_owner_hits_without_collapsing_results():
    explorer = VideoExplorer.__new__(VideoExplorer)
    hits = [
        {
            "bvid": f"base-{idx}",
            "owner": {"mid": idx, "name": f"owner-{idx}"},
        }
        for idx in range(12)
    ]
    hits.extend(
        [
            {
                "bvid": "owner-1",
                "owner": {"mid": 1629347259, "name": "红警HBK08"},
                "owner_intent_rank": 0,
            },
            {
                "bvid": "base-12",
                "owner": {"mid": 12, "name": "owner-12"},
            },
            {
                "bvid": "owner-2",
                "owner": {"mid": 1629347259, "name": "红警HBK08"},
                "owner_intent_rank": 1,
            },
        ]
    )

    search_res = explorer._blend_owner_intent_hits(
        search_res={"hits": hits},
        owner_intent_info={"owner": {"mid": 1629347259, "name": "红警HBK08"}},
    )

    top_ten_bvids = [hit["bvid"] for hit in search_res["hits"][:10]]
    assert top_ten_bvids[1] == "owner-1"
    assert top_ten_bvids[4] == "owner-2"
    assert len([bvid for bvid in top_ten_bvids if bvid.startswith("owner-")]) == 2
    assert search_res["owner_intent_blend"]["inserted_hits"] == 2


def test_owner_intent_author_group_rerank_uses_multi_owner_candidates_and_group_strength():
    authors = [
        {
            "mid": 370204215,
            "name": "罗辑的宇宙",
            "sum_count": 1,
            "sum_rank_score": 1.0,
            "top_rank_score": 1.0,
            "first_appear_order": 0,
        },
        {
            "mid": 31961422,
            "name": "红警土豆_",
            "sum_count": 45,
            "sum_rank_score": 25.4775,
            "top_rank_score": 0.9925,
            "first_appear_order": 3,
        },
        {
            "mid": 1629347259,
            "name": "红警HBK08",
            "sum_count": 74,
            "sum_rank_score": 47.4475,
            "top_rank_score": 0.985,
            "first_appear_order": 6,
        },
        {
            "mid": 1055866657,
            "name": "红警烧碱酱",
            "sum_count": 8,
            "sum_rank_score": 5.5575,
            "top_rank_score": 0.995,
            "first_appear_order": 2,
        },
    ]

    reranked = VideoExplorer._promote_owner_intent_author_group(
        authors_list=authors,
        owner_intent_info={
            "owners": [
                {
                    "mid": 17528467,
                    "name": "红警360",
                    "score": 333.5,
                    "sources": ["name", "topic"],
                },
                {
                    "mid": 1629347259,
                    "name": "红警HBK08",
                    "score": 290.0,
                    "sources": ["name", "topic"],
                },
                {
                    "mid": 31961422,
                    "name": "红警土豆_",
                    "score": 285.5,
                    "sources": ["name", "topic"],
                },
            ]
        },
    )

    assert [author["name"] for author in reranked[:3]] == [
        "红警HBK08",
        "红警土豆_",
        "红警烧碱酱",
    ]
    assert reranked[0]["intent_match"] is True
    assert reranked[1]["intent_match"] is True
