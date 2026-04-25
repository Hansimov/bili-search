from dsl.rewrite import DslExprRewriter
from dsl.elastic import DslExprToElasticConverter
from dsl.filter import QueryDslDictFilterMerger
from elastics.videos.results.reranking import rerank_focused_title_hits
from elastics.videos.searcher_v2 import VideoSearcherV2
from video_searcher_test_utils import make_searcher


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


def test_search_applies_asset_backed_alias_rule():
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

    assert captured["dsl_query"] == "ComfyUI 教程"
    assert captured["suggest_info"] == {}
    assert result["semantic_rewrite_info"]["alias_rewritten"] is True
    assert result["semantic_rewrite_info"]["applied_query"] == "ComfyUI 教程"


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

    assert captured["relation_kwargs"]["text"] == "ComfyUI 教程"
    assert captured["dsl_query"] == "ComfyUI 教程"
    assert captured["suggest_info"] == {
        "group_replaces_count": [
            [["教程", "教学"], 920],
        ]
    }
    assert result["semantic_rewrite_info"]["relation_rewritten"] is True
    assert result["semantic_rewrite_info"]["applied_query"] == "ComfyUI 教学"


def test_search_builds_semantic_suggest_info_for_model_code_attribute_query_mod():
    searcher, captured = make_searcher({"owners": []})

    class _StubRelationsClient:
        @staticmethod
        def related_tokens_by_tokens(**kwargs):
            captured["relation_kwargs"] = kwargs
            return {
                "mode": kwargs.get("mode", "semantic"),
                "options": [
                    {"text": "h20 gpu", "score": 800.0},
                    {"text": "h2h显卡", "score": 900.0},
                ],
            }

    def _capture_get_info(**kwargs):
        captured["dsl_query"] = kwargs["query"]
        captured["suggest_info"] = kwargs["suggest_info"]
        return (
            captured.setdefault("query_info", {}),
            {"rewrited_word_exprs": ["h20 gpu"]},
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    searcher._relations_client = _StubRelationsClient()
    searcher.get_info_of_query_rewrite_dsl = _capture_get_info

    result = searcher.search("h20 显卡 q=vr", limit=5)

    assert captured["relation_kwargs"]["text"] == "h20 显卡"
    assert captured["dsl_query"] == "h20 显卡 q=vr"
    assert captured["suggest_info"] == {
        "group_replaces_count": [
            [["显卡", "gpu"], 800],
        ]
    }
    assert result["semantic_rewrite_info"]["relation_rewritten"] is True
    assert result["semantic_rewrite_info"]["relation_query"] == "h20 显卡"
    assert result["semantic_rewrite_info"]["applied_query"] == "h20 gpu"


def test_model_code_score_cliff_filter_prunes_weak_ambiguous_tail():
    result = {
        "hits": [
            {"title": "B200 GPU 价格曲线", "score": 12.0},
            {"title": "奔驰 B200", "score": 0.5},
            {"title": "B200 网关", "score": 0.4},
            {"title": "B200 外设", "score": 0.3},
            {"title": "B200 汽车维修", "score": 0.2},
        ],
        "return_hits": 5,
    }

    info = VideoSearcherV2._apply_model_code_score_cliff_filter(
        result,
        query="b200 价格",
        relation_rewritten=False,
    )

    assert info["applied"] is True
    assert info["min_keep"] == 3
    assert result["return_hits"] == 3
    assert [hit["title"] for hit in result["hits"]] == [
        "B200 GPU 价格曲线",
        "奔驰 B200",
        "B200 网关",
    ]


def test_search_keeps_owner_candidates_when_asset_alias_rewrites_query():
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

    assert captured["dsl_query"] == "ComfyUI 工作流"
    assert captured["extra_filters"] == [
        {"terms": {"owner.mid": [14813517, 3494377187969081]}}
    ]
    assert "owners" in result["intent_info"]
    assert "owner_filter" not in result["intent_info"]
    assert result["semantic_rewrite_info"]["alias_rewritten"] is True


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
