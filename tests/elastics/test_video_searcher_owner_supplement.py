"""Owner-intent supplement and grouping tests for VideoExplorer."""

from __future__ import annotations

from dsl.fields.word import get_auto_require_short_han_exact_mode
from elastics.videos.explorer import VideoExplorer
from recalls.base import RecallPool
from video_searcher_test_utils import make_explorer


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
