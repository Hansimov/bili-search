from elastics.videos.explorer import VideoExplorer
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

    searcher._owner_searcher = _StubOwnerSearcher()
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


def make_explorer(
    owner_result: dict, qmod: list[str] | None = None
) -> tuple[VideoExplorer, dict]:
    explorer = VideoExplorer.__new__(VideoExplorer)
    captured: dict = {}

    class _StubOwnerSearcher:
        @staticmethod
        def search(query, mode="name", size=5):
            return owner_result

    def _capture_call(**kwargs):
        captured["query"] = kwargs["query"]
        captured["extra_filters"] = kwargs["extra_filters"]
        captured["owner_intent_info"] = kwargs.get("owner_intent_info")
        return {
            "query": kwargs["query"],
            "status": "finished",
            "data": [{"output": {"hits": [], "total_hits": 0}}],
        }

    explorer._owner_searcher = _StubOwnerSearcher()
    explorer.get_qmod_from_query = lambda query: qmod or ["word"]
    explorer.explore_v2 = _capture_call
    explorer.knn_explore_v2 = _capture_call
    explorer.hybrid_explore_v2 = _capture_call
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
    assert top_ten_bvids[3] == "owner-1"
    assert top_ten_bvids[7] == "owner-2"
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
