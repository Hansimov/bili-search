from dsl.fields.word import get_auto_require_short_han_exact_mode
from dsl.rewrite import DslExprRewriter
from dsl.elastic import DslExprToElasticConverter
from dsl.filter import QueryDslDictFilterMerger
from elastics.videos.explorer import VideoExplorer
from elastics.videos.constants import SEARCH_BOOSTED_FIELDS
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


def test_search_applies_alias_rewrite_before_building_query_dsl():
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


def test_search_builds_semantic_suggest_info_for_alias_query_relation_rewrite():
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
