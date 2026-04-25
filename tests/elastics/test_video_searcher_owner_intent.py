"""Owner-intent and explore tests for VideoSearcherV2."""

from __future__ import annotations

from dsl.fields.word import get_auto_require_short_han_exact_mode
from elastics.videos.constants import SEARCH_BOOSTED_FIELDS
from video_searcher_test_utils import make_explorer, make_searcher


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
    captured["auto_exact_modes"] = []

    def _get_info(**kwargs):
        captured["dsl_queries"].append(kwargs["query"])
        captured["auto_exact_modes"].append(get_auto_require_short_han_exact_mode())
        return (
            {"words_expr": kwargs["query"]},
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
    assert captured["auto_exact_modes"] == ["all", "model_code"]
    assert result["total_hits"] == 5
    assert result["retry_info"]["relaxed_auto_exact_segments"] is True
    assert result["retry_info"]["kept_model_code_exact"] is True


def test_search_retries_model_code_attribute_query_with_qmod_marker():
    searcher, captured = make_searcher({"owners": []})
    captured["auto_exact_modes"] = []

    def _get_info(**kwargs):
        captured["auto_exact_modes"].append(get_auto_require_short_han_exact_mode())
        return (
            {"words_expr": "h20 显卡"},
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    def _parse(*args, **kwargs):
        if len(captured["auto_exact_modes"]) == 1:
            return {"hits": [], "total_hits": 0, "return_hits": 0}
        return {
            "hits": [{"title": "model-code relaxed hit"}],
            "total_hits": 1,
            "return_hits": 1,
        }

    searcher.get_info_of_query_rewrite_dsl = _get_info
    searcher.hit_parser.parse = _parse

    result = searcher.search("h20 显卡 q=vr", limit=5)

    assert captured["auto_exact_modes"] == ["all", "model_code"]
    assert result["return_hits"] == 1
    assert result["retry_info"]["kept_model_code_exact"] is True


def test_search_embedding_denoise_overfetches_model_code_retry():
    searcher, captured = make_searcher({"owners": []})
    captured["auto_exact_modes"] = []
    captured["limits"] = []

    def _get_info(**kwargs):
        captured["auto_exact_modes"].append(get_auto_require_short_han_exact_mode())
        return (
            {"words_expr": "h20 显卡"},
            captured.setdefault("rewrite_info", {}),
            captured.setdefault("query_dsl_dict", {"match_all": {}}),
        )

    def _construct_search_body(**kwargs):
        captured["limits"].append(kwargs["limit"])
        return {"query": kwargs["query_dsl_dict"]}

    def _parse(*args, **kwargs):
        if len(captured["auto_exact_modes"]) == 1:
            return {"hits": [], "total_hits": 0, "return_hits": 0}
        hits = [{"title": f"candidate {idx}"} for idx in range(12)]
        return {"hits": hits, "total_hits": 12, "return_hits": len(hits)}

    class _FakeReranker:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def rerank(**kwargs):
            captured["rerank_kwargs"] = kwargs
            hits = list(reversed(kwargs["hits"]))
            return hits, {
                "reranked_count": len(hits),
                "valid_passages": len(hits),
                "total_ms": 1,
            }

    searcher.get_info_of_query_rewrite_dsl = _get_info
    searcher.construct_search_body = _construct_search_body
    searcher.hit_parser.parse = _parse
    searcher._get_embedding_denoise_reranker = lambda: _FakeReranker()

    result = searcher.search("h20 显卡 q=vr", limit=5)

    assert captured["auto_exact_modes"] == ["all", "model_code"]
    assert captured["limits"] == [5, 160]
    assert captured["rerank_kwargs"]["query"] == "h20 显卡"
    assert captured["rerank_kwargs"]["max_rerank"] == 160
    assert [hit["title"] for hit in result["hits"]] == [
        "candidate 11",
        "candidate 10",
        "candidate 9",
        "candidate 8",
        "candidate 7",
    ]
    assert result["return_hits"] == 5
    assert result["embedding_denoise_info"]["applied"] is True
    assert result["retry_info"]["embedding_denoise_applied"] is True


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


def test_resolve_vector_auto_constraint_query_keeps_model_code_topic_query():
    owner_result = {
        "owners": [
            {
                "mid": 664452009,
                "name": "显卡维修圆",
                "score": 334.5,
                "sample_view": 6308,
                "sources": ["name", "topic"],
            }
        ]
    }
    searcher, _ = make_searcher(owner_result)

    assert searcher._resolve_vector_auto_constraint_query("显卡 h20") == "显卡 h20"
    assert searcher._resolve_spaced_owner_intent_info("显卡 h20") == {}


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
