"""Shared helpers for VideoSearcherV2 intent tests."""

from __future__ import annotations

from dsl.rewrite import DslExprRewriter
from elastics.videos.explorer import VideoExplorer
from elastics.videos.searcher_v2 import VideoSearcherV2


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
