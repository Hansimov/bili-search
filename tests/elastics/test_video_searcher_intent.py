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
