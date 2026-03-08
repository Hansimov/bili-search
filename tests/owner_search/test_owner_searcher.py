import sys
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elastics.owners.hits import OwnerHitsParser
from elastics.owners.searcher import OwnerSearcher


class StubOwnerSearcher(OwnerSearcher):
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls = []
        self.hit_parser = OwnerHitsParser()
        self._latin_token_re = re.compile(r"[a-z0-9][a-z0-9_\-\.]{1,}")
        self._cjk_span_re = re.compile(r"[\u4e00-\u9fff]+")

    def _submit(self, body: dict, context: str = "search") -> dict:
        self.calls.append({"context": context, "body": body})
        return self.responses.pop(0)


def _make_hit(mid: int, name: str, score: float, **extra) -> dict:
    source = {
        "mid": mid,
        "name": name,
        "total_videos": extra.pop("total_videos", 0),
        "total_view": extra.pop("total_view", 0),
        "influence_score": extra.pop("influence_score", 0.0),
        "quality_score": extra.pop("quality_score", 0.0),
        "activity_score": extra.pop("activity_score", 0.0),
        "profile_domain_ready": extra.pop("profile_domain_ready", False),
        "core_tokenizer_version": extra.pop("core_tokenizer_version", "coretok-dev"),
    }
    source.update(extra)
    return {"_source": source, "_score": score}


def _make_response(*hits: dict) -> dict:
    return {
        "hits": {
            "hits": list(hits),
            "total": {"value": len(hits)},
            "max_score": max((hit.get("_score") for hit in hits), default=None),
        }
    }


def test_search_relevance_name_route_only_uses_name_query():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(_make_hit(1, "影视飓风", 1.0)),
            _make_response(
                _make_hit(
                    1,
                    "影视飓风",
                    45.0,
                    total_videos=500,
                    total_view=2500000000,
                    influence_score=0.92,
                    quality_score=0.78,
                    activity_score=0.66,
                ),
                _make_hit(
                    2,
                    "飓多多StormCrew",
                    11.0,
                    total_videos=120,
                    total_view=320000000,
                    influence_score=0.71,
                    quality_score=0.61,
                    activity_score=0.58,
                ),
            ),
        ]
    )

    result = searcher.search("影视飓风", sort_by="relevance", limit=5, compact=True)

    assert result["query_type"] == "name"
    assert [hit["mid"] for hit in result["hits"]] == [1, 2]
    assert result["hits"][0]["_score"] > result["hits"][1]["_score"]
    assert result["hits"][0]["name"] == "影视飓风"
    assert len(searcher.calls) == 2
    assert searcher.calls[0]["context"] == "route.exact_name"
    assert searcher.calls[1]["context"] == "search.name"
    assert result["domain_status"] == "name_route"


def test_search_non_relevance_domain_query_requires_profile_tokens():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(),
        ]
    )

    result = searcher.search("黑神话悟空", sort_by="influence", limit=3, compact=True)

    assert result["query_route"] == "domain"
    assert result["hits"] == []
    assert result["domain_status"] == "query_tokens_missing"
    assert len(searcher.calls) == 1
    assert searcher.calls[0]["context"] == "route.exact_name"


def test_search_by_domain_uses_profile_token_placeholders_when_provided():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(),
            _make_response(_make_hit(10, "黑神话研究所", 4.0, influence_score=0.63)),
        ]
    )

    result = searcher.search_by_domain(
        "黑神话悟空",
        sort_by="influence",
        limit=3,
        compact=True,
        tag_token_ids=[101, 202, 202],
        text_token_ids=[301, 302],
    )

    assert result["query_route"] == "domain"
    assert result["domain_status"] == "query_tokens_used"
    assert len(searcher.calls) == 2
    assert searcher.calls[0]["context"] == "route.exact_name"
    assert searcher.calls[1]["context"] == "search_by_domain.tokens"
    body = searcher.calls[1]["body"]
    assert body["sort"][0] == {"influence_score": {"order": "desc"}}
    should = body["query"]["bool"]["should"]
    assert any(
        clause.get("constant_score", {})
        .get("filter", {})
        .get("terms", {})
        .get("core_tag_token_ids")
        == [101, 202]
        for clause in should
    )
    assert any(
        clause.get("constant_score", {})
        .get("filter", {})
        .get("terms", {})
        .get("core_text_token_ids")
        == [301, 302]
        for clause in should
    )


def test_search_by_domain_without_tokens_returns_empty_placeholder_result():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(),
        ]
    )

    result = searcher.search_by_domain(
        "当你半年不上线的账号打一把王者排位的时候",
        sort_by="influence",
        limit=3,
        compact=True,
    )

    assert searcher.calls == []
    assert result["query_route"] == "phrase"
    assert result["domain_status"] == "query_tokens_missing"
    assert result["hits"] == []


def test_search_by_domain_name_like_query_routes_to_strict_name_query():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(_make_hit(1, "影视飓风", 10.0)),
            _make_response(_make_hit(1, "影视飓风", 8.0, influence_score=0.9)),
        ]
    )

    result = searcher.search_by_domain(
        "影视飓风",
        sort_by="influence",
        limit=3,
        compact=True,
    )

    assert result["query_route"] == "name"
    assert len(searcher.calls) == 2
    assert searcher.calls[0]["context"] == "route.exact_name"
    assert searcher.calls[1]["context"] == "search_by_domain.name_routed"
    strict_query = searcher.calls[1]["body"]["query"]
    should = strict_query["bool"]["should"]
    assert any(
        "term" in clause and "name.keyword" in clause["term"] for clause in should
    )
    assert any(
        "match" in clause and "name.words" in clause["match"] for clause in should
    )
    assert result["domain_status"] == "name_route"


def test_search_relevance_domain_route_uses_profile_tokens_when_provided():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(),
            _make_response(
                _make_hit(
                    1,
                    "纳塔剧情研究员",
                    5.0,
                    influence_score=0.95,
                    quality_score=0.82,
                    activity_score=0.41,
                    profile_domain_ready=True,
                ),
            ),
        ]
    )

    result = searcher.search(
        "纳塔剧情分析",
        sort_by="relevance",
        limit=3,
        compact=True,
        text_token_ids=[701, 702, 703],
    )

    assert len(searcher.calls) == 2
    assert searcher.calls[0]["context"] == "route.exact_name"
    assert searcher.calls[1]["context"] == "search.domain_tokens"
    assert result["query_route"] == "domain"
    assert result["domain_status"] == "query_tokens_used"
    assert result["hits"][0]["mid"] == 1


@pytest.mark.parametrize(
    ("query", "exact_name_hit", "expected_route"),
    [
        ("影视飓风", True, "name"),
        ("老番茄", True, "name"),
        ("黑神话悟空", False, "domain"),
        ("当你半年不上线的账号打一把王者排位的时候", False, "phrase"),
        ("原神纳塔剧情解析", False, "phrase"),
    ],
)
def test_detect_query_route_covers_head_and_tail_queries(
    query: str,
    exact_name_hit: bool,
    expected_route: str,
):
    searcher = StubOwnerSearcher(responses=[])

    assert (
        searcher._detect_query_route(query, exact_name_hit=exact_name_hit)
        == expected_route
    )
