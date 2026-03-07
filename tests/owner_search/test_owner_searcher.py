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
        "top_tags": extra.pop("top_tags", ""),
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


def test_search_relevance_fuses_name_and_domain_hits():
    searcher = StubOwnerSearcher(
        responses=[
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
                    top_tags="科技 数码 影视",
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
                    top_tags="科技 Vlog",
                ),
            ),
            _make_response(
                _make_hit(
                    1,
                    "影视飓风",
                    6.0,
                    total_videos=500,
                    total_view=2500000000,
                    influence_score=0.92,
                    quality_score=0.78,
                    activity_score=0.66,
                    top_tags="科技 数码 影视",
                ),
                _make_hit(
                    3,
                    "科技老张",
                    8.0,
                    total_videos=210,
                    total_view=410000000,
                    influence_score=0.69,
                    quality_score=0.73,
                    activity_score=0.71,
                    top_tags="影视 器材 评测",
                ),
            ),
        ]
    )

    result = searcher.search("影视飓风", sort_by="relevance", limit=5, compact=True)

    assert result["query_type"] == "name"
    assert [hit["mid"] for hit in result["hits"]] == [1, 2, 3]
    assert result["hits"][0]["_score"] > result["hits"][1]["_score"]
    assert result["hits"][0]["name"] == "影视飓风"
    assert len({hit["mid"] for hit in result["hits"]}) == 3
    assert len(searcher.calls) == 2
    assert searcher.calls[0]["context"] == "search.name"
    assert searcher.calls[1]["context"] == "search.domain"


def test_search_non_relevance_uses_combined_query_and_es_sort():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(),
            _make_response(
                _make_hit(
                    10,
                    "黑神话研究所",
                    4.0,
                    total_videos=88,
                    total_view=90000000,
                    influence_score=0.63,
                )
            ),
        ]
    )

    result = searcher.search("黑神话悟空", sort_by="influence", limit=3, compact=True)

    assert result["hits"][0]["mid"] == 10
    assert result["query_route"] == "domain"
    assert len(searcher.calls) == 2
    assert searcher.calls[0]["context"] == "route.exact_name"
    body = searcher.calls[1]["body"]
    assert body["sort"][0] == {"influence_score": {"order": "desc"}}
    assert body["query"]["bool"]["should"]


def test_search_non_relevance_head_domain_uses_strict_domain_gate():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(),
            _make_response(_make_hit(10, "黑神话研究所", 4.0, influence_score=0.63)),
        ]
    )

    searcher.search("黑神话悟空", sort_by="influence", limit=3, compact=True)

    domain_query = searcher.calls[1]["body"]["query"]["bool"]["should"][1]
    assert domain_query["bool"]["must"]
    strict_query = domain_query["bool"]["must"][0]["bool"]
    assert strict_query["minimum_should_match"] == 1
    assert any(
        clause.get("match", {}).get("semantic_terms.words", {}).get("operator") == "and"
        or clause.get("match", {}).get("topic_phrases.words", {}).get("operator")
        == "and"
        or clause.get("match", {}).get("domain_text.words", {}).get("operator") == "and"
        or clause.get("match", {}).get("top_tags.words", {}).get("operator") == "and"
        for clause in strict_query["should"]
    )


def test_search_by_domain_long_phrase_requires_strict_phrase_match_before_sort():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(
                _make_hit(
                    10,
                    "幸运的盆子",
                    3.5,
                    total_videos=88,
                    total_view=90000000,
                    influence_score=0.63,
                )
            )
        ]
    )

    searcher.search_by_domain(
        "当你半年不上线的账号打一把王者排位的时候",
        sort_by="influence",
        limit=3,
        compact=True,
    )

    assert len(searcher.calls) == 1
    body = searcher.calls[0]["body"]
    assert "sort" not in body
    assert body["size"] >= 24
    domain_query = body["query"]
    assert domain_query["bool"]["must"]
    strict_query = domain_query["bool"]["must"][0]["bool"]
    assert strict_query["minimum_should_match"] == 1
    assert any("match_phrase" in clause for clause in strict_query["should"])


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


def test_search_phrase_route_uses_sparse_semantic_rerank():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(
                _make_hit(
                    1,
                    "高质量但不够相关",
                    6.2,
                    quality_score=0.91,
                    topic_phrases="影视分析",
                    semantic_terms="影视 希区柯克 电影 分析",
                ),
                _make_hit(
                    2,
                    "镜头语言研究室",
                    4.0,
                    quality_score=0.63,
                    topic_phrases="希区柯克镜头语言",
                    semantic_terms="希区柯克 镜头 语言 希区柯克镜头 镜头语言",
                ),
            )
        ]
    )

    result = searcher.search(
        "希区柯克镜头语言", sort_by="quality", limit=2, compact=True
    )

    assert len(searcher.calls) == 1
    assert searcher.calls[0]["context"] == "search.phrase_routed"
    assert searcher.calls[0]["body"]["size"] >= 16
    assert "sort" not in searcher.calls[0]["body"]
    assert result["query_route"] == "phrase"
    assert result["hits"][0]["mid"] == 2


def test_search_phrase_route_falls_back_to_relaxed_semantic_query_on_zero_hits():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(),
            _make_response(
                _make_hit(
                    7,
                    "纳塔剧情工坊",
                    3.0,
                    quality_score=0.62,
                    semantic_terms="原神 纳塔 剧情 解析 纳塔剧情 剧情解析",
                )
            ),
        ]
    )

    result = searcher.search(
        "原神纳塔剧情解析", sort_by="quality", limit=3, compact=True
    )

    assert len(searcher.calls) == 2
    assert searcher.calls[0]["context"] == "search.phrase_routed"
    assert searcher.calls[1]["context"] == "search.phrase_routed.fallback"
    assert result["query_route"] == "phrase"
    assert result["hits"][0]["mid"] == 7


def test_search_domain_route_can_use_controlled_semantic_rerank():
    searcher = StubOwnerSearcher(
        responses=[
            _make_response(),
            _make_response(
                _make_hit(
                    1,
                    "高影响力泛作者",
                    6.5,
                    quality_score=0.55,
                    influence_score=0.95,
                    semantic_terms="游戏 热门 综合",
                ),
                _make_hit(
                    2,
                    "纳塔剧情研究员",
                    4.0,
                    quality_score=0.82,
                    influence_score=0.41,
                    semantic_terms="原神 纳塔 剧情 解析 纳塔剧情 剧情解析",
                ),
            ),
        ]
    )

    result = searcher.search_by_domain(
        "纳塔剧情分析",
        sort_by="quality",
        limit=3,
        compact=True,
    )

    assert len(searcher.calls) == 2
    assert searcher.calls[0]["context"] == "route.exact_name"
    assert searcher.calls[1]["context"] == "search_by_domain.domain_semantic"
    assert "sort" not in searcher.calls[1]["body"]
    assert result["query_route"] == "domain"
    assert result["hits"][0]["mid"] == 2


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
