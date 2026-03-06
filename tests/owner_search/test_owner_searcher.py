import sys
from pathlib import Path

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
            _make_response(
                _make_hit(
                    10,
                    "黑神话研究所",
                    4.0,
                    total_videos=88,
                    total_view=90000000,
                    influence_score=0.63,
                )
            )
        ]
    )

    result = searcher.search("黑神话悟空", sort_by="influence", limit=3, compact=True)

    assert result["hits"][0]["mid"] == 10
    assert len(searcher.calls) == 1
    body = searcher.calls[0]["body"]
    assert body["sort"][0] == {"influence_score": {"order": "desc"}}
    assert body["query"]["bool"]["should"]
