from elastics.owners.searcher import OwnerSearcher


def test_build_name_search_body_uses_only_indexed_owner_name_fields():
    searcher = OwnerSearcher.__new__(OwnerSearcher)
    body = searcher._build_name_search_body(query="影视飓风", size=5)

    should = body["query"]["bool"]["should"]
    multi_match = next(item["multi_match"] for item in should if "multi_match" in item)

    assert multi_match["fields"] == [
        "owner.name.words^8",
        "owner.name.suggest^6",
    ]


def test_prepare_query_normalizes_punctuation_and_spacing():
    searcher = OwnerSearcher.__new__(OwnerSearcher)

    assert (
        searcher._prepare_query("  影视飓风：有哪些关联账号？  ", "relation")
        == "影视飓风 有哪些关联账号"
    )


def test_search_topic_mode_skips_name_candidates():
    searcher = OwnerSearcher.__new__(OwnerSearcher)
    called = {"name": False}

    def mark_name(*args, **kwargs):
        called["name"] = True
        return []

    searcher._prepare_query = lambda query, mode: query
    searcher._search_name_candidates = mark_name
    searcher._search_topic_candidates = lambda query, size: [
        {"mid": 1, "name": "黑神话作者", "score": 80.0, "sources": ["topic"]}
    ]
    searcher._search_relation_candidates = lambda query, name_hits, size: []
    searcher._load_owner_metadata = lambda mids: {}

    result = searcher.search("黑神话悟空", mode="topic", size=5)

    assert called["name"] is False
    assert result["owners"][0]["sources"] == ["topic"]


def test_search_relation_mode_prefers_relation_results_over_seed_name_hits():
    searcher = OwnerSearcher.__new__(OwnerSearcher)

    searcher._prepare_query = lambda query, mode: query
    searcher._search_name_candidates = lambda query, size: [
        {"mid": 1, "name": "影视飓风", "score": 120.0, "sources": ["name"]}
    ]
    searcher._search_relation_candidates = lambda query, name_hits, size: [
        {"mid": 2, "name": "油管科技TV", "score": 98.0, "sources": ["relation"]}
    ]
    searcher._search_topic_candidates = lambda query, size: []
    searcher._load_owner_metadata = lambda mids: {}

    result = searcher.search("影视飓风有哪些关联账号", mode="relation", size=5)

    assert [owner["name"] for owner in result["owners"]] == ["油管科技TV"]


def test_search_hydrates_owner_face_and_sample_from_metadata_lookup():
    searcher = OwnerSearcher.__new__(OwnerSearcher)

    searcher._prepare_query = lambda query, mode: query
    searcher._search_name_candidates = lambda query, size: []
    searcher._search_topic_candidates = lambda query, size: [
        {"mid": 946974, "name": "影视飓风", "score": 86.0, "sources": ["topic"]}
    ]
    searcher._search_relation_candidates = lambda query, name_hits, size: []
    searcher._load_owner_metadata = lambda mids: {
        946974: {
            "face": "https://example.com/owner-face.jpg",
            "sample_title": "一条代表作",
            "sample_bvid": "BV1owner1",
            "sample_pic": "https://example.com/sample-cover.jpg",
            "sample_view": 123456,
        }
    }

    result = searcher.search("影视飓风", mode="topic", size=5)

    assert result["owners"][0]["face"] == "https://example.com/owner-face.jpg"
    assert result["owners"][0]["sample_title"] == "一条代表作"
    assert result["owners"][0]["sample_bvid"] == "BV1owner1"
    assert result["owners"][0]["sample_pic"] == "https://example.com/sample-cover.jpg"
    assert result["owners"][0]["sample_view"] == 123456
