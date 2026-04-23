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


def test_search_relation_mode_keeps_seed_name_hit_ahead_of_relation_results():
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

    assert [owner["name"] for owner in result["owners"]] == [
        "影视飓风",
        "油管科技TV",
    ]


def test_owner_intent_match_prefers_tail_entity_over_generic_topic_prefix():
    searcher = OwnerSearcher.__new__(OwnerSearcher)

    assert searcher._owner_intent_match_score(
        "王者荣耀张大仙",
        "指法芬芳张大仙",
    ) > searcher._owner_intent_match_score(
        "王者荣耀张大仙",
        "王者荣耀-大仙",
    )


def test_search_name_mode_enriches_with_topic_hits_for_owner_intent_query():
    searcher = OwnerSearcher.__new__(OwnerSearcher)

    searcher._prepare_query = lambda query, mode: query
    searcher._search_name_candidates = lambda query, size: [
        {"mid": 1, "name": "王者荣耀-大仙", "score": 150.0, "sources": ["name"]}
    ]
    searcher._search_topic_candidates = lambda query, size: [
        {"mid": 2, "name": "指法芬芳张大仙", "score": 100.0, "sources": ["topic"]}
    ]
    searcher._search_relation_candidates = lambda query, name_hits, size: []
    searcher._load_owner_metadata = lambda mids: {}

    result = searcher.search("王者荣耀张大仙", mode="name", size=5)

    assert result["owners"][0]["name"] == "指法芬芳张大仙"
    assert result["owners"][0]["sources"] == ["topic"]


def test_search_name_mode_skips_topic_enrichment_when_exact_name_hit_exists():
    searcher = OwnerSearcher.__new__(OwnerSearcher)
    called = {"topic": False}

    searcher._prepare_query = lambda query, mode: query
    searcher._search_name_candidates = lambda query, size: [
        {"mid": 1, "name": "老师好我叫何同学", "score": 150.0, "sources": ["name"]}
    ]

    def mark_topic(*args, **kwargs):
        called["topic"] = True
        return []

    searcher._search_topic_candidates = mark_topic
    searcher._search_relation_candidates = lambda query, name_hits, size: []
    searcher._load_owner_metadata = lambda mids: {}

    result = searcher.search("老师好我叫何同学", mode="name", size=5)

    assert called["topic"] is False
    assert result["owners"][0]["name"] == "老师好我叫何同学"


def test_search_name_mode_enriches_short_alias_when_only_prefix_noise_exists():
    searcher = OwnerSearcher.__new__(OwnerSearcher)

    searcher._prepare_query = lambda query, mode: query
    searcher._search_name_candidates = lambda query, size: [
        {"mid": 1, "name": "何同学的周末", "score": 150.0, "sources": ["name"]}
    ]
    searcher._search_topic_candidates = lambda query, size: [
        {"mid": 2, "name": "老师好我叫何同学", "score": 100.0, "sources": ["topic"]}
    ]
    searcher._search_relation_candidates = lambda query, name_hits, size: []
    searcher._load_owner_metadata = lambda mids: {}

    result = searcher.search("何同学", mode="name", size=5)

    assert result["owners"][0]["name"] == "老师好我叫何同学"


def test_search_name_mode_prefers_name_supported_canonical_owner_over_topic_only_variant():
    searcher = OwnerSearcher.__new__(OwnerSearcher)

    searcher._prepare_query = lambda query, mode: query
    searcher._search_name_candidates = lambda query, size: [
        {"mid": 1, "name": "何同学的周末", "score": 150.0, "sources": ["name"]},
        {"mid": 2, "name": "老师好我叫何同学", "score": 141.0, "sources": ["name"]},
    ]
    searcher._search_topic_candidates = lambda query, size: [
        {"mid": 3, "name": "何同学工作室", "score": 100.0, "sources": ["topic"]},
        {"mid": 2, "name": "老师好我叫何同学", "score": 97.0, "sources": ["topic"]},
    ]
    searcher._search_relation_candidates = lambda query, name_hits, size: []
    searcher._load_owner_metadata = lambda mids: {}

    result = searcher.search("何同学", mode="name", size=5)

    assert result["owners"][0]["name"] == "老师好我叫何同学"


def test_search_name_mode_uses_stable_name_recall_floor_for_owner_intent():
    searcher = OwnerSearcher.__new__(OwnerSearcher)

    searcher._prepare_query = lambda query, mode: query

    def fake_name_candidates(query: str, size: int) -> list[dict]:
        hits = [
            {"mid": 1, "name": "何同学的周末", "score": 150.0, "sources": ["name"]},
            {"mid": 3, "name": "原神何同学", "score": 149.0, "sources": ["name"]},
        ]
        if size >= 16:
            hits.append(
                {
                    "mid": 2,
                    "name": "老师好我叫何同学",
                    "score": 141.0,
                    "sources": ["name"],
                }
            )
        return hits

    searcher._search_name_candidates = fake_name_candidates
    searcher._search_topic_candidates = lambda query, size: [
        {"mid": 4, "name": "何同学工作室", "score": 100.0, "sources": ["topic"]},
        {"mid": 2, "name": "老师好我叫何同学", "score": 97.0, "sources": ["topic"]},
    ]
    searcher._search_relation_candidates = lambda query, name_hits, size: []
    searcher._load_owner_metadata = lambda mids: {}

    result = searcher.search("何同学", mode="name", size=5)

    assert result["owners"][0]["name"] == "老师好我叫何同学"


def test_search_name_mode_uses_stable_topic_recall_floor_for_owner_intent():
    searcher = OwnerSearcher.__new__(OwnerSearcher)

    searcher._prepare_query = lambda query, mode: query
    searcher._search_name_candidates = lambda query, size: [
        {"mid": 1, "name": "红警土豆_", "score": 151.5, "sources": ["name"]},
        {"mid": 2, "name": "红警HBK08", "score": 150.0, "sources": ["name"]},
    ]

    def fake_topic_candidates(query: str, size: int) -> list[dict]:
        hits = [
            {"mid": 2, "name": "红警HBK08", "score": 140.0, "sources": ["topic"]},
            {"mid": 1, "name": "红警土豆_", "score": 134.0, "sources": ["topic"]},
        ]
        if size >= 16:
            hits.insert(
                0,
                {"mid": 3, "name": "红警360", "score": 155.0, "sources": ["topic"]},
            )
        return hits

    searcher._search_topic_candidates = fake_topic_candidates
    searcher._search_relation_candidates = lambda query, name_hits, size: []
    searcher._load_owner_metadata = lambda mids: {}

    result = searcher.search("红警", mode="name", size=5)

    assert any(owner["name"] == "红警360" for owner in result["owners"])


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
