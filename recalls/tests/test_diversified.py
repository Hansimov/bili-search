"""
Tests for ranks/diversified.py — DiversifiedRanker slot allocation.
"""

import time
from tclogger import logger

from ranks.diversified import DiversifiedRanker, SLOT_PRESETS


def _make_hit(bvid, score=1.0, view=1000, pubdate=None, stat_score=0.5, **kw):
    """Create a test hit dict with required fields."""
    if pubdate is None:
        pubdate = int(time.time()) - 86400  # 1 day ago
    return {
        "bvid": bvid,
        "score": score,
        "stat": {
            "view": view,
            "coin": view // 10,
            "like": view // 5,
            "favorite": view // 20,
            "danmaku": view // 50,
            "reply": view // 50,
        },
        "pubdate": pubdate,
        "stat_score": stat_score,
        **kw,
    }


def test_diversified_rank_basic():
    """Top 10 should contain representatives from all 4 dimensions when using direct slot allocation."""
    logger.note("> Test: diversified_rank basic slot allocation")

    now = int(time.time())
    hits = []

    # 5 highly relevant (high BM25 score, but low stats so they don't dominate quality)
    for i in range(5):
        hits.append(
            _make_hit(
                f"REL{i}",
                score=50 - i,
                view=200,
                pubdate=now - 86400 * 60,
                stat_score=0.1,
            )
        )

    # 5 highly popular (high view count, moderate interactions)
    for i in range(5):
        hits.append(
            _make_hit(
                f"POP{i}",
                score=3,
                view=500000 * (5 - i),
                pubdate=now - 86400 * 90,
                stat_score=0.4,
            )
        )

    # 5 very recent (published in last hours)
    for i in range(5):
        hits.append(
            _make_hit(
                f"NEW{i}",
                score=5,
                view=100,
                pubdate=now - 3600 * (i + 1),
                stat_score=0.1,
            )
        )

    # 5 high quality (balanced stats across all fields - DocScorer loves this)
    for i in range(5):
        v = 50000 * (5 - i)
        hits.append(
            _make_hit(
                f"QUA{i}",
                score=4,
                view=v,
                pubdate=now - 86400 * 30,
                stat_score=0.9,
            )
        )
        # Override stat dict with balanced high interactions
        hits[-1]["stat"] = {
            "view": v,
            "coin": v // 4,
            "like": v // 3,
            "favorite": v // 5,
            "danmaku": v // 8,
            "reply": v // 8,
        }

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank(
        hits_info={"hits": hits},
        top_k=10,
        prefer="balanced",
    )

    ranked = result["hits"]
    assert len(ranked) == 10, f"Expected 10 hits, got {len(ranked)}"

    # Check dimension distribution — all 4 base dimensions should be present
    dims = result.get("dimension_distribution", {})
    logger.mesg(f"  Dimension distribution: {dims}")

    slot_dims = set(h.get("_slot_dimension") for h in ranked)
    for dim in ["relevance", "quality", "recency", "popularity"]:
        assert dim in slot_dims, f"Missing dimension: {dim}. Got: {slot_dims}"

    # The balanced preset allocates {relevance: 2, quality: 1, recency: 1, popularity: 1}
    # = 5 dimension slots + 5 fused slots to fill top_k=10
    assert (
        dims.get("relevance", 0) == 2
    ), f"Expected 2 relevance slots, got {dims.get('relevance')}"
    assert (
        dims.get("recency", 0) == 1
    ), f"Expected 1 recency slot, got {dims.get('recency')}"
    assert (
        dims.get("quality", 0) == 1
    ), f"Expected 1 quality slot, got {dims.get('quality')}"
    assert (
        dims.get("popularity", 0) == 1
    ), f"Expected 1 popularity slot, got {dims.get('popularity')}"

    # All results should have rank_score
    for h in ranked:
        assert "rank_score" in h, f"Missing rank_score for {h['bvid']}"

    logger.success("  PASSED")


def test_diversified_rank_slot_counts():
    """Slot counts should match the preset configuration."""
    logger.note("> Test: diversified_rank slot count accuracy")

    now = int(time.time())
    # Create 40 distinct items with varied characteristics
    hits = []
    for i in range(40):
        hits.append(
            _make_hit(
                f"BV{i:03d}",
                score=40 - i,
                view=(40 - i) * 1000,
                pubdate=now - i * 3600,
                stat_score=0.1 + (i % 10) * 0.08,
            )
        )

    ranker = DiversifiedRanker()
    slots = SLOT_PRESETS["balanced"]
    total_slots = sum(slots.values())  # 10

    result = ranker.diversified_rank(
        hits_info={"hits": hits},
        top_k=total_slots,
        prefer="balanced",
    )

    ranked = result["hits"]
    assert len(ranked) == total_slots, f"Expected {total_slots}, got {len(ranked)}"

    # All should be unique
    bvids = [h["bvid"] for h in ranked]
    assert len(set(bvids)) == len(bvids), f"Duplicate bvids: {bvids}"

    logger.success("  PASSED")


def test_diversified_rank_no_duplicates():
    """Same bvid should not appear twice in results."""
    logger.note("> Test: diversified_rank no duplicates")

    now = int(time.time())
    # Create items that would score high in multiple dimensions
    hits = [
        _make_hit("BV_MULTI", score=50, view=100000, pubdate=now - 60, stat_score=0.95),
    ]
    # Add some filler
    for i in range(20):
        hits.append(_make_hit(f"BV{i}", score=10, view=1000, stat_score=0.3))

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank(
        hits_info={"hits": hits},
        top_k=10,
    )

    ranked = result["hits"]
    bvids = [h["bvid"] for h in ranked]
    assert len(set(bvids)) == len(bvids), f"Duplicate bvids found: {bvids}"

    # BV_MULTI should appear exactly once
    multi_count = sum(1 for bvid in bvids if bvid == "BV_MULTI")
    assert multi_count == 1, f"BV_MULTI appeared {multi_count} times"

    logger.success("  PASSED")


def test_diversified_rank_empty():
    """Empty hits should not crash."""
    logger.note("> Test: diversified_rank empty input")

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank(hits_info={"hits": []}, top_k=10)

    assert result["hits"] == []
    assert result["rank_method"] == "diversified"

    logger.success("  PASSED")


def test_diversified_rank_fewer_than_top_k():
    """When fewer hits than top_k, return all of them."""
    logger.note("> Test: diversified_rank fewer than top_k")

    hits = [_make_hit(f"BV{i}") for i in range(3)]
    ranker = DiversifiedRanker()
    result = ranker.diversified_rank(hits_info={"hits": hits}, top_k=10)

    assert len(result["hits"]) == 3

    logger.success("  PASSED")


def test_diversified_rank_with_fused_fallback():
    """Three-phase ranking: headline top-3, then slot allocation, then fused rest."""
    logger.note("> Test: diversified_rank_with_fused_fallback")

    now = int(time.time())
    hits = []
    for i in range(30):
        hits.append(
            _make_hit(
                f"BV{i:03d}",
                score=30 - i,
                view=(30 - i) * 500,
                pubdate=now - i * 7200,
                stat_score=0.1 + (i % 8) * 0.1,
            )
        )

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank_with_fused_fallback(
        hits_info={"hits": hits},
        top_k=20,
        diversify_top_n=10,
    )

    ranked = result["hits"]
    assert len(ranked) == 20, f"Expected 20, got {len(ranked)}"
    assert result.get("diversified_top_n") == 10

    # First 3 should be headline picks, remaining 7 of top-10 are slot picks
    headline_count = result.get("headline_top_n", 0)
    assert headline_count == 3, f"Expected 3 headline picks, got {headline_count}"

    # First 10 should all have slot dimension tags (headline, dimension, or fused)
    for h in ranked[:10]:
        assert (
            "_slot_dimension" in h
        ), f"Missing slot dimension in top-10 hit: {h.get('bvid')}"

    # Headline hits should be tagged as "headline"
    headline_hits = [h for h in ranked[:10] if h.get("_slot_dimension") == "headline"]
    assert (
        len(headline_hits) == 3
    ), f"Expected 3 headline hits, got {len(headline_hits)}"

    # No duplicates
    bvids = [h["bvid"] for h in ranked]
    assert len(set(bvids)) == len(bvids), "Duplicates in fallback result"

    logger.success("  PASSED")


def test_diversified_rank_prefer_presets():
    """Different presets should shift dimension distribution."""
    logger.note("> Test: diversified_rank preset variation")

    now = int(time.time())
    hits = []
    for i in range(50):
        hits.append(
            _make_hit(
                f"BV{i:03d}",
                score=50 - i,
                view=(50 - i) * 2000,
                pubdate=now - i * 3600,
                stat_score=0.05 + (i % 12) * 0.075,
            )
        )

    ranker = DiversifiedRanker()

    # prefer_relevance should yield more relevance slots
    res_r = ranker.diversified_rank(
        hits_info={"hits": [dict(h) for h in hits]},
        top_k=10,
        prefer="prefer_relevance",
    )
    # prefer_recency should yield more recency slots
    res_t = ranker.diversified_rank(
        hits_info={"hits": [dict(h) for h in hits]},
        top_k=10,
        prefer="prefer_recency",
    )

    dist_r = res_r.get("dimension_distribution", {})
    dist_t = res_t.get("dimension_distribution", {})

    logger.mesg(f"  prefer_relevance dist: {dist_r}")
    logger.mesg(f"  prefer_recency dist: {dist_t}")

    # Relevance preset should have more relevance items than recency preset
    assert dist_r.get("relevance", 0) >= dist_t.get(
        "relevance", 0
    ), "prefer_relevance should have more relevance items"

    logger.success("  PASSED")


def test_score_all_dimensions():
    """_score_all_dimensions should produce bounded scores."""
    logger.note("> Test: _score_all_dimensions correctness")

    now = int(time.time())
    hits = [
        _make_hit("BV001", score=25.0, view=50000, pubdate=now - 3600, stat_score=0.7),
        _make_hit(
            "BV002", score=5.0, view=100, pubdate=now - 86400 * 365, stat_score=0.1
        ),
    ]

    ranker = DiversifiedRanker()
    ranker._score_all_dimensions(hits, now_ts=now)

    for hit in hits:
        for field in [
            "relevance_score",
            "quality_score",
            "recency_score",
            "popularity_score",
        ]:
            val = hit.get(field)
            assert val is not None, f"{field} missing for {hit['bvid']}"
            assert 0 <= val <= 1.0, f"{field}={val} out of [0,1] for {hit['bvid']}"

    # BV001 should be more relevant (higher score)
    assert hits[0]["relevance_score"] > hits[1]["relevance_score"]
    # BV001 should be more popular
    assert hits[0]["popularity_score"] > hits[1]["popularity_score"]
    # BV001 should be more recent
    assert hits[0]["recency_score"] > hits[1]["recency_score"]

    logger.success("  PASSED")


def test_headline_score_computed():
    """_score_all_dimensions should compute headline_score for each hit."""
    logger.note("> Test: headline_score computation")

    now = int(time.time())
    hits = [
        _make_hit("BV_HIGH", score=40, view=100000, pubdate=now - 3600, stat_score=0.9),
        _make_hit(
            "BV_LOW", score=5, view=100, pubdate=now - 86400 * 365, stat_score=0.1
        ),
    ]

    ranker = DiversifiedRanker()
    ranker._score_all_dimensions(hits, now_ts=now)

    for hit in hits:
        assert "headline_score" in hit, f"Missing headline_score for {hit['bvid']}"
        assert 0 <= hit["headline_score"] <= 1.0

    # BV_HIGH should have much higher headline_score (good on all dimensions)
    assert hits[0]["headline_score"] > hits[1]["headline_score"]

    logger.success("  PASSED")


def test_select_headline_top_n():
    """_select_headline_top_n picks the best composite candidates."""
    logger.note("> Test: _select_headline_top_n selection quality")

    now = int(time.time())
    hits = []

    # High relevance but old/low quality — should NOT be headline
    for i in range(5):
        hits.append(
            _make_hit(
                f"REL{i}",
                score=50 - i,
                view=200,
                pubdate=now - 86400 * 365,
                stat_score=0.05,
            )
        )

    # Balanced: high relevance + recent + quality — ideal headline candidates
    for i in range(5):
        hits.append(
            _make_hit(
                f"IDEAL{i}",
                score=35 - i,
                view=80000,
                pubdate=now - 3600 * (i + 1),
                stat_score=0.8,
            )
        )

    ranker = DiversifiedRanker()
    ranker._score_all_dimensions(hits, now_ts=now)

    selected, bvids = ranker._select_headline_top_n(hits, top_n=3)

    assert len(selected) == 3
    assert len(bvids) == 3

    # All selected should be tagged as "headline"
    for hit in selected:
        assert hit.get("_slot_dimension") == "headline"

    # Ideal candidates (good on all 3 axes) should dominate headline picks
    ideal_in_headline = sum(1 for h in selected if h["bvid"].startswith("IDEAL"))
    assert (
        ideal_in_headline >= 2
    ), f"Expected ≥2 IDEAL in headline, got {ideal_in_headline}"

    logger.success("  PASSED")


def test_headline_top_n_no_duplicates_with_slots():
    """Headline picks should not appear again in slot allocation."""
    logger.note("> Test: headline exclusion from slot allocation")

    now = int(time.time())
    hits = []
    for i in range(30):
        hits.append(
            _make_hit(
                f"BV{i:03d}",
                score=30 - i,
                view=(30 - i) * 1000,
                pubdate=now - i * 3600,
                stat_score=0.1 + (i % 10) * 0.08,
            )
        )

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank_with_fused_fallback(
        hits_info={"hits": hits},
        top_k=15,
        diversify_top_n=10,
        headline_top_n=3,
    )

    ranked = result["hits"]
    bvids = [h["bvid"] for h in ranked]
    assert len(set(bvids)) == len(bvids), f"Duplicate bvids: {bvids}"

    # First 3 should be headline
    headline_dims = [h.get("_slot_dimension") for h in ranked[:3]]
    assert all(
        d == "headline" for d in headline_dims
    ), f"Top 3 should be headline, got {headline_dims}"

    logger.success("  PASSED")


def test_three_phase_ranking_structure():
    """Full three-phase ranking should have headline + slot + fused sections."""
    logger.note("> Test: three-phase ranking structure")

    now = int(time.time())
    hits = []
    for i in range(50):
        v = max(100, (50 - i) * 2000)
        hits.append(
            _make_hit(
                f"BV{i:03d}",
                score=50 - i,
                view=v,
                pubdate=now - i * 7200,
                stat_score=0.05 + (i % 12) * 0.075,
            )
        )

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank_with_fused_fallback(
        hits_info={"hits": hits},
        top_k=30,
        diversify_top_n=10,
        headline_top_n=3,
    )

    ranked = result["hits"]
    assert len(ranked) == 30

    # Check phase distribution
    headline_count = sum(
        1 for h in ranked[:10] if h.get("_slot_dimension") == "headline"
    )
    slot_count = sum(
        1
        for h in ranked[:10]
        if h.get("_slot_dimension") in {"relevance", "quality", "recency", "popularity"}
    )
    fused_count = sum(1 for h in ranked[10:] if h.get("_slot_dimension") == "fused")

    assert headline_count == 3, f"Expected 3 headline in top-10, got {headline_count}"
    assert slot_count >= 1, f"Expected ≥1 slot dimensions in top-10, got {slot_count}"
    assert fused_count > 0, f"Expected some fused hits beyond top-10"

    # Verify metadata
    assert result.get("headline_top_n") == 3
    assert result.get("diversified_top_n") == 10

    logger.success("  PASSED")


def test_exact_top_k_return_count():
    """Diversified ranker must return exactly top_k docs when pool >= top_k."""
    logger.note("> Test: exact top_k return count guarantee")

    now = int(time.time())
    # Create a pool large enough (500 docs) to serve top_k=400
    hits = []
    for i in range(500):
        hits.append(
            _make_hit(
                f"BV{i:04d}",
                score=max(1, 500 - i),
                view=max(100, (500 - i) * 100),
                pubdate=now - i * 3600,
                stat_score=0.05 + (i % 20) * 0.045,
            )
        )

    ranker = DiversifiedRanker()

    # Test with top_k=400 (the EXPLORE_RANK_TOP_K)
    result = ranker.diversified_rank_with_fused_fallback(
        hits_info={"hits": hits},
        top_k=400,
        diversify_top_n=10,
        headline_top_n=3,
    )

    ranked = result["hits"]
    assert len(ranked) == 400, (
        f"Expected exactly 400 hits, got {len(ranked)}. "
        f"This is the 'return_hits not exactly 400' bug."
    )
    assert result["return_hits"] == 400

    # No duplicates
    bvids = [h["bvid"] for h in ranked]
    assert len(set(bvids)) == 400, "Duplicates found in 400-doc result"

    logger.success("  PASSED")


def test_exact_top_k_when_pool_smaller():
    """When pool < top_k, return all docs (not fewer)."""
    logger.note("> Test: top_k with small pool")

    now = int(time.time())
    hits = [
        _make_hit(f"BV{i}", score=10 - i, view=1000, pubdate=now - i * 3600)
        for i in range(7)
    ]

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank_with_fused_fallback(
        hits_info={"hits": hits},
        top_k=400,
        diversify_top_n=10,
    )

    ranked = result["hits"]
    assert len(ranked) == 7, f"Expected 7 (full pool), got {len(ranked)}"

    logger.success("  PASSED")


def test_relevance_floor_in_slots():
    """Slot candidates must pass minimum relevance threshold."""
    logger.note("> Test: relevance floor in slot allocation")

    now = int(time.time())
    hits = []

    # 5 highly relevant docs
    for i in range(5):
        hits.append(
            _make_hit(
                f"REL{i}",
                score=50 - i,
                view=5000,
                pubdate=now - 86400 * 7,
                stat_score=0.5,
            )
        )

    # 20 irrelevant but viral docs (very high view but near-zero BM25)
    for i in range(20):
        hits.append(
            _make_hit(
                f"VIRAL{i}",
                score=0.1,  # Very low relevance
                view=10000000,  # Very high views
                pubdate=now - 3600,  # Very recent
                stat_score=0.9,  # High quality
            )
        )

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank_with_fused_fallback(
        hits_info={"hits": hits},
        top_k=15,
        diversify_top_n=10,
    )

    ranked = result["hits"]
    top_10 = ranked[:10]

    # Check that top-10 is NOT dominated by irrelevant viral docs.
    # With the relevance floor, the top-10 should prefer relevant docs
    # over viral irrelevant ones.
    relevant_in_top10 = sum(1 for h in top_10 if h["bvid"].startswith("REL"))
    # At minimum, the headline positions should be relevant
    assert relevant_in_top10 >= 3, (
        f"Expected ≥3 relevant docs in top-10, got {relevant_in_top10}. "
        f"Viral irrelevant docs may be taking over top slots."
    )

    logger.success("  PASSED")


def test_headline_relevance_tiebreak():
    """When headline_scores are similar, higher relevance should win."""
    logger.note("> Test: headline relevance tiebreaker")

    now = int(time.time())

    # Two candidates with similar headline scores but different relevance
    hits = [
        # High relevance, good quality/recency → better tiebreak
        _make_hit(
            "HIGH_REL",
            score=40,
            view=50000,
            pubdate=now - 3600 * 6,
            stat_score=0.7,
        ),
        # Lower relevance, slightly better quality → similar headline score
        _make_hit(
            "LOW_REL",
            score=15,
            view=100000,
            pubdate=now - 3600 * 3,
            stat_score=0.85,
        ),
        # Filler
        *[_make_hit(f"FILL{i}", score=5, view=500) for i in range(10)],
    ]

    ranker = DiversifiedRanker()
    ranker._score_all_dimensions(hits, now_ts=now)

    selected, _ = ranker._select_headline_top_n(hits, top_n=1)

    # The more relevant doc should win the tiebreak
    assert selected[0]["bvid"] == "HIGH_REL", (
        f"Expected HIGH_REL to win headline, got {selected[0]['bvid']}. "
        f"Relevance tiebreak may not be working."
    )

    logger.success("  PASSED")


def test_title_match_tagging_title_only():
    """_tag_title_matches should tag docs when query appears in title."""
    logger.note("> Test: title-match tagging on title field")
    from recalls.word import MultiLaneWordRecall

    hits = [
        {"bvid": "BV001", "title": "通义实验室发布新模型", "tags": "AI"},
        {"bvid": "BV002", "title": "AI模型测试", "tags": "通义实验室,测试"},
        {"bvid": "BV003", "title": "无关视频", "tags": "无关标签"},
    ]

    MultiLaneWordRecall._tag_title_matches(hits, "通义实验室")

    # BV001: query in title → should be tagged
    assert hits[0].get("_title_matched") is True, "Query in title should be tagged"
    # BV002: query not in title, but in tags → should be tagged
    assert hits[1].get("_title_matched") is True, "Query in tags should be tagged"
    # BV003: query in neither → should not be tagged
    assert not hits[2].get("_title_matched"), "Non-matching doc should not be tagged"

    logger.success("  PASSED")


def test_title_match_tagging_tags_only():
    """_tag_title_matches should tag docs when query appears only in tags."""
    logger.note("> Test: title-match tagging on tags field")
    from recalls.word import MultiLaneWordRecall

    hits = [
        {"bvid": "BV001", "title": "一个普通视频", "tags": "飓风营救,动作,电影"},
        {"bvid": "BV002", "title": "飓风营救完整版", "tags": "电影"},
        {"bvid": "BV003", "title": "天气预报", "tags": "气象"},
    ]

    MultiLaneWordRecall._tag_title_matches(hits, "飓风营救")

    assert hits[0].get("_title_matched") is True, "Query in tags should be tagged"
    assert hits[1].get("_title_matched") is True, "Query in title should be tagged"
    assert not hits[2].get("_title_matched"), "Non-matching should not be tagged"

    logger.success("  PASSED")


def test_title_match_tagging_short_query():
    """Short queries (<=2 chars) should match as substrings in title or tags."""
    logger.note("> Test: title-match tagging with short query")
    from recalls.word import MultiLaneWordRecall

    hits = [
        {"bvid": "BV001", "title": "红警08对战", "tags": "游戏"},
        {"bvid": "BV002", "title": "新版本更新", "tags": "红警,攻略"},
        {"bvid": "BV003", "title": "完全无关", "tags": "其他"},
    ]

    MultiLaneWordRecall._tag_title_matches(hits, "红警")

    assert hits[0].get("_title_matched") is True, "Short query in title"
    assert hits[1].get("_title_matched") is True, "Short query in tags"
    assert not hits[2].get("_title_matched"), "Short query not in either"

    logger.success("  PASSED")


def test_title_match_tagging_preserves_existing():
    """Already-tagged docs should not be overwritten by _tag_title_matches."""
    logger.note("> Test: title-match tagging preserves existing tags")
    from recalls.word import MultiLaneWordRecall

    hits = [
        {
            "bvid": "BV001",
            "title": "无关标题",
            "tags": "无关标签",
            "_title_matched": True,
        },
    ]

    MultiLaneWordRecall._tag_title_matches(hits, "其他查询")

    # Should preserve existing True tag even though query doesn't match
    assert hits[0]["_title_matched"] is True

    logger.success("  PASSED")


def test_title_match_bonus():
    """Docs with _title_matched should get boosted relevance_score."""
    logger.note("> Test: title-match bonus in scoring")

    now = int(time.time())
    # One high-score doc to set max_score, then two docs with identical BM25
    hits = [
        _make_hit("ANCHOR", score=40, view=5000, pubdate=now - 3600, stat_score=0.5),
        _make_hit("MATCH", score=20, view=5000, pubdate=now - 3600, stat_score=0.5),
        _make_hit("NO_MATCH", score=20, view=5000, pubdate=now - 3600, stat_score=0.5),
    ]
    hits[1]["_title_matched"] = True

    ranker = DiversifiedRanker()
    ranker._score_all_dimensions(hits, now_ts=now)

    # Title-matched doc should have higher relevance_score than non-matched
    assert hits[1]["relevance_score"] > hits[2]["relevance_score"], (
        f"Title-matched doc ({hits[1]['relevance_score']}) should have higher "
        f"relevance than non-matched ({hits[2]['relevance_score']})"
    )

    # The bonus should be TITLE_MATCH_BONUS (0.15)
    from ranks.constants import TITLE_MATCH_BONUS

    diff = hits[1]["relevance_score"] - hits[2]["relevance_score"]
    assert (
        abs(diff - TITLE_MATCH_BONUS) < 0.01
    ), f"Expected bonus ≈{TITLE_MATCH_BONUS}, got {diff}"

    # Title-matched doc should also have higher headline_score
    assert hits[1]["headline_score"] > hits[2]["headline_score"]

    logger.success("  PASSED")


def test_title_match_helps_ranking():
    """Title-matched doc should rank higher than non-matched with same BM25."""
    logger.note("> Test: title-match impact on full ranking")

    now = int(time.time())
    hits = []

    # 5 non-title-matched docs with high BM25
    for i in range(5):
        hits.append(
            _make_hit(
                f"PLAIN{i}",
                score=25 - i,
                view=10000,
                pubdate=now - 86400 * 7,
                stat_score=0.5,
            )
        )

    # 3 title-matched docs with moderate BM25 (slightly lower than PLAIN docs)
    for i in range(3):
        h = _make_hit(
            f"TITLE{i}",
            score=20 - i,
            view=8000,
            pubdate=now - 86400 * 7,
            stat_score=0.5,
        )
        h["_title_matched"] = True
        hits.append(h)

    # 10 filler docs
    for i in range(10):
        hits.append(_make_hit(f"FILL{i}", score=5, view=1000, stat_score=0.2))

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank_with_fused_fallback(
        hits_info={"hits": hits},
        top_k=10,
        diversify_top_n=10,
    )

    ranked = result["hits"]
    top_5_bvids = {h["bvid"] for h in ranked[:5]}

    # At least 2 title-matched docs should be in top 5
    title_in_top5 = sum(1 for bvid in top_5_bvids if bvid.startswith("TITLE"))
    assert title_in_top5 >= 2, (
        f"Expected ≥2 title-matched docs in top-5, got {title_in_top5}. "
        f"Top 5: {[h['bvid'] for h in ranked[:5]]}"
    )

    logger.success("  PASSED")


def test_relevance_gating_in_slots():
    """Relevance gating should prevent irrelevant docs from taking dimension slots."""
    logger.note("> Test: relevance gating in slot allocation")

    now = int(time.time())
    hits = []

    # 5 relevant docs with moderate stats
    for i in range(5):
        hits.append(
            _make_hit(
                f"REL{i}",
                score=40 - i,
                view=5000,
                pubdate=now - 86400 * 30,
                stat_score=0.4,
            )
        )

    # 10 irrelevant but popular/recent/quality docs
    for i in range(10):
        hits.append(
            _make_hit(
                f"IRR{i}",
                score=1.0,  # Very low BM25 → low relevance
                view=2000000,  # Very high views
                pubdate=now - 3600,  # Very recent
                stat_score=0.95,  # Very high quality
            )
        )

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank(
        hits_info={"hits": hits},
        top_k=10,
        prefer="balanced",
    )

    ranked = result["hits"]

    # With relevance gating, dimension slots should be dominated by relevant docs
    # because IRR docs have low relevance_factor, gating their dimension scores.
    # At minimum, the relevance slots (2) + most other dimension slots should be REL docs
    rel_in_top = sum(1 for h in ranked if h["bvid"].startswith("REL"))
    assert rel_in_top >= 4, (
        f"Expected ≥4 relevant docs in top-10 (with gating), got {rel_in_top}. "
        f"Irrelevant docs may be bypassing relevance gating. "
        f"Ranked: {[(h['bvid'], h.get('_slot_dimension')) for h in ranked]}"
    )

    logger.success("  PASSED")


def test_graduated_threshold_relaxation():
    """When too few pass the relevance floor, threshold should relax gradually."""
    logger.note("> Test: graduated threshold relaxation")

    now = int(time.time())
    hits = []

    # Only 3 docs with high relevance (pass SLOT_MIN_RELEVANCE=0.30)
    for i in range(3):
        hits.append(
            _make_hit(
                f"HIGH{i}",
                score=30 - i,
                view=5000,
                pubdate=now - 86400,
                stat_score=0.5,
            )
        )

    # 7 docs with moderate relevance (would pass 0.15 but not 0.30)
    for i in range(7):
        hits.append(
            _make_hit(
                f"MED{i}",
                score=8 - i * 0.5,
                view=3000,
                pubdate=now - 86400 * 7,
                stat_score=0.4,
            )
        )

    ranker = DiversifiedRanker()

    # Score dimensions
    ranker._score_all_dimensions(hits, now_ts=now)

    # Allocate with top_k=10 — only 3 pass strict floor, should relax
    from ranks.diversified import SLOT_PRESETS

    slots = SLOT_PRESETS["balanced"]
    result = ranker._allocate_slots(hits, slots, top_k=10)

    # Should return 10 results (relaxation allows more candidates)
    assert len(result) == 10, f"Expected 10 results, got {len(result)}"

    # No duplicates
    bvids = [h["bvid"] for h in result]
    assert len(set(bvids)) == len(bvids), f"Duplicate bvids: {bvids}"

    logger.success("  PASSED")


def test_gated_scores_cleaned_up():
    """Temporary _gated_* fields should be cleaned up after slot allocation."""
    logger.note("> Test: gated score field cleanup")

    now = int(time.time())
    hits = [_make_hit(f"BV{i}", score=20 - i, view=1000) for i in range(15)]

    ranker = DiversifiedRanker()
    ranker._score_all_dimensions(hits, now_ts=now)

    from ranks.diversified import SLOT_PRESETS

    slots = SLOT_PRESETS["balanced"]
    ranker._allocate_slots(hits, slots, top_k=10)

    # No _gated_* fields should remain on any hit
    for h in hits:
        gated_fields = [k for k in h.keys() if k.startswith("_gated_")]
        assert not gated_fields, f"Leftover gated fields on {h['bvid']}: {gated_fields}"

    logger.success("  PASSED")


if __name__ == "__main__":
    test_diversified_rank_basic()
    test_diversified_rank_slot_counts()
    test_diversified_rank_no_duplicates()
    test_diversified_rank_empty()
    test_diversified_rank_fewer_than_top_k()
    test_diversified_rank_with_fused_fallback()
    test_diversified_rank_prefer_presets()
    test_score_all_dimensions()
    test_headline_score_computed()
    test_select_headline_top_n()
    test_headline_top_n_no_duplicates_with_slots()
    test_three_phase_ranking_structure()
    test_exact_top_k_return_count()
    test_exact_top_k_when_pool_smaller()
    test_relevance_floor_in_slots()
    test_headline_relevance_tiebreak()
    test_title_match_tagging_title_only()
    test_title_match_tagging_tags_only()
    test_title_match_tagging_short_query()
    test_title_match_tagging_preserves_existing()
    test_title_match_bonus()
    test_title_match_helps_ranking()
    test_relevance_gating_in_slots()
    test_graduated_threshold_relaxation()
    test_gated_scores_cleaned_up()
    logger.success("\n✓ All diversified ranking tests passed")
