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

    # The balanced preset allocates {relevance: 2, quality: 2, recency: 2, popularity: 1}
    # = 7 dimension slots + 3 fused slots to fill top_k=10
    assert (
        dims.get("relevance", 0) == 2
    ), f"Expected 2 relevance slots, got {dims.get('relevance')}"
    assert (
        dims.get("recency", 0) == 2
    ), f"Expected 2 recency slots, got {dims.get('recency')}"
    assert (
        dims.get("quality", 0) == 2
    ), f"Expected 2 quality slots, got {dims.get('quality')}"
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
    logger.success("\n✓ All diversified ranking tests passed")
