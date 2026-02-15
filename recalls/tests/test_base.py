"""
Tests for recalls/base.py — RecallResult, RecallPool merging and deduplication.
"""

from tclogger import logger

from recalls.base import RecallResult, RecallPool


def _make_hit(bvid: str, score: float = 0.0, **kwargs) -> dict:
    """Helper to create a minimal hit dict."""
    return {"bvid": bvid, "score": score, **kwargs}


def test_recall_pool_merge_basic():
    """RecallPool.merge should deduplicate docs across lanes."""
    logger.note("> Test: RecallPool.merge basic dedup")

    r1 = RecallResult(
        hits=[_make_hit("BV001", 10.0), _make_hit("BV002", 8.0)],
        lane="relevance",
        total_hits=1000,
        took_ms=50.0,
    )
    r2 = RecallResult(
        hits=[_make_hit("BV002", 5.0), _make_hit("BV003", 3.0)],
        lane="popularity",
        total_hits=500,
        took_ms=30.0,
    )

    pool = RecallPool.merge(r1, r2)

    # Should have 3 unique bvids
    bvids = [h["bvid"] for h in pool.hits]
    assert len(bvids) == 3, f"Expected 3 hits, got {len(bvids)}: {bvids}"
    assert set(bvids) == {"BV001", "BV002", "BV003"}

    # Lane tags
    assert pool.lane_tags["BV001"] == {"relevance"}
    assert pool.lane_tags["BV002"] == {"relevance", "popularity"}
    assert pool.lane_tags["BV003"] == {"popularity"}

    # Metadata
    assert pool.total_hits == 1000  # max of 1000, 500
    assert pool.took_ms == 50.0  # max of 50, 30
    assert pool.timed_out is False

    # lanes_info
    assert "relevance" in pool.lanes_info
    assert "popularity" in pool.lanes_info
    assert pool.lanes_info["relevance"]["hit_count"] == 2
    assert pool.lanes_info["popularity"]["hit_count"] == 2

    logger.success("  PASSED")


def test_recall_pool_merge_preserves_first_occurrence():
    """When merging, the first occurrence's scores should be preserved,
    while additional lane ranks are added."""
    logger.note("> Test: RecallPool.merge preserves first occurrence")

    r1 = RecallResult(
        hits=[_make_hit("BV001", score=10.0, title="First Title")],
        lane="relevance",
    )
    r2 = RecallResult(
        hits=[_make_hit("BV001", score=5.0, title="Different Title")],
        lane="recency",
    )

    pool = RecallPool.merge(r1, r2)

    # Should have 1 hit
    assert len(pool.hits) == 1
    hit = pool.hits[0]

    # First occurrence's main data preserved
    assert hit["score"] == 10.0
    assert hit["title"] == "First Title"

    # Both lane ranks present
    assert hit["relevance_rank"] == 0
    assert hit["recency_rank"] == 0

    # Tagged with both lanes
    assert pool.lane_tags["BV001"] == {"relevance", "recency"}

    logger.success("  PASSED")


def test_recall_pool_merge_rank_tracking():
    """Lane ranks should reflect position within each lane."""
    logger.note("> Test: RecallPool.merge rank tracking")

    r1 = RecallResult(
        hits=[
            _make_hit("BV001", 10.0),
            _make_hit("BV002", 8.0),
            _make_hit("BV003", 6.0),
        ],
        lane="relevance",
    )

    pool = RecallPool.merge(r1)

    assert pool.hits[0]["relevance_rank"] == 0
    assert pool.hits[1]["relevance_rank"] == 1
    assert pool.hits[2]["relevance_rank"] == 2

    logger.success("  PASSED")


def test_recall_pool_merge_timeout_propagation():
    """Timeout flag should be set if any lane timed out."""
    logger.note("> Test: RecallPool.merge timeout propagation")

    r1 = RecallResult(hits=[], lane="a", timed_out=False)
    r2 = RecallResult(hits=[], lane="b", timed_out=True)

    pool = RecallPool.merge(r1, r2)
    assert pool.timed_out is True

    pool2 = RecallPool.merge(r1)
    assert pool2.timed_out is False

    logger.success("  PASSED")


def test_recall_pool_merge_empty():
    """Merging empty results should produce an empty pool."""
    logger.note("> Test: RecallPool.merge empty")

    r1 = RecallResult(hits=[], lane="empty")
    pool = RecallPool.merge(r1)

    assert len(pool.hits) == 0
    assert pool.total_hits == 0

    logger.success("  PASSED")


def test_recall_pool_merge_skips_no_bvid():
    """Hits without bvid should be skipped."""
    logger.note("> Test: RecallPool.merge skips no-bvid hits")

    r1 = RecallResult(
        hits=[{"score": 5.0}, _make_hit("BV001", 10.0)],
        lane="relevance",
    )

    pool = RecallPool.merge(r1)
    assert len(pool.hits) == 1
    assert pool.hits[0]["bvid"] == "BV001"

    logger.success("  PASSED")


def test_recall_pool_merge_four_lanes():
    """Test merging all 4 lanes as in the multi-lane recall scenario."""
    logger.note("> Test: RecallPool.merge 4-lane scenario")

    relevance_hits = [_make_hit(f"BV{i:02d}", score=100 - i) for i in range(5)]
    popularity_hits = [_make_hit(f"BV{i:02d}") for i in [3, 4, 10, 11]]
    recency_hits = [_make_hit(f"BV{i:02d}") for i in [0, 1, 20, 21]]
    quality_hits = [_make_hit(f"BV{i:02d}") for i in [2, 30, 31, 32]]

    pool = RecallPool.merge(
        RecallResult(hits=relevance_hits, lane="relevance", total_hits=10000),
        RecallResult(hits=popularity_hits, lane="popularity", total_hits=5000),
        RecallResult(hits=recency_hits, lane="recency", total_hits=8000),
        RecallResult(hits=quality_hits, lane="quality", total_hits=3000),
    )

    # Count unique bvids
    all_input = set()
    for bvid_str in [
        "BV00",
        "BV01",
        "BV02",
        "BV03",
        "BV04",
        "BV03",
        "BV04",
        "BV10",
        "BV11",
        "BV00",
        "BV01",
        "BV20",
        "BV21",
        "BV02",
        "BV30",
        "BV31",
        "BV32",
    ]:
        all_input.add(bvid_str)

    assert len(pool.hits) == len(
        all_input
    ), f"Expected {len(all_input)} unique, got {len(pool.hits)}"

    # BV03 should be in both relevance and popularity
    assert "relevance" in pool.lane_tags["BV03"]
    assert "popularity" in pool.lane_tags["BV03"]

    # BV00 should be in both relevance and recency
    assert "relevance" in pool.lane_tags["BV00"]
    assert "recency" in pool.lane_tags["BV00"]

    # total_hits should be max across lanes
    assert pool.total_hits == 10000

    logger.success("  PASSED")


if __name__ == "__main__":
    test_recall_pool_merge_basic()
    test_recall_pool_merge_preserves_first_occurrence()
    test_recall_pool_merge_rank_tracking()
    test_recall_pool_merge_timeout_propagation()
    test_recall_pool_merge_empty()
    test_recall_pool_merge_skips_no_bvid()
    test_recall_pool_merge_four_lanes()
    logger.success("\n✓ All recalls/base tests passed")
