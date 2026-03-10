"""Unit tests for recall optimization features."""

import pytest
from recalls.base import RecallPool, RecallResult, NoiseFilter


# =============================================================================
# Noise Filter Target Count Guarantee
# =============================================================================


class TestNoiseFilterTargetCount:
    """Test that filter_noise respects target_count."""

    def _make_pool(self, hits: list[dict]) -> RecallPool:
        tags = {h["bvid"]: {"test"} for h in hits}
        return RecallPool(
            hits=hits,
            lanes_info={},
            total_hits=len(hits),
            took_ms=0,
            timed_out=False,
            lane_tags=tags,
        )

    def test_target_count_keeps_minimum(self):
        """When filtering would drop below target, backfill re-adds best removed docs."""
        hits = [
            {
                "bvid": f"BV{i}",
                "score": 100 - i,
                "title": f"Title {i}" * 5,
                "desc": "desc " * 10,
            }
            for i in range(50)
        ]
        pool = self._make_pool(hits)
        # Without target: aggressive filtering might remove many
        filtered_no_target = pool.filter_noise(score_ratio=0.5, target_count=None)
        # With target=40: must keep at least 40
        filtered_with_target = pool.filter_noise(score_ratio=0.5, target_count=40)
        assert len(filtered_with_target.hits) >= 40

    def test_target_count_not_needed(self):
        """When enough docs pass, target_count doesn't change behavior."""
        hits = [
            {
                "bvid": f"BV{i}",
                "score": 90 + i,
                "title": f"Title {i}" * 5,
                "desc": "desc " * 10,
            }
            for i in range(50)
        ]
        pool = self._make_pool(hits)
        filtered = pool.filter_noise(score_ratio=0.1, target_count=30)
        # All docs have high scores, so all pass
        assert len(filtered.hits) == 50

    def test_target_count_backfill_tracked(self):
        """Backfill count is recorded in lanes_info."""
        # High-scoring docs that pass, plus low-scoring ones that don't
        # Need > NOISE_MIN_HITS_FOR_FILTER (30) total hits for filter to engage
        hits = [
            {
                "bvid": f"BV_high_{i}",
                "score": 100,
                "title": "Good title " * 3,
                "desc": "desc " * 10,
            }
            for i in range(10)
        ] + [
            {
                "bvid": f"BV_low_{i}",
                "score": 5,
                "title": "Low title " * 3,
                "desc": "desc " * 10,
            }
            for i in range(30)
        ]
        pool = self._make_pool(hits)
        filtered = pool.filter_noise(score_ratio=0.5, target_count=25)
        # Should backfill 15 low-scoring docs (10 pass + 15 backfill = 25)
        noise_info = filtered.lanes_info.get("_noise_filter", {})
        assert noise_info.get("backfilled", 0) > 0
        assert len(filtered.hits) >= 25

    def test_empty_pool_unchanged(self):
        """Empty pool returns empty pool."""
        pool = RecallPool(
            hits=[],
            lanes_info={},
            total_hits=0,
            took_ms=0,
            timed_out=False,
            lane_tags={},
        )
        filtered = pool.filter_noise(target_count=100)
        assert len(filtered.hits) == 0


# =============================================================================
# RecallPool Merge Metadata Propagation
# =============================================================================


class TestRecallPoolMerge:
    """Test that merge preserves metadata flags."""

    def test_merge_preserves_title_matched(self):
        """_title_matched flag survives merge."""
        r1 = RecallResult(
            hits=[{"bvid": "BV1", "score": 10, "_title_matched": True}],
            lane="a",
            total_hits=1,
            took_ms=0,
            timed_out=False,
        )
        r2 = RecallResult(
            hits=[{"bvid": "BV2", "score": 8}],
            lane="b",
            total_hits=1,
            took_ms=0,
            timed_out=False,
        )
        pool = RecallPool.merge(r1, r2)
        bv1 = next(h for h in pool.hits if h["bvid"] == "BV1")
        assert bv1.get("_title_matched") is True

    def test_merge_dedup_keeps_best_score(self):
        """When both results have same bvid, keep the one with higher score."""
        r1 = RecallResult(
            hits=[{"bvid": "BV1", "score": 10}],
            lane="a",
            total_hits=1,
            took_ms=0,
            timed_out=False,
        )
        r2 = RecallResult(
            hits=[{"bvid": "BV1", "score": 20}],
            lane="b",
            total_hits=1,
            took_ms=0,
            timed_out=False,
        )
        pool = RecallPool.merge(r1, r2)
        assert len(pool.hits) == 1
        assert pool.hits[0]["score"] == 20


# =============================================================================
# NoiseFilter Content Quality Penalty
# =============================================================================


class TestNoiseFilterContentQuality:
    """Test content quality penalty application."""

    def test_short_text_penalty(self):
        """Short title+desc gets score penalty."""
        hits = [
            {"bvid": "BV1", "score": 100, "title": "Hi", "desc": ""},
            {
                "bvid": "BV2",
                "score": 100,
                "title": "Normal title " * 5,
                "desc": "Description " * 10,
            },
        ]
        NoiseFilter.apply_content_quality_penalty(hits)
        assert hits[0]["score"] < hits[1]["score"]  # Short text penalized

    def test_low_engagement_penalty(self):
        """Low-view docs get score penalty."""
        hits = [
            {
                "bvid": "BV1",
                "score": 100,
                "title": "T" * 30,
                "desc": "D" * 30,
                "stat": {"view": 10},
            },
            {
                "bvid": "BV2",
                "score": 100,
                "title": "T" * 30,
                "desc": "D" * 30,
                "stat": {"view": 100000},
            },
        ]
        NoiseFilter.apply_content_quality_penalty(hits)
        assert hits[0]["score"] < hits[1]["score"]  # Low engagement penalized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
