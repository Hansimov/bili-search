"""
Tests for recall noise filtering.

Tests cover:
1. NoiseFilter.filter_by_score_ratio — lane-level BM25 noise removal
2. NoiseFilter.filter_knn_by_score_ratio — KNN bit vector noise removal
3. RecallPool.filter_noise — pool-level multi-lane noise filtering
4. NoiseFilter.apply_content_quality_penalty — BM25 short-text / low-engagement penalties
5. RecallPool.filter_noise with content quality signals
"""

import pytest
from recalls.base import RecallResult, RecallPool, NoiseFilter
from ranks.constants import (
    NOISE_SCORE_RATIO_GATE,
    NOISE_KNN_SCORE_RATIO,
    NOISE_MIN_HITS_FOR_FILTER,
    NOISE_SHORT_TEXT_MIN_LENGTH,
    NOISE_SHORT_TEXT_PENALTY,
    NOISE_MIN_ENGAGEMENT_VIEWS,
    NOISE_LOW_ENGAGEMENT_PENALTY,
)


# ---- Helpers ----


def make_hits(scores: list[float], bvid_prefix: str = "BV") -> list[dict]:
    """Create hit dicts with given scores."""
    return [
        {"bvid": f"{bvid_prefix}{i}", "score": score} for i, score in enumerate(scores)
    ]


def make_pool(
    lane_hits: dict[str, list[dict]],
) -> RecallPool:
    """Create a RecallPool from lane name -> hits mapping."""
    results = []
    for lane_name, hits in lane_hits.items():
        results.append(
            RecallResult(
                hits=hits,
                lane=lane_name,
                total_hits=len(hits),
            )
        )
    return RecallPool.merge(*results)


# ============================================================================
# NoiseFilter.filter_by_score_ratio tests
# ============================================================================


class TestFilterByScoreRatio:
    """Test lane-level score ratio filtering."""

    def test_removes_low_scores(self):
        """Hits below ratio * max_score are removed."""
        hits = make_hits([10.0, 8.0, 5.0, 1.0, 0.5])
        # gate = 10.0 * 0.12 = 1.2
        filtered = NoiseFilter.filter_by_score_ratio(hits, ratio=0.12, min_hits=3)
        scores = [h["score"] for h in filtered]
        assert 0.5 not in scores
        assert 1.0 not in scores  # 1.0 < 1.2
        assert 10.0 in scores
        assert 8.0 in scores
        assert 5.0 in scores

    def test_preserves_all_when_too_few(self):
        """Don't filter if total hits below min_hits."""
        hits = make_hits([10.0, 0.1, 0.01])
        filtered = NoiseFilter.filter_by_score_ratio(hits, ratio=0.5, min_hits=10)
        assert len(filtered) == 3  # All preserved

    def test_handles_zero_scores(self):
        """Handles case where all scores are 0."""
        hits = make_hits([0.0, 0.0, 0.0])
        filtered = NoiseFilter.filter_by_score_ratio(hits, ratio=0.5, min_hits=2)
        assert len(filtered) == 3  # Can't filter when max=0

    def test_handles_empty(self):
        """Handles empty list."""
        filtered = NoiseFilter.filter_by_score_ratio([])
        assert filtered == []

    def test_custom_score_field(self):
        """Uses custom score field."""
        hits = [
            {"bvid": "BV1", "rerank_score": 1.0},
            {"bvid": "BV2", "rerank_score": 0.5},
            {"bvid": "BV3", "rerank_score": 0.05},
        ]
        # gate = 1.0 * 0.12 = 0.12
        filtered = NoiseFilter.filter_by_score_ratio(
            hits, score_field="rerank_score", ratio=0.12, min_hits=2
        )
        assert len(filtered) == 2
        assert filtered[0]["bvid"] == "BV1"
        assert filtered[1]["bvid"] == "BV2"

    def test_score_ratio_boundary(self):
        """Hits exactly at the gate threshold pass."""
        hits = make_hits([10.0, 5.0, 1.2, 1.1999])
        # gate = 10.0 * 0.12 = 1.2
        filtered = NoiseFilter.filter_by_score_ratio(hits, ratio=0.12, min_hits=3)
        scores = [h["score"] for h in filtered]
        assert 1.2 in scores  # Exactly at gate passes
        assert 1.1999 not in scores  # Just below gate removed

    def test_aggressive_ratio(self):
        """Higher ratio removes more docs."""
        hits = make_hits([10.0, 8.0, 5.0, 3.0, 1.0])
        # ratio=0.5 → gate = 5.0
        filtered = NoiseFilter.filter_by_score_ratio(hits, ratio=0.5, min_hits=3)
        assert len(filtered) == 3  # 10, 8, 5
        assert all(h["score"] >= 5.0 for h in filtered)


# ============================================================================
# NoiseFilter.filter_knn_by_score_ratio tests
# ============================================================================


class TestFilterKNNScoreRatio:
    """Test KNN-specific noise filtering (stricter ratio)."""

    def test_stricter_than_bm25(self):
        """KNN ratio is stricter (0.5 default vs 0.12 for BM25)."""
        hits = make_hits([0.020, 0.018, 0.015, 0.012, 0.010, 0.005])
        # KNN gate = 0.020 * 0.5 = 0.010
        knn_filtered = NoiseFilter.filter_knn_by_score_ratio(hits, min_hits=3)
        # Score 0.005 < 0.010, removed
        assert len(knn_filtered) == 5
        assert all(h["score"] >= 0.010 for h in knn_filtered)

    def test_knn_narrow_range(self):
        """KNN scores cluster in narrow range — filters tail effectively."""
        # Simulates LSH hamming scores: 0.019, 0.018, 0.017, ..., 0.001
        scores = [0.019 - i * 0.001 for i in range(19)]
        hits = make_hits(scores)
        # gate = 0.019 * 0.5 = 0.0095
        filtered = NoiseFilter.filter_knn_by_score_ratio(hits, min_hits=10)
        # Keep scores >= 0.0095: 0.019, 0.018, ..., 0.010 = 10 hits
        assert len(filtered) == 10

    def test_knn_preserves_top_hits(self):
        """Good KNN matches are always preserved."""
        hits = make_hits([0.025, 0.023, 0.020, 0.018])
        filtered = NoiseFilter.filter_knn_by_score_ratio(hits, min_hits=3)
        # gate = 0.025 * 0.5 = 0.0125 — all pass
        assert len(filtered) == 4


# ============================================================================
# RecallPool.filter_noise tests
# ============================================================================


class TestRecallPoolFilterNoise:
    """Test pool-level noise filtering with multi-lane awareness."""

    def test_removes_low_confidence_docs(self):
        """Docs with low scores are removed from the pool."""
        # Relevance lane: high scores
        relevance_hits = make_hits([20.0, 18.0, 15.0, 12.0, 10.0], "rel")
        # Popularity lane: mix of high and low score docs
        popularity_hits = [
            {"bvid": "pop0", "score": 15.0, "stat": {"view": 1000000}},
            {"bvid": "pop1", "score": 12.0, "stat": {"view": 500000}},
            {"bvid": "pop2", "score": 1.0, "stat": {"view": 100000}},  # Low BM25
            {"bvid": "pop3", "score": 0.5, "stat": {"view": 50000}},  # Noise
        ]

        pool = make_pool(
            {
                "relevance": relevance_hits,
                "popularity": popularity_hits,
            }
        )

        # Before filter: 9 unique hits
        assert len(pool.hits) == 9

        filtered = pool.filter_noise(score_ratio=0.12, min_hits=5)

        # gate = 20.0 * 0.12 = 2.4
        # pop2 (1.0) and pop3 (0.5) should be removed
        assert len(filtered.hits) < len(pool.hits)
        bvids = {h["bvid"] for h in filtered.hits}
        assert "pop3" not in bvids  # 0.5 < 2.4

    def test_multi_lane_docs_get_lower_threshold(self):
        """Docs appearing in multiple lanes survive with lower scores."""
        # Doc "shared" appears in both lanes
        lane1_hits = [
            {"bvid": "shared", "score": 5.0},
            {"bvid": "lane1_only", "score": 8.0},
        ]
        lane2_hits = [
            {"bvid": "shared", "score": 5.0},
            {"bvid": "lane2_only", "score": 3.0},
            {"bvid": "lane2_low", "score": 0.5},
        ]

        # Add enough hits to exceed min_hits threshold
        for i in range(30):
            lane1_hits.append({"bvid": f"filler1_{i}", "score": 10.0})

        pool = make_pool({"lane1": lane1_hits, "lane2": lane2_hits})

        # gate = 10.0 * 0.12 = 1.2
        # multi-lane gate = 1.2 * 0.5 = 0.6
        filtered = pool.filter_noise(
            score_ratio=0.12, min_hits=20, multi_lane_factor=0.5
        )

        bvids = {h["bvid"] for h in filtered.hits}
        # "shared" has score 5.0, multi-lane gate 0.6 → passes
        assert "shared" in bvids
        # "lane2_low" has score 0.5, single-lane gate 1.2 → removed
        assert "lane2_low" not in bvids

    def test_preserves_all_when_pool_small(self):
        """Don't filter if pool has fewer hits than min_hits."""
        hits = make_hits([10.0, 0.1, 0.01])
        pool = make_pool({"relevance": hits})
        filtered = pool.filter_noise(min_hits=100)
        assert len(filtered.hits) == len(pool.hits)

    def test_empty_pool(self):
        """Empty pool returns empty pool."""
        pool = RecallPool()
        filtered = pool.filter_noise()
        assert len(filtered.hits) == 0

    def test_updates_lanes_info(self):
        """Noise filter stats are added to lanes_info."""
        hits = make_hits([10.0] * 40 + [0.1] * 10)
        pool = make_pool({"relevance": hits})
        filtered = pool.filter_noise(score_ratio=0.12, min_hits=30)

        assert "_noise_filter" in filtered.lanes_info
        info = filtered.lanes_info["_noise_filter"]
        assert info["removed"] > 0
        assert info["kept"] == len(filtered.hits)
        assert info["gate"] == pytest.approx(1.2, abs=0.01)  # 10.0 * 0.12

    def test_lane_tags_updated(self):
        """Filtered pool has correct lane_tags for remaining hits."""
        relevance_hits = make_hits([20.0, 15.0, 10.0], "rel")
        popularity_hits = make_hits([18.0, 0.5], "pop")

        # Add fillers to exceed min_hits
        for i in range(30):
            relevance_hits.append({"bvid": f"filler_{i}", "score": 12.0})

        pool = make_pool({"relevance": relevance_hits, "popularity": popularity_hits})
        filtered = pool.filter_noise(score_ratio=0.12, min_hits=20)

        # Verify all remaining hits have lane_tags
        for hit in filtered.hits:
            bvid = hit.get("bvid")
            if bvid:
                assert bvid in filtered.lane_tags


# ============================================================================
# Integration scenario tests
# ============================================================================


class TestNoiseFilterScenarios:
    """Test noise filtering with realistic scenarios."""

    def test_bm25_rare_keyword_noise(self):
        """Simulates BM25 matching on rare unimportant keywords.

        Query: "影视飓风" (a UP主 name)
        BM25 over-matches on "飓风" (hurricane) — a rare term with high IDF.
        Irrelevant docs about actual hurricanes get high-ish BM25 scores
        from the "飓风" term alone, but are semantically unrelated.
        """
        # Relevant docs: high scores, matching full entity
        relevant = [{"bvid": f"rel_{i}", "score": 15.0 + i} for i in range(10)]
        # Noise: docs matching just "飓风" with moderate BM25 scores
        noise = [{"bvid": f"noise_{i}", "score": 2.0 + i * 0.3} for i in range(20)]
        # Deep noise: very low scores
        deep_noise = [{"bvid": f"deep_{i}", "score": 0.3 + i * 0.1} for i in range(10)]

        all_hits = relevant + noise + deep_noise
        pool = make_pool({"relevance": all_hits})

        # gate = 24.0 * 0.12 = 2.88
        filtered = pool.filter_noise(score_ratio=0.12, min_hits=20)

        # All relevant docs preserved (scores 15-24)
        rel_count = sum(1 for h in filtered.hits if h["bvid"].startswith("rel_"))
        assert rel_count == 10

        # Deep noise removed (scores 0.3-1.2, all < 2.88)
        deep_count = sum(1 for h in filtered.hits if h["bvid"].startswith("deep_"))
        assert deep_count == 0

    def test_knn_bit_vector_noise(self):
        """Simulates KNN bit vector returning many irrelevant docs.

        LSH hamming distance search returns 1000 docs, but many
        are completely irrelevant due to quantization noise in the
        2048-bit vector space. Scores cluster in a narrow range.
        """
        # Top relevant docs: highest hamming similarity
        relevant = make_hits(
            [0.022, 0.021, 0.020, 0.019, 0.018] * 4, "rel"  # 20 relevant
        )
        # Noise: cluster of irrelevant docs with similar-ish scores
        noise = make_hits(
            [0.008, 0.007, 0.006, 0.005, 0.004, 0.003] * 10, "noise"  # 60 noise
        )

        all_hits = relevant + noise
        # Simulate a KNN pool
        pool = make_pool({"knn": all_hits})

        # KNN gate = 0.022 * 0.5 = 0.011
        filtered = pool.filter_noise(score_ratio=0.5, min_hits=30)

        # All relevant docs preserved
        rel_count = sum(1 for h in filtered.hits if h["bvid"].startswith("rel"))
        assert rel_count == 20

        # All noise removed (max noise score 0.008 < 0.011)
        noise_count = sum(1 for h in filtered.hits if h["bvid"].startswith("noise"))
        assert noise_count == 0


# ============================================================================
# NoiseFilter.apply_content_quality_penalty tests
# ============================================================================


def _make_rich_hit(bvid, score, title="", desc="", views=1000):
    """Create a hit with title, desc, and stat fields."""
    return {
        "bvid": bvid,
        "score": score,
        "title": title,
        "desc": desc,
        "stat": {"view": views},
    }


class TestApplyContentQualityPenalty:
    """Test BM25 short-text and low-engagement penalty adjustments."""

    def test_short_text_gets_penalized(self):
        """Docs with very short title+desc should have reduced scores."""
        hits = [
            _make_rich_hit("BV1", 10.0, title="短标题", desc=""),  # 3 chars < 15
            _make_rich_hit(
                "BV2",
                10.0,
                title="这是一个足够长的正常视频标题不会触发短文本惩罚",
                desc="",
            ),  # > 15 chars
        ]
        NoiseFilter.apply_content_quality_penalty(hits)
        # BV1 has short text (3 < NOISE_SHORT_TEXT_MIN_LENGTH), should be penalized
        assert hits[0]["score"] < 10.0
        assert hits[0].get("_quality_penalty") is not None
        # BV2 has long enough text, should be untouched
        assert hits[1]["score"] == 10.0

    def test_no_penalty_when_no_content(self):
        """Docs with empty title+desc (content_length=0) are not penalized."""
        hits = [
            _make_rich_hit("BV1", 10.0, title="", desc=""),
        ]
        NoiseFilter.apply_content_quality_penalty(hits)
        # content_length == 0, the condition "0 < content_length" is False
        assert hits[0]["score"] == 10.0

    def test_low_engagement_gets_penalized(self):
        """Docs with near-zero views should have reduced scores."""
        long_title = "这是一个足够长的正常视频标题不会触发短文本惩罚"
        hits = [
            _make_rich_hit("BV1", 10.0, title=long_title, views=5),
            _make_rich_hit("BV2", 10.0, title=long_title, views=5000),
        ]
        NoiseFilter.apply_content_quality_penalty(hits)
        assert hits[0]["score"] < 10.0  # Low views penalty
        assert hits[1]["score"] == 10.0  # Sufficient views

    def test_double_penalty_stacks(self):
        """Short text + low engagement should compound penalties."""
        hits = [
            _make_rich_hit("BV1", 10.0, title="短", desc="", views=5),  # Both penalties
        ]
        NoiseFilter.apply_content_quality_penalty(hits)
        expected = 10.0 * NOISE_SHORT_TEXT_PENALTY * NOISE_LOW_ENGAGEMENT_PENALTY
        assert abs(hits[0]["score"] - expected) < 0.001

    def test_no_stat_dict_skips_engagement_penalty(self):
        """Hits without stat dict should not get engagement penalty."""
        hits = [
            {
                "bvid": "BV1",
                "score": 10.0,
                "title": "这是一个足够长的正常视频标题不会触发短文本惩罚",
            }
        ]
        NoiseFilter.apply_content_quality_penalty(hits)
        assert hits[0]["score"] == 10.0  # No penalty applied

    def test_modifies_in_place(self):
        """The method should modify hits in-place and also return them."""
        hits = [_make_rich_hit("BV1", 10.0, title="短", views=5)]
        result = NoiseFilter.apply_content_quality_penalty(hits)
        assert result is hits
        assert hits[0]["score"] < 10.0


# ============================================================================
# RecallPool.filter_noise with content quality signals
# ============================================================================


class TestFilterNoiseWithContentQuality:
    """Test pool-level noise filtering with short-text and low-engagement signals."""

    def test_short_text_docs_filtered_as_noise(self):
        """Short-text docs near the score gate should be removed by penalty."""
        short_doc = _make_rich_hit("BV_SHORT", 3.0, title="短标题", views=10000)
        normal_doc = _make_rich_hit(
            "BV_NORM",
            20.0,
            title="这是一个足够长的正常视频标题不会触发短文本惩罚",
            views=10000,
        )
        filler = [
            _make_rich_hit(
                f"BV_F{i}",
                15.0 - i * 0.3,
                title="这是一个足够长的正常视频标题不会触发短文本惩罚",
                views=5000,
            )
            for i in range(35)
        ]

        all_hits = [normal_doc, short_doc] + filler
        pool = make_pool({"relevance": all_hits})

        # gate = 20.0 * 0.12 = 2.4
        # BV_SHORT effective = 3.0 * SHORT_TEXT_PENALTY = 3.0 * 0.3 = 0.9 < 2.4
        filtered = pool.filter_noise(score_ratio=0.12)

        short_present = any(h["bvid"] == "BV_SHORT" for h in filtered.hits)
        assert not short_present, "Short-text doc should have been filtered out"

        normal_present = any(h["bvid"] == "BV_NORM" for h in filtered.hits)
        assert normal_present, "Normal doc should have survived"

    def test_low_engagement_docs_filtered_as_noise(self):
        """Docs with near-zero views near the gate should be removed."""
        low_view_doc = _make_rich_hit(
            "BV_LOW",
            3.0,
            title="这是一个足够长的正常视频标题不会触发短文本惩罚",
            views=10,
        )
        high_view_doc = _make_rich_hit(
            "BV_HIGH",
            20.0,
            title="这是一个足够长的正常视频标题不会触发短文本惩罚",
            views=100000,
        )
        filler = [
            _make_rich_hit(
                f"BV_F{i}",
                15.0 - i * 0.3,
                title="这是一个足够长的正常视频标题不会触发短文本惩罚",
                views=5000,
            )
            for i in range(35)
        ]

        all_hits = [high_view_doc, low_view_doc] + filler
        pool = make_pool({"relevance": all_hits})

        # gate = 20.0 * 0.12 = 2.4
        # BV_LOW effective = 3.0 * LOW_ENGAGEMENT_PENALTY = 3.0 * 0.15 = 0.45 < 2.4
        filtered = pool.filter_noise(score_ratio=0.12)

        low_present = any(h["bvid"] == "BV_LOW" for h in filtered.hits)
        assert not low_present, "Low-engagement doc should have been filtered out"

    def test_noise_filter_stats_include_penalty_counts(self):
        """Filter stats should report short-text and low-engagement penalized counts."""
        hits = [
            _make_rich_hit(
                "BV1",
                20.0,
                title="这是一个足够长的正常视频标题不会触发短文本惩罚",
                views=10000,
            ),
            _make_rich_hit("BV2", 3.0, title="短", views=10000),  # short text
            _make_rich_hit(
                "BV3",
                3.0,
                title="这是一个足够长的正常视频标题不会触发短文本惩罚",
                views=5,
            ),  # low engagement
        ] + [
            _make_rich_hit(
                f"BV_F{i}",
                10.0,
                title="这是一个足够长的正常视频标题不会触发短文本惩罚",
                views=5000,
            )
            for i in range(35)
        ]

        pool = make_pool({"relevance": hits})
        filtered = pool.filter_noise(score_ratio=0.12)

        stats = filtered.lanes_info.get("_noise_filter", {})
        assert stats.get("short_text_penalized", 0) >= 1
        assert stats.get("low_engagement_penalized", 0) >= 1
