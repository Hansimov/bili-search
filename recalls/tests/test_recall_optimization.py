"""
Unit tests for recall optimization features.

Tests for:
1. Noise filter target_count guarantee (backfill)
2. Owner intent detection (_detect_owner_intent)
3. Owner match tagging (_tag_owner_matches)
4. RecallPool merge metadata propagation
"""

import pytest
from recalls.base import RecallPool, RecallResult, NoiseFilter
from recalls.manager import RecallManager
from recalls.word import MultiLaneWordRecall


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
# Owner Intent Detection
# =============================================================================


class TestOwnerIntentDetection:
    """Test _detect_owner_intent in RecallManager."""

    def test_exact_token_match(self):
        """'红警08' matches owner '红警HBK08' (tokens: 红警+08 ⊆ 红警+hbk+08)."""
        hits = [
            {"owner": {"name": "红警HBK08", "mid": 1}},
            {"owner": {"name": "红警HBK08", "mid": 1}},
            {"owner": {"name": "红警V神", "mid": 2}},
            {"owner": {"name": "普通用户", "mid": 3}},
        ]
        names = RecallManager._detect_owner_intent("红警08", hits)
        assert "红警HBK08" in names
        assert "红警V神" not in names
        assert "普通用户" not in names

    def test_no_match_partial_tokens(self):
        """'通义实验室' should NOT match owner '酷玩实验室' (missing '通义')."""
        hits = [
            {"owner": {"name": "酷玩实验室", "mid": 1}},
            {"owner": {"name": "灵镜实验室", "mid": 2}},
            {"owner": {"name": "超前实验室", "mid": 3}},
        ]
        names = RecallManager._detect_owner_intent("通义实验室", hits)
        assert len(names) == 0

    def test_english_query_match(self):
        """'chatgpt' matches owner 'ChatGPT-官方' (token: chatgpt ⊆ chatgpt+官方)."""
        hits = [
            {"owner": {"name": "ChatGPT-官方", "mid": 1}},
            {"owner": {"name": "ChatGPT-官方", "mid": 1}},
            {"owner": {"name": "AI助手", "mid": 2}},
        ]
        names = RecallManager._detect_owner_intent("chatgpt", hits)
        assert "ChatGPT-官方" in names

    def test_multi_token_query(self):
        """'吴恩达大模型' matches owner '吴恩达大模型' exactly."""
        hits = [
            {"owner": {"name": "吴恩达大模型", "mid": 1}},
            {"owner": {"name": "吴恩达LLM", "mid": 2}},
        ]
        names = RecallManager._detect_owner_intent("吴恩达大模型", hits)
        # "吴恩达大模型": query_tokens = {"吴恩达", "大模型"}
        # owner "吴恩达大模型": tokens = {"吴恩达", "大模型"} → match ✓
        # owner "吴恩达LLM": tokens = {"吴恩达", "llm"} → no "大模型" → no match ✗
        assert "吴恩达大模型" in names
        assert "吴恩达LLM" not in names

    def test_empty_query(self):
        """Empty query returns empty."""
        hits = [{"owner": {"name": "Test", "mid": 1}}]
        assert RecallManager._detect_owner_intent("", hits) == []

    def test_empty_hits(self):
        """Empty hits returns empty."""
        assert RecallManager._detect_owner_intent("test", []) == []

    def test_sorted_by_frequency(self):
        """Results sorted by frequency descending."""
        hits = [
            {"owner": {"name": "红警HBK08", "mid": 1}},
            {"owner": {"name": "红警HBK08", "mid": 1}},
            {"owner": {"name": "红警HBK08", "mid": 1}},
        ]
        names = RecallManager._detect_owner_intent("红警08", hits)
        assert names[0] == "红警HBK08"

    def test_missing_owner_field(self):
        """Hits without owner field are skipped."""
        hits = [
            {"title": "No owner field"},
            {"owner": "not a dict"},
            {"owner": {"name": "红警HBK08", "mid": 1}},
        ]
        names = RecallManager._detect_owner_intent("红警08", hits)
        assert "红警HBK08" in names


# =============================================================================
# Owner Match Tagging
# =============================================================================


class TestOwnerMatchTagging:
    """Test _tag_owner_matches in MultiLaneWordRecall."""

    def test_tags_matching_owners(self):
        """Hits from matching owners get tagged."""
        hits = [
            {"owner": {"name": "红警HBK08"}, "title": "红警08视频"},
            {"owner": {"name": "红警V神"}, "title": "红警视频"},
            {"owner": {"name": "普通用户"}, "title": "其他视频"},
        ]
        MultiLaneWordRecall._tag_owner_matches(hits, "红警08")
        assert hits[0].get("_owner_matched") is True
        assert hits[0].get("_matched_owner_name") == "红警HBK08"
        assert hits[1].get("_owner_matched") is None  # 红警V神 doesn't match
        assert hits[2].get("_owner_matched") is None

    def test_no_match_no_tags(self):
        """When no owner matches, no tags are set."""
        hits = [
            {"owner": {"name": "通义大模型"}, "title": "通义视频"},
        ]
        MultiLaneWordRecall._tag_owner_matches(hits, "红警08")
        assert hits[0].get("_owner_matched") is None

    def test_empty_query_no_tags(self):
        """Empty query doesn't tag anything."""
        hits = [{"owner": {"name": "Test"}}]
        MultiLaneWordRecall._tag_owner_matches(hits, "")
        assert hits[0].get("_owner_matched") is None


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

    def test_merge_preserves_owner_matched(self):
        """_owner_matched and _matched_owner_name survive merge."""
        r1 = RecallResult(
            hits=[
                {
                    "bvid": "BV1",
                    "score": 10,
                    "_owner_matched": True,
                    "_matched_owner_name": "红警HBK08",
                }
            ],
            lane="a",
            total_hits=1,
            took_ms=0,
            timed_out=False,
        )
        pool = RecallPool.merge(r1)
        bv1 = pool.hits[0]
        assert bv1.get("_owner_matched") is True
        assert bv1.get("_matched_owner_name") == "红警HBK08"

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
