"""
Diversified Slot-Based Ranker

Instead of fusing all signals into one continuous score (which causes
homogeneous top-N results), this ranker allocates "slots" to different
dimensions, ensuring that the top-K results contain representatives
from each dimension: relevant, popular, recent, and high-quality.

Algorithm:
1. Score all candidates on each dimension independently
2. Allocate slots: pick top-N from each dimension (skipping already selected)
3. Fill remaining slots with best overall score
4. Final stable sort by rank_score for display

This guarantees that "相关的"、"热度高的"、"质量高的"、"时间近的"
all appear in the top results.

Scoring improvements over naive max-normalization:
- Popularity uses log-scale normalization because view counts follow
  a power-law distribution. Without log-scale, a 100M-view video
  crushes all 50K-view videos to near-zero popularity scores.
- Fused weights are configurable via DIVERSIFIED_FUSED_WEIGHTS.
"""

import math
import time as _time
from typing import Literal

from tclogger import dict_get

from ranks.scorers import StatsScorer, PubdateScorer
from ranks.constants import (
    RANK_TOP_K,
    RANK_PREFER_TYPE,
    RANK_PREFER,
    DIVERSIFIED_FUSED_WEIGHTS,
)

# Slot allocation presets
SLOT_PRESETS = {
    "balanced": {
        "relevance": 3,
        "quality": 2,
        "recency": 3,
        "popularity": 2,
    },
    "prefer_relevance": {
        "relevance": 5,
        "quality": 2,
        "recency": 2,
        "popularity": 1,
    },
    "prefer_quality": {
        "relevance": 2,
        "quality": 4,
        "recency": 2,
        "popularity": 2,
    },
    "prefer_recency": {
        "relevance": 2,
        "quality": 1,
        "recency": 5,
        "popularity": 2,
    },
}

# Score field names for each dimension
DIMENSION_SCORE_FIELDS = {
    "relevance": "relevance_score",
    "quality": "quality_score",
    "recency": "recency_score",
    "popularity": "popularity_score",
}


class DiversifiedRanker:
    """Slot-based diversified ranking for comprehensive top-K results.

    This ranker ensures that the top-K results contain documents that are:
    - Most relevant (by BM25/embedding score)
    - Most popular (by view count)
    - Most recent (by publish date)
    - Highest quality (by stat_score)

    The key insight: continuous score fusion (weighted sum) creates
    "average" results where nothing stands out. Slot allocation guarantees
    that exceptional items in each dimension are represented.

    Example:
        >>> ranker = DiversifiedRanker()
        >>> result = ranker.diversified_rank(
        ...     hits_info={"hits": candidates},
        ...     top_k=10,
        ...     prefer="balanced",
        ... )
        >>> # Top 10 now has: 3 most relevant, 2 most popular,
        >>> # 3 most recent, 2 highest quality (with dedup)
    """

    def __init__(self):
        self.stats_scorer = StatsScorer()
        self.pubdate_scorer = PubdateScorer()

    def _score_all_dimensions(self, hits: list[dict], now_ts: float = None) -> None:
        """Score each hit on all four dimensions.

        Computes and stores dimension scores in each hit dict:
        - relevance_score: Normalized BM25/hybrid/rerank score [0, 1]
        - quality_score: Stat quality from DocScorer [0, 1)
        - recency_score: Normalized time factor [0, 1]
        - popularity_score: Log-normalized view count [0, 1]

        Popularity uses log-scale normalization because view counts follow
        a power-law distribution. Without log-scale, a single 100M-view
        video crushes all other videos' popularity to near-zero:
            log1p(100,000,000) / log1p(50,000) = 18.4 / 10.8 = 1.7x
        vs linear:
            100,000,000 / 50,000 = 2000x

        Args:
            hits: List of hit dicts to score.
            now_ts: Current timestamp for recency calculation.
        """
        if now_ts is None:
            now_ts = _time.time()

        # Collect raw values for normalization
        raw_scores = []
        raw_views = []
        for h in hits:
            raw_scores.append(
                h.get("rerank_score") or h.get("hybrid_score") or h.get("score") or 0
            )
            raw_views.append(dict_get(h, "stat.view", 0) or 0)

        # Max-normalization for relevance (BM25/cosine scores are well-distributed)
        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score <= 0:
            max_score = 1.0

        # Log-scale normalization for popularity (power-law distributed)
        log_views = [math.log1p(v) for v in raw_views]
        max_log_view = max(log_views) if log_views else 1.0
        if max_log_view <= 0:
            max_log_view = 1.0

        for i, hit in enumerate(hits):
            # Relevance: use best available score
            hit["relevance_score"] = round(
                min(raw_scores[i] / max_score, 1.0) if max_score > 0 else 0.0, 4
            )

            # Quality: bounded [0, 1) from DocScorer
            stats = dict_get(hit, "stat", {})
            hit["quality_score"] = round(self.stats_scorer.calc(stats), 4)

            # Recency: normalized time factor [0, 1]
            pubdate = dict_get(hit, "pubdate", 0)
            time_factor = self.pubdate_scorer.calc(pubdate, now_ts=now_ts)
            hit["recency_score"] = round(self.pubdate_scorer.normalize(time_factor), 4)
            hit["time_factor"] = round(time_factor, 4)

            # Popularity: log-scale normalization
            hit["popularity_score"] = round(log_views[i] / max_log_view, 4)

    def _allocate_slots(
        self,
        hits: list[dict],
        slots: dict[str, int],
        top_k: int,
    ) -> list[dict]:
        """Allocate slots from each dimension, then fill remaining.

        Algorithm:
        1. For each dimension, pick top-N items not already selected
        2. Tag each selected item with its selection reason
        3. Fill remaining top_k slots with best fused score
        4. Assign rank_score for stable ordering

        Args:
            hits: Scored hits (must have dimension scores from _score_all_dimensions).
            slots: Dict mapping dimension name to slot count.
            top_k: Total items to return.

        Returns:
            Ordered list of top_k hits with rank_score assigned.
        """
        selected_bvids: set = set()
        result: list[dict] = []
        slot_order = 0

        # Phase 1: Pick top items from each dimension
        for dimension, count in slots.items():
            score_field = DIMENSION_SCORE_FIELDS.get(dimension, f"{dimension}_score")
            # Sort by this dimension's score
            sorted_hits = sorted(
                hits, key=lambda h: h.get(score_field, 0), reverse=True
            )

            added = 0
            for hit in sorted_hits:
                bvid = hit.get("bvid")
                if bvid and bvid not in selected_bvids and added < count:
                    selected_bvids.add(bvid)
                    hit["_slot_dimension"] = dimension
                    hit["_slot_order"] = slot_order
                    result.append(hit)
                    added += 1
                    slot_order += 1

        # Phase 2: Fill remaining slots with best overall fused score
        remaining = max(0, top_k - len(result))
        if remaining > 0:
            # Fused score: weighted combination of all dimensions
            w = DIVERSIFIED_FUSED_WEIGHTS
            for hit in hits:
                if hit.get("bvid") not in selected_bvids:
                    hit["_fused_score"] = (
                        hit.get("relevance_score", 0) * w["relevance"]
                        + hit.get("quality_score", 0) * w["quality"]
                        + hit.get("recency_score", 0) * w["recency"]
                        + hit.get("popularity_score", 0) * w["popularity"]
                    )

            fused_sorted = sorted(
                [h for h in hits if h.get("bvid") not in selected_bvids],
                key=lambda h: h.get("_fused_score", 0),
                reverse=True,
            )

            for hit in fused_sorted[:remaining]:
                bvid = hit.get("bvid")
                if bvid:
                    selected_bvids.add(bvid)
                    hit["_slot_dimension"] = "fused"
                    hit["_slot_order"] = slot_order
                    result.append(hit)
                    slot_order += 1

        # Phase 3: Assign rank_score for stable ordering
        # Within slot top-K: use slot order (preserves dimension diversity)
        # Beyond slot top-K: use fused score
        for i, hit in enumerate(result):
            # Higher rank_score = higher position
            # slot_order 0 -> highest, slot_order N -> lowest in top section
            hit["rank_score"] = round(1.0 - (i / max(len(result), 1)), 6)

        return result

    def diversified_rank(
        self,
        hits_info: dict,
        top_k: int = RANK_TOP_K,
        prefer: RANK_PREFER_TYPE = RANK_PREFER,
        slot_preset: str = None,
        custom_slots: dict = None,
    ) -> dict:
        """Rank hits with diversified slot allocation.

        Args:
            hits_info: Dict with "hits" list.
            top_k: Total results to return.
            prefer: Preference mode (maps to slot preset).
            slot_preset: Override slot preset name.
            custom_slots: Custom slot allocation dict.

        Returns:
            hits_info with diversified-ranked hits.
        """
        hits = hits_info.get("hits", [])
        if not hits:
            hits_info["rank_method"] = "diversified"
            return hits_info

        now_ts = _time.time()

        # Score all dimensions
        self._score_all_dimensions(hits, now_ts=now_ts)

        # Determine slot allocation
        if custom_slots:
            slots = custom_slots
        else:
            preset_name = slot_preset or prefer or "balanced"
            slots = SLOT_PRESETS.get(preset_name, SLOT_PRESETS["balanced"])

        # Allocate slots and rank
        ranked_hits = self._allocate_slots(hits, slots, top_k)

        hits_info["hits"] = ranked_hits
        hits_info["return_hits"] = len(ranked_hits)
        hits_info["rank_method"] = "diversified"
        hits_info["slot_allocation"] = slots

        # Summary info
        dimension_counts = {}
        for hit in ranked_hits:
            dim = hit.get("_slot_dimension", "unknown")
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        hits_info["dimension_distribution"] = dimension_counts

        return hits_info

    def diversified_rank_with_fused_fallback(
        self,
        hits_info: dict,
        top_k: int = RANK_TOP_K,
        prefer: RANK_PREFER_TYPE = RANK_PREFER,
        diversify_top_n: int = 10,
    ) -> dict:
        """Diversified ranking for top-N, then fused score for rest.

        This is the recommended mode: diversify the first page (top 10),
        then use continuous fused scoring for the remaining results.

        Args:
            hits_info: Dict with "hits" list.
            top_k: Total results to return.
            prefer: Preference mode.
            diversify_top_n: How many items to diversify (first page).

        Returns:
            hits_info with hybrid ranked hits.
        """
        hits = hits_info.get("hits", [])
        if not hits:
            hits_info["rank_method"] = "diversified"
            return hits_info

        now_ts = _time.time()

        # Score all dimensions
        self._score_all_dimensions(hits, now_ts=now_ts)

        # Get slot preset
        preset_name = prefer or "balanced"
        slots = SLOT_PRESETS.get(preset_name, SLOT_PRESETS["balanced"])

        # Phase 1: Diversified selection for top-N
        top_hits = self._allocate_slots(hits, slots, diversify_top_n)
        top_bvids = {h.get("bvid") for h in top_hits}

        # Phase 2: Fused score ranking for the rest
        w = DIVERSIFIED_FUSED_WEIGHTS
        remaining_hits = [h for h in hits if h.get("bvid") not in top_bvids]
        for hit in remaining_hits:
            hit["rank_score"] = round(
                hit.get("relevance_score", 0) * w["relevance"]
                + hit.get("quality_score", 0) * w["quality"]
                + hit.get("recency_score", 0) * w["recency"]
                + hit.get("popularity_score", 0) * w["popularity"],
                6,
            )
            hit["_slot_dimension"] = "fused"

        remaining_hits.sort(key=lambda h: h.get("rank_score", 0), reverse=True)

        # Combine
        final_hits = top_hits + remaining_hits[: max(0, top_k - len(top_hits))]

        hits_info["hits"] = final_hits
        hits_info["return_hits"] = len(final_hits)
        hits_info["rank_method"] = "diversified"
        hits_info["diversified_top_n"] = diversify_top_n
        hits_info["slot_allocation"] = slots

        return hits_info
