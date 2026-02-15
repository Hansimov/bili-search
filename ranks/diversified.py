"""
Diversified Slot-Based Ranker

Instead of fusing all signals into one continuous score (which causes
homogeneous top-N results), this ranker uses a two-phase approach:

Phase 1 — Headline selection (top 3):
    The most visible positions use a composite "headline quality" score
    that balances relevance with quality and recency. This ensures the
    top-3 results are both highly relevant AND high quality, not just
    the highest BM25 score (which may be a low-quality short-text doc).

Phase 2 — Diversified slot allocation (positions 4-10):
    Allocates "slots" to different dimensions, ensuring that the
    remaining top-K positions contain representatives from each dimension:
    relevant, popular, recent, and high-quality.

Phase 3 — Fused scoring (beyond top-10):
    Remaining positions are filled using a weighted combination of all
    dimension scores.

Scoring improvements over naive max-normalization:
- Popularity uses log-scale normalization because view counts follow
  a power-law distribution. Without log-scale, a 100M-view video
  crushes all 50K-view videos to near-zero popularity scores.
- Short-duration videos (<30s) get a quality penalty to avoid clickbait.
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
    HEADLINE_TOP_N,
    HEADLINE_WEIGHTS,
    RANK_SHORT_DURATION_THRESHOLD,
    RANK_SHORT_DURATION_PENALTY,
)

# Slot allocation presets
# NOTE: These now specify slots for positions AFTER the headline top-N.
# E.g., if HEADLINE_TOP_N=3, "balanced" allocates 7 more slots (3+7=10).
SLOT_PRESETS = {
    "balanced": {
        "relevance": 2,
        "quality": 2,
        "recency": 2,
        "popularity": 1,
    },
    "prefer_relevance": {
        "relevance": 3,
        "quality": 2,
        "recency": 1,
        "popularity": 1,
    },
    "prefer_quality": {
        "relevance": 1,
        "quality": 3,
        "recency": 1,
        "popularity": 2,
    },
    "prefer_recency": {
        "relevance": 1,
        "quality": 1,
        "recency": 4,
        "popularity": 1,
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
    """Two-phase diversified ranking for comprehensive top-K results.

    Phase 1 — Headline quality (top 3):
        Picks the best candidates using a composite score that balances
        relevance with quality and recency. This ensures the most visible
        positions are occupied by results that are both relevant AND
        high quality, not just the highest BM25 score.

    Phase 2 — Slot allocation (positions 4-10):
        Ensures remaining top positions contain representatives from
        each dimension: relevant, popular, recent, high-quality.

    Phase 3 — Fused scoring (beyond top 10):
        Uses weighted combination of all dimension scores.

    The key insight: pure relevance ranking often puts short-text,
    low-quality docs at the top because BM25 inflates their scores.
    Headline quality scoring fixes this by requiring quality AND
    relevance together for the most prominent positions.

    Example:
        >>> ranker = DiversifiedRanker()
        >>> result = ranker.diversified_rank(
        ...     hits_info={"hits": candidates},
        ...     top_k=10,
        ...     prefer="balanced",
        ... )
        >>> # Top 3: best headline quality (relevant + quality + recent)
        >>> # Positions 4-10: diversified slots
        >>> # Beyond 10: fused scoring
    """

    def __init__(self):
        self.stats_scorer = StatsScorer()
        self.pubdate_scorer = PubdateScorer()

    def _score_all_dimensions(self, hits: list[dict], now_ts: float = None) -> None:
        """Score each hit on all four dimensions + headline quality.

        Computes and stores dimension scores in each hit dict:
        - relevance_score: Normalized BM25/hybrid/rerank score [0, 1]
        - quality_score: Stat quality from DocScorer [0, 1), with content penalty
        - recency_score: Normalized time factor [0, 1]
        - popularity_score: Log-normalized view count [0, 1]
        - headline_score: Composite score for top-3 selection [0, 1]

        Quality score adjustments:
        - Short-duration videos (<30s) get a penalty because they are
          often low-effort content that shouldn't occupy top positions.

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
            quality = self.stats_scorer.calc(stats)

            # Apply short-duration penalty
            duration = hit.get("duration", 0) or 0
            if 0 < duration < RANK_SHORT_DURATION_THRESHOLD:
                quality *= RANK_SHORT_DURATION_PENALTY

            hit["quality_score"] = round(quality, 4)

            # Recency: normalized time factor [0, 1]
            pubdate = dict_get(hit, "pubdate", 0)
            time_factor = self.pubdate_scorer.calc(pubdate, now_ts=now_ts)
            hit["recency_score"] = round(self.pubdate_scorer.normalize(time_factor), 4)
            hit["time_factor"] = round(time_factor, 4)

            # Popularity: log-scale normalization
            hit["popularity_score"] = round(log_views[i] / max_log_view, 4)

            # Headline quality: composite score for top-3 selection
            # This balances relevance with quality/recency to ensure top
            # positions are both relevant AND high quality
            w = HEADLINE_WEIGHTS
            hit["headline_score"] = round(
                w["relevance"] * hit["relevance_score"]
                + w["quality"] * hit["quality_score"]
                + w["recency"] * hit["recency_score"]
                + w["popularity"] * hit["popularity_score"],
                4,
            )

    def _select_headline_top_n(
        self,
        hits: list[dict],
        top_n: int = HEADLINE_TOP_N,
        min_relevance: float = 0.3,
    ) -> tuple[list[dict], set[str]]:
        """Select top-N headline positions using composite quality score.

        The headline positions (typically top 3) are the most visible
        results. Instead of purely using relevance (which favors short-text
        BM25 artifacts), this method picks candidates that are:
        - Highly relevant (must pass minimum relevance threshold)
        - High quality (good stats, reasonable duration)
        - Reasonably recent

        This ensures the first impression is strong: relevant, trustworthy,
        and timely content.

        Args:
            hits: Scored hits (must have scores from _score_all_dimensions).
            top_n: Number of headline positions to fill.
            min_relevance: Minimum relevance_score to qualify for headline.

        Returns:
            Tuple of (selected headline hits, set of selected bvids).
        """
        # Filter candidates: must have reasonable relevance
        candidates = [h for h in hits if h.get("relevance_score", 0) >= min_relevance]

        if not candidates:
            candidates = hits  # Fallback: use all if none pass threshold

        # Sort by headline quality score
        candidates.sort(key=lambda h: h.get("headline_score", 0), reverse=True)

        selected = []
        selected_bvids = set()
        for hit in candidates:
            if len(selected) >= top_n:
                break
            bvid = hit.get("bvid")
            if bvid and bvid not in selected_bvids:
                selected_bvids.add(bvid)
                hit["_slot_dimension"] = "headline"
                hit["_slot_order"] = len(selected)
                selected.append(hit)

        return selected, selected_bvids

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
        headline_top_n: int = HEADLINE_TOP_N,
    ) -> dict:
        """Three-phase ranking: headline → diversified → fused.

        Phase 1 — Headline quality (top 3):
            Picks the best candidates using composite headline_score that
            balances relevance + quality + recency. This ensures the most
            visible positions are occupied by both relevant AND
            high-quality results.

        Phase 2 — Diversified slot allocation (positions 4-10):
            Fills remaining diversified positions with dimension representatives
            (relevance, quality, recency, popularity).

        Phase 3 — Fused scoring (beyond top 10):
            Remaining positions use continuous fused scoring.

        Args:
            hits_info: Dict with "hits" list.
            top_k: Total results to return.
            prefer: Preference mode.
            diversify_top_n: Total items for diversification (phases 1+2).
            headline_top_n: How many items to pick by headline quality.

        Returns:
            hits_info with three-phase ranked hits.
        """
        hits = hits_info.get("hits", [])
        if not hits:
            hits_info["rank_method"] = "diversified"
            return hits_info

        now_ts = _time.time()

        # Score all dimensions (relevance, quality, recency, popularity, headline)
        self._score_all_dimensions(hits, now_ts=now_ts)

        # Phase 1: Headline selection (top 3)
        headline_hits, headline_bvids = self._select_headline_top_n(
            hits, top_n=headline_top_n
        )

        # Phase 2: Diversified slot allocation for remaining positions
        preset_name = prefer or "balanced"
        slots = SLOT_PRESETS.get(preset_name, SLOT_PRESETS["balanced"])

        remaining_for_slots = max(0, diversify_top_n - len(headline_hits))
        slot_hits = []
        slot_bvids = set(headline_bvids)  # Exclude already-selected headlines

        if remaining_for_slots > 0:
            slot_hits = self._allocate_slots(
                [h for h in hits if h.get("bvid") not in slot_bvids],
                slots,
                remaining_for_slots,
            )
            for hit in slot_hits:
                bvid = hit.get("bvid")
                if bvid:
                    slot_bvids.add(bvid)

        top_hits = headline_hits + slot_hits
        top_bvids = slot_bvids

        # Phase 3: Fused score ranking for the rest
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

        # Combine all phases
        final_hits = top_hits + remaining_hits[: max(0, top_k - len(top_hits))]

        # Assign rank_score for stable ordering in top section
        for i, hit in enumerate(final_hits):
            if i < len(top_hits):
                hit["rank_score"] = round(1.0 - (i / max(len(final_hits), 1)), 6)

        hits_info["hits"] = final_hits
        hits_info["return_hits"] = len(final_hits)
        hits_info["rank_method"] = "diversified"
        hits_info["diversified_top_n"] = diversify_top_n
        hits_info["headline_top_n"] = len(headline_hits)
        hits_info["slot_allocation"] = slots

        # Summary info
        dimension_counts = {}
        for hit in final_hits[:diversify_top_n]:
            dim = hit.get("_slot_dimension", "unknown")
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        hits_info["dimension_distribution"] = dimension_counts

        return hits_info
