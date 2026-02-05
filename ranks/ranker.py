"""
Video Hits Ranker

This module provides the main VideoHitsRanker class that implements various
ranking methods for video search results.

Ranking Methods:
    - heads: Simple truncation without ranking
    - rrf: Reciprocal Rank Fusion across multiple metrics
    - stats: Stats-weighted ranking (popularity + recency + relevance)
    - relevance: Pure relevance/similarity ranking
    - tiered: Two-zone ranking (high relevance gets stats boost)

Usage:
    >>> from ranks import VideoHitsRanker
    >>> ranker = VideoHitsRanker()
    >>> ranked_hits = ranker.stats_rank(hits_info, top_k=100)
"""

import heapq
import math

from tclogger import dict_get

from ranks.constants import (
    RANK_TOP_K,
    RRF_K,
    RRF_HEAP_SIZE,
    RRF_HEAP_RATIO,
    RRF_WEIGHTS,
    RELEVANCE_MIN_SCORE,
    RELEVANCE_SCORE_POWER,
    TIERED_HIGH_RELEVANCE_THRESHOLD,
    TIERED_SIMILARITY_THRESHOLD,
    TIERED_STATS_WEIGHT,
    TIERED_RECENCY_WEIGHT,
)
from ranks.scorers import (
    StatsScorer,
    PubdateScorer,
    RelateScorer,
    transform_relevance_score,
)
from ranks.fusion import ScoreFuser


class VideoHitsRanker:
    """Main ranker class providing multiple ranking strategies.

    This class unifies all ranking methods in one place, making it easy to
    switch between different ranking strategies based on search mode.

    Example:
        >>> ranker = VideoHitsRanker()
        >>> # Stats-based ranking for word search
        >>> result = ranker.stats_rank(hits_info, top_k=100)
        >>> # Pure relevance for vector search
        >>> result = ranker.relevance_rank(hits_info, top_k=100)
        >>> # Tiered ranking for hybrid search
        >>> result = ranker.tiered_rank(hits_info, top_k=100)
    """

    def __init__(self):
        """Initialize ranker with component scorers."""
        self.stats_scorer = StatsScorer()
        self.pubdate_scorer = PubdateScorer()
        self.relate_scorer = RelateScorer()
        self.score_fuser = ScoreFuser()

    def heads(self, hits_info: dict, top_k: int = RANK_TOP_K) -> dict:
        """Get first k hits without ranking.

        Simple truncation - useful when results are already sorted
        or when ranking is not needed.

        Args:
            hits_info: Dict with "hits" list.
            top_k: Number of hits to keep.

        Returns:
            hits_info with truncated hits.
        """
        hits = hits_info.get("hits", [])
        if top_k and len(hits) > top_k:
            hits = hits[:top_k]
            hits_info["hits"] = hits
            hits_info["return_hits"] = len(hits)
            hits_info["rank_method"] = "heads"
        return hits_info

    def get_top_hits(
        self, hits: list[dict], top_k: int, sort_field: str = "rank_score"
    ) -> list[dict]:
        """Get top k hits by a score field using heap.

        Args:
            hits: List of hit dicts.
            top_k: Number of top hits to return.
            sort_field: Field name to sort by.

        Returns:
            List of top k hits, sorted descending by sort_field.
        """
        return heapq.nlargest(top_k, hits, key=lambda x: x.get(sort_field, 0))

    def rrf_rank(
        self,
        hits_info: dict,
        top_k: int = RANK_TOP_K,
        rrf_weights: dict = RRF_WEIGHTS,
        rrf_k: int = RRF_K,
        heap_size: int = RRF_HEAP_SIZE,
        heap_ratio: int = RRF_HEAP_RATIO,
    ) -> dict:
        """Rank by Reciprocal Rank Fusion across multiple metrics.

        RRF formula: score = sum(weight[i] / (k + rank[i])) for each metric i

        This method combines rankings from multiple signals (views, pubdate,
        relevance, etc.) into a single ranking using RRF fusion.

        Args:
            hits_info: Dict with "hits" list.
            top_k: Number of top results to return.
            rrf_weights: Dict mapping metric names to weights.
            rrf_k: RRF constant k (higher = smoother fusion).
            heap_size: Max items to consider per metric.
            heap_ratio: heap_size multiplier relative to top_k.

        Returns:
            hits_info with RRF-ranked hits.
        """
        hits: list[dict] = hits_info.get("hits", [])
        hits_num = len(hits)
        top_k = min(top_k, hits_num)
        heap_size = min(max(heap_size, top_k * heap_ratio), hits_num)

        # Get values for each metric
        metric_keys = list(rrf_weights.keys())
        metric_vals: dict[str, list] = {}
        for key in metric_keys:
            mvals = [dict_get(hit, key, 0) for hit in hits]
            metric_vals[key] = mvals

        # Calculate ranks for each metric using heap
        metric_rank_dict: dict[str, dict[int, int]] = {}
        for mkey, mvals in metric_vals.items():
            top_idxs = heapq.nlargest(
                heap_size, range(hits_num), key=lambda i: mvals[i]
            )
            metric_rank_dict[mkey] = {
                idx: rank + 1 for rank, idx in enumerate(top_idxs)
            }

        # Calculate RRF scores by ranks of all metrics
        rrf_scores: list[float] = [0.0] * hits_num
        last_rank = heap_size + 1
        for mkey, mvals in metric_vals.items():
            w = float(rrf_weights.get(mkey, 1.0) or 0.0)
            mrank_dict = metric_rank_dict.get(mkey, {})
            for i in range(hits_num):
                rank = mrank_dict.get(i, last_rank)
                rrf_scores[i] += w / (rrf_k + rank)

        # Get top_k hits by RRF scores
        for i, hit in enumerate(hits):
            hit["rank_score"] = round(rrf_scores[i], 6)

        top_hits = self.get_top_hits(hits, top_k=top_k, sort_field="rank_score")
        hits_info["hits"] = top_hits
        hits_info["return_hits"] = len(top_hits)
        hits_info["rank_method"] = "rrf"

        return hits_info

    def get_hits_metrics(
        self, hits: list[dict]
    ) -> tuple[list[dict], list[int], list[float]]:
        """Extract metrics from hits for scoring.

        Args:
            hits: List of hit dicts.

        Returns:
            Tuple of (stats_list, pubdate_list, relate_list).
        """
        stats_list = [dict_get(hit, "stat", {}) for hit in hits]
        pubdate_list = [dict_get(hit, "pubdate", 0) for hit in hits]
        relate_list = [dict_get(hit, "score", 0.0) for hit in hits]
        return stats_list, pubdate_list, relate_list

    def stats_rank(
        self,
        hits_info: dict,
        top_k: int = RANK_TOP_K,
        apply_score_transform: bool = True,
    ) -> dict:
        """Rank by combined stats, pubdate, and relevance scores.

        This is the default ranking method for keyword search.
        It balances popularity, recency, and relevance.

        Args:
            hits_info: Dict with "hits" list.
            top_k: Number of top results to return.
            apply_score_transform: Whether to apply score transformation
                                   for relevance amplification.

        Returns:
            hits_info with stats-ranked hits.
        """
        hits: list[dict] = hits_info.get("hits", [])
        hits_num = len(hits)
        top_k = min(top_k, hits_num)
        stats_list, pubdate_list, relate_list = self.get_hits_metrics(hits)
        self.relate_scorer.update_relate_gate(relate_list)

        # Get max relate for normalization in transform
        max_relate = max(relate_list) if relate_list else 1.0
        if max_relate <= 0:
            max_relate = 1.0

        for hit, stats, pubdate, relate in zip(
            hits, stats_list, pubdate_list, relate_list
        ):
            stats_score = self.stats_scorer.calc(stats)
            pubdate_score = self.pubdate_scorer.calc(pubdate)
            relate_score = self.relate_scorer.calc(relate)

            # Apply score transformation to amplify high-relevance items
            if apply_score_transform:
                transformed_relate = transform_relevance_score(
                    relate, max_score=max_relate
                )
                # Blend transformed relate with gate-based relate_score
                relate_score = max(relate_score, transformed_relate)

            rank_score = self.score_fuser.fuse(
                stats_score=stats_score,
                pubdate_score=pubdate_score,
                relate_score=relate_score,
            )
            hit["rank_score"] = rank_score

        top_hits = self.get_top_hits(hits, top_k=top_k, sort_field="rank_score")
        hits_info["hits"] = top_hits
        hits_info["return_hits"] = len(top_hits)
        hits_info["rank_method"] = "stats"

        return hits_info

    def relevance_rank(
        self,
        hits_info: dict,
        top_k: int = RANK_TOP_K,
        min_score: float = RELEVANCE_MIN_SCORE,
        score_power: float = RELEVANCE_SCORE_POWER,
        score_field: str = "score",
    ) -> dict:
        """Rank purely by relevance score (vector similarity).

        This is the preferred ranking method for vector search results.
        It does NOT use stats or pubdate weighting - only similarity matters.

        Args:
            hits_info: Dict with "hits" list.
            top_k: Number of top results to return.
            min_score: Minimum normalized score threshold.
            score_power: Power transform exponent.
            score_field: Field containing the relevance score.

        Returns:
            hits_info with relevance-ranked hits.
        """
        hits: list[dict] = hits_info.get("hits", [])
        if not hits:
            hits_info["rank_method"] = "relevance"
            return hits_info

        # Get max score for normalization
        scores = [dict_get(hit, score_field, 0.0) or 0.0 for hit in hits]
        max_score = max(scores) if scores else 1.0
        if max_score <= 0:
            max_score = 1.0

        # Calculate relative min_score threshold
        relative_min = min_score / max_score if max_score > 0 else min_score

        # Filter and transform scores
        filtered_hits = []
        for hit in hits:
            score = dict_get(hit, score_field, 0.0) or 0.0

            # Normalize score to 0-1 based on max_score
            normalized = score / max_score if max_score > 0 else 0.0

            # Skip hits below relative minimum threshold
            if normalized < relative_min:
                continue

            # Apply power transform to amplify differences
            transformed = normalized**score_power

            hit["rank_score"] = round(transformed, 6)
            hit["normalized_score"] = round(normalized, 4)
            filtered_hits.append(hit)

        # Sort by rank_score (transformed relevance)
        filtered_hits.sort(key=lambda x: x.get("rank_score", 0), reverse=True)

        # Take top_k
        top_k = min(top_k, len(filtered_hits))
        top_hits = filtered_hits[:top_k]

        hits_info["hits"] = top_hits
        hits_info["return_hits"] = len(top_hits)
        hits_info["filtered_count"] = len(hits) - len(filtered_hits)
        hits_info["rank_method"] = "relevance"

        return hits_info

    def tiered_rank(
        self,
        hits_info: dict,
        top_k: int = RANK_TOP_K,
        relevance_field: str = "hybrid_score",
        high_relevance_threshold: float = TIERED_HIGH_RELEVANCE_THRESHOLD,
        similarity_threshold: float = TIERED_SIMILARITY_THRESHOLD,
        stats_weight: float = TIERED_STATS_WEIGHT,
        recency_weight: float = TIERED_RECENCY_WEIGHT,
    ) -> dict:
        """Tiered ranking: high-relevance zone gets popularity boost, rest by relevance.

        This method divides results into two zones:
        1. HIGH RELEVANCE ZONE: score >= threshold * max_score
           - Sorted by (stats + recency) weighted score
           - Allows popular/recent content to surface among highly relevant results
        2. LOW RELEVANCE ZONE: score < threshold
           - Strictly sorted by relevance score
           - No popularity boost for less relevant content

        Args:
            hits_info: Dict with "hits" list.
            top_k: Number of top results to return.
            relevance_field: Field containing the relevance score.
            high_relevance_threshold: Threshold for high relevance zone.
            similarity_threshold: Items within this diff are "equally relevant".
            stats_weight: Weight for popularity in secondary sort.
            recency_weight: Weight for recency in secondary sort.

        Returns:
            hits_info with tiered-ranked hits.
        """
        hits: list[dict] = hits_info.get("hits", [])
        if not hits:
            hits_info["rank_method"] = "tiered"
            return hits_info

        # Step 1: Get max score and normalize
        scores = [hit.get(relevance_field, 0) or 0 for hit in hits]
        max_score = max(scores) if scores else 1.0
        if max_score <= 0:
            max_score = 1.0

        threshold_score = max_score * high_relevance_threshold

        # Step 2: Calculate scores and split into zones
        high_zone = []
        low_zone = []

        for hit in hits:
            rel_score = hit.get(relevance_field, 0) or 0
            rel_norm = rel_score / max_score if max_score > 0 else 0
            hit["relevance_norm"] = round(rel_norm, 4)
            hit["relevance_score_raw"] = rel_score

            # Calculate stats score (popularity)
            stats = hit.get("stat", {})
            stats_score = self.stats_scorer.calc(stats)
            hit["stats_score"] = stats_score

            # Calculate recency score
            pubdate = hit.get("pubdate", 0)
            recency_score = self.pubdate_scorer.calc(pubdate)
            hit["recency_score"] = recency_score

            # Normalize stats_score (typically ranges 1-1000+)
            stats_norm = math.log10(stats_score + 1) / 3.0  # ~0-1 for stats 1-1000
            recency_norm = recency_score / 4.0  # recency_score is ~0-4

            # Combined secondary score for high relevance zone
            hit["secondary_score"] = (
                stats_weight * stats_norm + recency_weight * recency_norm
            )

            # Split into zones
            if rel_score >= threshold_score:
                high_zone.append(hit)
            else:
                low_zone.append(hit)

        # Step 3: Sort high zone by relevance first, then by secondary within tiers
        high_zone.sort(key=lambda x: x.get("relevance_norm", 0), reverse=True)

        if high_zone and similarity_threshold > 0:
            sorted_high = []
            tier_start = 0
            tier_base = high_zone[0].get("relevance_norm", 1)

            for i, hit in enumerate(high_zone):
                rel_norm = hit.get("relevance_norm", 0)
                rel_diff = (tier_base - rel_norm) / tier_base if tier_base > 0 else 1

                if rel_diff > similarity_threshold:
                    # New tier: sort previous tier by secondary_score
                    tier = high_zone[tier_start:i]
                    tier.sort(key=lambda x: x.get("secondary_score", 0), reverse=True)
                    for t_hit in tier:
                        t_hit["zone"] = "high"
                    sorted_high.extend(tier)

                    # Start new tier
                    tier_start = i
                    tier_base = rel_norm

            # Last tier
            tier = high_zone[tier_start:]
            tier.sort(key=lambda x: x.get("secondary_score", 0), reverse=True)
            for t_hit in tier:
                t_hit["zone"] = "high"
            sorted_high.extend(tier)

            high_zone = sorted_high
        else:
            high_zone.sort(key=lambda x: x.get("secondary_score", 0), reverse=True)
            for hit in high_zone:
                hit["zone"] = "high"

        # Step 4: Sort low zone strictly by relevance
        low_zone.sort(key=lambda x: x.get("relevance_norm", 0), reverse=True)
        for hit in low_zone:
            hit["zone"] = "low"

        # Step 5: Concatenate zones
        result_hits = high_zone + low_zone

        # Take top_k
        top_k = min(top_k, len(result_hits))
        top_hits = result_hits[:top_k]

        # Set rank_score and display score
        for hit in top_hits:
            rel_norm = hit.get("relevance_norm", 0)
            rel_raw = hit.get("relevance_score_raw", 0)
            secondary = hit.get("secondary_score", 0)

            if hit.get("zone") == "high":
                hit["rank_score"] = round(rel_norm + 0.1 * secondary, 6)
            else:
                hit["rank_score"] = round(rel_norm, 6)

            hit["score"] = round(rel_raw, 2)

        hits_info["hits"] = top_hits
        hits_info["return_hits"] = len(top_hits)
        hits_info["high_zone_count"] = len(high_zone)
        hits_info["low_zone_count"] = len(low_zone)
        hits_info["max_relevance_score"] = round(max_score, 2)
        hits_info["rank_method"] = "tiered"

        return hits_info

    def rank(
        self,
        hits_info: dict,
        method: str = "stats",
        top_k: int = RANK_TOP_K,
        **kwargs,
    ) -> dict:
        """Unified ranking entry point.

        Args:
            hits_info: Dict with "hits" list.
            method: Ranking method ("heads", "rrf", "stats", "relevance", "tiered").
            top_k: Number of top results to return.
            **kwargs: Additional arguments for specific ranking methods.

        Returns:
            hits_info with ranked hits.
        """
        if method == "heads":
            return self.heads(hits_info, top_k=top_k)
        elif method == "rrf":
            return self.rrf_rank(hits_info, top_k=top_k, **kwargs)
        elif method == "stats":
            return self.stats_rank(hits_info, top_k=top_k, **kwargs)
        elif method == "relevance":
            return self.relevance_rank(hits_info, top_k=top_k, **kwargs)
        elif method == "tiered":
            return self.tiered_rank(hits_info, top_k=top_k, **kwargs)
        else:
            return self.stats_rank(hits_info, top_k=top_k, **kwargs)
