"""
Base Types for Recall System

Provides data classes for recall results, merged recall pools,
and noise filtering to remove low-confidence candidates.
"""

from dataclasses import dataclass, field
from typing import Optional

from ranks.constants import (
    NOISE_SCORE_RATIO_GATE,
    NOISE_KNN_SCORE_RATIO,
    NOISE_MIN_HITS_FOR_FILTER,
    NOISE_MULTI_LANE_GATE_FACTOR,
    NOISE_SHORT_TEXT_MIN_LENGTH,
    NOISE_SHORT_TEXT_PENALTY,
    NOISE_MIN_ENGAGEMENT_VIEWS,
    NOISE_LOW_ENGAGEMENT_PENALTY,
)


@dataclass
class RecallResult:
    """Result from a single recall lane.

    Attributes:
        hits: List of candidate documents (dicts with at least 'bvid').
        lane: Name of the recall lane (e.g., 'relevance', 'popularity').
        total_hits: Estimated total matching documents in ES.
        took_ms: Time taken for this recall in milliseconds.
        timed_out: Whether the ES query timed out.
    """

    hits: list[dict]
    lane: str
    total_hits: int = 0
    took_ms: float = 0.0
    timed_out: bool = False


@dataclass
class RecallPool:
    """Merged, deduplicated pool from multiple recall lanes.

    Attributes:
        hits: Deduplicated list of candidate documents.
        lanes_info: Dict mapping lane name to RecallResult metadata.
        total_hits: Max total_hits from any lane (best estimate).
        took_ms: Total wall-clock time for all recalls.
        timed_out: Whether any lane timed out.
        lane_tags: Dict mapping bvid -> set of lane names it came from.
    """

    hits: list[dict] = field(default_factory=list)
    lanes_info: dict = field(default_factory=dict)
    total_hits: int = 0
    took_ms: float = 0.0
    timed_out: bool = False
    lane_tags: dict = field(default_factory=dict)

    @staticmethod
    def merge(*results: RecallResult) -> "RecallPool":
        """Merge multiple RecallResults into a deduplicated RecallPool.

        Documents appearing in multiple lanes get tagged with all their
        source lanes. This information is used by the diversified ranker
        to understand which dimensions each candidate covers.

        Args:
            *results: RecallResult instances to merge.

        Returns:
            Merged RecallPool with deduplicated hits.
        """
        seen_bvids: dict[str, int] = {}  # bvid -> index in hits
        merged_hits: list[dict] = []
        lane_tags: dict[str, set] = {}  # bvid -> set of lane names
        lanes_info: dict[str, dict] = {}
        total_hits = 0
        max_took = 0.0
        any_timeout = False

        for result in results:
            lane_name = result.lane
            lanes_info[lane_name] = {
                "hit_count": len(result.hits),
                "total_hits": result.total_hits,
                "took_ms": result.took_ms,
                "timed_out": result.timed_out,
            }
            total_hits = max(total_hits, result.total_hits)
            max_took = max(max_took, result.took_ms)
            any_timeout = any_timeout or result.timed_out

            for rank, hit in enumerate(result.hits):
                bvid = hit.get("bvid")
                if not bvid:
                    continue

                # Tag lane rank for this hit
                rank_key = f"{lane_name}_rank"
                hit[rank_key] = rank

                if bvid not in seen_bvids:
                    # New document
                    seen_bvids[bvid] = len(merged_hits)
                    merged_hits.append(hit)
                    lane_tags[bvid] = {lane_name}
                else:
                    # Already seen - merge lane tag and rank info
                    idx = seen_bvids[bvid]
                    merged_hits[idx][rank_key] = rank
                    lane_tags[bvid].add(lane_name)

        # Store lane tags in hits for downstream use
        for hit in merged_hits:
            bvid = hit.get("bvid")
            if bvid and bvid in lane_tags:
                hit["_recall_lanes"] = lane_tags[bvid]

        return RecallPool(
            hits=merged_hits,
            lanes_info=lanes_info,
            total_hits=total_hits,
            took_ms=max_took,
            timed_out=any_timeout,
            lane_tags=lane_tags,
        )

    def filter_noise(
        self,
        score_ratio: float = NOISE_SCORE_RATIO_GATE,
        min_hits: int = NOISE_MIN_HITS_FOR_FILTER,
        multi_lane_factor: float = NOISE_MULTI_LANE_GATE_FACTOR,
    ) -> "RecallPool":
        """Remove low-confidence candidates from the pool.

        Three-signal noise removal:
        1. Score-ratio gating: docs with score < ratio * max_score are noise.
        2. Short-text penalty: BM25 inflates scores for short docs; apply
           penalty factor to their effective score before gating.
        3. Low-engagement penalty: docs with near-zero views/engagement are
           likely spam/junk; penalize their effective score.

        Multi-lane docs (appearing in 2+ recall lanes) get a reduced threshold
        since cross-lane appearance is strong evidence of relevance.

        Args:
            score_ratio: Minimum score as fraction of max_score.
            min_hits: Don't filter if pool has fewer hits than this.
            multi_lane_factor: Multiply threshold by this for multi-lane docs.

        Returns:
            New RecallPool with noise removed.
        """
        if len(self.hits) <= min_hits:
            return self

        # Get scores for gating
        scores = [h.get("score", 0) or 0 for h in self.hits]
        max_score = max(scores) if scores else 0
        if max_score <= 0:
            return self

        gate = max_score * score_ratio

        filtered_hits = []
        filtered_tags = {}
        removed_count = 0
        short_text_penalized = 0
        low_engagement_penalized = 0

        for hit in self.hits:
            score = hit.get("score", 0) or 0
            bvid = hit.get("bvid", "")
            lanes = self.lane_tags.get(bvid, set())
            lane_count = len(lanes)

            # Multi-lane docs get a lower threshold
            effective_gate = gate * multi_lane_factor if lane_count >= 2 else gate

            # Apply short-text penalty to effective score
            effective_score = score
            title = hit.get("title", "") or ""
            desc = hit.get("desc", "") or ""
            content_length = len(title) + len(desc)
            if content_length < NOISE_SHORT_TEXT_MIN_LENGTH and content_length > 0:
                effective_score *= NOISE_SHORT_TEXT_PENALTY
                short_text_penalized += 1

            # Apply low-engagement penalty (only when stat data is available)
            stat = hit.get("stat")
            if isinstance(stat, dict):
                views = stat.get("view", 0) or 0
                if views < NOISE_MIN_ENGAGEMENT_VIEWS:
                    effective_score *= NOISE_LOW_ENGAGEMENT_PENALTY
                    low_engagement_penalized += 1

            if effective_score >= effective_gate:
                filtered_hits.append(hit)
                if bvid:
                    filtered_tags[bvid] = lanes
            else:
                removed_count += 1

        # Update lanes_info with filter stats
        lanes_info = dict(self.lanes_info)
        lanes_info["_noise_filter"] = {
            "removed": removed_count,
            "kept": len(filtered_hits),
            "gate": round(gate, 4),
            "max_score": round(max_score, 4),
            "short_text_penalized": short_text_penalized,
            "low_engagement_penalized": low_engagement_penalized,
        }

        return RecallPool(
            hits=filtered_hits,
            lanes_info=lanes_info,
            total_hits=self.total_hits,
            took_ms=self.took_ms,
            timed_out=self.timed_out,
            lane_tags=filtered_tags,
        )


class NoiseFilter:
    """Static methods for lane-level score filtering.

    Used within individual recall strategies (word, vector) to filter
    noisy hits before they enter the merge pool.
    """

    @staticmethod
    def filter_by_score_ratio(
        hits: list[dict],
        score_field: str = "score",
        ratio: float = NOISE_SCORE_RATIO_GATE,
        min_hits: int = NOISE_MIN_HITS_FOR_FILTER,
    ) -> list[dict]:
        """Remove hits scoring below ratio * max_score.

        Args:
            hits: List of hit dicts.
            score_field: Field containing the score.
            ratio: Minimum score as fraction of max.
            min_hits: Don't filter if fewer hits than this.

        Returns:
            Filtered list of hits.
        """
        if len(hits) <= min_hits:
            return hits

        scores = [h.get(score_field, 0) or 0 for h in hits]
        max_score = max(scores) if scores else 0
        if max_score <= 0:
            return hits

        gate = max_score * ratio
        return [h for h in hits if (h.get(score_field, 0) or 0) >= gate]

    @staticmethod
    def filter_knn_by_score_ratio(
        hits: list[dict],
        score_field: str = "score",
        ratio: float = NOISE_KNN_SCORE_RATIO,
        min_hits: int = NOISE_MIN_HITS_FOR_FILTER,
    ) -> list[dict]:
        """Remove KNN hits scoring below ratio * max_score.

        KNN bit vector (LSH hamming) scores have narrow ranges where
        many irrelevant docs cluster near the relevant ones. A stricter
        ratio is needed compared to BM25 scores.

        Args:
            hits: List of KNN hit dicts.
            score_field: Field containing the score.
            ratio: Minimum score as fraction of max (stricter than BM25).
            min_hits: Don't filter if fewer hits than this.

        Returns:
            Filtered list of hits.
        """
        return NoiseFilter.filter_by_score_ratio(
            hits, score_field=score_field, ratio=ratio, min_hits=min_hits
        )

    @staticmethod
    def apply_content_quality_penalty(
        hits: list[dict],
        score_field: str = "score",
        short_text_min_length: int = NOISE_SHORT_TEXT_MIN_LENGTH,
        short_text_penalty: float = NOISE_SHORT_TEXT_PENALTY,
        min_engagement_views: int = NOISE_MIN_ENGAGEMENT_VIEWS,
        low_engagement_penalty: float = NOISE_LOW_ENGAGEMENT_PENALTY,
    ) -> list[dict]:
        """Apply content quality penalties to hit scores in-place.

        Addresses BM25's bias toward short texts: BM25 field-length
        normalization gives disproportionately high scores to docs with
        very short title/desc. This method penalizes such docs by
        reducing their effective score.

        Also penalizes docs with near-zero engagement (views < threshold),
        which are typically spam, test uploads, or very low-quality content.

        Args:
            hits: List of hit dicts (modified in-place).
            score_field: Field containing the score to adjust.
            short_text_min_length: Min chars (title+desc) for substantial content.
            short_text_penalty: Multiply score by this for short content.
            min_engagement_views: Minimum views to pass quality gate.
            low_engagement_penalty: Score multiplier for very low engagement.

        Returns:
            Same list with adjusted scores (modified in-place).
        """
        for hit in hits:
            penalty = 1.0

            # Short text penalty
            title = hit.get("title", "") or ""
            desc = hit.get("desc", "") or ""
            content_length = len(title) + len(desc)
            if 0 < content_length < short_text_min_length:
                penalty *= short_text_penalty

            # Low engagement penalty (only when stat data is available)
            stat = hit.get("stat")
            if isinstance(stat, dict):
                views = stat.get("view", 0) or 0
                if views < min_engagement_views:
                    penalty *= low_engagement_penalty

            if penalty < 1.0:
                score = hit.get(score_field, 0) or 0
                hit[score_field] = score * penalty
                hit["_quality_penalty"] = round(penalty, 4)

        return hits
