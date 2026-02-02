import heapq
import math

from tclogger import dict_get

from elastics.videos.constants import RANK_TOP_K
from elastics.videos.constants import RELEVANCE_MIN_SCORE, RELEVANCE_SCORE_POWER
from elastics.videos.constants import RRF_K, RRF_HEAP_SIZE, RRF_HEAP_RATIO, RRF_WEIGHTS
from elastics.videos.constants import (
    RELATE_GATE_RATIO,
    RELATE_GATE_COUNT,
    RELATE_SCORE_POWER,
)
from elastics.videos.constants import (
    TIERED_HIGH_RELEVANCE_THRESHOLD,
    TIERED_SIMILARITY_THRESHOLD,
    TIERED_STATS_WEIGHT,
    TIERED_RECENCY_WEIGHT,
)

# Score transformation parameters for vector search
# Uses power transform: transformed = ((score - min) / (max - min)) ^ power
SCORE_TRANSFORM_POWER = 3.0  # higher = more emphasis on top scores
SCORE_TRANSFORM_MIN = 0.4  # scores below this are treated as 0
SCORE_TRANSFORM_MAX = 1.0  # max possible score

# High relevance boost threshold and multiplier
# Scores above this threshold get extra boost to ensure they rank first
HIGH_RELEVANCE_THRESHOLD = 0.85  # top 15% of normalized score range
HIGH_RELEVANCE_BOOST = 2.0  # multiplier for high relevance scores

# Pubdate scoring parameters
PUBDATE_BASE = 1262275200  # 2010-01-01 00:00:00
SECONDS_PER_DAY = 86400
ZERO_DAY_SCORE = 4.0  # score for videos published most recently
INFT_DAY_SCORE = 0.25  # score for videos published before base
PUBDATE_SCORE_POINTS = [(0, 4.0), (7, 1.0), (30, 0.6), (365, 0.3)]

# Stats scoring parameters
STAT_FIELDS = ["view", "favorite", "coin", "reply", "share", "danmaku"]
STAT_LOGX_OFFSETS = {
    "view": 10,
    "favorite": 2,
    "coin": 2,
    "reply": 2,
    "share": 2,
    "danmaku": 2,
}


def log_x(x: int, base: float = 10.0, offset: int = 10) -> float:
    x = max(x, 0)
    return math.log(x + offset, base)


def transform_relevance_score(
    score: float,
    min_score: float = SCORE_TRANSFORM_MIN,
    max_score: float = SCORE_TRANSFORM_MAX,
    power: float = SCORE_TRANSFORM_POWER,
    high_threshold: float = HIGH_RELEVANCE_THRESHOLD,
    high_boost: float = HIGH_RELEVANCE_BOOST,
) -> float:
    """Transform relevance score to amplify differences between high and low scores.

    Uses a power transform to stretch the score distribution, then applies
    an additional boost to scores above the high relevance threshold.

    Args:
        score: Raw relevance score (typically 0-1 for vector search).
        min_score: Scores below this are treated as 0.
        max_score: Maximum possible score for normalization.
        power: Exponent for power transform (higher = more emphasis on top scores).
        high_threshold: Scores above this get extra boost.
        high_boost: Multiplier for high relevance scores.

    Returns:
        Transformed score with amplified differences.
    """
    if score <= min_score:
        return 0.0

    # Normalize to 0-1 range
    normalized = (score - min_score) / (max_score - min_score)
    normalized = min(max(normalized, 0.0), 1.0)

    # Apply power transform to stretch distribution
    # This makes high scores much higher relative to medium scores
    transformed = normalized**power

    # Apply extra boost for very high relevance scores
    if normalized >= high_threshold:
        # Additional boost that increases with how far above threshold
        boost_factor = 1.0 + (high_boost - 1.0) * (
            (normalized - high_threshold) / (1.0 - high_threshold)
        )
        transformed *= boost_factor

    return transformed


class StatsScorer:
    def __init__(
        self,
        stat_fields: list = STAT_FIELDS,
        stat_logx_offsets: dict = STAT_LOGX_OFFSETS,
    ):
        self.stat_fields = stat_fields
        self.stat_logx_offsets = stat_logx_offsets

    def calc_stats_score_by_prod_logx(self, stats: dict) -> float:
        """Product of log(x+offset) of stats fields"""
        return math.prod(
            log_x(
                x=stats.get(field, 0),
                base=10,
                offset=self.stat_logx_offsets.get(field, 2),
            )
            for field in self.stat_fields
        )

    def calc(self, stats: dict) -> float:
        stats_score = self.calc_stats_score_by_prod_logx(stats)
        return stats_score


class PubdateScorer:
    def __init__(
        self,
        day_score_points: list[tuple[float, float]] = PUBDATE_SCORE_POINTS,
        zero_day_score: float = ZERO_DAY_SCORE,
        inft_day_score: float = INFT_DAY_SCORE,
    ):
        """
        - day_score_points: (days, score) pairs
            - (0, 4.0): score 4.0 for videos published most recently
            - (7, 1.0): score 1.0 for videos published 7 days ago
            - (30, 0.6): score 0.6 for videos published 30 days ago
            - (365, 0.3): score 0.3 for videos published 1 year ago
        - zero_day_score: score for videos published after now (default 4.0)
        - inft_day_score: score for videos published before base (default 0.25)
        """
        self.day_score_points = sorted(day_score_points)
        self.zero_day_score = zero_day_score
        self.inft_day_score = inft_day_score
        self.slope_offsets = self.pre_calc_slope_offsets(day_score_points)

    def pre_calc_slope_offsets(self, points: list[tuple[float, float]]):
        slope_offsets = []
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            slope = (y2 - y1) / (x2 - x1)
            offset = y1 - slope * x1
            slope_offsets.append((slope, offset))
        return slope_offsets

    def calc_pass_days(self, pubdate: int) -> float:
        return (pubdate - PUBDATE_BASE) / SECONDS_PER_DAY

    def calc_pubdate_score_by_slope_offsets(self, pubdate: int) -> float:
        """Segmented linear functions by slopes and offsets"""
        pass_days = self.calc_pass_days(pubdate)
        points = self.day_score_points
        if pass_days <= points[0][0]:
            return self.zero_day_score
        if pass_days >= points[-1][0]:
            return self.inft_day_score
        for i in range(1, len(points)):
            if pass_days <= points[i][0]:
                slope, offset = self.slope_offsets[i - 1]
                score = slope * pass_days + offset
                return score
        return self.inft_day_score

    def calc(self, pubdate: int) -> float:
        pubdate_score = self.calc_pubdate_score_by_slope_offsets(pubdate)
        return pubdate_score


class RelateScorer:
    def __init__(self, relate_gate: float = None):
        self.relate_gate = relate_gate

    def set_relate_gate(self, relate_gate: float):
        self.relate_gate = relate_gate

    def calc_relate_gate_by_ratio(
        self, relate_list: list[float], ratio: float = RELATE_GATE_RATIO
    ):
        """relate under ratio * max_relate would be set to 0"""
        if not relate_list:
            return None
        max_relate = max(relate_list)
        relate_gate = max_relate * ratio
        return relate_gate

    def calc_relate_gate_by_count(
        self, relate_list: list[float], count: int = RELATE_GATE_COUNT
    ):
        """relate after N-th highest relate would be set to 0"""
        if not relate_list:
            return None
        if len(relate_list) <= count:
            return None
        sorted_list = sorted(relate_list, reverse=True)
        relate_gate = sorted_list[count - 1]
        return relate_gate

    def update_relate_gate(self, relate_list: list[float]):
        gate_by_ratio = self.calc_relate_gate_by_ratio(relate_list)
        gate_by_count = self.calc_relate_gate_by_count(relate_list)
        if gate_by_ratio is None and gate_by_count is None:
            return
        if gate_by_ratio is None:
            relate_gate = gate_by_count
        elif gate_by_count is None:
            relate_gate = gate_by_ratio
        else:
            # min() means allow more hits
            relate_gate = min(gate_by_ratio, gate_by_count)
        self.set_relate_gate(relate_gate)

    def is_pass_gate(self, relate: float) -> bool:
        return self.relate_gate is None or relate >= self.relate_gate

    def is_set_gate(self) -> bool:
        return self.relate_gate is not None and self.relate_gate > 0

    def calc(self, relate: float) -> float:
        if not self.is_pass_gate(relate):
            return 0.0
        if self.is_set_gate():
            return (relate / self.relate_gate) ** RELATE_SCORE_POWER
        return relate


class ScoreFuser:
    def calc_fuse_score_by_prod(
        self, stats_score: float, pubdate_score: float, relate_score: float
    ) -> float:
        """Fuse scores using product formula with strong relevance emphasis.

        The formula is: stats_score * pubdate_score * (relate_score ^ 3)
        The cube on relate_score makes relevance strongly dominate the ranking,
        ensuring highly relevant results appear first even if less popular.
        """
        # Cube relate_score to strongly amplify its importance
        relate_emphasis = relate_score**3
        return round(stats_score * pubdate_score * relate_emphasis, 6)

    def fuse(
        self, stats_score: float, pubdate_score: float, relate_score: float
    ) -> float:
        fuse_score = self.calc_fuse_score_by_prod(
            stats_score=stats_score,
            pubdate_score=pubdate_score,
            relate_score=relate_score,
        )
        return fuse_score


class VideoHitsRanker:
    def __init__(self):
        self.stats_scorer = StatsScorer()
        self.pubdate_scorer = PubdateScorer()
        self.relate_scorer = RelateScorer()
        self.score_fuser = ScoreFuser()

    def heads(self, hits_info: dict, top_k: int = RANK_TOP_K) -> dict:
        """Get first k hits without rank"""
        hits = hits_info.get("hits", [])
        if top_k and len(hits) > top_k:
            hits = hits[:top_k]
            hits_info["hits"] = hits
            hits_info["return_hits"] = len(hits)
            hits_info["rank_method"] = "tops"
        return hits_info

    def get_top_hits(
        self, hits: list[dict], top_k: int, sort_field: str = "rank_score"
    ) -> list[dict]:
        return heapq.nlargest(top_k, hits, key=lambda x: x[sort_field])

    def rrf_rank(
        self,
        hits_info: dict,
        top_k: int = RANK_TOP_K,
        rrf_weights: dict = RRF_WEIGHTS,
        rrf_k: int = RRF_K,
        heap_size: int = RRF_HEAP_SIZE,
        heap_ratio: int = RRF_HEAP_RATIO,
    ) -> dict:
        """Get top k hits by RRF (Reciprocal Rank Fusion) rank with metrics and weights.
        Format of hits_info: * LINK: elastics/videos/hits.py
        """
        hits: list[dict] = hits_info.get("hits", [])
        hits_num = len(hits)
        top_k = min(top_k, hits_num)
        heap_size = min(max(heap_size, top_k * heap_ratio), hits_num)

        # get values for each metric
        metric_keys = list(rrf_weights.keys())
        metric_vals: dict[str, list[int, float]] = {}
        for key in metric_keys:
            mvals = [dict_get(hit, key, 0) for hit in hits]
            metric_vals[key] = mvals

        # calc ranks for each metric with heap
        metric_rank_dict: dict[str, dict[int, int]] = {}
        for mkey, mvals in metric_vals.items():
            top_idxs = heapq.nlargest(
                heap_size, range(hits_num), key=lambda i: mvals[i]
            )
            metric_rank_dict[mkey] = {
                idx: rank + 1 for rank, idx in enumerate(top_idxs)
            }

        # calc RRF scores by ranks of all metrics
        rrf_scores: list[float] = [0.0] * hits_num
        last_rank = heap_size + 1
        for mkey, mvals in metric_vals.items():
            w = float(rrf_weights.get(mkey, 1.0) or 0.0)
            mrank_dict = metric_rank_dict.get(mkey, {})
            for i in range(hits_num):
                rank = mrank_dict.get(i, last_rank)
                rrf_scores[i] += w / (rrf_k + rank)

        # get top_k hits by RRF scores
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
        stats_list = [dict_get(hit, "stat", {}) for hit in hits]
        pubdate_list = [dict_get(hit, "pubdate", 0) for hit in hits]
        relate_list = [dict_get(hit, "score", 0.0) for hit in hits]
        return stats_list, pubdate_list, relate_list

    def stats_rank(
        self,
        hits_info: dict,
        top_k: int = RANK_TOP_K,
        apply_score_transform: bool = True,
    ):
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
                # Use transformed_relate to boost high-relevance items
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
        """Rank hits purely by relevance score (vector similarity).

        This is the preferred ranking method for vector search results.
        It does NOT use stats (view, favorite, etc.) or pubdate weighting.
        Only the vector similarity score matters - ensuring the most relevant
        results appear first regardless of popularity or recency.

        The method:
        1. Filters out hits below the minimum relevance threshold
        2. Applies power transform to amplify score differences
        3. Sorts purely by transformed relevance score

        Args:
            hits_info: Dict containing hits list and metadata.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score threshold (0-1).
                       Hits below this are considered irrelevant noise.
            score_power: Power for score transformation.
                         Higher values = more separation between high/low scores.
            score_field: Field name containing the relevance score.

        Returns:
            hits_info with filtered and sorted hits.
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

        # Calculate relative min_score threshold based on max_score
        # If max_score is 0.7 and min_score is 0.5, relative threshold = 0.5/0.7 = 0.71
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
            # This makes the gap between 0.9 and 0.8 much larger than 0.5 and 0.4
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

        This ranking method divides results into two zones:
        1. HIGH RELEVANCE ZONE: score >= high_relevance_threshold * max_score
           - Within this zone, results are sorted by (stats + recency) weighted score
           - This allows popular/recent content to surface among highly relevant results
        2. LOW RELEVANCE ZONE: score < threshold
           - Strictly sorted by relevance score
           - No popularity boost for less relevant content

        Algorithm:
        1. Normalize all scores by max_score
        2. Split into high/low relevance zones based on threshold
        3. High zone: sort by (stats_weight * stats_score + recency_weight * recency_score)
        4. Low zone: sort by relevance score
        5. Concatenate: high zone first, then low zone

        Score Display:
        - The score field is set to the original hybrid_score (preserving word score magnitude)
        - This provides consistent display across word-only and hybrid searches

        Args:
            hits_info: Dict containing hits list and metadata.
            top_k: Maximum number of results to return.
            relevance_field: Field containing relevance score (default: hybrid_score).
            high_relevance_threshold: Only items with normalized score >= this get boost.
            similarity_threshold: Within high zone, items within this diff are "equal".
            stats_weight: Weight for popularity score.
            recency_weight: Weight for recency score.

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

        # Calculate threshold score for high relevance zone
        threshold_score = max_score * high_relevance_threshold

        # Step 2: Calculate scores and split into zones
        high_zone = []  # High relevance items (get popularity boost)
        low_zone = []  # Low relevance items (sorted by relevance only)

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
            # Use log scale for better distribution
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

        # Step 3: Sort high zone by secondary_score (popularity + recency)
        # But first sort by relevance to establish sub-tiers, then by secondary within tiers
        high_zone.sort(key=lambda x: x.get("relevance_norm", 0), reverse=True)

        # Within high zone, group by similarity and sort by secondary
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

            # Don't forget the last tier
            tier = high_zone[tier_start:]
            tier.sort(key=lambda x: x.get("secondary_score", 0), reverse=True)
            for t_hit in tier:
                t_hit["zone"] = "high"
            sorted_high.extend(tier)

            high_zone = sorted_high
        else:
            # No similarity threshold: just sort by secondary
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

        # Set rank_score for sorting and score for display
        # Score preserves the original hybrid_score magnitude (similar to word search scores)
        for i, hit in enumerate(top_hits):
            rel_norm = hit.get("relevance_norm", 0)
            rel_raw = hit.get("relevance_score_raw", 0)
            secondary = hit.get("secondary_score", 0)

            # rank_score: combines relevance with secondary for internal sorting reference
            # High zone items get boosted base score
            if hit.get("zone") == "high":
                hit["rank_score"] = round(rel_norm + 0.1 * secondary, 6)
            else:
                hit["rank_score"] = round(rel_norm, 6)

            # Set display score to preserve original hybrid_score magnitude
            # hybrid_score is already in word-search score range (e.g., 0-30)
            # This provides consistent scoring across word-only and hybrid searches
            hit["score"] = round(rel_raw, 2)

        hits_info["hits"] = top_hits
        hits_info["return_hits"] = len(top_hits)
        hits_info["high_zone_count"] = len(high_zone)
        hits_info["low_zone_count"] = len(low_zone)
        hits_info["max_relevance_score"] = round(max_score, 2)
        hits_info["rank_method"] = "tiered"

        return hits_info
