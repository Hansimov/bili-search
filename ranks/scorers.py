"""
Scoring Classes for Video Ranking

This module provides scorer classes that convert raw metrics into normalized scores.
Each scorer handles a specific aspect of video quality/relevance.

Scorers:
    - StatsScorer: Popularity scoring based on view, coin, favorite, etc.
    - PubdateScorer: Recency scoring based on publish date
    - RelateScorer: Relevance gating and scoring

Functions:
    - transform_relevance_score: Power transform for vector similarity scores
    - log_x: Logarithmic transform with offset
"""

import math
from typing import Literal

from ranks.constants import (
    # Stats scoring
    STAT_FIELDS,
    STAT_LOGX_OFFSETS,
    # Pubdate scoring
    PUBDATE_BASE,
    SECONDS_PER_DAY,
    ZERO_DAY_SCORE,
    INFT_DAY_SCORE,
    PUBDATE_SCORE_POINTS,
    # Relevance scoring
    RELATE_GATE_RATIO,
    RELATE_GATE_COUNT,
    RELATE_SCORE_POWER,
    SCORE_TRANSFORM_POWER,
    SCORE_TRANSFORM_MIN,
    SCORE_TRANSFORM_MAX,
    HIGH_RELEVANCE_THRESHOLD,
    HIGH_RELEVANCE_BOOST,
)


def log_x(x: int, base: float = 10.0, offset: int = 10) -> float:
    """Logarithmic transform with offset for robust handling of small values.

    Formula: log_base(x + offset)

    Args:
        x: Input value (will be clamped to non-negative).
        base: Logarithm base (default 10).
        offset: Added to x before log (default 10, prevents log(0)).

    Returns:
        Transformed value.
    """
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

    This transformation serves several purposes:
    1. Filter out noise: Scores below min_score become 0
    2. Amplify differences: Power transform makes gaps between similar scores larger
    3. Boost top results: Very high scores get extra multiplier

    Args:
        score: Raw relevance score (typically 0-1 for vector search).
        min_score: Scores below this are treated as 0.
        max_score: Maximum possible score for normalization.
        power: Exponent for power transform (higher = more emphasis on top scores).
        high_threshold: Scores above this get extra boost.
        high_boost: Multiplier for high relevance scores.

    Returns:
        Transformed score with amplified differences.

    Example:
        >>> transform_relevance_score(0.9)  # High score
        0.729 * 1.5 = ~1.1
        >>> transform_relevance_score(0.5)  # Medium score
        0.0185  # Much lower after power transform
        >>> transform_relevance_score(0.3)  # Below threshold
        0.0  # Filtered out
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
    """Popularity scorer based on video statistics.

    Calculates a composite score from view count, favorites, coins, etc.
    Uses log transform to handle wide range of values gracefully.

    The scoring formula is:
        product(log(stat + offset) for each stat field)

    This gives higher scores to videos that are popular across multiple metrics.

    Example:
        >>> scorer = StatsScorer()
        >>> scorer.calc({"view": 100000, "coin": 1000, "favorite": 500})
        ~150.0
    """

    def __init__(
        self,
        stat_fields: list = STAT_FIELDS,
        stat_logx_offsets: dict = STAT_LOGX_OFFSETS,
    ):
        """Initialize stats scorer.

        Args:
            stat_fields: List of stat field names to include in scoring.
            stat_logx_offsets: Dict mapping field name to log offset.
        """
        self.stat_fields = stat_fields
        self.stat_logx_offsets = stat_logx_offsets

    def calc_stats_score_by_prod_logx(self, stats: dict) -> float:
        """Calculate score as product of log-transformed stats.

        Args:
            stats: Dict of stat values (view, coin, favorite, etc.).

        Returns:
            Product of log(x+offset) for all configured fields.
        """
        return math.prod(
            log_x(
                x=stats.get(field, 0),
                base=10,
                offset=self.stat_logx_offsets.get(field, 2),
            )
            for field in self.stat_fields
        )

    def calc(self, stats: dict) -> float:
        """Calculate stats score.

        Args:
            stats: Dict of stat values.

        Returns:
            Stats score (higher = more popular).
        """
        return self.calc_stats_score_by_prod_logx(stats)


class PubdateScorer:
    """Recency scorer based on video publish date.

    Uses piecewise linear interpolation between defined points to
    calculate a score that decays with video age.

    Score decay curve (default):
        - Today: 4.0
        - 1 week old: 1.0
        - 1 month old: 0.6
        - 1 year old: 0.3
        - Older: 0.25

    Example:
        >>> scorer = PubdateScorer()
        >>> scorer.calc(1704067200)  # 2024-01-01
        ~0.35 (depending on current date)
    """

    def __init__(
        self,
        day_score_points: list[tuple[float, float]] = PUBDATE_SCORE_POINTS,
        zero_day_score: float = ZERO_DAY_SCORE,
        inft_day_score: float = INFT_DAY_SCORE,
    ):
        """Initialize pubdate scorer.

        Args:
            day_score_points: List of (days_old, score) points for interpolation.
            zero_day_score: Score for videos from the future (edge case).
            inft_day_score: Score for very old videos.
        """
        self.day_score_points = sorted(day_score_points)
        self.zero_day_score = zero_day_score
        self.inft_day_score = inft_day_score
        self.slope_offsets = self.pre_calc_slope_offsets(day_score_points)

    def pre_calc_slope_offsets(self, points: list[tuple[float, float]]):
        """Pre-calculate slopes and offsets for piecewise linear interpolation.

        Args:
            points: List of (x, y) points.

        Returns:
            List of (slope, offset) tuples for each segment.
        """
        slope_offsets = []
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            slope = (y2 - y1) / (x2 - x1)
            offset = y1 - slope * x1
            slope_offsets.append((slope, offset))
        return slope_offsets

    def calc_pass_days(self, pubdate: int) -> float:
        """Calculate days since pubdate relative to base date.

        Args:
            pubdate: Unix timestamp of publish date.

        Returns:
            Days passed since PUBDATE_BASE (can be negative for old videos).
        """
        return (pubdate - PUBDATE_BASE) / SECONDS_PER_DAY

    def calc_pubdate_score_by_slope_offsets(self, pubdate: int) -> float:
        """Calculate score using piecewise linear interpolation.

        Args:
            pubdate: Unix timestamp of publish date.

        Returns:
            Recency score.
        """
        pass_days = self.calc_pass_days(pubdate)
        points = self.day_score_points

        # Handle edge cases
        if pass_days <= points[0][0]:
            return self.zero_day_score
        if pass_days >= points[-1][0]:
            return self.inft_day_score

        # Find the segment and interpolate
        for i in range(1, len(points)):
            if pass_days <= points[i][0]:
                slope, offset = self.slope_offsets[i - 1]
                score = slope * pass_days + offset
                return score

        return self.inft_day_score

    def calc(self, pubdate: int) -> float:
        """Calculate pubdate score.

        Args:
            pubdate: Unix timestamp of publish date.

        Returns:
            Recency score (higher = more recent).
        """
        return self.calc_pubdate_score_by_slope_offsets(pubdate)


class RelateScorer:
    """Relevance gating scorer.

    Filters and transforms relevance scores based on a dynamic gate threshold.
    The gate is calculated from the score distribution to keep only
    sufficiently relevant results.

    Two methods for calculating the gate:
    1. Ratio-based: gate = max_score * ratio
    2. Count-based: keep top N scores

    The final gate is the minimum of both (more permissive).

    Example:
        >>> scorer = RelateScorer()
        >>> scores = [0.9, 0.85, 0.7, 0.5, 0.3]
        >>> scorer.update_relate_gate(scores)
        >>> scorer.calc(0.9)  # Above gate
        1.5
        >>> scorer.calc(0.3)  # Below gate
        0.0
    """

    def __init__(self, relate_gate: float = None):
        """Initialize relate scorer.

        Args:
            relate_gate: Initial gate value. If None, will be calculated.
        """
        self.relate_gate = relate_gate

    def set_relate_gate(self, relate_gate: float):
        """Set the relevance gate threshold.

        Args:
            relate_gate: Gate value (scores below this are filtered).
        """
        self.relate_gate = relate_gate

    def calc_relate_gate_by_ratio(
        self, relate_list: list[float], ratio: float = RELATE_GATE_RATIO
    ) -> float:
        """Calculate gate as ratio of max score.

        Args:
            relate_list: List of relevance scores.
            ratio: Gate ratio (0-1).

        Returns:
            Gate threshold, or None if list is empty.
        """
        if not relate_list:
            return None
        max_relate = max(relate_list)
        relate_gate = max_relate * ratio
        return relate_gate

    def calc_relate_gate_by_count(
        self, relate_list: list[float], count: int = RELATE_GATE_COUNT
    ) -> float:
        """Calculate gate to keep top N scores.

        Args:
            relate_list: List of relevance scores.
            count: Number of scores to keep.

        Returns:
            Gate threshold (score at Nth position), or None if insufficient scores.
        """
        if not relate_list:
            return None
        if len(relate_list) <= count:
            return None
        sorted_list = sorted(relate_list, reverse=True)
        relate_gate = sorted_list[count - 1]
        return relate_gate

    def update_relate_gate(self, relate_list: list[float]):
        """Update gate based on score distribution.

        Uses minimum of ratio-based and count-based gates (more permissive).

        Args:
            relate_list: List of relevance scores.
        """
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
        """Check if score passes the gate.

        Args:
            relate: Relevance score to check.

        Returns:
            True if score >= gate (or no gate set).
        """
        return self.relate_gate is None or relate >= self.relate_gate

    def is_set_gate(self) -> bool:
        """Check if gate is set and positive.

        Returns:
            True if gate is set and > 0.
        """
        return self.relate_gate is not None and self.relate_gate > 0

    def calc(self, relate: float) -> float:
        """Calculate gated relevance score.

        Args:
            relate: Raw relevance score.

        Returns:
            0 if below gate, otherwise transformed score.
        """
        if not self.is_pass_gate(relate):
            return 0.0
        if self.is_set_gate():
            return (relate / self.relate_gate) ** RELATE_SCORE_POWER
        return relate
