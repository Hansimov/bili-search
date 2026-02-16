"""
Scoring Classes for Video Ranking

This module provides scorer classes that convert raw metrics into normalized scores.
Each scorer handles a specific aspect of video quality/relevance.

Scorers:
    - StatsScorer: Popularity scoring based on view, coin, favorite, etc.
    - PubdateScorer: Real-time recency scoring using blux.doc_score time factor
    - RelateScorer: Relevance gating and scoring

Functions:
    - transform_relevance_score: Power transform for vector similarity scores
"""

import math
import time as _time

from ranks.constants import (
    # Recency scoring (from blux time factor)
    TIME_FACTOR_MIN,
    TIME_FACTOR_MAX,
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

    Supports two scoring modes:
    1. "doc_score" (default): Uses blux.doc_score.DocScorer for saturated,
       anomaly-aware scoring. Returns bounded values ∈ [0, 1).

    The doc_score mode is preferred because:
    - Saturated per-field scoring prevents any single stat from dominating
    - Anomaly detection penalizes artificially inflated stats
    - Bounded output simplifies score fusion and comparison
    - Consistent with the stat_score stored in ES index

    Example:
        >>> scorer = StatsScorer()
        >>> scorer.calc({"view": 100000, "coin": 1000, "favorite": 500})
        ~0.65  # bounded [0, 1)
    """

    def __init__(self):
        """Initialize stats scorer with lazy-loaded DocScorer."""
        self._doc_scorer = None

    @property
    def doc_scorer(self):
        """Lazy-init DocScorer from blux (avoids import at module level)."""
        if self._doc_scorer is None:
            from blux.doc_score import DocScorer

            self._doc_scorer = DocScorer()
        return self._doc_scorer

    def calc_stat_quality(self, stats: dict) -> float:
        """Calculate bounded stat quality score using DocScorer.

        Uses blux.doc_score.DocScorer's saturated scoring with anomaly detection.
        Returns the pure stat quality without time factor.

        Score = stat_score × anomaly_factor, where:
        - stat_score ∈ [0, 1): weighted average of saturated per-field scores
        - anomaly_factor ∈ [0.3, 1.0]: penalizes inconsistent stats

        Args:
            stats: Dict of stat values (view, coin, favorite, like, danmaku, reply).

        Returns:
            Bounded quality score ∈ [0, 1).
        """
        scorer = self.doc_scorer
        stat_score = scorer._calc_stat_score(stats)
        anomaly_factor = scorer._calc_anomaly_factor(stats)
        return stat_score * anomaly_factor

    def calc(self, stats: dict, hit: dict = None) -> float:
        """Calculate stats score.

        If the hit has a pre-computed stat_score from ES (via blux.doc_score),
        extracts just the stat quality component. Otherwise computes fresh
        using DocScorer's saturated scoring.

        Args:
            stats: Dict of stat values.
            hit: Optional hit dict that may contain pre-computed "stat_score".

        Returns:
            Stats quality score ∈ [0, 1).
        """
        return self.calc_stat_quality(stats)


class PubdateScorer:
    """Real-time recency scorer using blux's DocScorer time factor.

    Computes freshness based on current time minus publish date,
    reusing blux.doc_score.DocScorer._calc_time_factor() for consistent
    time decay behavior.

    Previous implementation used an absolute-days-since-2010 calculation
    that produced a constant 0.25 for all modern videos. This version
    uses real-time age (now - pubdate) for meaningful time decay.

    Time decay curve (from DocScorer):
        - <= 1 hour old:  1.30 (fresh boost)
        - 1 day old:      1.10
        - 3 days old:     0.90
        - 7 days old:     0.70
        - 15 days old:    0.55
        - >= 30 days old: 0.45

    Returns time_factor in [0.45, 1.30], normalized to [0, 1] via normalize().

    Example:
        >>> scorer = PubdateScorer()
        >>> scorer.calc(time.time() - 3600)    # 1 hour ago
        1.30
        >>> scorer.calc(time.time() - 86400)   # 1 day ago
        1.10
        >>> scorer.calc(time.time() - 2592000) # 30 days ago
        0.45
    """

    def __init__(self):
        """Initialize pubdate scorer with lazy-loaded DocScorer."""
        self._doc_scorer = None

    @property
    def doc_scorer(self):
        """Lazy-init DocScorer from blux (avoids import at module level)."""
        if self._doc_scorer is None:
            from blux.doc_score import DocScorer

            self._doc_scorer = DocScorer()
        return self._doc_scorer

    def calc(self, pubdate: int, now_ts: float = None) -> float:
        """Calculate real-time recency score.

        Uses blux.doc_score.DocScorer._calc_time_factor() with the
        real-time video age (now - pubdate) as input.

        Args:
            pubdate: Unix timestamp of publish date.
            now_ts: Current timestamp. Defaults to time.time().

        Returns:
            Time factor in [TIME_FACTOR_MIN, TIME_FACTOR_MAX] = [0.45, 1.30].
            Higher values mean more recent.
        """
        if now_ts is None:
            now_ts = _time.time()
        age_seconds = max(0, now_ts - pubdate)
        return self.doc_scorer._calc_time_factor(age_seconds)

    def normalize(self, time_factor: float) -> float:
        """Normalize time factor to [0, 1] range.

        Maps [TIME_FACTOR_MIN, TIME_FACTOR_MAX] = [0.45, 1.30] to [0, 1].

        Args:
            time_factor: Raw time factor from calc().

        Returns:
            Normalized value in [0, 1].
        """
        denom = TIME_FACTOR_MAX - TIME_FACTOR_MIN
        if denom <= 0:
            return 0.5
        return max(0.0, min(1.0, (time_factor - TIME_FACTOR_MIN) / denom))


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
