"""
Score Fusion Strategies

This module provides methods for combining multiple scores into a final ranking score.

Main class:
    - ScoreFuser: Combines stats, pubdate, and relevance scores

The fusion formula emphasizes relevance to ensure that highly relevant results
rank first, even if they have lower popularity or recency scores.
"""

from typing import Literal


class ScoreFuser:
    """Fuses multiple scores into a single ranking score.

    The default fusion strategy uses a product formula with strong relevance emphasis:
        final_score = stats_score * pubdate_score * (relate_score ^ 3)

    The cube on relate_score makes relevance strongly dominate the ranking,
    ensuring highly relevant results appear first even if less popular.

    Example:
        >>> fuser = ScoreFuser()
        >>> # High relevance wins even with lower stats
        >>> fuser.fuse(stats=10, pubdate=1.0, relate=0.9)  # relate^3 = 0.729
        7.29
        >>> fuser.fuse(stats=100, pubdate=1.0, relate=0.5)  # relate^3 = 0.125
        12.5  # Higher stats but lower relevance = similar final score
    """

    def __init__(self, relate_power: float = 3.0):
        """Initialize score fuser.

        Args:
            relate_power: Exponent applied to relate_score (default 3.0).
                         Higher values give more weight to relevance.
        """
        self.relate_power = relate_power

    def calc_fuse_score_by_prod(
        self, stats_score: float, pubdate_score: float, relate_score: float
    ) -> float:
        """Fuse scores using product formula with strong relevance emphasis.

        Formula: stats_score * pubdate_score * (relate_score ^ relate_power)

        Args:
            stats_score: Popularity score from StatsScorer.
            pubdate_score: Recency score from PubdateScorer.
            relate_score: Relevance score from RelateScorer.

        Returns:
            Fused score, rounded to 6 decimal places.
        """
        # Power transform on relate_score to strongly amplify its importance
        relate_emphasis = relate_score**self.relate_power
        return round(stats_score * pubdate_score * relate_emphasis, 6)

    def calc_fuse_score_by_weighted_sum(
        self,
        stats_score: float,
        pubdate_score: float,
        relate_score: float,
        stats_weight: float = 0.3,
        pubdate_weight: float = 0.2,
        relate_weight: float = 0.5,
    ) -> float:
        """Fuse scores using weighted sum.

        Formula: stats_weight * stats + pubdate_weight * pubdate + relate_weight * relate

        This is an alternative to the product formula, useful when you want
        more predictable score combinations.

        Args:
            stats_score: Popularity score (should be normalized 0-1).
            pubdate_score: Recency score (typically 0-4).
            relate_score: Relevance score (typically 0-1).
            stats_weight: Weight for stats (default 0.3).
            pubdate_weight: Weight for pubdate (default 0.2).
            relate_weight: Weight for relate (default 0.5).

        Returns:
            Weighted sum of scores.
        """
        return round(
            stats_weight * stats_score
            + pubdate_weight * pubdate_score
            + relate_weight * relate_score,
            6,
        )

    def fuse(
        self,
        stats_score: float,
        pubdate_score: float,
        relate_score: float,
        method: Literal["product", "weighted_sum"] = "product",
        **kwargs,
    ) -> float:
        """Fuse multiple scores into a single ranking score.

        Args:
            stats_score: Popularity score from StatsScorer.
            pubdate_score: Recency score from PubdateScorer.
            relate_score: Relevance score from RelateScorer.
            method: Fusion method ("product" or "weighted_sum").
            **kwargs: Additional arguments for specific fusion methods.

        Returns:
            Fused score for ranking.
        """
        if method == "weighted_sum":
            return self.calc_fuse_score_by_weighted_sum(
                stats_score=stats_score,
                pubdate_score=pubdate_score,
                relate_score=relate_score,
                **kwargs,
            )
        else:
            return self.calc_fuse_score_by_prod(
                stats_score=stats_score,
                pubdate_score=pubdate_score,
                relate_score=relate_score,
            )
