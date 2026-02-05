"""
DEPRECATED: This module has been moved to ranks/

All ranking functionality has been consolidated into the `ranks` package.
This file is kept for backward compatibility only.

Please update your imports:
    OLD: from elastics.videos.ranker import VideoHitsRanker
    NEW: from ranks.ranker import VideoHitsRanker

    OLD: from elastics.videos.ranker import StatsScorer, PubdateScorer
    NEW: from ranks.scorers import StatsScorer, PubdateScorer
"""

import warnings

# Emit deprecation warning when this module is imported
warnings.warn(
    "elastics.videos.ranker is deprecated. Use ranks.ranker instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from new location for backward compatibility
from ranks.constants import (
    RANK_TOP_K,
    RELEVANCE_MIN_SCORE,
    RELEVANCE_SCORE_POWER,
    RRF_K,
    RRF_HEAP_SIZE,
    RRF_HEAP_RATIO,
    RRF_WEIGHTS,
    RELATE_GATE_RATIO,
    RELATE_GATE_COUNT,
    RELATE_SCORE_POWER,
    TIERED_HIGH_RELEVANCE_THRESHOLD,
    TIERED_SIMILARITY_THRESHOLD,
    TIERED_STATS_WEIGHT,
    TIERED_RECENCY_WEIGHT,
    SCORE_TRANSFORM_POWER,
    SCORE_TRANSFORM_MIN,
    SCORE_TRANSFORM_MAX,
    HIGH_RELEVANCE_THRESHOLD,
    HIGH_RELEVANCE_BOOST,
    PUBDATE_BASE,
    SECONDS_PER_DAY,
    ZERO_DAY_SCORE,
    INFT_DAY_SCORE,
    PUBDATE_SCORE_POINTS,
    STAT_FIELDS,
    STAT_LOGX_OFFSETS,
)

from ranks.scorers import (
    log_x,
    transform_relevance_score,
    StatsScorer,
    PubdateScorer,
    RelateScorer,
)

from ranks.fusion import ScoreFuser

from ranks.ranker import VideoHitsRanker

__all__ = [
    # Constants
    "RANK_TOP_K",
    "RELEVANCE_MIN_SCORE",
    "RELEVANCE_SCORE_POWER",
    "RRF_K",
    "RRF_HEAP_SIZE",
    "RRF_HEAP_RATIO",
    "RRF_WEIGHTS",
    "RELATE_GATE_RATIO",
    "RELATE_GATE_COUNT",
    "RELATE_SCORE_POWER",
    "TIERED_HIGH_RELEVANCE_THRESHOLD",
    "TIERED_SIMILARITY_THRESHOLD",
    "TIERED_STATS_WEIGHT",
    "TIERED_RECENCY_WEIGHT",
    "SCORE_TRANSFORM_POWER",
    "SCORE_TRANSFORM_MIN",
    "SCORE_TRANSFORM_MAX",
    "HIGH_RELEVANCE_THRESHOLD",
    "HIGH_RELEVANCE_BOOST",
    "PUBDATE_BASE",
    "SECONDS_PER_DAY",
    "ZERO_DAY_SCORE",
    "INFT_DAY_SCORE",
    "PUBDATE_SCORE_POINTS",
    "STAT_FIELDS",
    "STAT_LOGX_OFFSETS",
    # Functions
    "log_x",
    "transform_relevance_score",
    # Classes
    "StatsScorer",
    "PubdateScorer",
    "RelateScorer",
    "ScoreFuser",
    "VideoHitsRanker",
]
