"""
Ranking Constants and Configuration

This module contains all constants related to ranking, scoring, and reranking.
These were previously scattered across `elastics.videos.constants` and other files.

Organization:
    1. Ranking Method Types - method selection
    2. Ranking Limits - result count limits
    3. RRF Configuration - Reciprocal Rank Fusion parameters
    4. Stats Scoring - popularity-based scoring
    5. Pubdate Scoring - recency-based scoring
    6. Relevance Scoring - similarity-based scoring
    7. Tiered Ranking - hybrid search parameters
    8. Reranking - embedding-based rerank parameters
    9. Hybrid Search - word+vector fusion parameters
    10. Author Grouping - author aggregation parameters
"""

from typing import Literal

# =============================================================================
# Ranking Method Types
# =============================================================================

RANK_METHOD_TYPE = Literal["heads", "rrf", "stats", "relevance", "tiered"]
RANK_METHOD = "stats"

# =============================================================================
# Ranking Preference Types
# =============================================================================

# Preference modes control the relative importance of quality, relevance,
# and recency in the final ranking score. Used by all ranking methods.
RANK_PREFER_TYPE = Literal[
    "balanced", "prefer_quality", "prefer_relevance", "prefer_recency"
]
RANK_PREFER = "balanced"  # Default preference

# Preference weight presets: quality (stats), relevance (BM25/embedding), recency (time)
# All weights sum to 1.0 for each preset
RANK_PREFER_PRESETS = {
    "balanced": {"quality": 0.30, "relevance": 0.50, "recency": 0.20},
    "prefer_quality": {"quality": 0.50, "relevance": 0.30, "recency": 0.20},
    "prefer_relevance": {"quality": 0.15, "relevance": 0.70, "recency": 0.15},
    "prefer_recency": {"quality": 0.20, "relevance": 0.30, "recency": 0.50},
}

# =============================================================================
# BM25 + Embedding Relevance Blending
# =============================================================================

# When reranking is used (q=wr, q=vr, q=wvr), blend BM25 keyword-match
# scores with embedding cosine similarity for more robust relevance.
# Semantic weight + BM25 weight should sum to ~1.0
BLEND_SEMANTIC_WEIGHT = 0.6  # Weight for embedding cosine similarity
BLEND_BM25_WEIGHT = 0.3  # Weight for BM25 keyword-match score
BLEND_KEYWORD_BONUS = 0.1  # Max bonus from keyword presence boost

# =============================================================================
# Ranking Limits
# =============================================================================

# General ranking limits
RANK_TOP_K = 50  # Default top-k for ranking

# Explore-specific limits
EXPLORE_RANK_TOP_K = 400  # Max results to return after ranking in explore
EXPLORE_GROUP_OWNER_LIMIT = 25  # Max author groups to return
EXPLORE_MOST_RELEVANT_LIMIT = 10000  # Max docs to scan for relevance

# =============================================================================
# RRF (Reciprocal Rank Fusion) Configuration
# =============================================================================

RRF_K = 60  # RRF constant k (higher = smoother rank fusion)
RRF_HEAP_SIZE = 2000  # Max items to consider per metric
RRF_HEAP_RATIO = 5  # heap_size = max(input, top_k * ratio)

# RRF weights for different metrics
# Higher weight = more influence on final ranking
RRF_WEIGHTS = {
    "pubdate": 2.0,  # Publish date timestamp
    "stat.view": 1.0,  # View count
    "stat.favorite": 1.0,  # Favorite count
    "stat.coin": 1.0,  # Coin count
    "score": 5.0,  # Relevance score (highest weight)
}

# Hybrid search RRF parameter
HYBRID_RRF_K = 60  # k parameter for hybrid search RRF fusion

# =============================================================================
# Stats-based Scoring Configuration
# =============================================================================

# Stat fields used for stats scoring
STAT_FIELDS = ["view", "favorite", "coin", "reply", "share", "danmaku"]

# Log offsets for different stat fields
# Higher offset = less emphasis on low values
STAT_LOGX_OFFSETS = {
    "view": 10,
    "favorite": 2,
    "coin": 2,
    "reply": 2,
    "share": 2,
    "danmaku": 2,
}

# =============================================================================
# Pubdate (Recency) Scoring Configuration
# =============================================================================

PUBDATE_BASE = 1262275200  # 2010-01-01 00:00:00 (reference timestamp)
SECONDS_PER_DAY = 86400

# Score for videos at different ages
ZERO_DAY_SCORE = 4.0  # Score for videos published most recently
INFT_DAY_SCORE = 0.25  # Score for videos published before base

# Piecewise linear function points: (days_old, score)
# Interpolates between these points for smooth decay
PUBDATE_SCORE_POINTS = [
    (0, 4.0),  # Today: 4.0
    (7, 1.0),  # 1 week: 1.0
    (30, 0.6),  # 1 month: 0.6
    (365, 0.3),  # 1 year: 0.3
]

# =============================================================================
# Relevance (Similarity) Scoring Configuration
# =============================================================================

# Minimum normalized score to be considered relevant
# Scores below this are filtered out in relevance ranking
RELEVANCE_MIN_SCORE = 0.4

# Power transform exponent for relevance scores
# Higher values = more separation between high and low scores
RELEVANCE_SCORE_POWER = 2.0

# Relevance gating parameters (for stats ranking)
# Results with score < RELATE_GATE_RATIO * max_score are penalized
RELATE_GATE_RATIO = 0.5  # Higher = more selective
RELATE_GATE_COUNT = 2000  # Max results to keep
RELATE_SCORE_POWER = 4  # Power transform for gated relevance

# Score transformation parameters for vector search
SCORE_TRANSFORM_POWER = 3.0  # Higher = more emphasis on top scores
SCORE_TRANSFORM_MIN = 0.4  # Scores below this treated as 0
SCORE_TRANSFORM_MAX = 1.0  # Max possible score

# High relevance boost threshold and multiplier
HIGH_RELEVANCE_THRESHOLD = 0.85  # Top 15% of normalized range
HIGH_RELEVANCE_BOOST = 2.0  # Multiplier for high relevance scores

# =============================================================================
# Tiered Ranking Configuration (Hybrid Search)
# =============================================================================

# Tiered ranking divides results into two zones:
# - High relevance zone: gets stats/recency boost
# - Low relevance zone: sorted strictly by relevance

# Threshold for high relevance zone (relative to max score)
# 0.7 means only top 30% by relevance score qualify for popularity boost
TIERED_HIGH_RELEVANCE_THRESHOLD = 0.7

# Within high relevance zone, items within this relative diff are "equally relevant"
TIERED_SIMILARITY_THRESHOLD = 0.05  # 5% relative difference

# Weights for secondary sort within high relevance zone
TIERED_STATS_WEIGHT = 0.7  # Weight for popularity (view, coin, etc.)
TIERED_RECENCY_WEIGHT = 0.3  # Weight for recency (pubdate)

# =============================================================================
# Reranking Configuration
# =============================================================================

# Whether to enable reranking by default
RERANK_ENABLED = True

# Maximum number of hits to rerank
# Higher values improve recall but increase latency
# With supplemental word recall, pool is ~2000-3000 hits (KNN + word)
RERANK_MAX_HITS = 2000

# Boost factors for keyword matching during rerank
RERANK_KEYWORD_BOOST = 1.5  # Boost when keyword found in tags/desc
RERANK_TITLE_KEYWORD_BOOST = 2.0  # Higher boost for title matches

# Text fields to use for document embedding during rerank
RERANK_TEXT_FIELDS = ["title", "tags", "desc", "owner.name"]

# Timeout for rerank operation
RERANK_TIMEOUT = 30  # seconds

# Maximum passage length for embedding
RERANK_MAX_PASSAGE_LENGTH = 4096  # characters

# Score for non-reranked hits (ensures they rank below reranked ones)
NON_RERANKED_SCORE_PENALTY = 0.01

# =============================================================================
# Hybrid Search Configuration
# =============================================================================

# Weight for word-based score in hybrid mode
HYBRID_WORD_WEIGHT = 0.5

# Weight for vector-based score in hybrid mode
HYBRID_VECTOR_WEIGHT = 0.5

# =============================================================================
# Author Grouping Configuration
# =============================================================================

# =============================================================================
# Recency Scoring (from blux.doc_score time factor)
# =============================================================================

# Real-time time factor boundaries (from DocScorer.TIME_ANCHORS)
# Used for normalization to [0, 1] range
TIME_FACTOR_MIN = 0.45  # Score for videos >= 30 days old
TIME_FACTOR_MAX = 1.30  # Score for videos <= 1 hour old

# =============================================================================
# Author Grouping Configuration
# =============================================================================

# Available sort fields for author grouping
AUTHOR_SORT_FIELD_TYPE = Literal[
    "sum_count",  # Total videos by author
    "sum_view",  # Total views across videos
    "sum_sort_score",  # Sum of sort scores
    "sum_rank_score",  # Sum of rank scores
    "top_rank_score",  # Max rank score
    "first_appear_order",  # Order by first video appearance
]

# Default sort field for author grouping
# "first_appear_order" ensures author order matches video list order
AUTHOR_SORT_FIELD = "first_appear_order"

# Maximum authors to return
AUTHOR_GROUP_LIMIT = 25
