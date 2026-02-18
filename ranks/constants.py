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

RANK_METHOD_TYPE = Literal[
    "heads", "rrf", "stats", "relevance", "tiered", "diversified"
]
RANK_METHOD = "diversified"

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
# Diversified Ranking Fused Score Weights
# =============================================================================

# Weights for fused scoring in diversified ranker (Phase 3: beyond top-N slots).
# Relevance dominates to prevent irrelevant docs from surfacing.
# Quality and recency provide secondary differentiation.
DIVERSIFIED_FUSED_WEIGHTS = {
    "relevance": 0.50,
    "quality": 0.20,
    "recency": 0.18,
    "popularity": 0.12,
}

# =============================================================================
# Headline Quality Scoring (Top-3 Selection)
# =============================================================================

# Top-N "headline" positions should be both relevant AND high quality.
# Instead of pure relevance slot allocation, the top positions use a
# composite "headline quality" score that balances relevance with
# quality and recency, ensuring the best first impression.
HEADLINE_TOP_N = 3  # How many top positions use headline quality scoring
HEADLINE_WEIGHTS = {
    "relevance": 0.50,  # Must be highly relevant (dominant signal)
    "quality": 0.30,  # Must be high quality — increased to push low-quality content down
    "recency": 0.10,  # Prefer recent
    "popularity": 0.10,  # Slight popularity bias
}

# Minimum relevance score required for headline and slot candidates.
# Prevents low-relevance docs from occupying top-10 positions even if
# they score high on popularity/recency. This is the primary fix for
# the "some top-10 slots have irrelevant docs" problem.
# Set to 0.40: docs must have at least 40% of max BM25 score
# to be eligible for any top-10 position.
SLOT_MIN_RELEVANCE = 0.55

# Minimum relevance for headline (top-3) positions — even stricter.
HEADLINE_MIN_RELEVANCE = 0.50

# Content quality signals for ranking penalty
# Short-duration penalty: very short videos (<30s) are often low-effort
RANK_SHORT_DURATION_THRESHOLD = 30  # seconds
RANK_SHORT_DURATION_PENALTY = 0.7  # multiply quality_score by this
RANK_VERY_SHORT_DURATION_THRESHOLD = 15  # seconds — very short videos
RANK_VERY_SHORT_DURATION_PENALTY = 0.3  # much harsher penalty for < 15s

# Short-title penalty: BM25 gives disproportionately high scores to docs
# with very short titles. Penalize these in ranking quality scoring.
# This fixes the 'chatgpt' problem where title='ChatGpt' scores highest.
RANK_SHORT_TITLE_THRESHOLD = 25  # characters — titles shorter than this are penalized
RANK_SHORT_TITLE_PENALTY = 0.4  # multiply quality_score by this for short titles

# Content depth penalty: penalizes relevance_score when the title adds
# almost no information beyond the query keywords. e.g., query='gta' and
# title='gta' → the title IS the query, providing zero additional context.
# This is a RELEVANCE penalty (not just quality) since BM25 falsely inflates
# scores for very short docs that are essentially keyword matches.
# The penalty is: relevance *= max(RANK_CONTENT_DEPTH_MIN_FACTOR,
#   (title_chars_beyond_query) / RANK_CONTENT_DEPTH_NORM_LENGTH)
RANK_CONTENT_DEPTH_MIN_FACTOR = 0.30  # minimum factor (prevents zeroing out)
RANK_CONTENT_DEPTH_NORM_LENGTH = 20  # chars beyond query for full relevance

# Very short duration disqualifies from headline/slot positions entirely.
# Videos under this threshold should NEVER occupy headline (top-3) slots
# unless nothing else is available.
RANK_HEADLINE_MIN_DURATION = 30  # seconds — minimum for headline positions
RANK_SLOT_MIN_DURATION = 15  # seconds — minimum for slot (top-10) positions

# Low engagement penalty in ranking: docs with low views are likely
# low-quality even if BM25 scores them high.
RANK_LOW_ENGAGEMENT_THRESHOLD = 500  # views
RANK_LOW_ENGAGEMENT_PENALTY = 0.7  # multiply quality_score by this

# =============================================================================
# Title Match Bonus
# =============================================================================

# Title match is a strong relevance signal: if the query appears in the title,
# the doc is very likely relevant. This bonus is added to relevance_score
# before normalization, acting as a multiplicative boost.
TITLE_MATCH_BONUS = 0.20  # Added to normalized relevance when title matches query

# Owner match bonus: when query partially matches an owner/UP主 name,
# docs from that owner get a relevance bonus. This helps entity queries
# like '红警08' (owner '红警HBK08') surface the creator's content.
# ONLY applied when the doc's title also contains query keywords
# (prevents boosting irrelevant uploads from matching owners).
OWNER_MATCH_BONUS = 0.30  # Added to normalized relevance for owner-matched docs

# Title-keyword overlap: when no query keyword appears in the title,
# the BM25 score comes entirely from non-title fields (owner.name, desc, tags).
# This suggests the doc's actual content is NOT about the query,
# so we penalize relevance. Example: query "吴恩达大模型", owner "吴恩达大模型课程"
# but title "《喜羊羊与灰太狼》" — BM25 is boosted by owner.name match, not content.
RANK_NO_TITLE_KEYWORD_PENALTY = 0.50  # multiply relevance when 0 query tokens in title

# Owner concentration thresholds: used to distinguish "owner queries"
# (where the user wants a specific creator) from "topic queries" (where
# the query happens to match many owner names).
# An owner is "dominant" when their doc count exceeds these thresholds.
OWNER_DOMINANT_MIN_DOCS = 5  # Minimum docs from one owner to consider dominant
OWNER_DOMINANT_RATIO = 0.15  # Docs from owner must be >= 15% of matched-owner pool
OWNER_DISPERSE_MAX_OWNERS = 6  # If >= this many owners match, it's a topic query

# =============================================================================
# Relevance Decay for Slot Candidates
# =============================================================================

# Below this relevance threshold, apply exponential decay to dimension scores.
# This prevents low-relevance docs from occupying dimension slots even if they
# are the "most popular" or "most recent" in the pool.
SLOT_RELEVANCE_DECAY_THRESHOLD = 0.45
SLOT_RELEVANCE_DECAY_POWER = 2.0  # Quadratic decay below threshold

# Quality tiebreaker: small weight to break ties when multiple docs have
# identical dimension scores (e.g., all chatgpt docs have relevance=1.0).
# This prevents low-quality docs from occupying slots arbitrarily.
SLOT_QUALITY_TIEBREAKER = 0.05

# =============================================================================
# Recall Noise Filtering
# =============================================================================

# Minimum BM25 score for non-relevance recall lanes.
# Docs scoring below this are noise — they match query syntax but not meaning.
# BM25 score of ~3.0 means at least one meaningful term matches.
# (Raised from 2.0: many barely-matching docs had scores 2.0-3.0)
MIN_BM25_SCORE = 3.0

# Score-ratio gate: remove docs with score < ratio * max_score in their lane.
# Applied after each recall lane returns, before merge.
# 0.10 means docs must score at least 10% of the best hit's score.
# (Lowered from 0.18: was too aggressive, causing insufficient candidate pools
# especially for single-keyword queries like 'chatgpt'. Quality control is
# now handled primarily by the diversified ranker's content quality scoring.)
NOISE_SCORE_RATIO_GATE = 0.10

# KNN score ratio: stricter threshold for noisy LSH hamming distance scores.
# LSH bit vectors have narrow score ranges, so many irrelevant docs score similarly.
NOISE_KNN_SCORE_RATIO = 0.50

# Don't apply noise filtering if total hits below this count.
# Small result sets need all candidates.
NOISE_MIN_HITS_FOR_FILTER = 30

# Gate reduction for multi-lane docs (appear in 2+ recall lanes).
# Multi-lane appearance is strong evidence of relevance, so apply lower threshold.
NOISE_MULTI_LANE_GATE_FACTOR = 0.5

# Content quality noise filtering:
# Short-text penalty: BM25 inflates scores for very short docs (field-length normalization).
# Penalize docs where combined content is too short — they are likely low-effort.
NOISE_SHORT_TEXT_MIN_LENGTH = 15  # Min chars (title+desc) to be considered substantial
NOISE_SHORT_TEXT_PENALTY = 0.4  # Multiply score by this when content is too short

# Quality engagement floor: even if BM25 score is high, docs with near-zero
# engagement are likely low-quality (spam, empty, test uploads).
NOISE_MIN_ENGAGEMENT_VIEWS = 50  # Minimum views to pass quality gate
NOISE_LOW_ENGAGEMENT_PENALTY = 0.25  # Score penalty factor for very low engagement

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

# Stats-based scoring is now handled by StatsScorer (DocScorer-based)
# in ranks/scorers.py. Individual stat fields and log offsets are
# configured in the DocScorer library, not here.

# =============================================================================
# Pubdate (Recency) Scoring Configuration
# =============================================================================

# Recency scoring is now handled by PubdateScorer in ranks/scorers.py,
# which uses the real-time time factor from DocScorer.TIME_ANCHORS.
# See TIME_FACTOR_MIN and TIME_FACTOR_MAX below for normalization bounds.

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

# Recency Scoring (from blux.doc_score time factor)
# Real-time time factor boundaries (from DocScorer.TIME_ANCHORS)
# Used for normalization to [0, 1] range
TIME_FACTOR_MIN = 0.45  # Score for videos >= 30 days old
TIME_FACTOR_MAX = 1.30  # Score for videos <= 1 hour old

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
