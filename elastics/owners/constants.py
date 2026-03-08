"""Constants for Owner search index and queries.

Defines index names, field configurations, boost weights,
scoring parameters, and search limits for the owners ES index.
"""

from typing import Literal

from math import log1p

# =============================================================================
# Index Configuration
# =============================================================================

ELASTIC_OWNERS_INDEX = "bili_owners_v1"
ELASTIC_OWNERS_DEV_INDEX = "bili_owners_dev1"

# Use the same elastic env names as videos
ELASTIC_DEV = "elastic_dev"
ELASTIC_PRO = "elastic_pro"

# =============================================================================
# Source Fields (fields to return in search results)
# =============================================================================

SOURCE_FIELDS = [
    "mid",
    "name",
    # influence
    "total_videos",
    "total_view",
    "total_like",
    "total_coin",
    "total_favorite",
    "influence_score",
    # quality
    "avg_favorite_rate",
    "avg_coin_rate",
    "avg_like_rate",
    "avg_stat_score",
    "quality_score",
    # activity
    "latest_pubdate",
    "latest_bvid",
    "recent_7d_videos",
    "recent_30d_videos",
    "publish_freq",
    "days_since_last",
    "activity_score",
    # profile-token placeholders
    "profile_domain_ready",
    "core_tokenizer_version",
    "core_tag_token_ids",
    "core_tag_token_weights",
    "core_text_token_ids",
    "core_text_token_weights",
    # relations
    "mentioned_mids",
    "mentioned_names",
    # meta
    "index_at",
    "update_at",
]

# Compact source fields for suggest / lightweight queries
SOURCE_FIELDS_COMPACT = [
    "mid",
    "name",
    "total_videos",
    "total_view",
    "influence_score",
    "quality_score",
    "activity_score",
    "latest_pubdate",
    "profile_domain_ready",
    "core_tokenizer_version",
]

# =============================================================================
# Search Match Fields & Boosts
# =============================================================================

# Name search fields with boosts
NAME_MATCH_BOOSTS = {
    "name.keyword": 50.0,  # Exact match — highest priority
    "name.words": 10.0,  # Token BM25 match
    "mentioned_names.words": 1.5,  # Related user names — auxiliary
}

# Future domain retrieval is token-based. These token-id fields are placeholders
# until CoreTagTokenizer / CoreTexTokenizer are trained and wired online.
PROFILE_TOKEN_MATCH_BOOSTS = {
    "core_tag_token_ids": 4.0,
    "core_text_token_ids": 5.0,
}
PROFILE_TOKEN_SCORE_DENOM = sum(PROFILE_TOKEN_MATCH_BOOSTS.values())
DOMAIN_PHRASE_QUERY_MIN_CHARS = 8

# =============================================================================
# Sort Fields
# =============================================================================

SORT_FIELD_TYPE = Literal[
    "relevance",
    "influence",
    "quality",
    "activity",
    "total_view",
    "total_videos",
]
SORT_FIELD_DEFAULT = "relevance"

SORT_FIELD_MAP = {
    "relevance": "_score",
    "influence": "influence_score",
    "quality": "quality_score",
    "activity": "activity_score",
    "total_view": "total_view",
    "total_videos": "total_videos",
}

# =============================================================================
# Limits & Timeouts
# =============================================================================

SEARCH_LIMIT = 20
SUGGEST_LIMIT = 10
TOP_OWNERS_LIMIT = 50
SEARCH_TIMEOUT = 3  # seconds
SUGGEST_TIMEOUT = 1.5  # seconds

# =============================================================================
# Scoring Constants (used in scorer.py)
# =============================================================================

# log_normalize max values (for influence scoring)
MAX_VIEW = 1e10
MAX_VIDEOS = 10000
MAX_LIKE = 1e8
MAX_COIN = 1e7
LOG_MAX_VIEW = log1p(MAX_VIEW)
LOG_MAX_VIDEOS = log1p(MAX_VIDEOS)

# Influence weights
INFLUENCE_WEIGHTS = {
    "view": 0.40,
    "scale": 0.20,
    "like": 0.25,
    "coin": 0.15,
}

# Quality rate ranges (for bounded_normalize)
QUALITY_RANGES = {
    "favorite_rate": {"low": 0.001, "high": 0.03},
    "coin_rate": {"low": 0.0005, "high": 0.015},
    "like_rate": {"low": 0.01, "high": 0.08},
}

# Quality weights
QUALITY_WEIGHTS = {
    "favorite_rate": 0.35,
    "coin_rate": 0.25,
    "like_rate": 0.20,
    "stat_quality": 0.20,
}
QUALITY_CONFIDENCE_MIN_VIDEOS = 20  # Full confidence at this video count
QUALITY_CONFIDENCE_FLOOR = 0.3  # Minimum multiplier for low-video owners

# Activity weights
ACTIVITY_WEIGHTS = {
    "recency": 0.45,
    "frequency": 0.25,
    "persistence": 0.15,
    "volume": 0.15,
}
ACTIVITY_RECENCY_TAU = 60  # Exponential decay τ in days
ACTIVITY_PERSISTENCE_DAYS = 180  # Full persistence score at this span
ACTIVITY_MIN_VIDEOS = 5  # Minimum videos for volume gate

# Rank weight profiles by query type
RANK_WEIGHT_PROFILES = {
    "name": {
        "name_match": 0.50,
        "domain": 0.05,
        "influence": 0.25,
        "quality": 0.10,
        "activity": 0.10,
    },
    "domain": {
        "name_match": 0.05,
        "domain": 0.35,
        "influence": 0.25,
        "quality": 0.20,
        "activity": 0.15,
    },
    "mixed": {
        "name_match": 0.25,
        "domain": 0.20,
        "influence": 0.25,
        "quality": 0.15,
        "activity": 0.15,
    },
}

# Name match score normalization denominator
NAME_MATCH_NORM_DENOM = 30.0

# NOTE: Index settings and mappings are defined in bili-scraper:
#   converters/elastic/owner_index_settings_v1.py
