"""Constants for Owner search index and queries.

Defines index names, field configurations, boost weights,
scoring parameters, and search limits for the owners ES index.
"""

from typing import Literal

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
    "total_danmaku",
    "total_reply",
    "total_share",
    "total_duration",
    "influence_score",
    # quality
    "avg_view",
    "avg_favorite_rate",
    "avg_coin_rate",
    "avg_like_rate",
    "avg_stat_score",
    "quality_score",
    # activity
    "latest_pubdate",
    "earliest_pubdate",
    "latest_bvid",
    "publish_freq",
    "days_since_last",
    "activity_score",
    # domain
    "top_tags",
    "primary_tid",
    "primary_ptid",
    # profile substitute
    "latest_pic",
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
    "top_tags",
    "latest_pic",
    "latest_pubdate",
    "primary_tid",
    "primary_ptid",
]

# =============================================================================
# Search Match Fields & Boosts
# =============================================================================

# Name search fields with boosts
NAME_MATCH_BOOSTS = {
    "name.keyword": 50.0,  # Exact match — highest priority
    "name.words": 10.0,  # Token BM25 match
    "top_tags.words": 3.0,  # Domain tags — auxiliary
    "mentioned_names.words": 1.5,  # Related user names — auxiliary
}

# Domain search fields with boosts
DOMAIN_MATCH_BOOSTS = {
    "top_tags.words": 5.0,  # Tags are the primary domain signal
    "name.words": 2.0,  # Name might contain domain keywords
    "mentioned_names.words": 1.0,  # Related users in similar domain
}

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
