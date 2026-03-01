"""Owner scoring functions for influence, quality, and activity metrics.

All scoring uses log/bounded normalization to handle the extreme power-law
distribution of internet data, where a few head creators dominate most traffic.
"""

import math

from elastics.owners.constants import (
    MAX_VIEW,
    MAX_VIDEOS,
    MAX_LIKE,
    MAX_COIN,
    INFLUENCE_WEIGHTS,
    QUALITY_RANGES,
    QUALITY_WEIGHTS,
    QUALITY_CONFIDENCE_MIN_VIDEOS,
    QUALITY_CONFIDENCE_FLOOR,
    ACTIVITY_WEIGHTS,
    ACTIVITY_RECENCY_TAU,
    ACTIVITY_PERSISTENCE_DAYS,
    ACTIVITY_MIN_VIDEOS,
    RANK_WEIGHT_PROFILES,
    NAME_MATCH_NORM_DENOM,
)


def log_normalize(value: float, max_val: float) -> float:
    """Log-scale normalization: log10(value+1) / log10(max_val+1) → [0, 1].

    Uses logarithmic scaling to compress the extreme power-law distribution
    so mid-tier creators still get reasonable scores.
    """
    if value <= 0:
        return 0.0
    return min(math.log10(value + 1) / math.log10(max_val + 1), 1.0)


def bounded_normalize(value: float, low: float, high: float) -> float:
    """Bounded linear normalization: maps [low, high] → [0, 1], clamped.

    Values below `low` map to 0, above `high` map to 1.
    """
    if high <= low:
        return 0.0
    return max(0.0, min((value - low) / (high - low), 1.0))


def compute_influence(
    total_view: int,
    total_videos: int,
    total_like: int,
    total_coin: int,
) -> float:
    """Compute owner influence score [0, 1].

    influence = exposure × creation_scale × community_approval

    Since Card API is banned (no follower count), influence is based entirely
    on video stats. total_view + interaction counts are more reliable signals
    of real content influence than follower count.
    """
    view_score = log_normalize(total_view, MAX_VIEW)
    scale_score = log_normalize(total_videos, MAX_VIDEOS)
    like_score = log_normalize(total_like, MAX_LIKE)
    coin_score = log_normalize(total_coin, MAX_COIN)

    w = INFLUENCE_WEIGHTS
    influence = (
        w["view"] * view_score
        + w["scale"] * scale_score
        + w["like"] * like_score
        + w["coin"] * coin_score
    )
    return round(influence, 4)


def compute_quality(
    avg_favorite_rate: float,
    avg_coin_rate: float,
    avg_like_rate: float,
    avg_stat_score: float,
    total_videos: int,
) -> float:
    """Compute owner creation quality score [0, 1].

    quality = interaction_depth × content_value × sample_confidence

    Favorites and coins are deeper interaction signals than views/likes.
    Sample confidence ensures low-video owners don't get inflated scores.
    """
    r = QUALITY_RANGES
    fav_score = bounded_normalize(
        avg_favorite_rate, r["favorite_rate"]["low"], r["favorite_rate"]["high"]
    )
    coin_score = bounded_normalize(
        avg_coin_rate, r["coin_rate"]["low"], r["coin_rate"]["high"]
    )
    like_score = bounded_normalize(
        avg_like_rate, r["like_rate"]["low"], r["like_rate"]["high"]
    )
    stat_quality = min(avg_stat_score / 100.0, 1.0)

    # Sample size confidence — more videos = more reliable quality estimate
    confidence = min(total_videos / QUALITY_CONFIDENCE_MIN_VIDEOS, 1.0)

    w = QUALITY_WEIGHTS
    raw_quality = (
        w["favorite_rate"] * fav_score
        + w["coin_rate"] * coin_score
        + w["like_rate"] * like_score
        + w["stat_quality"] * stat_quality
    )

    # Apply confidence decay — low-video owners get discounted quality
    quality = raw_quality * (
        QUALITY_CONFIDENCE_FLOOR + (1.0 - QUALITY_CONFIDENCE_FLOOR) * confidence
    )
    return round(quality, 4)


def compute_activity(
    days_since_last: float,
    publish_freq: float,
    total_videos: int,
    days_span: float,
) -> float:
    """Compute owner activity score [0, 1].

    activity = recency × frequency × persistence × volume_gate

    Recency uses exponential decay (τ=60 days):
      0 days → 1.0, 30 days → ~0.61, 90 days → ~0.22, 365 days → ~0.002
    """
    # Recency: exponential decay
    recency = math.exp(-days_since_last / ACTIVITY_RECENCY_TAU)

    # Frequency: videos per day, normalized
    freq_score = bounded_normalize(publish_freq, low=1 / 90, high=1.0)

    # Persistence: active for at least ACTIVITY_PERSISTENCE_DAYS
    persistence = min(days_span / ACTIVITY_PERSISTENCE_DAYS, 1.0)

    # Volume gate: at least ACTIVITY_MIN_VIDEOS to be meaningful
    volume_gate = min(total_videos / ACTIVITY_MIN_VIDEOS, 1.0)

    w = ACTIVITY_WEIGHTS
    activity = (
        w["recency"] * recency
        + w["frequency"] * freq_score
        + w["persistence"] * persistence
        + w["volume"] * volume_gate
    )
    return round(activity, 4)


def compute_owner_rank_score(
    name_match_score: float,
    domain_score: float,
    influence_score: float,
    quality_score: float,
    activity_score: float,
    query_type: str = "name",
) -> float:
    """Compute final owner ranking score by weighted fusion.

    Args:
        name_match_score: BM25 name match score [0, ~50].
        domain_score: Domain relevance [0, 1] (from text/vector match).
        influence_score: Pre-computed influence [0, 1].
        quality_score: Pre-computed quality [0, 1].
        activity_score: Pre-computed activity [0, 1].
        query_type: "name" / "domain" / "mixed" — controls weight profile.

    Returns:
        Final ranking score [0, 1].
    """
    w = RANK_WEIGHT_PROFILES.get(query_type, RANK_WEIGHT_PROFILES["mixed"])

    # Normalize BM25 name_match_score to [0, 1]
    name_norm = min(name_match_score / NAME_MATCH_NORM_DENOM, 1.0)

    final_score = (
        w["name_match"] * name_norm
        + w["domain"] * domain_score
        + w["influence"] * influence_score
        + w["quality"] * quality_score
        + w["activity"] * activity_score
    )
    return round(final_score, 4)


def detect_owner_query_type(
    query: str,
    name_hits: list[dict],
    domain_hits: list[dict],
) -> str:
    """Detect query type based on name/domain hit patterns.

    Returns:
        "name"   — query precisely matches an owner name (e.g. "影视飓风")
        "domain" — query describes a topic/domain (e.g. "黑神话悟空 UP 主")
        "mixed"  — ambiguous (e.g. "红警" — both a name and a topic)
    """
    has_exact_name = any(
        hit.get("name", "") == query for hit in name_hits
    )
    has_strong_name = (
        len(name_hits) > 0 and name_hits[0].get("_score", 0) > 20.0
    )
    has_domain_hits = len(domain_hits) > 3

    if has_exact_name or has_strong_name:
        return "name" if not has_domain_hits else "mixed"
    elif has_domain_hits:
        return "domain"
    else:
        return "mixed"
