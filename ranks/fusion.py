"""
Score Fusion Strategies

This module provides methods for combining multiple normalized scores
into a final ranking score, supporting preference-based weight adjustment.

Main class:
    - ScoreFuser: Combines quality, relevance, and recency scores

All input scores are expected to be normalized to [0, 1]:
    - quality ∈ [0, 1): stat quality from DocScorer
    - relevance ∈ [0, 1]: normalized relevance (BM25, cosine, or blended)
    - recency ∈ [0, 1]: normalized time factor from PubdateScorer

Preference modes control relative weight of each signal:
    - balanced: relevance-first, quality second, recency third
    - prefer_quality: emphasize document popularity/engagement
    - prefer_relevance: emphasize search match quality
    - prefer_recency: emphasize freshness
"""

import math

from ranks.constants import (
    RANK_PREFER_TYPE,
    RANK_PREFER,
    RANK_PREFER_PRESETS,
    BLEND_SEMANTIC_WEIGHT,
    BLEND_BM25_WEIGHT,
    BLEND_KEYWORD_BONUS,
)


class ScoreFuser:
    """Fuses multiple normalized scores into a single ranking score.

    Supports two fusion strategies:
    1. Preference-based weighted sum (primary, recommended):
       score = w_quality * quality + w_relevance * relevance + w_recency * recency

    2. Legacy product formula (kept for compatibility):
       score = (base + stats) * pubdate * relate^power

    The preference-based approach is preferred because:
    - Each signal contributes independently and transparently
    - Weights directly map to user-visible preference modes
    - Normalized inputs ensure balanced contribution

    Example:
        >>> fuser = ScoreFuser()
        >>> fuser.fuse_with_preference(quality=0.8, relevance=0.9, recency=0.5)
        0.69  # balanced: 0.3*0.8 + 0.5*0.9 + 0.2*0.5
        >>> fuser.fuse_with_preference(quality=0.8, relevance=0.9, recency=0.5,
        ...                            prefer="prefer_quality")
        0.77  # 0.5*0.8 + 0.3*0.9 + 0.2*0.5
    """

    def fuse_with_preference(
        self,
        quality: float,
        relevance: float,
        recency: float,
        prefer: RANK_PREFER_TYPE = RANK_PREFER,
    ) -> float:
        """Fuse scores using preference-weighted sum.

        All inputs should be normalized to [0, 1].

        Formula: w_q * quality + w_r * relevance + w_t * recency

        Args:
            quality: Quality score from StatsScorer ∈ [0, 1).
            relevance: Normalized relevance ∈ [0, 1].
            recency: Normalized recency from PubdateScorer ∈ [0, 1].
            prefer: Preference mode controlling weight distribution.

        Returns:
            Fused score in [0, 1], rounded to 6 decimal places.
        """
        weights = RANK_PREFER_PRESETS.get(prefer, RANK_PREFER_PRESETS["balanced"])
        score = (
            weights["quality"] * quality
            + weights["relevance"] * relevance
            + weights["recency"] * recency
        )
        return round(min(score, 1.0), 6)

    @staticmethod
    def blend_relevance(
        cosine_similarity: float,
        bm25_norm: float = 0.0,
        keyword_boost: float = 1.0,
    ) -> float:
        """Blend embedding cosine similarity with BM25 keyword-match score.

        Used after reranking to produce a unified relevance score that
        combines semantic understanding (from embeddings) with precise
        keyword matching (from BM25).

        Formula when BM25 is available:
            relevance = semantic_w * cosine + bm25_w * bm25_norm + keyword_bonus

        Formula when only embedding is available:
            relevance = cosine * (semantic_w + bm25_w) + keyword_bonus

        The keyword_bonus is derived from the keyword_boost multiplier
        (typically 1.0 for no match, up to ~6.0+ for multiple matches).
        It provides a small uplift for documents containing exact query terms.

        Args:
            cosine_similarity: Embedding cosine similarity ∈ [0, 1].
            bm25_norm: Normalized BM25 score ∈ [0, 1]. 0 if unavailable.
            keyword_boost: Keyword boost multiplier from reranker (>= 1.0).

        Returns:
            Blended relevance score ∈ [0, 1].
        """
        # Keyword bonus: log-compress the boost multiplier to [0, BLEND_KEYWORD_BONUS]
        # keyword_boost=1.0 → 0, keyword_boost=3.0 → ~0.07, keyword_boost=6.0 → ~0.10
        if keyword_boost > 1.0:
            bonus = min(
                math.log1p(keyword_boost - 1.0) / 2.0 * BLEND_KEYWORD_BONUS / 0.1,
                BLEND_KEYWORD_BONUS,
            )
        else:
            bonus = 0.0

        if bm25_norm > 0:
            # Both signals available: weighted blend
            blended = (
                BLEND_SEMANTIC_WEIGHT * cosine_similarity
                + BLEND_BM25_WEIGHT * bm25_norm
                + bonus
            )
        else:
            # Only embedding: scale semantic to fill full weight
            blended = (
                cosine_similarity * (BLEND_SEMANTIC_WEIGHT + BLEND_BM25_WEIGHT) + bonus
            )

        return min(max(blended, 0.0), 1.0)

    # ---- Removed legacy methods ----
    # calc_fuse_score_by_prod() and fuse() have been removed.
    # Use fuse_with_preference() for all score fusion needs.
