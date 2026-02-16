"""
Ranks Module - Centralized Ranking and Scoring System

This module provides a unified ranking infrastructure for video search results.
It consolidates all ranking, scoring, and reranking functionality.

Core Design:
    The ranking system uses a three-phase diversified approach:
    1. Headline selection (top 3): composite score balancing relevance + quality
    2. Relevance-gated slot allocation (positions 4-10): dimension representatives
       with relevance gating — slots are scored as dim_score * relevance_factor,
       preventing irrelevant docs from occupying visible positions
    3. Fused scoring (beyond 10): weighted combination with relevance at 0.50

    Title-match awareness: docs with _title_matched flag receive TITLE_MATCH_BONUS
    added to relevance_score, ensuring title-matching docs are strongly preferred.

    Graduated threshold relaxation in slot allocation:
    - First: only docs with relevance >= SLOT_MIN_RELEVANCE (0.30)
    - If too few: relax to SLOT_MIN_RELEVANCE * 0.5
    - If still too few: allow all

Module Structure:
    - constants.py: All ranking-related constants and configuration
    - scorers.py: Score calculation classes (stats quality, recency, relevance gating)
    - fusion.py: Score fusion and combination (preference-weighted sum)
    - diversified.py: Three-phase diversified slot-based ranker (default)
    - ranker.py: Main VideoHitsRanker class routing to all ranking strategies
    - reranker.py: Embedding-based reranker for precise similarity ranking
    - grouper.py: Author grouping and aggregation logic

Usage:
    from ranks.ranker import VideoHitsRanker
    from ranks.constants import RANK_METHOD_TYPE, RANK_TOP_K
    from ranks.reranker import get_reranker
    from ranks.grouper import AuthorGrouper
"""
