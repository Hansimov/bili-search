"""
Ranks Module - Centralized Ranking and Scoring System

This module provides a unified ranking infrastructure for video search results.
It consolidates all ranking, scoring, and reranking functionality.

Module Structure:
    - constants.py: All ranking-related constants and configuration
    - scorers.py: Score calculation classes (stats, pubdate, relevance)
    - fusion.py: Score fusion and combination strategies
    - ranker.py: Main VideoHitsRanker class for various ranking methods
    - reranker.py: Embedding-based reranker for precise similarity ranking
    - grouper.py: Author grouping and aggregation logic

Usage:
    Import directly from submodules:
        from ranks.constants import RANK_METHOD_TYPE, RANK_TOP_K
        from ranks.ranker import VideoHitsRanker
        from ranks.reranker import get_reranker
        from ranks.grouper import AuthorGrouper
"""
