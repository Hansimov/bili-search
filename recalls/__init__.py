"""
Recalls Module - Multi-Lane Recall System

This module provides the recall (粗召) layer for video search.
Instead of a single search that tries to optimize one dimension,
multi-lane recall runs parallel searches optimized for different signals:

- **Relevance lane**: BM25 keyword match, sorted by _score
- **Popularity lane**: Keyword match + sorted by stat.view desc
- **Recency lane**: Keyword match + sorted by pubdate desc
- **Quality lane**: Keyword match + sorted by stat_score desc

By recalling from all four dimensions, we ensure that "相关的"、"热度高的"、
"质量高的"、"时间近的" documents are all present in the candidate pool
BEFORE the ranking stage.

Module Structure:
    - base.py: RecallResult, RecallPool data classes
    - word.py: Multi-lane word-based recall
    - vector.py: KNN vector-based recall
    - manager.py: RecallManager orchestrating strategies by query mode

Usage:
    from recalls.manager import RecallManager
    manager = RecallManager(searcher)
    pool = manager.recall(query="黑神话", mode="word", ...)
"""
