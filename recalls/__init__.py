"""
Recalls Module - Multi-Lane Recall System

This module provides the recall (粗召) layer for video search.
Instead of a single search that tries to optimize one dimension,
multi-lane recall runs parallel searches optimized for different signals:

- **Relevance lane**: BM25 keyword match, sorted by _score (text match quality)
- **Title-match lane**: BM25 on title+tags — high precision for entity queries
- **Popularity lane**: Keyword match + sorted by stat.view desc (raw reach)
- **Recency lane**: Keyword match + sorted by pubdate desc (freshness)
- **Quality lane**: Keyword match + sorted by stat_score desc (content quality)

Design rationale for 5 lanes:
- Each lane captures a genuinely distinct signal dimension.
- The title_match lane fixes entity query problems: queries like
  '通义实验室', '飓风营救', '红警08' need title-level precision.
- Popularity (view count) measures raw reach/awareness, while quality
  (stat_score) measures content satisfaction — two distinct signals.

Title-match tagging:
After recall, all hits are tagged with `_title_matched` based on whether
the query terms appear in the document title or tags. This tag is used by the
downstream ranker to apply a relevance bonus (TITLE_MATCH_BONUS).

Module Structure:
    - base.py: RecallResult, RecallPool data classes, NoiseFilter
    - word.py: Multi-lane word-based recall (5 lanes)
    - vector.py: KNN vector-based recall
    - manager.py: RecallManager orchestrating strategies by query mode

Usage:
    from recalls.manager import RecallManager
    manager = RecallManager()
    pool = manager.recall(searcher=searcher, query="黑神话", mode="word")
"""
