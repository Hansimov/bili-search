"""
Recall Pool Optimizer

Extracts features and signals from recall pool to optimize downstream
ranking and query refinement. Designed to bridge multi-round recall with
the diversified ranking phase.

Key capabilities:
1. **Pool statistics**: Summarize score and view distributions.
2. **Adaptive scoring hints**: Provide signal-based adjustments for the
    ranker and downstream diagnostics.
3. **Feature extraction**: Extract top keywords, dominant tags, and
   content type distributions from the pool for query refinement.

Usage:
    optimizer = RecallPoolOptimizer()
    hints = optimizer.analyze(pool, query)
    # hints.dominant_tags → list of (tag, count)
    # hints.title_keywords → list of (keyword, freq)
    # hints.content_type_bias → dict of type → ratio
    # hints.score_distribution → ScoreDistribution
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from tclogger import dict_get


# ── Data structures ───────────────────────────────────────────────────────


@dataclass
class ScoreDistribution:
    """Statistics about the score distribution in the pool."""

    max_score: float = 0.0
    min_score: float = 0.0
    median_score: float = 0.0
    p25_score: float = 0.0
    p75_score: float = 0.0
    mean_score: float = 0.0
    std_score: float = 0.0
    count: int = 0

    @classmethod
    def from_scores(cls, scores: list[float]) -> "ScoreDistribution":
        if not scores:
            return cls()
        sorted_scores = sorted(scores, reverse=True)
        n = len(sorted_scores)
        mean = sum(sorted_scores) / n
        variance = sum((s - mean) ** 2 for s in sorted_scores) / n
        return cls(
            max_score=sorted_scores[0],
            min_score=sorted_scores[-1],
            median_score=sorted_scores[n // 2],
            p25_score=sorted_scores[min(int(n * 0.25), n - 1)],
            p75_score=sorted_scores[min(int(n * 0.75), n - 1)],
            mean_score=round(mean, 4),
            std_score=round(variance**0.5, 4),
            count=n,
        )


@dataclass
class ContentTypeDistribution:
    """Distribution of content categories in the pool."""

    # Based on tag frequency analysis
    dominant_category: str = ""
    category_ratios: dict[str, float] = field(default_factory=dict)
    # Top co-occurring tags across pool
    top_tags: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class PoolHints:
    """Optimization hints extracted from analyzing the recall pool.

    These hints are passed to the ranking phase for adaptive scoring
    and to the recall manager for potential query refinement.
    """

    # Content signals
    content_types: ContentTypeDistribution = field(
        default_factory=ContentTypeDistribution
    )
    title_keywords: list[tuple[str, int]] = field(default_factory=list)

    # Score signals
    score_dist: ScoreDistribution = field(default_factory=ScoreDistribution)
    view_dist: ScoreDistribution = field(default_factory=ScoreDistribution)

    # Adaptive parameters for ranking
    suggested_title_bonus: float = 0.20
    noise_gate_ratio: float = 0.10

    # Feature summary (for logging/debugging)
    summary: str = ""


# ── CJK tokenizer helper ─────────────────────────────────────────────────

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+")
_STOP_TAGS = {
    "日常",
    "生活",
    "原创",
    "搞笑",
    "娱乐",
    "科技",
    "知识",
    "教程",
    "教学",
    "入门",
    "新手",
    "小白",
    "合集",
    "全集",
    "转载",
    "自制",
    "必剪创作",
    "必剪",
}


def _tokenize(text: str) -> list[str]:
    """Extract CJK and alphanumeric tokens."""
    return _CJK_RE.findall(text.lower()) if text else []


# ── Main optimizer ────────────────────────────────────────────────────────


class RecallPoolOptimizer:
    """Analyzes recall pool to produce optimization hints.

    Designed to be called after round-1 recall and before ranking.
    The hints can also guide round-2 supplementary recall.

    Example:
        optimizer = RecallPoolOptimizer()
        hints = optimizer.analyze(pool_hits, query="红警08")
        # Use hints.title_keywords for query expansion
    """

    def analyze(
        self,
        hits: list[dict],
        query: str,
        lane_tags: dict[str, set] = None,
    ) -> PoolHints:
        """Analyze the recall pool and produce optimization hints.

        Args:
            hits: Recall pool hit dicts.
            query: Original search query.
            lane_tags: Optional bvid → set of lane names.

        Returns:
            PoolHints with extracted features and adaptive parameters.
        """
        hints = PoolHints()

        if not hits:
            return hints

        query_lower = query.lower().strip() if query else ""
        q_tokens = set(_tokenize(query_lower))

        # 1. Score distribution
        scores = [h.get("score", 0) or 0 for h in hits]
        hints.score_dist = ScoreDistribution.from_scores([s for s in scores if s > 0])

        # 2. View distribution
        views = [max(dict_get(h, "stat.view", 0) or 0, 0) for h in hits]
        hints.view_dist = ScoreDistribution.from_scores(views)

        # 3. Title keyword extraction
        hints.title_keywords = self._extract_title_keywords(hits, q_tokens, top_k=20)

        # 4. Tag analysis
        hints.content_types = self._analyze_tags(hits, top_k=30)

        # 5. Adaptive parameters
        hints.noise_gate_ratio = self._compute_noise_gate(hints.score_dist)

        # 6. Summary
        hints.summary = (
            f"score=[{hints.score_dist.min_score:.1f}"
            f"..{hints.score_dist.max_score:.1f}], "
            f"top_tags={[t for t, _ in hints.content_types.top_tags[:5]]}"
        )

        return hints

    def _extract_title_keywords(
        self,
        hits: list[dict],
        q_tokens: set[str],
        top_k: int = 20,
    ) -> list[tuple[str, int]]:
        """Extract most frequent non-query keywords from titles.

        This identifies the dominant topics in the pool beyond the query
        itself, useful for query expansion and intent analysis.
        """
        keyword_counts: dict[str, int] = {}
        for hit in hits:
            title = (hit.get("title") or "").lower()
            tokens = _tokenize(title)
            for token in tokens:
                if token not in q_tokens and len(token) >= 2:
                    keyword_counts[token] = keyword_counts.get(token, 0) + 1

        # Sort by frequency
        sorted_kw = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_kw[:top_k]

    def _analyze_tags(
        self,
        hits: list[dict],
        top_k: int = 30,
    ) -> ContentTypeDistribution:
        """Analyze tag distribution across the pool."""
        tag_counts: dict[str, int] = {}
        for hit in hits:
            tags_str = hit.get("tags", "") or ""
            if isinstance(tags_str, str):
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            elif isinstance(tags_str, list):
                tags = tags_str
            else:
                tags = []

            for tag in tags:
                tag_lower = tag.lower()
                if tag_lower not in _STOP_TAGS and len(tag_lower) >= 2:
                    tag_counts[tag_lower] = tag_counts.get(tag_lower, 0) + 1

        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

        dist = ContentTypeDistribution()
        dist.top_tags = sorted_tags[:top_k]
        if sorted_tags:
            dist.dominant_category = sorted_tags[0][0]

        return dist

    @staticmethod
    def _compute_noise_gate(score_dist: ScoreDistribution) -> float:
        """Compute adaptive noise gate ratio based on score spread.

        Wide spread → higher gate (more noise to remove).
        Narrow spread → lower gate (scores are clustered).
        """
        if score_dist.count == 0 or score_dist.max_score <= 0:
            return 0.10

        # Coefficient of variation: std / mean
        cv = score_dist.std_score / max(score_dist.mean_score, 0.01)
        if cv > 1.5:
            return 0.15  # Very spread → stricter gate
        elif cv > 0.8:
            return 0.12
        else:
            return 0.08  # Tight cluster → lenient gate
