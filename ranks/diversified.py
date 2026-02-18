"""
Diversified Slot-Based Ranker

Instead of fusing all signals into one continuous score (which causes
homogeneous top-N results), this ranker uses a three-phase approach:

Phase 1 — Headline selection (top 3):
    Composite "headline quality" score balancing relevance, quality, and
    recency (weights from HEADLINE_WEIGHTS, relevance dominates at 0.55).
    Minimum relevance threshold (HEADLINE_MIN_RELEVANCE) prevents
    low-relevance docs from occupying headline positions.

Phase 2 — Relevance-gated diversified slot allocation (positions 4-10):
    Allocates slots to dimensions (relevance, quality, recency, popularity)
    with a critical improvement: **relevance gating**. Each dimension score
    is multiplied by a relevance factor derived from SLOT_RELEVANCE_DECAY_*
    constants. This means a doc can only score high in a dimension slot if
    it is ALSO relevant to the query. An irrelevant viral video cannot
    take a "popularity" slot.

    Graduated threshold relaxation:
    - First: only docs with relevance >= SLOT_MIN_RELEVANCE
    - If too few: relax to relevance >= SLOT_MIN_RELEVANCE * 0.5
    - If still too few: allow all

Phase 3 — Fused scoring (beyond top-10):
    Remaining positions filled by weighted combination of all dimension
    scores, with relevance dominating (0.50 weight).

Title-match bonus:
    Docs tagged with `_title_matched` (from recall title_match lane) receive
    TITLE_MATCH_BONUS added to their relevance_score during dimension scoring.
    This ensures docs whose titles match the query are strongly preferred.

Scoring notes:
- Popularity uses log-scale normalization (power-law distribution).
- Short-duration videos (<30s) get a quality penalty to avoid clickbait.
- Slot presets: balanced (5 slots), relevance (4), quality (5).
"""

import math
import re
import time as _time
from typing import Literal

from tclogger import dict_get

from ranks.scorers import StatsScorer, PubdateScorer
from ranks.constants import (
    RANK_TOP_K,
    RANK_PREFER_TYPE,
    RANK_PREFER,
    DIVERSIFIED_FUSED_WEIGHTS,
    HEADLINE_TOP_N,
    HEADLINE_WEIGHTS,
    HEADLINE_MIN_RELEVANCE,
    SLOT_MIN_RELEVANCE,
    RANK_SHORT_DURATION_THRESHOLD,
    RANK_SHORT_DURATION_PENALTY,
    RANK_VERY_SHORT_DURATION_THRESHOLD,
    RANK_VERY_SHORT_DURATION_PENALTY,
    RANK_SHORT_TITLE_THRESHOLD,
    RANK_SHORT_TITLE_PENALTY,
    RANK_CONTENT_DEPTH_MIN_FACTOR,
    RANK_CONTENT_DEPTH_NORM_LENGTH,
    RANK_HEADLINE_MIN_DURATION,
    RANK_SLOT_MIN_DURATION,
    RANK_LOW_ENGAGEMENT_THRESHOLD,
    RANK_LOW_ENGAGEMENT_PENALTY,
    TITLE_MATCH_BONUS,
    OWNER_MATCH_BONUS,
    RANK_NO_TITLE_KEYWORD_PENALTY,
    SLOT_RELEVANCE_DECAY_THRESHOLD,
    SLOT_RELEVANCE_DECAY_POWER,
    SLOT_QUALITY_TIEBREAKER,
)

# Slot allocation presets
# NOTE: These specify slots for positions AFTER the headline top-N.
# E.g., if HEADLINE_TOP_N=3, "balanced" allocates 7 more slots (3+7=10).
# Reduced from previous totals to leave room for fused scoring:
# 2R+1Q+1T+1P = 5 dimension slots + 2 fused = 7 total (+ 3 headline = 10)
SLOT_PRESETS = {
    "balanced": {
        "relevance": 2,
        "quality": 1,
        "recency": 1,
        "popularity": 1,
    },
    "prefer_relevance": {
        "relevance": 3,
        "quality": 1,
        "recency": 1,
        "popularity": 1,
    },
    "prefer_quality": {
        "relevance": 1,
        "quality": 2,
        "recency": 1,
        "popularity": 1,
    },
    "prefer_recency": {
        "relevance": 1,
        "quality": 1,
        "recency": 3,
        "popularity": 1,
    },
}

# Score field names for each dimension
DIMENSION_SCORE_FIELDS = {
    "relevance": "relevance_score",
    "quality": "quality_score",
    "recency": "recency_score",
    "popularity": "popularity_score",
}


class DiversifiedRanker:
    """Three-phase diversified ranking for comprehensive top-K results.

    Phase 1 — Headline quality (top 3):
        Picks the best candidates using a composite score that balances
        relevance with quality and recency. All candidates must pass a
        minimum relevance threshold. This ensures the most visible
        positions are occupied by results that are both relevant AND
        high quality, not just the highest BM25 score.

    Phase 2 — Slot allocation (positions 4-10):
        Ensures remaining top positions contain representatives from
        each dimension: relevant, popular, recent, high-quality.
        All candidates must pass SLOT_MIN_RELEVANCE threshold.

    Phase 3 — Fused scoring (beyond top 10):
        Uses weighted combination of all dimension scores, with
        relevance having the largest weight (0.40).

    The key insights:
    1. Pure relevance ranking often puts short-text, low-quality docs
       at the top because BM25 inflates their scores. Headline quality
       scoring fixes this.
    2. Without a relevance floor, viral but irrelevant videos could
       occupy "popularity" slots despite having no query relevance.
       SLOT_MIN_RELEVANCE prevents this.

    Example:
        >>> ranker = DiversifiedRanker()
        >>> result = ranker.diversified_rank_with_fused_fallback(
        ...     hits_info={"hits": candidates},
        ...     top_k=400,
        ...     prefer="balanced",
        ... )
        >>> # Top 3: best headline quality (relevant + quality + recent)
        >>> # Positions 4-10: diversified slots with relevance floor
        >>> # Beyond 10: fused scoring
        >>> len(result["hits"])  # Exactly 400 (if pool >= 400)
        400
    """

    def __init__(self):
        self.stats_scorer = StatsScorer()
        self.pubdate_scorer = PubdateScorer()

    @staticmethod
    def _compute_tag_affinity(hits: list[dict], top_k: int = 20) -> dict[str, float]:
        """Extract dominant tags from top-K docs for pseudo-relevance feedback.

        Analyzes the tags of the highest-scoring docs to identify
        "query-associated tags" — tags that frequently co-occur with
        relevant content. Returns a tag→weight map used to boost docs
        that share these tags in the fused scoring phase.

        This implements a form of pseudo-relevance feedback:
        - Round 1 produced the initial results
        - We extract features (tags) from the best results
        - We use those features to boost similar docs in ranking

        Args:
            hits: Scored hits (must have relevance_score set).
            top_k: Number of top docs to analyze.

        Returns:
            Dict mapping tag→affinity_weight (higher = more associated).
        """
        # Sort by relevance_score, tiebreak by quality_score
        sorted_hits = sorted(
            hits,
            key=lambda h: (
                h.get("relevance_score", 0),
                h.get("quality_score", 0),
            ),
            reverse=True,
        )
        top_docs = sorted_hits[:top_k]

        # Count tag frequency in top docs
        tag_freq: dict[str, int] = {}
        for hit in top_docs:
            tags_str = hit.get("tags", "") or ""
            for tag in tags_str.split(","):
                tag = tag.strip()
                if tag and len(tag) >= 2:
                    tag_freq[tag] = tag_freq.get(tag, 0) + 1

        if not tag_freq:
            return {}

        # Only keep tags that appear in >= 3 top docs (signal, not noise)
        min_freq = min(3, max(1, top_k // 5))
        affinity_tags = {
            tag: freq / top_k for tag, freq in tag_freq.items() if freq >= min_freq
        }

        return affinity_tags

    def _score_all_dimensions(
        self, hits: list[dict], now_ts: float = None, query: str = ""
    ) -> None:
        """Score each hit on all four dimensions + headline quality.

        Computes and stores dimension scores in each hit dict:
        - relevance_score: Normalized BM25/hybrid/rerank score [0, 1],
          with title-match and owner-match bonuses applied, and content
          depth penalty for ultra-short titles
        - quality_score: Stat quality from DocScorer [0, 1), with content
          penalties for short titles, short duration, low engagement
        - recency_score: Normalized time factor [0, 1]
        - popularity_score: Log-normalized view count [0, 1]
        - headline_score: Composite score for top-3 selection [0, 1]

        Content depth penalty (NEW):
        Penalizes relevance_score when the title adds almost no information
        beyond the query keywords. e.g., query='gta' and title='gta' → the
        title IS the query. The penalty reduces relevance proportionally to
        how much meaningful content exists beyond the query terms.

        Title-match bonus:
        Docs tagged with _title_matched=True get TITLE_MATCH_BONUS added
        to their normalized relevance score (capped at 1.0).

        Owner-match bonus:
        Docs tagged with _owner_matched=True get OWNER_MATCH_BONUS added
        to their normalized relevance score (capped at 1.0). This ensures
        docs from creators matching the query are strongly preferred.

        Content quality penalties:
        - Short titles (< RANK_SHORT_TITLE_THRESHOLD chars): reduce quality
        - Short duration (< 30s): reduce quality
        - Low engagement (< RANK_LOW_ENGAGEMENT_THRESHOLD views): reduce quality

        Args:
            hits: List of hit dicts to score.
            now_ts: Current timestamp for recency calculation.
            query: Original search query for content depth analysis.
        """
        if now_ts is None:
            now_ts = _time.time()

        # Collect raw values for normalization
        raw_scores = []
        raw_views = []
        for h in hits:
            raw_scores.append(
                h.get("rerank_score") or h.get("hybrid_score") or h.get("score") or 0
            )
            raw_views.append(dict_get(h, "stat.view", 0) or 0)

        # Max-normalization for relevance (BM25/cosine scores are well-distributed)
        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score <= 0:
            max_score = 1.0

        # Log-scale normalization for popularity (power-law distributed)
        log_views = [math.log1p(v) for v in raw_views]
        max_log_view = max(log_views) if log_views else 1.0
        if max_log_view <= 0:
            max_log_view = 1.0

        for i, hit in enumerate(hits):
            # Relevance: use best available score, normalized to [0, 1]
            rel_norm = min(raw_scores[i] / max_score, 1.0) if max_score > 0 else 0.0

            # Content depth penalty: BM25 inflates scores for ultra-short titles
            # that are essentially just the query keywords. Penalize relevance
            # proportionally to how much meaningful content the title adds
            # beyond the query terms.
            title = (hit.get("title") or "").strip()
            query_lower = query.lower().strip()
            title_lower = title.lower() if title else ""

            # Check how many query tokens appear in the title.
            # For CJK compound queries like "小红书推荐系统", use the longest
            # contiguous CJK substring match to handle partial matches.
            # E.g., "小红书推荐用户及冷启动" contains "小红书推荐" (5/6 CJK chars
            # from "小红书推荐系统", coverage 83%) → keyword match.
            title_has_query_keywords = False
            if title_lower and query_lower:
                # Extract CJK characters from query
                query_cjk = re.sub(r"[^\u4e00-\u9fff]", "", query_lower)
                if len(query_cjk) >= 4:
                    # Longest contiguous CJK substring match
                    max_match = 0
                    for start in range(len(query_cjk)):
                        for end in range(start + 2, len(query_cjk) + 1):
                            if query_cjk[start:end] in title_lower:
                                max_match = max(max_match, end - start)
                    title_has_query_keywords = max_match >= len(query_cjk) * 0.5
                else:
                    # Short/non-CJK query: full-string match
                    q_terms = [
                        t for t in re.split(r"[\s\-_,，。、/\\|]+", query_lower) if t
                    ]
                    title_has_query_keywords = any(t in title_lower for t in q_terms)

            if title and query_lower:
                # Remove query-like substrings from title to measure "depth"
                remaining = title_lower
                for term in re.split(r"[\s\-_,，。、/\\|]+", query_lower):
                    if term:
                        remaining = remaining.replace(term, "")
                # Strip non-alphanumeric debris
                meaningful_chars = len(re.sub(r"[^\u4e00-\u9fff\w]", "", remaining))
                depth_factor = min(
                    meaningful_chars / RANK_CONTENT_DEPTH_NORM_LENGTH, 1.0
                )
                depth_factor = max(depth_factor, RANK_CONTENT_DEPTH_MIN_FACTOR)
                rel_norm *= depth_factor

            # Title-keyword overlap penalty: if NO query keywords appear in the
            # title, the BM25 score comes from owner.name/desc/tags — the doc's
            # actual content is likely NOT about the query.
            if title_lower and query_lower and not title_has_query_keywords:
                rel_norm *= RANK_NO_TITLE_KEYWORD_PENALTY

            # Apply title-match bonus: docs with query in title get a boost
            if hit.get("_title_matched"):
                rel_norm = min(rel_norm + TITLE_MATCH_BONUS, 1.0)

            # Apply owner-match bonus: docs from creators matching query.
            # Only apply if title also contains query keywords — prevents
            # boosting irrelevant uploads (e.g., 喜羊羊 from UP "吴恩达大模型课程")
            if hit.get("_owner_matched") and title_has_query_keywords:
                rel_norm = min(rel_norm + OWNER_MATCH_BONUS, 1.0)

            hit["relevance_score"] = round(rel_norm, 4)

            # Quality: bounded [0, 1) from DocScorer
            stats = dict_get(hit, "stat", {})
            quality = self.stats_scorer.calc(stats)

            # Apply tiered short-duration penalty
            duration = hit.get("duration", 0) or 0
            if 0 < duration < RANK_VERY_SHORT_DURATION_THRESHOLD:
                quality *= RANK_VERY_SHORT_DURATION_PENALTY
            elif 0 < duration < RANK_SHORT_DURATION_THRESHOLD:
                quality *= RANK_SHORT_DURATION_PENALTY

            # Apply short-title penalty: very short titles indicate low-effort content
            title = hit.get("title", "") or ""
            if 0 < len(title) < RANK_SHORT_TITLE_THRESHOLD:
                quality *= RANK_SHORT_TITLE_PENALTY

            # Apply low-engagement penalty: few views indicate low-value content
            views = dict_get(hit, "stat.view", 0) or 0
            if views < RANK_LOW_ENGAGEMENT_THRESHOLD:
                quality *= RANK_LOW_ENGAGEMENT_PENALTY

            hit["quality_score"] = round(quality, 4)

            # Recency: normalized time factor [0, 1]
            pubdate = dict_get(hit, "pubdate", 0)
            time_factor = self.pubdate_scorer.calc(pubdate, now_ts=now_ts)
            hit["recency_score"] = round(self.pubdate_scorer.normalize(time_factor), 4)
            hit["time_factor"] = round(time_factor, 4)

            # Popularity: log-scale normalization
            hit["popularity_score"] = round(log_views[i] / max_log_view, 4)

            # Headline quality: composite score for top-3 selection
            w = HEADLINE_WEIGHTS
            headline = (
                w["relevance"] * hit["relevance_score"]
                + w["quality"] * hit["quality_score"]
                + w["recency"] * hit["recency_score"]
                + w["popularity"] * hit["popularity_score"]
            )
            # Owner match bonus in headline: boost owner-matched docs visibility
            # (only if the title contains query keywords — same guard as relevance)
            if hit.get("_owner_matched") and title_has_query_keywords:
                headline += 0.10
            hit["headline_score"] = round(headline, 4)

    def _select_headline_top_n(
        self,
        hits: list[dict],
        top_n: int = HEADLINE_TOP_N,
        min_relevance: float = HEADLINE_MIN_RELEVANCE,
    ) -> tuple[list[dict], set[str]]:
        """Select top-N headline positions using composite quality score.

        The headline positions (typically top 3) are the most visible
        results. Instead of purely using relevance (which favors short-text
        BM25 artifacts), this method picks candidates that are:
        - Highly relevant (must pass minimum relevance threshold)
        - High quality (good stats, reasonable duration)
        - Reasonably recent
        - NOT ultra-short content (duration >= RANK_HEADLINE_MIN_DURATION)

        The selection process:
        1. Filter candidates by minimum relevance AND minimum duration
        2. Sort by headline_score (composite of relevance + quality + recency)
        3. Among top candidates with similar headline scores, prefer
           those with higher relevance to break ties

        Args:
            hits: Scored hits (must have scores from _score_all_dimensions).
            top_n: Number of headline positions to fill.
            min_relevance: Minimum relevance_score to qualify for headline.

        Returns:
            Tuple of (selected headline hits, set of selected bvids).
        """
        # Filter candidates: must have reasonable relevance AND duration
        candidates = [
            h
            for h in hits
            if h.get("relevance_score", 0) >= min_relevance
            and (h.get("duration", 0) or 0) >= RANK_HEADLINE_MIN_DURATION
        ]

        if len(candidates) < top_n:
            # Fallback: relax duration requirement
            candidates = [
                h for h in hits if h.get("relevance_score", 0) >= min_relevance
            ]

        if not candidates:
            # Fallback: relax threshold to half, then to all
            candidates = [
                h for h in hits if h.get("relevance_score", 0) >= min_relevance * 0.5
            ]
            if not candidates:
                candidates = hits

        # Sort by headline quality score, then by relevance as tiebreaker
        candidates.sort(
            key=lambda h: (
                h.get("headline_score", 0),
                h.get("relevance_score", 0),
            ),
            reverse=True,
        )

        selected = []
        selected_bvids = set()
        for hit in candidates:
            if len(selected) >= top_n:
                break
            bvid = hit.get("bvid")
            if bvid and bvid not in selected_bvids:
                selected_bvids.add(bvid)
                hit["_slot_dimension"] = "headline"
                hit["_slot_order"] = len(selected)
                selected.append(hit)

        return selected, selected_bvids

    def _allocate_slots(
        self,
        hits: list[dict],
        slots: dict[str, int],
        top_k: int,
        min_relevance: float = SLOT_MIN_RELEVANCE,
    ) -> list[dict]:
        """Allocate slots from each dimension with relevance gating.

        Algorithm:
        1. Filter candidates by minimum relevance threshold
        2. For each dimension, compute relevance-gated dimension score:
           gated_score = dimension_score * relevance_factor
           This prevents irrelevant docs from occupying slots.
        3. Pick top-N items per dimension not already selected
        4. Fill remaining top_k slots with best fused score
        5. Assign rank_score for stable ordering

        The relevance gating ensures dimension slots are only filled by
        docs that are BOTH high on that dimension AND reasonably relevant.

        Args:
            hits: Scored hits (must have dimension scores).
            slots: Dict mapping dimension name to slot count.
            top_k: Total items to return.
            min_relevance: Minimum relevance_score for slot candidates.

        Returns:
            Ordered list of top_k hits with rank_score assigned.
        """
        selected_bvids: set = set()
        result: list[dict] = []
        slot_order = 0

        # Candidates must pass relevance floor for diversified slots
        eligible = [
            h
            for h in hits
            if h.get("relevance_score", 0) >= min_relevance
            and (h.get("duration", 0) or 0) >= RANK_SLOT_MIN_DURATION
        ]
        # If too few pass the floor, relax duration requirement
        if len(eligible) < top_k:
            eligible = [h for h in hits if h.get("relevance_score", 0) >= min_relevance]
        # If still too few, relax to half threshold
        if len(eligible) < top_k:
            eligible = [
                h for h in hits if h.get("relevance_score", 0) >= min_relevance * 0.5
            ]
        # If still too few, allow all
        if len(eligible) < top_k:
            eligible = hits

        # Phase 1: Pick top items from each dimension with relevance gating
        for dimension, count in slots.items():
            score_field = DIMENSION_SCORE_FIELDS.get(dimension, f"{dimension}_score")

            # Compute relevance-gated dimension score
            for h in eligible:
                dim_score = h.get(score_field, 0)
                rel_score = h.get("relevance_score", 0)

                if rel_score >= SLOT_RELEVANCE_DECAY_THRESHOLD:
                    relevance_factor = 1.0
                elif rel_score > 0:
                    ratio = rel_score / SLOT_RELEVANCE_DECAY_THRESHOLD
                    relevance_factor = ratio**SLOT_RELEVANCE_DECAY_POWER
                else:
                    relevance_factor = 0.0

                h[f"_gated_{dimension}"] = (
                    dim_score * relevance_factor
                    + SLOT_QUALITY_TIEBREAKER * h.get("quality_score", 0)
                )

            gated_field = f"_gated_{dimension}"
            sorted_hits = sorted(
                eligible, key=lambda h: h.get(gated_field, 0), reverse=True
            )

            added = 0
            for hit in sorted_hits:
                bvid = hit.get("bvid")
                if bvid and bvid not in selected_bvids and added < count:
                    selected_bvids.add(bvid)
                    hit["_slot_dimension"] = dimension
                    hit["_slot_order"] = slot_order
                    result.append(hit)
                    added += 1
                    slot_order += 1

        # Phase 2: Fill remaining slots with best overall fused score
        remaining = max(0, top_k - len(result))
        if remaining > 0:
            w = DIVERSIFIED_FUSED_WEIGHTS
            for hit in hits:
                if hit.get("bvid") not in selected_bvids:
                    hit["_fused_score"] = (
                        hit.get("relevance_score", 0) * w["relevance"]
                        + hit.get("quality_score", 0) * w["quality"]
                        + hit.get("recency_score", 0) * w["recency"]
                        + hit.get("popularity_score", 0) * w["popularity"]
                    )

            fused_sorted = sorted(
                [h for h in hits if h.get("bvid") not in selected_bvids],
                key=lambda h: h.get("_fused_score", 0),
                reverse=True,
            )

            for hit in fused_sorted[:remaining]:
                bvid = hit.get("bvid")
                if bvid:
                    selected_bvids.add(bvid)
                    hit["_slot_dimension"] = "fused"
                    hit["_slot_order"] = slot_order
                    result.append(hit)
                    slot_order += 1

        # Phase 3: Assign rank_score for stable ordering
        for i, hit in enumerate(result):
            hit["rank_score"] = round(1.0 - (i / max(len(result), 1)), 6)

        # Clean up temporary gated score fields
        for h in eligible:
            for dimension in slots:
                h.pop(f"_gated_{dimension}", None)

        return result

    def diversified_rank(
        self,
        hits_info: dict,
        top_k: int = RANK_TOP_K,
        prefer: RANK_PREFER_TYPE = RANK_PREFER,
        slot_preset: str = None,
        custom_slots: dict = None,
        query: str = "",
    ) -> dict:
        """Rank hits with diversified slot allocation.

        Args:
            hits_info: Dict with "hits" list.
            top_k: Total results to return.
            prefer: Preference mode (maps to slot preset).
            slot_preset: Override slot preset name.
            custom_slots: Custom slot allocation dict.
            query: Original search query for content depth analysis.

        Returns:
            hits_info with diversified-ranked hits.
        """
        hits = hits_info.get("hits", [])
        if not hits:
            hits_info["rank_method"] = "diversified"
            return hits_info

        now_ts = _time.time()

        # Score all dimensions
        self._score_all_dimensions(hits, now_ts=now_ts, query=query)

        # Determine slot allocation
        if custom_slots:
            slots = custom_slots
        else:
            preset_name = slot_preset or prefer or "balanced"
            slots = SLOT_PRESETS.get(preset_name, SLOT_PRESETS["balanced"])

        # Allocate slots and rank
        ranked_hits = self._allocate_slots(hits, slots, top_k)

        hits_info["hits"] = ranked_hits
        hits_info["return_hits"] = len(ranked_hits)
        hits_info["rank_method"] = "diversified"
        hits_info["slot_allocation"] = slots

        # Summary info
        dimension_counts = {}
        for hit in ranked_hits:
            dim = hit.get("_slot_dimension", "unknown")
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        hits_info["dimension_distribution"] = dimension_counts

        return hits_info

    def diversified_rank_with_fused_fallback(
        self,
        hits_info: dict,
        top_k: int = RANK_TOP_K,
        prefer: RANK_PREFER_TYPE = RANK_PREFER,
        diversify_top_n: int = 10,
        headline_top_n: int = HEADLINE_TOP_N,
        query: str = "",
    ) -> dict:
        """Three-phase ranking: headline → diversified → fused.

        Phase 1 — Headline quality (top 3):
            Picks the best candidates using composite headline_score that
            balances relevance + quality + recency. All candidates must
            pass a minimum relevance threshold AND have sufficient duration.

        Phase 2 — Diversified slot allocation (positions 4-10):
            Fills remaining diversified positions with dimension representatives
            (relevance, quality, recency, popularity). All candidates must
            pass SLOT_MIN_RELEVANCE threshold and duration floor.

        Phase 3 — Fused scoring (beyond top 10):
            Remaining positions use continuous fused scoring with relevance
            dominating the weight distribution.

        IMPORTANT: This method guarantees returning exactly min(top_k, len(hits))
        documents. If the pool has >= top_k docs, exactly top_k are returned.

        Args:
            hits_info: Dict with "hits" list.
            top_k: Total results to return (guaranteed if pool is large enough).
            prefer: Preference mode.
            diversify_top_n: Total items for diversification (phases 1+2).
            headline_top_n: How many items to pick by headline quality.
            query: Original search query for content depth analysis.

        Returns:
            hits_info with three-phase ranked hits.
        """
        hits = hits_info.get("hits", [])
        if not hits:
            hits_info["rank_method"] = "diversified"
            return hits_info

        now_ts = _time.time()

        # Score all dimensions (relevance, quality, recency, popularity, headline)
        self._score_all_dimensions(hits, now_ts=now_ts, query=query)

        # Pseudo-relevance feedback: extract dominant tags from top docs
        # to boost topically coherent results in phase 3 fused scoring.
        affinity_tags = self._compute_tag_affinity(hits, top_k=20)

        # Phase 1: Headline selection (top 3)
        headline_hits, headline_bvids = self._select_headline_top_n(
            hits, top_n=headline_top_n
        )

        # Phase 2: Diversified slot allocation for remaining positions
        preset_name = prefer or "balanced"
        slots = SLOT_PRESETS.get(preset_name, SLOT_PRESETS["balanced"])

        remaining_for_slots = max(0, diversify_top_n - len(headline_hits))
        slot_hits = []
        all_selected_bvids = set(headline_bvids)

        if remaining_for_slots > 0:
            slot_hits = self._allocate_slots(
                [h for h in hits if h.get("bvid") not in all_selected_bvids],
                slots,
                remaining_for_slots,
            )
            for hit in slot_hits:
                bvid = hit.get("bvid")
                if bvid:
                    all_selected_bvids.add(bvid)

        top_hits = headline_hits + slot_hits

        # Phase 3: Fused score ranking for the rest
        # Includes tag-affinity bonus from pseudo-relevance feedback.
        w = DIVERSIFIED_FUSED_WEIGHTS
        remaining_hits = [h for h in hits if h.get("bvid") not in all_selected_bvids]
        for hit in remaining_hits:
            base_fused = (
                hit.get("relevance_score", 0) * w["relevance"]
                + hit.get("quality_score", 0) * w["quality"]
                + hit.get("recency_score", 0) * w["recency"]
                + hit.get("popularity_score", 0) * w["popularity"]
            )
            # Tag affinity: small bonus for docs sharing tags with top docs
            if affinity_tags:
                tags_str = hit.get("tags", "") or ""
                doc_tags = {t.strip() for t in tags_str.split(",") if t.strip()}
                tag_overlap = sum(affinity_tags.get(t, 0) for t in doc_tags)
                # Cap at 0.10 to prevent tag-rich docs from dominating
                tag_bonus = min(tag_overlap * 0.05, 0.10)
                base_fused += tag_bonus
            hit["rank_score"] = round(base_fused, 6)
            hit["_slot_dimension"] = "fused"

        remaining_hits.sort(key=lambda h: h.get("rank_score", 0), reverse=True)

        # Combine: top-N diversified + rest by fused score
        # Guarantee exactly min(top_k, total_pool) results
        target_count = min(top_k, len(hits))
        fused_quota = max(0, target_count - len(top_hits))
        final_hits = top_hits + remaining_hits[:fused_quota]

        # Assign rank_score for stable ordering in top section
        for i, hit in enumerate(final_hits):
            if i < len(top_hits):
                hit["rank_score"] = round(1.0 - (i / max(len(final_hits), 1)), 6)

        hits_info["hits"] = final_hits
        hits_info["return_hits"] = len(final_hits)
        hits_info["rank_method"] = "diversified"
        hits_info["diversified_top_n"] = diversify_top_n
        hits_info["headline_top_n"] = len(headline_hits)
        hits_info["slot_allocation"] = slots

        # Summary info
        dimension_counts = {}
        for hit in final_hits[:diversify_top_n]:
            dim = hit.get("_slot_dimension", "unknown")
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        hits_info["dimension_distribution"] = dimension_counts

        return hits_info
