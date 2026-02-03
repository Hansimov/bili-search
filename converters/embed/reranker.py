"""Embedding-based Reranker for KNN Search Results

This module provides a reranker that uses tfmx.rerank() to refine
initial KNN search results obtained via LSH bit vectors.

The problem with LSH bit vectors:
- LSH compresses high-dimensional float embeddings into bit vectors
- This introduces quantization noise and reduces precision
- Similar concepts become nearly indistinguishable in hamming distance

The solution:
- Use bit vector KNN for fast initial recall (coarse filtering)
- Use tfmx.rerank() for precise cosine similarity (fine ranking)
- Optionally boost results that contain exact keyword matches

Key improvement over old implementation:
- Old: 1 network call for query embedding + 1 batch call for doc embeddings
- New: 1 network call total via tfmx.rerank() (server-side computation)

Performance optimizations:
- Passage construction: Following TextDocItem pattern for consistent text building
- Efficient text extraction: Using dict_get for nested field access
- Memory management: Clear intermediate data structures promptly
- Keyword extraction: Use DSL parser instead of regex for accuracy
"""

import time
from tclogger import logger, dict_get

from converters.embed.embed_client import TextEmbedClient, get_embed_client

# Performance tuning constants
MAX_PASSAGE_LENGTH = 4096  # Truncate passages longer than this (chars)
RERANK_TIMEOUT = 30  # Timeout in seconds for rerank operation

# Score constants for non-reranked hits
# Non-reranked hits get a score penalty to ensure they rank below reranked ones
NON_RERANKED_SCORE_PENALTY = 0.01  # Score assigned to non-reranked hits


def check_keyword_match(
    text: str,
    keywords: list[str],
    case_sensitive: bool = False,
) -> tuple[bool, int]:
    """Check if text contains any of the keywords.

    Args:
        text: Text to search in.
        keywords: List of keywords to look for.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        Tuple of (has_match, match_count).
    """
    if not text or not keywords:
        return False, 0

    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]

    match_count = sum(1 for k in keywords if k in text)
    return match_count > 0, match_count


def extract_keywords_from_expr_tree(expr_tree) -> list[str]:
    """Extract keywords from DSL expression tree.

    Uses the DSL parser's word_expr nodes to extract actual search keywords,
    avoiding false positives from filter expressions like q=v, date=2024.

    Args:
        expr_tree: DslExprNode tree from DSL parser.

    Returns:
        List of keyword strings for matching.
    """
    if expr_tree is None:
        return []

    keywords = []
    word_expr_nodes = expr_tree.find_all_childs_with_key("word_expr")

    for word_expr_node in word_expr_nodes:
        # Get word_val_single nodes (the actual word values)
        word_val_nodes = word_expr_node.find_all_childs_with_key("word_val_single")
        for word_val_node in word_val_nodes:
            # Skip date-formatted words (they're filter conditions, not keywords)
            if word_val_node.extras.get("is_date_format", False):
                continue
            word = word_val_node.get_deepest_node_value()
            if word and len(word.strip()) >= 2:
                keywords.append(word.strip())

    return keywords


def compute_passage(
    hit: dict,
    max_passage_len: int = MAX_PASSAGE_LENGTH,
) -> str:
    """Extract and combine text from a hit for embedding.

    Following TextDocItem.build_sentence() pattern for consistent text building.
    Uses dict_get for safe nested field access.

    Args:
        hit: Hit document with text fields.
        max_passage_len: Maximum total passage length.

    Returns:
        Combined text string for embedding.
    """
    parts = []

    # owner.name - with semantic tag like TextDocItem
    owner_name = dict_get(hit, "owner.name", default="", sep=".")
    if isinstance(owner_name, str):
        owner_name = owner_name.strip()
        if owner_name:
            parts.append(f"<UP主>{owner_name}</UP主>")

    # title - core content
    title = dict_get(hit, "title", default="", sep=".")
    if isinstance(title, str):
        title = title.strip()
        if title:
            parts.append(title)

    # tags - wrapped in parentheses like TextDocItem
    tags = dict_get(hit, "tags", default="", sep=".")
    if isinstance(tags, str):
        tags = tags.strip()
        if tags:
            parts.append(f"({tags})")

    # desc - skip if just "-"
    desc = dict_get(hit, "desc", default="", sep=".")
    if isinstance(desc, str):
        desc = desc.strip()
        if desc and desc != "-":
            parts.append(desc)

    if not parts:
        # Fallback to bvid
        bvid = hit.get("bvid", "")
        return f"video {bvid}" if bvid else ""

    combined = " ".join(parts)

    # Final truncation
    if len(combined) > max_passage_len:
        combined = combined[:max_passage_len]

    return combined


class EmbeddingReranker:
    """Reranker using tfmx.rerank() for precise similarity calculation.

    This reranker improves recall@k for vector search by:
    1. Taking initial KNN results (from LSH bit vector search)
    2. Computing precise cosine similarity via tfmx.rerank() (single call)
    3. Optionally boosting results containing exact keyword matches
    4. Returning reranked results with improved precision

    IMPORTANT: tfmx.rerank() must process all passages in a single call
    to compute correct global rankings. Batch splitting would break
    ranking consistency.

    Score normalization:
    - Reranked hits: cosine_similarity * keyword_boost (typically 0.3-1.5)
    - Non-reranked hits: assigned NON_RERANKED_SCORE_PENALTY (0.01)
    - This ensures reranked hits always rank above non-reranked ones

    Example:
        reranker = EmbeddingReranker()
        reranked_hits = reranker.rerank(
            query="基洛夫",
            hits=initial_hits,
            keywords=["基洛夫"],  # From DSL parser
            keyword_boost=2.0,
        )
    """

    def __init__(
        self,
        embed_client: TextEmbedClient = None,
        lazy_init: bool = True,
    ):
        """Initialize the reranker.

        Args:
            embed_client: Embedding client to use. If None, uses singleton.
            lazy_init: If True, delay client initialization until first use.
        """
        self._embed_client = embed_client
        self._lazy_init = lazy_init

    @property
    def embed_client(self) -> TextEmbedClient:
        """Get or create embed client."""
        if self._embed_client is None:
            self._embed_client = get_embed_client()
        return self._embed_client

    def is_available(self) -> bool:
        """Check if the reranker is available."""
        return self.embed_client.is_available()

    def rerank(
        self,
        query: str,
        hits: list[dict],
        keywords: list[str] = None,
        expr_tree=None,
        keyword_boost: float = 1.5,
        title_keyword_boost: float = 2.0,
        max_rerank: int = 1000,
        score_field: str = "rerank_score",
        verbose: bool = False,
    ) -> tuple[list[dict], dict]:
        """Rerank hits using tfmx.rerank() and optional keyword matching.

        The reranking formula:
            final_score = cosine_similarity * keyword_boost_multiplier

        For title matches, an additional boost is applied:
            boost *= (1 + title_keyword_boost * match_count)

        IMPORTANT: All passages must be processed in a single tfmx.rerank()
        call to ensure correct global ranking. Batch splitting is NOT used
        as it would break ranking consistency.

        Score normalization:
        - Reranked hits get scores from 0.0 to ~3.0 (cosine * boost)
        - Non-reranked hits (beyond max_rerank) get NON_RERANKED_SCORE_PENALTY
        - This ensures proper ordering: all reranked > all non-reranked

        Args:
            query: Original query string (used for embedding).
            hits: List of hit documents to rerank.
            keywords: Pre-extracted keywords from DSL parser. If None, will
                     try to extract from expr_tree, otherwise uses query.
            expr_tree: DslExprNode tree for keyword extraction. Optional.
            keyword_boost: Boost factor when keywords match in tags/desc.
            title_keyword_boost: Boost factor when keywords match in title.
            max_rerank: Maximum number of hits to rerank (default 1000).
            score_field: Field name to store the rerank score.
            verbose: Enable verbose logging.

        Returns:
            Tuple of (reranked hits, perf_info dict with timing details).
        """
        perf_info = {
            "total_ms": 0,
            "passage_prep_ms": 0,
            "rerank_call_ms": 0,
            "keyword_scoring_ms": 0,
            "sorting_ms": 0,
            "hits_count": len(hits),
            "reranked_count": 0,
            "valid_passages": 0,
        }

        total_start = time.perf_counter()

        if not hits:
            return hits, perf_info

        if not self.is_available():
            logger.warn("× Reranker not available, returning original order")
            return hits, perf_info

        # Extract keywords: prefer provided keywords > expr_tree > query
        if keywords is None:
            if expr_tree is not None:
                keywords = extract_keywords_from_expr_tree(expr_tree)
            else:
                # Fallback: simple split on whitespace
                keywords = [w.strip() for w in query.split() if len(w.strip()) >= 2]

        # Limit hits to rerank
        hits_to_rerank = hits[:max_rerank]
        remaining_hits = hits[max_rerank:]
        perf_info["reranked_count"] = len(hits_to_rerank)

        # Step 1: Prepare passages - build list directly without intermediate storage
        step_start = time.perf_counter()

        valid_indices = []
        valid_passages = []

        for i, hit in enumerate(hits_to_rerank):
            passage = compute_passage(hit)
            if passage:
                valid_indices.append(i)
                valid_passages.append(passage)

        perf_info["passage_prep_ms"] = round(
            (time.perf_counter() - step_start) * 1000, 2
        )
        perf_info["valid_passages"] = len(valid_indices)

        if not valid_indices:
            logger.warn("× No valid passages to rerank")
            return hits, perf_info

        # Step 2: Call tfmx.rerank() - SINGLE call for all passages
        # This is critical for correct global ranking
        step_start = time.perf_counter()

        try:
            rankings = self.embed_client.rerank(query, valid_passages)
        finally:
            # Immediately release passage memory after API call
            del valid_passages

        perf_info["rerank_call_ms"] = round(
            (time.perf_counter() - step_start) * 1000, 2
        )

        if not rankings:
            logger.warn("× Rerank call returned no results")
            return hits, perf_info

        # Build score map: original_hit_idx -> similarity_score
        similarity_scores = {}
        for valid_list_idx, original_idx in enumerate(valid_indices):
            if valid_list_idx < len(rankings):
                _, sim_score = rankings[valid_list_idx]
                similarity_scores[original_idx] = sim_score

        # Clear rankings to release memory
        del rankings
        del valid_indices

        # Step 3: Apply keyword boosts and assign scores
        step_start = time.perf_counter()

        for i, hit in enumerate(hits_to_rerank):
            # Get base similarity score
            similarity = similarity_scores.get(i, 0.1)

            # Compute keyword boost
            boost = 1.0

            # Title match (highest priority)
            title = dict_get(hit, "title", default="", sep=".")
            if isinstance(title, str):
                title_match, title_count = check_keyword_match(title, keywords)
                if title_match:
                    boost *= 1 + title_keyword_boost * title_count

            # Tags match
            tags = dict_get(hit, "tags", default="", sep=".")
            if isinstance(tags, str):
                tags_match, tags_count = check_keyword_match(tags, keywords)
                if tags_match:
                    boost *= 1 + keyword_boost * tags_count * 0.8

            # Desc match (lower priority)
            desc = dict_get(hit, "desc", default="", sep=".")
            if isinstance(desc, str):
                desc_match, desc_count = check_keyword_match(desc, keywords)
                if desc_match:
                    boost *= 1 + keyword_boost * desc_count * 0.3

            # Final score
            rerank_score = similarity * boost

            # Update hit with scores (minimal fields to reduce memory)
            hit[score_field] = round(rerank_score, 6)
            hit["cosine_similarity"] = round(similarity, 6)
            hit["keyword_boost"] = round(boost, 4)
            hit["rank_score"] = round(rerank_score, 6)
            hit["score"] = round(rerank_score, 6)
            hit["reranked"] = True  # Mark as reranked

        # Clear similarity scores
        del similarity_scores

        perf_info["keyword_scoring_ms"] = round(
            (time.perf_counter() - step_start) * 1000, 2
        )

        # Step 4: Sort by rerank score
        step_start = time.perf_counter()
        hits_to_rerank.sort(key=lambda x: x.get(score_field, 0), reverse=True)
        perf_info["sorting_ms"] = round((time.perf_counter() - step_start) * 1000, 2)

        # Mark remaining hits with penalty scores
        # CRITICAL: Non-reranked hits must have lower scores than all reranked hits
        # to avoid incorrect ordering when merged
        for hit in remaining_hits:
            hit[score_field] = NON_RERANKED_SCORE_PENALTY
            hit["rank_score"] = NON_RERANKED_SCORE_PENALTY
            hit["score"] = NON_RERANKED_SCORE_PENALTY
            hit["cosine_similarity"] = 0.0
            hit["keyword_boost"] = 0.0
            hit["reranked"] = False  # Mark as not reranked

        perf_info["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

        if verbose:
            logger.mesg(f"  Rerank perf: {perf_info}")

        return hits_to_rerank + remaining_hits, perf_info


# Singleton instance
_reranker: EmbeddingReranker = None


def get_reranker() -> EmbeddingReranker:
    """Get or create a singleton EmbeddingReranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = EmbeddingReranker()
    return _reranker
