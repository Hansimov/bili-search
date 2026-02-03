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
- Passage truncation: Long passages are truncated to save memory/bandwidth
- Efficient text extraction: Pre-compute and cache field access
- Memory management: Clear intermediate data structures promptly
"""

import re
import time
from tclogger import logger, dict_get

from converters.embed.embed_client import TextEmbedClient, get_embed_client

# Performance tuning constants
MAX_PASSAGE_LENGTH = 4096  # Truncate passages longer than this (chars)
MAX_FIELD_LENGTH = 150  # Max length per text field before truncation
RERANK_TIMEOUT = 30  # Timeout in seconds for rerank operation


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


def extract_keywords(query: str) -> list[str]:
    """Extract keywords from query string.

    Performs simple tokenization to extract meaningful keywords.
    Filters out DSL expressions (e.g., q=v, date=2024).

    Args:
        query: Query string, possibly with DSL expressions.

    Returns:
        List of keywords for matching.
    """
    # Remove patterns like q=xxx, date=xxx, etc.
    cleaned = re.sub(r"\b\w+=[^\s]+", "", query)

    # Split by whitespace and punctuation
    tokens = re.split(r"[\s,;，；、]+", cleaned)

    # Filter empty and very short tokens
    keywords = [t.strip() for t in tokens if t.strip() and len(t.strip()) >= 2]

    return keywords


def _get_nested_field(hit: dict, field: str) -> str:
    """Extract a nested field value from hit dict.

    Args:
        hit: Hit document dict.
        field: Field name, can be nested like "owner.name".

    Returns:
        Field value as string, or empty string if not found.
    """
    value = hit
    for part in field.split("."):
        if isinstance(value, dict):
            value = value.get(part, "")
        else:
            return ""
    if isinstance(value, str):
        return value.strip()
    return ""


def compute_passage(
    hit: dict,
    text_fields: list[str],
    max_passage_len: int = MAX_PASSAGE_LENGTH,
    max_field_len: int = MAX_FIELD_LENGTH,
) -> str:
    """Extract and combine text from a hit for embedding.

    Optimized for minimal memory allocation and fast execution.

    Args:
        hit: Hit document with text fields.
        text_fields: Fields to extract text from.
        max_passage_len: Maximum total passage length.
        max_field_len: Maximum length per field.

    Returns:
        Combined text string for embedding.
    """
    parts = []
    total_len = 0

    for field in text_fields:
        value = _get_nested_field(hit, field)
        if not value:
            continue

        # Truncate field if needed
        if len(value) > max_field_len:
            value = value[:max_field_len]

        parts.append(value)
        total_len += len(value) + 1  # +1 for space

        # Early exit if we have enough text
        if total_len >= max_passage_len:
            break

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

    Example:
        reranker = EmbeddingReranker()
        reranked_hits = reranker.rerank(
            query="基洛夫",
            hits=initial_hits,
            text_fields=["title", "tags", "desc"],
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
        text_fields: list[str] = None,
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

        Args:
            query: Original query string.
            hits: List of hit documents to rerank.
            text_fields: Fields to use for computing document text.
                        Defaults to ["title", "tags", "desc"].
            keyword_boost: Boost factor when keywords match in tags/desc.
            title_keyword_boost: Boost factor when keywords match in title.
            max_rerank: Maximum number of hits to rerank.
            score_field: Field name to store the rerank score.
            verbose: Enable verbose logging.

        Returns:
            Tuple of (reranked hits, perf_info dict with timing details).
        """
        if text_fields is None:
            text_fields = ["title", "tags", "desc"]

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

        # Limit hits to rerank
        hits_to_rerank = hits[:max_rerank]
        remaining_hits = hits[max_rerank:]
        perf_info["reranked_count"] = len(hits_to_rerank)

        # Step 1: Prepare passages efficiently
        step_start = time.perf_counter()

        passages = []
        valid_indices = []

        for i, hit in enumerate(hits_to_rerank):
            passage = compute_passage(hit, text_fields)
            passages.append(passage)
            if passage:
                valid_indices.append(i)

        perf_info["passage_prep_ms"] = round(
            (time.perf_counter() - step_start) * 1000, 2
        )
        perf_info["valid_passages"] = len(valid_indices)

        if not valid_indices:
            logger.warn("× No valid passages to rerank")
            return hits, perf_info

        # Extract only valid passages for API call
        valid_passages = [passages[i] for i in valid_indices]

        # Step 2: Call tfmx.rerank() - SINGLE call for all passages
        # This is critical for correct global ranking
        step_start = time.perf_counter()

        rankings = self.embed_client.rerank(query, valid_passages)

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

        # Extract keywords once for all hits
        keywords = extract_keywords(query)

        # Step 3: Apply keyword boosts and assign scores
        step_start = time.perf_counter()

        for i, hit in enumerate(hits_to_rerank):
            # Get base similarity score
            similarity = similarity_scores.get(i, 0.1)

            # Compute keyword boost
            boost = 1.0

            # Title match (highest priority)
            title = hit.get("title", "")
            title_match, title_count = check_keyword_match(title, keywords)
            if title_match:
                boost *= 1 + title_keyword_boost * title_count

            # Tags match
            tags = hit.get("tags", "")
            tags_match, tags_count = check_keyword_match(tags, keywords)
            if tags_match:
                boost *= 1 + keyword_boost * tags_count * 0.8

            # Desc match (lower priority)
            desc = hit.get("desc", "")
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

        perf_info["keyword_scoring_ms"] = round(
            (time.perf_counter() - step_start) * 1000, 2
        )

        # Step 4: Sort by rerank score
        step_start = time.perf_counter()
        hits_to_rerank.sort(key=lambda x: x.get(score_field, 0), reverse=True)
        perf_info["sorting_ms"] = round((time.perf_counter() - step_start) * 1000, 2)

        # Mark remaining hits with zero scores
        for hit in remaining_hits:
            hit[score_field] = 0.0
            hit["rank_score"] = 0.0
            hit["score"] = 0.0

        perf_info["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

        if verbose:
            logger.mesg(f"  Rerank perf: {perf_info}")

        # Clear intermediate data to help GC
        del passages
        del valid_passages
        del similarity_scores

        return hits_to_rerank + remaining_hits, perf_info


# Singleton instance
_reranker: EmbeddingReranker = None


def get_reranker() -> EmbeddingReranker:
    """Get or create a singleton EmbeddingReranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = EmbeddingReranker()
    return _reranker
