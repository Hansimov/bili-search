"""Embedding-based Reranker for KNN Search Results

This module provides a reranker that uses float embeddings to refine
initial KNN search results obtained via LSH bit vectors.

The problem with LSH bit vectors:
- LSH compresses high-dimensional float embeddings into bit vectors
- This introduces quantization noise and reduces precision
- Similar concepts (e.g., "基洛夫", "斯大林", "朱可夫") become nearly indistinguishable
- Typical symptom: all scores cluster around 0.7x with poor differentiation

The solution:
- Use bit vector KNN for fast initial recall (coarse filtering)
- Use float embeddings for precise reranking (fine ranking)
- Optionally boost results that contain exact keyword matches
"""

import math
import time
from typing import Union
from tclogger import logger

from converters.embed.embed_client import TextEmbedSearchClient, get_embed_client


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector.

    Returns:
        Cosine similarity score in range [-1, 1].
        Returns 0.0 if either vector is empty or zero-norm.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


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


class EmbeddingReranker:
    """Reranker using float embeddings for precise similarity calculation.

    This reranker is designed to improve recall@k for vector search:
    1. Takes initial KNN results (from LSH bit vector search)
    2. Computes precise cosine similarity using float embeddings
    3. Optionally boosts results containing exact keyword matches
    4. Returns reranked results with improved precision

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
        embed_client: TextEmbedSearchClient = None,
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
    def embed_client(self) -> TextEmbedSearchClient:
        """Get or create embed client."""
        if self._embed_client is None:
            self._embed_client = get_embed_client()
        return self._embed_client

    def is_available(self) -> bool:
        """Check if the reranker is available."""
        return self.embed_client.is_available()

    def compute_text_for_hit(
        self,
        hit: dict,
        text_fields: list[str] = None,
    ) -> str:
        """Extract and combine text from a hit for embedding.

        Args:
            hit: Hit document with text fields.
            text_fields: Fields to extract text from.
                        Defaults to ["title", "tags", "desc"].

        Returns:
            Combined text string for embedding.
            Returns a placeholder if all fields are empty to avoid TEI errors.
        """
        if text_fields is None:
            text_fields = ["title", "tags", "desc"]

        texts = []
        for field in text_fields:
            value = hit.get(field, "")
            if value and isinstance(value, str) and value.strip():
                texts.append(value.strip())

        combined = " ".join(texts)

        # If no text found, use bvid as fallback to avoid empty input
        # Empty strings cause TEI service errors
        if not combined.strip():
            bvid = hit.get("bvid", "")
            if bvid:
                return f"video {bvid}"
            return ""

        return combined

    def rerank(
        self,
        query: str,
        hits: list[dict],
        text_fields: list[str] = None,
        keyword_boost: float = 1.5,
        title_keyword_boost: float = 2.0,
        max_rerank: int = 200,
        score_field: str = "rerank_score",
        verbose: bool = False,
    ) -> tuple[list[dict], dict]:
        """Rerank hits using float embeddings and optional keyword matching.

        The reranking formula:
            final_score = cosine_similarity * (1 + keyword_boost * has_keyword_match)

        For title matches, an additional boost is applied:
            final_score *= (1 + title_keyword_boost) if title matches

        Args:
            query: Original query string.
            hits: List of hit documents to rerank.
            text_fields: Fields to use for computing document embeddings.
                        Defaults to ["title", "tags", "desc"].
            keyword_boost: Boost factor when keywords match in any field.
            title_keyword_boost: Additional boost when keywords match in title.
            max_rerank: Maximum number of hits to rerank (for efficiency).
            score_field: Field name to store the rerank score.
            verbose: Enable verbose logging.

        Returns:
            Tuple of (reranked hits, perf_info dict with timing details).
        """
        perf_info = {
            "total_ms": 0,
            "query_embed_ms": 0,
            "doc_text_extract_ms": 0,
            "doc_embed_ms": 0,
            "scoring_ms": 0,
            "sorting_ms": 0,
            "hits_count": len(hits),
            "reranked_count": 0,
        }

        total_start = time.perf_counter()

        if not hits:
            return hits, perf_info

        if not self.is_available():
            logger.warn("× Reranker not available, returning original order")
            return hits, perf_info

        # Limit hits to rerank for efficiency
        hits_to_rerank = hits[:max_rerank]
        remaining_hits = hits[max_rerank:]
        perf_info["reranked_count"] = len(hits_to_rerank)

        # Step 1: Get query embedding
        step_start = time.perf_counter()
        query_embedding = self.embed_client.text_to_embedding(query)
        perf_info["query_embed_ms"] = round(
            (time.perf_counter() - step_start) * 1000, 2
        )

        if not query_embedding:
            logger.warn("× Failed to get query embedding, returning original order")
            return hits, perf_info

        # Extract keywords from query (simple tokenization)
        keywords = self._extract_keywords(query)

        # Step 2: Extract doc texts for embedding
        step_start = time.perf_counter()
        doc_texts = [
            self.compute_text_for_hit(hit, text_fields) for hit in hits_to_rerank
        ]
        perf_info["doc_text_extract_ms"] = round(
            (time.perf_counter() - step_start) * 1000, 2
        )

        # Step 3: Compute document embeddings in batch
        step_start = time.perf_counter()
        doc_embeddings = self.embed_client.texts_to_embeddings(doc_texts)
        perf_info["doc_embed_ms"] = round((time.perf_counter() - step_start) * 1000, 2)

        # Step 4: Compute rerank scores
        step_start = time.perf_counter()
        for i, hit in enumerate(hits_to_rerank):
            doc_embedding = doc_embeddings[i] if i < len(doc_embeddings) else []

            # Cosine similarity between query and document
            if doc_embedding:
                similarity = cosine_similarity(query_embedding, doc_embedding)
            else:
                similarity = hit.get("score", 0.5)  # Fallback to original score

            # Keyword matching boost
            boost = 1.0

            # Check title match (highest priority)
            title = hit.get("title", "")
            title_match, title_count = check_keyword_match(title, keywords)
            if title_match:
                boost *= 1 + title_keyword_boost * title_count

            # Check tags match
            tags = hit.get("tags", "")
            tags_match, tags_count = check_keyword_match(tags, keywords)
            if tags_match:
                boost *= (
                    1 + keyword_boost * tags_count * 0.8
                )  # Slightly less than title

            # Check desc match (lower priority)
            desc = hit.get("desc", "")
            desc_match, desc_count = check_keyword_match(desc, keywords)
            if desc_match:
                boost *= 1 + keyword_boost * desc_count * 0.3  # Much less than title

            # Final score - keep original rerank score (not normalized)
            rerank_score = similarity * boost
            hit[score_field] = round(rerank_score, 6)
            hit["cosine_similarity"] = round(similarity, 6)
            hit["keyword_boost"] = round(boost, 4)
            # Use rerank_score directly for ranking (no normalization)
            hit["rank_score"] = round(rerank_score, 6)
            hit["score"] = round(rerank_score, 6)

            if verbose:
                has_match = title_match or tags_match or desc_match
                logger.mesg(
                    f"  [{i}] sim={similarity:.4f} boost={boost:.2f} "
                    f"score={rerank_score:.4f} match={has_match} "
                    f"title={title[:30]}..."
                )
        perf_info["scoring_ms"] = round((time.perf_counter() - step_start) * 1000, 2)

        # Step 5: Sort by rerank score
        step_start = time.perf_counter()
        hits_to_rerank.sort(key=lambda x: x.get(score_field, 0), reverse=True)
        perf_info["sorting_ms"] = round((time.perf_counter() - step_start) * 1000, 2)

        # Append remaining hits with lower priority (mark as not reranked)
        for i, hit in enumerate(remaining_hits):
            hit[score_field] = 0.0  # Mark as not reranked
            hit["rank_score"] = 0.0
            hit["score"] = 0.0

        perf_info["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

        if verbose:
            logger.mesg(f"  Rerank perf: {perf_info}")

        return hits_to_rerank + remaining_hits, perf_info

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query string.

        Performs simple tokenization to extract meaningful keywords.
        Filters out DSL expressions (e.g., q=v, date=2024).

        Args:
            query: Query string, possibly with DSL expressions.

        Returns:
            List of keywords for matching.
        """
        # Remove common DSL patterns
        import re

        # Remove patterns like q=xxx, date=xxx, etc.
        cleaned = re.sub(r"\b\w+=[^\s]+", "", query)

        # Split by whitespace and punctuation
        tokens = re.split(r"[\s,;，；、]+", cleaned)

        # Filter empty and very short tokens
        keywords = [t.strip() for t in tokens if t.strip() and len(t.strip()) >= 2]

        return keywords

    def rerank_with_stats(
        self,
        query: str,
        hits: list[dict],
        text_fields: list[str] = None,
        keyword_boost: float = 1.5,
        title_keyword_boost: float = 2.0,
        relevance_weight: float = 0.7,
        popularity_weight: float = 0.3,
        max_rerank: int = 200,
        verbose: bool = False,
    ) -> tuple[list[dict], dict]:
        """Rerank with both relevance and popularity considered.

        Uses a weighted combination of semantic relevance and popularity metrics.

        Final score = relevance_weight * rerank_score + popularity_weight * popularity_score

        Args:
            query: Original query string.
            hits: List of hit documents to rerank.
            text_fields: Fields for computing document embeddings.
            keyword_boost: Boost for keyword matches.
            title_keyword_boost: Boost for title keyword matches.
            relevance_weight: Weight for semantic relevance (0-1).
            popularity_weight: Weight for popularity score (0-1).
            max_rerank: Maximum hits to rerank.
            verbose: Enable verbose logging.

        Returns:
            Tuple of (reranked hits, perf_info dict).
        """
        # First do semantic reranking
        hits, perf_info = self.rerank(
            query=query,
            hits=hits,
            text_fields=text_fields,
            keyword_boost=keyword_boost,
            title_keyword_boost=title_keyword_boost,
            max_rerank=max_rerank,
            score_field="rerank_score",
            verbose=verbose,
        )

        # Normalize rerank scores for combining with popularity
        max_rerank_score = max(
            (h.get("rerank_score", 0) for h in hits[:max_rerank]), default=1.0
        )
        if max_rerank_score <= 0:
            max_rerank_score = 1.0

        # Calculate popularity score and combine
        for hit in hits[:max_rerank]:
            stat = hit.get("stat", {})
            view = stat.get("view", 0)
            coin = stat.get("coin", 0)
            favorite = stat.get("favorite", 0)

            # Log-scaled popularity (avoids extreme values dominating)
            popularity = (
                math.log10(max(view, 1)) * 0.5
                + math.log10(max(coin, 1) * 10) * 0.25
                + math.log10(max(favorite, 1) * 5) * 0.25
            )
            # Normalize to roughly 0-1 range (assuming max ~10M views)
            popularity_norm = min(popularity / 10.0, 1.0)

            rerank_norm = hit.get("rerank_score", 0) / max_rerank_score

            # Combined score
            final_score = (
                relevance_weight * rerank_norm + popularity_weight * popularity_norm
            )
            hit["final_score"] = round(final_score, 6)
            hit["popularity_score"] = round(popularity_norm, 4)

        # Sort by final score
        hits[:max_rerank] = sorted(
            hits[:max_rerank],
            key=lambda x: x.get("final_score", 0),
            reverse=True,
        )

        return hits, perf_info


# Singleton instance
_reranker: EmbeddingReranker = None


def get_reranker() -> EmbeddingReranker:
    """Get or create a singleton EmbeddingReranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = EmbeddingReranker()
    return _reranker
