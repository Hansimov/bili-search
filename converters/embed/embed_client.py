"""Text Embedding Client for KNN Search and Reranking

This module provides a unified client for:
1. Converting query text to LSH hex string for Elasticsearch KNN search
2. Reranking search results using cosine similarity via tfmx.rerank()

Uses TEIClients from tfmx library to connect to TEI (Text Embeddings Inference) services.

Connection Keep-alive:
    Long-running services should use the keep-alive feature to prevent
    connection timeouts. HTTP connections typically close after 60-300 seconds
    of inactivity. The keep-alive mechanism sends periodic health checks.

    Example:
        client = TextEmbedClient()
        client.start_keepalive()  # Start background keep-alive
        # ... use client normally ...
        client.stop_keepalive()   # Stop before shutdown
"""

import threading
import time
from tclogger import logger

from configs.envs import TEI_CLIENTS_ENDPOINTS

# Default LSH bit count - should match text_emb dims in ES index
KNN_LSH_BITN = 2048

# Default cache size for LSH results
LSH_CACHE_SIZE = 1024

# Keep-alive settings
KEEPALIVE_INTERVAL = 60  # Send keep-alive every 60 seconds
KEEPALIVE_TIMEOUT = 300  # Consider connection stale after 5 minutes of no activity


def get_tei_endpoints() -> list[str]:
    """Get TEI client endpoints from secrets config."""
    if not TEI_CLIENTS_ENDPOINTS:
        logger.warn("× No TEI endpoints configured in secrets.json")
    return TEI_CLIENTS_ENDPOINTS


class TextEmbedClient:
    """Unified client for text embedding operations.

    Provides:
    1. LSH (Locality Sensitive Hashing) for KNN vector search
    2. Reranking via cosine similarity (single network call)
    3. Float embeddings (if needed)

    Uses TEIClients for efficient multi-machine distribution.

    Example:
        client = TextEmbedClient()

        # For KNN search
        hex_vector = client.text_to_hex("游戏视频")

        # For reranking (returns similarity scores directly)
        rankings = client.rerank("红警", ["红警攻略", "星际争霸", "红警视频"])
        # rankings: [(1, 0.85), (2, 0.45), (0, 0.92)]
        # tuple is (rank_position, similarity_score)
    """

    def __init__(
        self,
        endpoints: list[str] = None,
        bitn: int = KNN_LSH_BITN,
        lazy_init: bool = True,
    ):
        """Initialize the text embedding client.

        Args:
            endpoints: List of TEI service endpoints. If None, reads from secrets.
            bitn: Number of bits for LSH (must match text_emb field dims).
            lazy_init: If True, delay TEIClients initialization until first use.
        """
        self.endpoints = endpoints or get_tei_endpoints()
        self.bitn = bitn
        self._clients = None
        self._initialized = False
        self._cache = {}  # Simple cache for text -> hex mappings
        self._cache_max_size = LSH_CACHE_SIZE

        # Keep-alive state
        self._last_activity_time = 0.0
        self._keepalive_thread: threading.Thread = None
        self._keepalive_stop_event = threading.Event()
        self._keepalive_lock = threading.Lock()

        if not lazy_init:
            self._ensure_initialized()

    def _update_activity_time(self) -> None:
        """Update the last activity timestamp."""
        self._last_activity_time = time.time()

    def _is_connection_stale(self) -> bool:
        """Check if connection might be stale due to inactivity."""
        if self._last_activity_time == 0:
            return True  # Never used, needs warming up
        elapsed = time.time() - self._last_activity_time
        return elapsed > KEEPALIVE_TIMEOUT

    def _ensure_initialized(self) -> bool:
        """Ensure TEIClients is initialized. Returns True if ready."""
        if self._initialized:
            return True

        if not self.endpoints:
            logger.warn("× Cannot initialize TextEmbedClient: no endpoints")
            return False

        try:
            from tfmx import TEIClients

            self._clients = TEIClients(endpoints=self.endpoints)
            self._initialized = True
            self._update_activity_time()
            return True
        except ImportError:
            logger.warn("× tfmx library not installed. Embedding features unavailable.")
            return False
        except Exception as e:
            logger.warn(f"× Failed to initialize TEIClients: {e}")
            return False

    def warmup(self, verbose: bool = False) -> bool:
        """Warmup the connection by sending a health check and test request.

        This is useful to call before the first real request to avoid
        cold-start latency. Should be called after long idle periods.

        Args:
            verbose: If True, log warmup details.

        Returns:
            True if warmup succeeded, False otherwise.
        """
        if not self._ensure_initialized():
            return False

        try:
            t0 = time.time()

            # Step 1: Health check to verify connectivity
            # Note: TEIClients.health() returns ClientsHealthResponse with
            # healthy_instances/total_instances (not healthy/total)
            health = self._clients.health()
            t1 = time.time()

            if verbose:
                logger.mesg(
                    f"  [warmup] Health check: {(t1-t0)*1000:.0f}ms, "
                    f"healthy_instances={health.healthy_instances}/{health.total_instances}"
                )

            # Step 2: Small LSH request to warm up the model
            _ = self._clients.lsh(["warmup test"], bitn=self.bitn)
            t2 = time.time()

            if verbose:
                logger.mesg(f"  [warmup] LSH test: {(t2-t1)*1000:.0f}ms")

            self._update_activity_time()
            return True

        except Exception as e:
            if verbose:
                logger.warn(f"  [warmup] Failed: {e}")
            return False

    def _keepalive_worker(self) -> None:
        """Background worker that sends periodic health checks."""
        while not self._keepalive_stop_event.is_set():
            # Wait for interval or stop signal
            if self._keepalive_stop_event.wait(timeout=KEEPALIVE_INTERVAL):
                break  # Stop event was set

            # Check if we need to send keep-alive
            with self._keepalive_lock:
                if not self._initialized or self._clients is None:
                    continue

                elapsed = time.time() - self._last_activity_time
                if elapsed < KEEPALIVE_INTERVAL:
                    # Recent activity, no need for keep-alive
                    continue

                try:
                    # Send health check to keep connection alive
                    self._clients.health()
                    self._update_activity_time()
                except Exception:
                    # Connection might be broken, will be refreshed on next use
                    pass

    def start_keepalive(self) -> None:
        """Start background keep-alive thread.

        This prevents HTTP connections from timing out during long idle periods.
        Call stop_keepalive() before shutdown.
        """
        with self._keepalive_lock:
            if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
                return  # Already running

            self._keepalive_stop_event.clear()
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_worker,
                name="TextEmbedClient-keepalive",
                daemon=True,  # Thread will be killed when main process exits
            )
            self._keepalive_thread.start()
            logger.mesg("  [keepalive] Started background keep-alive thread")

    def stop_keepalive(self) -> None:
        """Stop background keep-alive thread."""
        with self._keepalive_lock:
            if self._keepalive_thread is None:
                return

            self._keepalive_stop_event.set()
            self._keepalive_thread.join(timeout=5.0)
            self._keepalive_thread = None
            logger.mesg("  [keepalive] Stopped background keep-alive thread")

    def refresh_if_stale(self, verbose: bool = False) -> bool:
        """Refresh connection if it might be stale.

        Call this before a request if the service has been idle for a while.
        This is lighter than warmup() - just does a health check.

        Args:
            verbose: If True, log refresh details.

        Returns:
            True if connection is ready, False otherwise.
        """
        if not self._ensure_initialized():
            return False

        if not self._is_connection_stale():
            return True  # Connection is fresh

        try:
            t0 = time.time()
            # Note: TEIClients.health() returns ClientsHealthResponse
            health = self._clients.health()
            elapsed = (time.time() - t0) * 1000

            if verbose:
                logger.mesg(
                    f"  [refresh] Health check: {elapsed:.0f}ms, "
                    f"healthy_instances={health.healthy_instances}/{health.total_instances}"
                )

            self._update_activity_time()
            return health.healthy_instances > 0

        except Exception as e:
            if verbose:
                logger.warn(f"  [refresh] Failed: {e}")
            return False

    def is_available(self) -> bool:
        """Check if the client is available and ready."""
        return self._ensure_initialized()

    # ========== LSH Methods (for KNN Search) ==========

    def _compute_lsh(self, text: str) -> str:
        """Internal method to compute LSH."""
        if not text or not text.strip():
            return ""
        try:
            results = self._clients.lsh([text], bitn=self.bitn)
            self._update_activity_time()
            return results[0] if results else ""
        except Exception as e:
            logger.warn(f"× Failed to compute LSH for text: {e}")
            return ""

    def _get_from_cache(self, text: str) -> str:
        """Get cached result if available."""
        return self._cache.get(text)

    def _add_to_cache(self, text: str, hex_result: str) -> None:
        """Add result to cache with simple LRU-like eviction."""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest item (first key in dict - Python 3.7+ maintains order)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[text] = hex_result

    def text_to_hex(self, text: str, use_cache: bool = True) -> str:
        """Convert a single text to LSH hex string.

        Args:
            text: Input text to convert.
            use_cache: Whether to use cache for repeated queries.

        Returns:
            Hex string representation of the LSH bit vector.
            Returns empty string if client is not available.
        """
        if not self._ensure_initialized():
            return ""

        if use_cache:
            cached = self._get_from_cache(text)
            if cached is not None:
                return cached
            result = self._compute_lsh(text)
            if result:
                self._add_to_cache(text, result)
            return result
        else:
            return self._compute_lsh(text)

    def texts_to_hex(self, texts: list[str]) -> list[str]:
        """Convert multiple texts to LSH hex strings.

        Args:
            texts: List of input texts to convert.

        Returns:
            List of hex string representations.
            Returns empty string for empty/whitespace-only texts.
        """
        if not texts:
            return []

        if not self._ensure_initialized():
            return [""] * len(texts)

        # Filter out empty texts and track their positions
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        if not non_empty_texts:
            return [""] * len(texts)

        try:
            results = self._clients.lsh(non_empty_texts, bitn=self.bitn)

            # Reconstruct full result list
            full_results = [""] * len(texts)
            for result_idx, original_idx in enumerate(non_empty_indices):
                if result_idx < len(results):
                    full_results[original_idx] = results[result_idx]

            return full_results
        except Exception as e:
            logger.warn(f"× Failed to compute LSH for texts: {e}")
            return [""] * len(texts)

    # ========== Rerank Methods ==========

    def rerank(
        self,
        query: str,
        passages: list[str],
        verbose: bool = False,
    ) -> list[tuple[int, float]]:
        """Rerank passages by cosine similarity to query.

        Uses tfmx.rerank() for efficient server-side computation.
        Single network call - embeds all texts and computes similarity matrix.

        Args:
            query: Query text.
            passages: List of passage texts to rank.
            verbose: If True, log timing details.

        Returns:
            List of (rank_position, similarity_score) tuples in passage order.
            rank_position: 0 = best match (highest similarity).

        Example:
            rankings = client.rerank("红警", ["红警攻略", "星际争霸", "红警视频"])
            # rankings[0] = (1, 0.85)  # "红警攻略" is rank 1 with score 0.85
            # rankings[1] = (2, 0.45)  # "星际争霸" is rank 2 with score 0.45
            # rankings[2] = (0, 0.92)  # "红警视频" is rank 0 (best) with score 0.92
        """
        import time

        if not query or not passages:
            return []

        t_start = time.perf_counter()

        if not self._ensure_initialized():
            return []

        t_init = time.perf_counter()

        try:
            # tfmx.rerank returns list[list[tuple[int, float]]]
            # We only have one query, so take the first result
            if verbose:
                init_ms = (t_init - t_start) * 1000
                logger.mesg(
                    f"  [rerank] ensure_initialized: {init_ms:.2f}ms, "
                    f"query_len={len(query)}, passages={len(passages)}"
                )

            t_call_start = time.perf_counter()
            results = self._clients.rerank([query], passages)
            t_call_end = time.perf_counter()

            self._update_activity_time()

            if verbose:
                call_ms = (t_call_end - t_call_start) * 1000
                total_ms = (t_call_end - t_start) * 1000
                logger.mesg(
                    f"  [rerank] API call: {call_ms:.2f}ms, total: {total_ms:.2f}ms"
                )

            return results[0] if results else []
        except Exception as e:
            logger.warn(f"× Failed to rerank: {e}")
            return []

    def rerank_batch(
        self,
        queries: list[str],
        passages: list[str],
    ) -> list[list[tuple[int, float]]]:
        """Rerank passages for multiple queries.

        Args:
            queries: List of query texts.
            passages: List of passage texts to rank against each query.

        Returns:
            List of rankings per query. Each ranking is list of
            (rank_position, similarity_score) tuples in passage order.
        """
        if not queries or not passages:
            return []

        if not self._ensure_initialized():
            return []

        try:
            results = self._clients.rerank(queries, passages)
            self._update_activity_time()
            return results
        except Exception as e:
            logger.warn(f"× Failed to batch rerank: {e}")
            return []

    # ========== Embedding Methods (if needed) ==========

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Get float embeddings for texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        if not self._ensure_initialized():
            return []

        # Filter out empty texts
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        if not non_empty_texts:
            return [[] for _ in texts]

        try:
            results = self._clients.embed(non_empty_texts)

            # Reconstruct full result list
            full_results = [[] for _ in texts]
            for result_idx, original_idx in enumerate(non_empty_indices):
                if result_idx < len(results):
                    full_results[original_idx] = results[result_idx]

            return full_results
        except Exception as e:
            logger.warn(f"× Failed to embed texts: {e}")
            return [[] for _ in texts]

    # ========== Utility Methods ==========

    def hex_to_byte_array(self, hex_str: str) -> list[int]:
        """Convert hex string to byte array for Elasticsearch.

        Elasticsearch expects byte array (list of signed int8) for bit vectors.

        Args:
            hex_str: Hex string (e.g., "a1b2c3...")

        Returns:
            List of signed bytes (-128 to 127).
        """
        if not hex_str:
            return []

        try:
            byte_data = bytes.fromhex(hex_str)
            return [b if b < 128 else b - 256 for b in byte_data]
        except ValueError as e:
            logger.warn(f"× Invalid hex string: {e}")
            return []

    def close(self):
        """Close the underlying TEIClients connection and stop keepalive."""
        self.stop_keepalive()
        if self._clients is not None:
            try:
                self._clients.close()
            except Exception:
                pass
            self._clients = None
            self._initialized = False
            self._last_activity_time = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Singleton instance
_embed_client: TextEmbedClient = None


def get_embed_client(start_keepalive: bool = False) -> TextEmbedClient:
    """Get or create a singleton TextEmbedClient instance.

    Args:
        start_keepalive: If True and this is the first call, start keepalive thread.
                        Use this for long-running services like search_app.

    Returns:
        Singleton TextEmbedClient instance.
    """
    global _embed_client
    if _embed_client is None:
        _embed_client = TextEmbedClient()
        if start_keepalive:
            _embed_client.start_keepalive()
    return _embed_client


def init_embed_client_with_keepalive() -> TextEmbedClient:
    """Initialize the singleton embed client with keepalive and warmup.

    This is the recommended way to initialize the client for long-running
    services like search_app. It:
    1. Creates the singleton client
    2. Warms up the connection (health check + test LSH)
    3. Starts the keepalive background thread

    Returns:
        Initialized and warmed-up TextEmbedClient instance.
    """
    client = get_embed_client(start_keepalive=False)
    client.warmup(verbose=True)
    client.start_keepalive()
    return client


# Legacy alias for backward compatibility
TextEmbedSearchClient = TextEmbedClient
