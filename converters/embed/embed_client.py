"""Text Embedding Client for KNN Search

This module provides a client for converting query text to LSH hex string
for use with Elasticsearch KNN search on the text_emb field.

Uses TEIClients from tfmx library to connect to TEI (Text Embeddings Inference) services.
"""

from tclogger import logger
from typing import Union

from configs.envs import TEI_CLIENTS_ENDPOINTS

# Default LSH bit count - should match text_emb dims in ES index
KNN_LSH_BITN = 2048


def get_tei_endpoints() -> list[str]:
    """Get TEI client endpoints from secrets config."""
    if not TEI_CLIENTS_ENDPOINTS:
        logger.warn("× No TEI endpoints configured in secrets.json")
    return TEI_CLIENTS_ENDPOINTS


class TextEmbedSearchClient:
    """Client for converting query text to LSH hex string for KNN search.

    Uses TEIClients to compute LSH (Locality Sensitive Hashing) of input text,
    which produces a bit vector representation as a hex string.

    The hex string can be used directly with Elasticsearch's KNN search
    on dense_vector fields with element_type="bit".

    Example:
        client = TextEmbedSearchClient()
        query_vector = client.text_to_hex("红警HBK08 游戏视频")
        # query_vector: "a1b2c3d4..." (512 hex chars = 2048 bits)
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

        if not lazy_init:
            self._ensure_initialized()

    def _ensure_initialized(self) -> bool:
        """Ensure TEIClients is initialized. Returns True if ready."""
        if self._initialized:
            return True

        if not self.endpoints:
            logger.warn("× Cannot initialize TextEmbedSearchClient: no endpoints")
            return False

        try:
            from tfmx import TEIClients

            self._clients = TEIClients(endpoints=self.endpoints)
            self._initialized = True
            return True
        except ImportError:
            logger.warn("× tfmx library not installed. KNN search unavailable.")
            return False
        except Exception as e:
            logger.warn(f"× Failed to initialize TEIClients: {e}")
            return False

    def is_available(self) -> bool:
        """Check if the client is available and ready."""
        return self._ensure_initialized()

    def text_to_hex(self, text: str) -> str:
        """Convert a single text to LSH hex string.

        Args:
            text: Input text to convert.

        Returns:
            Hex string representation of the LSH bit vector.
            Returns empty string if client is not available.
        """
        if not self._ensure_initialized():
            return ""

        try:
            results = self._clients.lsh([text], bitn=self.bitn)
            return results[0] if results else ""
        except Exception as e:
            logger.warn(f"× Failed to compute LSH for text: {e}")
            return ""

    def texts_to_hex(self, texts: list[str]) -> list[str]:
        """Convert multiple texts to LSH hex strings.

        Args:
            texts: List of input texts to convert.

        Returns:
            List of hex string representations.
            Returns empty list if client is not available.
        """
        if not texts:
            return []

        if not self._ensure_initialized():
            return [""] * len(texts)

        try:
            results = self._clients.lsh(texts, bitn=self.bitn)
            return results
        except Exception as e:
            logger.warn(f"× Failed to compute LSH for texts: {e}")
            return [""] * len(texts)

    async def text_to_hex_async(self, text: str) -> str:
        """Async version of text_to_hex."""
        if not self._ensure_initialized():
            return ""

        try:
            results = await self._clients.lsh_async([text], bitn=self.bitn)
            return results[0] if results else ""
        except Exception as e:
            logger.warn(f"× Failed to compute LSH for text: {e}")
            return ""

    async def texts_to_hex_async(self, texts: list[str]) -> list[str]:
        """Async version of texts_to_hex."""
        if not texts:
            return []

        if not self._ensure_initialized():
            return [""] * len(texts)

        try:
            results = await self._clients.lsh_async(texts, bitn=self.bitn)
            return results
        except Exception as e:
            logger.warn(f"× Failed to compute LSH for texts: {e}")
            return [""] * len(texts)

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
            # Convert hex string to bytes, then to signed int8 list
            byte_data = bytes.fromhex(hex_str)
            # Convert to signed bytes (-128 to 127) as ES expects
            return [b if b < 128 else b - 256 for b in byte_data]
        except ValueError as e:
            logger.warn(f"× Invalid hex string: {e}")
            return []

    def close(self):
        """Close the underlying TEIClients connection."""
        if self._clients is not None:
            try:
                self._clients.close()
            except Exception:
                pass
            self._clients = None
            self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Singleton instance for convenience
_embed_client: TextEmbedSearchClient = None


def get_embed_client() -> TextEmbedSearchClient:
    """Get or create a singleton TextEmbedSearchClient instance."""
    global _embed_client
    if _embed_client is None:
        _embed_client = TextEmbedSearchClient()
    return _embed_client
