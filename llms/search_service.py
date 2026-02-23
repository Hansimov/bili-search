"""HTTP client for the Search App service.

Communicates with the SearchApp's FastAPI endpoints (search, explore, suggest)
over HTTP. This decouples the LLM module from direct Elasticsearch access,
making it deployable as an independent service.
"""

import requests

from tclogger import logger
from typing import Optional


class SearchServiceClient:
    """HTTP client for the Bili Search App service.

    Wraps the SearchApp's REST API endpoints for use by the LLM tools.

    Usage:
        client = SearchServiceClient("http://localhost:20001")
        results = client.explore("黑神话 :view>=1w")
        suggest_results = client.suggest("影视飓风")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:20001",
        timeout: float = 30,
        verbose: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose

    def _post(self, path: str, payload: dict) -> dict:
        """Send POST request and return JSON response."""
        url = f"{self.base_url}{path}"
        if self.verbose:
            logger.note(f"> Search API: POST {path}")

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.warn(f"× Search API timed out: {path}")
            return {"error": "timeout", "hits": [], "total_hits": 0}
        except requests.exceptions.ConnectionError:
            logger.warn(f"× Search API connection failed: {url}")
            return {"error": "connection_failed", "hits": [], "total_hits": 0}
        except requests.exceptions.RequestException as e:
            logger.warn(f"× Search API error: {e}")
            return {"error": str(e), "hits": [], "total_hits": 0}

    def explore(
        self,
        query: str,
        qmod: Optional[str] = None,
        verbose: bool = False,
    ) -> dict:
        """Call the /explore endpoint.

        Uses the unified_explore which auto-detects query mode and provides
        multi-lane recall + diversified ranking + author grouping.

        Args:
            query: Search query with optional DSL filters.
            qmod: Override query mode (w/v/wv/wvr).
            verbose: Verbose logging on server side.

        Returns:
            Explore result dict with steps and hits.
        """
        payload = {"query": query, "verbose": verbose}
        if qmod:
            payload["qmod"] = qmod
        return self._post("/explore", payload)

    def search(
        self,
        query: str,
        limit: int = 20,
        verbose: bool = False,
    ) -> dict:
        """Call the /search endpoint.

        Direct word-based search with highlighting and ranking.

        Args:
            query: Search query with optional DSL filters.
            limit: Maximum number of results.
            verbose: Verbose logging on server side.

        Returns:
            Search result dict with hits.
        """
        payload = {"query": query, "limit": limit, "verbose": verbose}
        return self._post("/search", payload)

    def suggest(
        self,
        query: str,
        limit: int = 25,
        verbose: bool = False,
    ) -> dict:
        """Call the /suggest endpoint.

        Lightweight search for quick suggestions and author detection.

        Args:
            query: Search query.
            limit: Maximum suggestions.
            verbose: Verbose logging on server side.

        Returns:
            Suggest result dict with hits.
        """
        payload = {"query": query, "limit": limit, "verbose": verbose}
        return self._post("/suggest", payload)

    def is_available(self) -> bool:
        """Check if the search service is reachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/docs", timeout=5, allow_redirects=True
            )
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False
