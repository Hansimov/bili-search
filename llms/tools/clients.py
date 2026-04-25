"""Search-service and external tool clients used by the LLM tool executor."""

from __future__ import annotations

import os
import requests

from tclogger import logger

from configs.envs import GOOGLE_HUB_ENVS
from llms.tools.defs import DEFAULT_SEARCH_CAPABILITIES, build_tool_definitions
from llms.tools.specs import SPEC_REGISTRY


class SearchService:
    """Direct search service wrapping VideoSearcherV2 and VideoExplorer."""

    def __init__(
        self,
        video_searcher,
        video_explorer,
        owner_searcher=None,
        relations_client=None,
        transcript_client=None,
        verbose: bool = False,
    ):
        self.video_searcher = video_searcher
        self.video_explorer = video_explorer
        self.owner_searcher = owner_searcher
        self.relations_client = relations_client
        self.transcript_client = transcript_client
        self.verbose = verbose
        self._capabilities = self._build_capabilities()

    def _build_capabilities(self) -> dict:
        return {
            **DEFAULT_SEARCH_CAPABILITIES,
            "service_type": "local",
            "service_name": "integrated_search",
            "supports_author_check": False,
            "supports_owner_search": self.owner_searcher is not None,
            "supports_multi_query": True,
            "supports_google_search": False,
            "supports_transcript_lookup": self.transcript_client is not None,
            "relation_endpoints": (
                [
                    "related_tokens_by_tokens",
                    "related_owners_by_tokens",
                    "related_videos_by_videos",
                    "related_owners_by_videos",
                    "related_videos_by_owners",
                    "related_owners_by_owners",
                ]
                if self.relations_client is not None
                else []
            ),
            "available_endpoints": [
                "/explore",
                "/video_lookup",
                "/suggest",
                "/search",
                "/random",
                "/latest",
                "/doc",
                "/knn_search",
                "/hybrid_search",
                "/search_owners",
                "/video_transcript",
                "/related_tokens_by_tokens",
                "/related_owners_by_tokens",
                "/related_videos_by_videos",
                "/related_owners_by_videos",
                "/related_videos_by_owners",
                "/related_owners_by_owners",
            ],
            "docs": list(SPEC_REGISTRY.keys()),
        }

    def capabilities(self, refresh: bool = False) -> dict:
        return dict(self._capabilities)

    def tool_definitions(self, include_read_spec: bool = False) -> list[dict]:
        return build_tool_definitions(
            self.capabilities(),
            include_read_spec=include_read_spec,
        )

    def explore(self, query: str, qmod=None, verbose: bool = False) -> dict:
        try:
            return self.video_explorer.unified_explore(
                query=query, qmod=qmod, verbose=verbose
            )
        except Exception as exc:
            logger.warn(f"× Search explore error: {exc}")
            return {"error": str(exc), "data": []}

    def suggest(self, query: str, limit: int = 25, verbose: bool = False) -> dict:
        try:
            return self.video_searcher.suggest(query, limit=limit, verbose=verbose)
        except Exception as exc:
            logger.warn(f"× Search suggest error: {exc}")
            return {"error": str(exc), "hits": [], "total_hits": 0}

    def search(self, query: str, limit: int = 25, verbose: bool = False) -> dict:
        try:
            return self.video_searcher.search(
                query,
                limit=limit,
                verbose=verbose,
            )
        except Exception as exc:
            logger.warn(f"× Search direct error: {exc}")
            return {"error": str(exc), "hits": [], "total_hits": 0}

    def search_owners(self, **kwargs) -> dict:
        if self.owner_searcher is None:
            return {"error": "Owner search unavailable", "owners": []}
        return self.owner_searcher.search(**kwargs)

    def get_video_transcript(self, video_id: str, **kwargs) -> dict:
        if self.transcript_client is None:
            return {"error": "Transcript lookup unavailable", "video_id": video_id}
        return self.transcript_client.get_video_transcript(video_id, request=kwargs)

    def lookup_videos(self, **kwargs) -> dict:
        if self.video_searcher is None:
            return {"error": "Video lookup unavailable", "hits": [], "total_hits": 0}
        return self.video_searcher.lookup_videos(**kwargs)

    def _relation_method(self, name: str):
        if self.relations_client is None:
            return None
        return getattr(self.relations_client, name, None)

    def related_tokens_by_tokens(self, **kwargs) -> dict:
        method = self._relation_method("related_tokens_by_tokens")
        if method is None:
            return {"error": "Relations service unavailable", "options": []}
        return method(**kwargs)

    def related_owners_by_tokens(self, **kwargs) -> dict:
        method = self._relation_method("related_owners_by_tokens")
        if method is None:
            return {"error": "Relations service unavailable", "owners": []}
        return method(**kwargs)

    def related_videos_by_videos(self, **kwargs) -> dict:
        method = self._relation_method("related_videos_by_videos")
        if method is None:
            return {"error": "Relations service unavailable", "videos": []}
        return method(**kwargs)

    def related_owners_by_videos(self, **kwargs) -> dict:
        method = self._relation_method("related_owners_by_videos")
        if method is None:
            return {"error": "Relations service unavailable", "owners": []}
        return method(**kwargs)

    def related_videos_by_owners(self, **kwargs) -> dict:
        method = self._relation_method("related_videos_by_owners")
        if method is None:
            return {"error": "Relations service unavailable", "videos": []}
        return method(**kwargs)

    def related_owners_by_owners(self, **kwargs) -> dict:
        method = self._relation_method("related_owners_by_owners")
        if method is None:
            return {"error": "Relations service unavailable", "owners": []}
        return method(**kwargs)


class SearchServiceClient:
    """HTTP-backed search service client for external search_app instances."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        verbose: bool = False,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self._capabilities_cache: dict | None = None

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warn(f"× Search service request error [{path}]: {exc}")
            return {"error": str(exc)}

    def health(self) -> dict:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warn(f"× Search service health error: {exc}")
            return {"error": str(exc)}

    def capabilities(self, refresh: bool = False) -> dict:
        if self._capabilities_cache is not None and not refresh:
            return dict(self._capabilities_cache)
        fallback = {
            **DEFAULT_SEARCH_CAPABILITIES,
            "service_type": "remote",
            "service_name": self.base_url,
            "base_url": self.base_url,
            "available_endpoints": [
                "/explore",
                "/video_lookup",
                "/suggest",
                "/health",
                "/capabilities",
                "/video_transcript",
            ],
            "docs": list(SPEC_REGISTRY.keys()),
        }
        try:
            response = requests.get(
                f"{self.base_url}/capabilities",
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                fallback.update(payload)
        except requests.RequestException as exc:
            logger.warn(f"× Search service capabilities error: {exc}")
        self._capabilities_cache = fallback
        return dict(self._capabilities_cache)

    def tool_definitions(self, include_read_spec: bool = False) -> list[dict]:
        return build_tool_definitions(
            self.capabilities(),
            include_read_spec=include_read_spec,
        )

    def explore(self, query: str, qmod=None, verbose: bool = False) -> dict:
        return self._post(
            "/explore",
            {"query": query, "qmod": qmod, "verbose": verbose},
        )

    def lookup_videos(self, **kwargs) -> dict:
        return self._post("/video_lookup", kwargs)

    def suggest(self, query: str, limit: int = 25, verbose: bool = False) -> dict:
        return self._post(
            "/suggest",
            {"query": query, "limit": limit, "verbose": verbose},
        )

    def search(self, query: str, limit: int = 25, verbose: bool = False) -> dict:
        return self._post(
            "/search",
            {"query": query, "limit": limit, "verbose": verbose},
        )

    def search_owners(self, **kwargs) -> dict:
        return self._post("/search_owners", kwargs)

    def get_video_transcript(self, **kwargs) -> dict:
        return self._post("/video_transcript", kwargs)

    def related_tokens_by_tokens(self, **kwargs) -> dict:
        return self._post("/related_tokens_by_tokens", kwargs)

    def related_owners_by_tokens(self, **kwargs) -> dict:
        return self._post("/related_owners_by_tokens", kwargs)

    def related_videos_by_videos(self, **kwargs) -> dict:
        return self._post("/related_videos_by_videos", kwargs)

    def related_owners_by_videos(self, **kwargs) -> dict:
        return self._post("/related_owners_by_videos", kwargs)

    def related_videos_by_owners(self, **kwargs) -> dict:
        return self._post("/related_videos_by_owners", kwargs)

    def related_owners_by_owners(self, **kwargs) -> dict:
        return self._post("/related_owners_by_owners", kwargs)


class GoogleSearchClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 45.0,
        verbose: bool = False,
        fallback_base_urls: list[str] | None = None,
        max_retries: int = 2,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.base_urls = [self.base_url]
        for candidate in fallback_base_urls or []:
            normalized = str(candidate).strip().rstrip("/")
            if normalized and normalized not in self.base_urls:
                self.base_urls.append(normalized)
        self.timeout = timeout
        self.verbose = verbose
        self.max_retries = max(1, int(max_retries))

    def _request(
        self, method: str, path: str, *, params: dict | None = None
    ) -> requests.Response:
        errors: list[str] = []
        for base_url in self.base_urls:
            for _attempt in range(self.max_retries):
                try:
                    response = requests.request(
                        method,
                        f"{base_url}{path}",
                        params=params,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    if base_url != self.base_url:
                        self.base_url = base_url
                    return response
                except requests.HTTPError as exc:
                    status_code = (
                        exc.response.status_code if exc.response is not None else None
                    )
                    errors.append(str(exc))
                    if status_code not in (502, 503, 504):
                        break
                except requests.RequestException as exc:
                    errors.append(str(exc))
                    break
        raise requests.RequestException(
            errors[-1] if errors else "Google hub request failed"
        )

    def health(self) -> dict:
        try:
            response = self._request("GET", "/health")
            return response.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def search(self, query: str, num: int = 5, lang: str | None = None) -> dict:
        params = {"q": query, "num": num}
        if lang:
            params["lang"] = lang
        try:
            response = self._request("GET", "/search", params=params)
            return response.json()
        except requests.RequestException as exc:
            logger.warn(f"× Google hub request error: {exc}")
            return {"success": False, "error": str(exc), "results": []}


def _split_google_hub_urls(raw_value: str | None) -> list[str]:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return []
    candidates = []
    for part in raw_text.replace("\n", ",").split(","):
        normalized = part.strip().rstrip("/")
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def create_google_search_client(
    *,
    base_url: str | None = None,
    timeout: float | None = None,
    verbose: bool = False,
    google_hub_envs: dict | None = None,
):
    envs = GOOGLE_HUB_ENVS if google_hub_envs is None else google_hub_envs
    if _truthy_env("BILI_GOOGLE_HUB_DISABLED"):
        return None

    configured_base_url = str(base_url or "").strip()
    if not configured_base_url:
        configured_base_url = str(os.getenv("BILI_GOOGLE_HUB_BASE_URL") or "").strip()
    if not configured_base_url:
        configured_base_url = str(
            envs.get("endpoint") or envs.get("base_url") or ""
        ).strip()

    resolved_urls = _split_google_hub_urls(configured_base_url)
    resolved_timeout = float(
        timeout
        if timeout is not None
        else os.getenv("BILI_GOOGLE_HUB_TIMEOUT")
        or envs.get("timeout")
        or 45
    )
    if not resolved_urls:
        return None
    return GoogleSearchClient(
        base_url=resolved_urls[0],
        fallback_base_urls=resolved_urls[1:],
        timeout=resolved_timeout,
        verbose=verbose,
    )


def create_search_service(
    *,
    video_searcher=None,
    video_explorer=None,
    owner_searcher=None,
    relations_client=None,
    transcript_client=None,
    base_url: str | None = None,
    timeout: float = 30.0,
    verbose: bool = False,
):
    if base_url:
        return SearchServiceClient(base_url=base_url, timeout=timeout, verbose=verbose)
    return SearchService(
        video_searcher=video_searcher,
        video_explorer=video_explorer,
        owner_searcher=owner_searcher,
        relations_client=relations_client,
        transcript_client=transcript_client,
        verbose=verbose,
    )
