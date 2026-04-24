"""Tool executor for dispatching and executing tool calls."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import requests
import time
from unittest.mock import Mock

from tclogger import logger

from configs.envs import GOOGLE_HUB_ENVS
from llms.contracts import ToolCallRequest
from llms.intent.focus import rewrite_known_term_aliases
from llms.messages import normalize_bvid_key
from llms.models import ToolCall
from llms.prompts.syntax import SEARCH_SYNTAX
from llms.tools.defs import DEFAULT_SEARCH_CAPABILITIES, build_tool_definitions
from llms.tools.names import canonical_tool_name
from llms.tools.video_lookup import coerce_search_video_lookup_arguments
from llms.tools.utils import (
    extract_explore_hits,
    filter_relevant_hits_for_llm,
    format_google_results,
    format_hits_for_llm,
    format_related_owners,
    format_related_token_options,
    format_related_videos,
)

# Whitelist-based spec registry — only these documents can be read
SPEC_REGISTRY = {
    "search_syntax": SEARCH_SYNTAX,
}

_ZERO_HIT_SOFT_TERMS = (
    "解读",
    "更新",
    "最新",
    "官方",
    "公告",
    "release notes",
    "release",
    "changelog",
)

_EXPAND_QUERY_UNSUPPORTED_SEMANTIC_MARKERS = (
    "mode must be",
    "semantic",
)

_OWNER_SOURCE_LABELS = {
    "name": "名字匹配",
    "topic": "主题发现",
    "relation": "关系发现",
    "related_tokens": "相关作者",
    "google_space": "Google 空间页",
}

_OWNER_SOURCE_WEIGHTS = {
    "name": 1.2,
    "topic": 1.0,
    "relation": 1.05,
    "related_tokens": 1.1,
    "google_space": 0.85,
}
_LLM_INTERVIEW_THEME_TOKENS = (
    "采访",
    "专访",
    "访谈",
)


def _normalize_query_spaces(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "")).strip()


def _strip_owner_search_filters(text: str) -> str:
    stripped = re.sub(r"q=[^\s]+", " ", str(text or ""), flags=re.IGNORECASE)
    stripped = re.sub(r":[^\s]+", " ", stripped)
    return _normalize_query_spaces(stripped)


def _coerce_owner_search_args(args: dict) -> dict:
    normalized = dict(args or {})
    text = str(normalized.get("text", "") or "").strip()

    if normalized.get("size") is None:
        for key in ("num", "limit"):
            if normalized.get(key) is not None:
                normalized["size"] = normalized.get(key)
                break

    if not text:
        alias_candidates = [
            ("topic", "topic"),
            ("name", "name"),
            ("relation", "relation"),
        ]
        for key, inferred_mode in alias_candidates:
            candidate = str(normalized.get(key, "") or "").strip()
            if not candidate:
                continue
            text = _strip_owner_search_filters(candidate)
            normalized["text"] = text
            normalized.setdefault("mode", inferred_mode)
            break

    if not text:
        query = normalized.get("query")
        queries = normalized.get("queries")
        candidates: list[str] = []
        if isinstance(query, str):
            candidates.append(query)
        if isinstance(queries, str):
            candidates.append(queries)
        elif isinstance(queries, list):
            candidates.extend(
                str(item).strip() for item in queries if str(item or "").strip()
            )
        for candidate in candidates:
            text = _strip_owner_search_filters(candidate)
            if text:
                normalized["text"] = text
                break

    mode = str(normalized.get("mode", "auto") or "auto")
    if (
        mode == "auto"
        and text
        and any(key in normalized for key in ("query", "queries"))
    ):
        normalized["mode"] = "topic"

    return normalized


def _normalize_seed_values(values: object) -> list[str]:
    if isinstance(values, str):
        text = values.strip()
        return [text] if text else []
    if isinstance(values, (list, tuple, set)):
        return [str(item).strip() for item in values if str(item or "").strip()]
    return []


def _strip_soft_filters(query: str) -> str:
    stripped = re.sub(r":date[<>=!]*[^\s]+", " ", str(query or ""))
    return _normalize_query_spaces(stripped)


def _drop_soft_terms(query: str) -> str:
    softened = str(query or "")
    for term in _ZERO_HIT_SOFT_TERMS:
        softened = re.sub(re.escape(term), " ", softened, flags=re.IGNORECASE)
    return _normalize_query_spaces(softened)


def _build_zero_hit_fallback_queries(query: str) -> list[str]:
    base = _normalize_query_spaces(query)
    candidates: list[str] = []

    stripped_filters = _strip_soft_filters(base)
    softened = _drop_soft_terms(stripped_filters)

    for candidate in [stripped_filters, softened]:
        candidate_text = _normalize_query_spaces(candidate)
        if (
            candidate_text
            and candidate_text != base
            and candidate_text not in candidates
        ):
            candidates.append(candidate_text)
        if (
            candidate_text
            and candidate_text != base
            and "q=" not in candidate_text
            and f"{candidate_text} q=vwr" not in candidates
        ):
            candidates.append(f"{candidate_text} q=vwr")

    return candidates


def _llm_theme_tokens_for_query(query: str) -> list[str]:
    normalized_query = _normalize_query_spaces(query)
    if any(token in normalized_query for token in _LLM_INTERVIEW_THEME_TOKENS):
        return list(_LLM_INTERVIEW_THEME_TOKENS)
    return []


def _hit_contains_llm_theme_token(hit: dict, theme_tokens: list[str]) -> bool:
    if not theme_tokens:
        return True
    searchable_text = "\n".join(
        [
            str(hit.get("title") or ""),
            str(hit.get("desc") or ""),
            (
                " ".join(hit.get("tags") or [])
                if isinstance(hit.get("tags"), list)
                else str(hit.get("tags") or "")
            ),
        ]
    )
    return any(token in searchable_text for token in theme_tokens)


def _filter_hits_by_llm_theme(
    query: str, hits: list[dict]
) -> tuple[list[dict], dict | None]:
    theme_tokens = _llm_theme_tokens_for_query(query)
    if not theme_tokens or not hits:
        return list(hits), None

    themed_hits = [
        hit for hit in hits if _hit_contains_llm_theme_token(hit, theme_tokens)
    ]
    if themed_hits:
        return themed_hits, {
            "theme_tokens": theme_tokens,
            "dropped_hits": len(hits) - len(themed_hits),
        }
    return [], {
        "theme_tokens": theme_tokens,
        "dropped_hits": len(hits),
        "warning": "No hit titles or tags matched the requested interview-style theme",
    }


def _normalize_owner_identity_key(owner: dict) -> str:
    mid = owner.get("mid")
    if mid not in (None, ""):
        return f"mid:{mid}"
    name = re.sub(r"\s+", "", str(owner.get("name", "") or "").lower())
    return f"name:{name}" if name else ""


def _extract_google_space_candidate_name(title: str) -> str:
    name = str(title or "").strip()
    if not name:
        return ""
    for marker in ("的个人空间", "个人空间", "主页"):
        if marker in name:
            name = name.split(marker, 1)[0]
    for separator in (" - ", " -", "-", "|", "｜"):
        if separator in name:
            name = name.split(separator, 1)[0]
    return name.strip(" -_|｜·，。！？?：:[]()（）")


class SearchService:
    """Direct search service wrapping VideoSearcherV2 and VideoExplorer.

    Provides the same explore/suggest interface as the old SearchServiceClient,
    but calls the search components directly instead of over HTTP.

    Usage:
        service = SearchService(video_searcher, video_explorer)
        result = service.explore("黑神话 :view>=1w")
        suggest = service.suggest("影视飓风")
    """

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
        """Call unified_explore directly."""
        try:
            return self.video_explorer.unified_explore(
                query=query, qmod=qmod, verbose=verbose
            )
        except Exception as e:
            logger.warn(f"× Search explore error: {e}")
            return {"error": str(e), "data": []}

    def suggest(self, query: str, limit: int = 25, verbose: bool = False) -> dict:
        """Call suggest directly."""
        try:
            return self.video_searcher.suggest(query, limit=limit, verbose=verbose)
        except Exception as e:
            logger.warn(f"× Search suggest error: {e}")
            return {"error": str(e), "hits": [], "total_hits": 0}

    def search(self, query: str, limit: int = 25, verbose: bool = False) -> dict:
        """Call direct search directly."""
        try:
            return self.video_searcher.search(
                query,
                limit=limit,
                verbose=verbose,
            )
        except Exception as e:
            logger.warn(f"× Search direct error: {e}")
            return {"error": str(e), "hits": [], "total_hits": 0}

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
        payload = {
            "query": query,
            "qmod": qmod,
            "verbose": verbose,
        }
        return self._post("/explore", payload)

    def lookup_videos(self, **kwargs) -> dict:
        return self._post("/video_lookup", kwargs)

    def suggest(self, query: str, limit: int = 25, verbose: bool = False) -> dict:
        payload = {
            "query": query,
            "limit": limit,
            "verbose": verbose,
        }
        return self._post("/suggest", payload)

    def search(self, query: str, limit: int = 25, verbose: bool = False) -> dict:
        payload = {
            "query": query,
            "limit": limit,
            "verbose": verbose,
        }
        return self._post("/search", payload)

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


def create_google_search_client(
    *,
    base_url: str | None = None,
    timeout: float | None = None,
    verbose: bool = False,
):
    configured_base_url = str(base_url or "").strip()
    if not configured_base_url:
        configured_base_url = str(os.getenv("BILI_GOOGLE_HUB_BASE_URL") or "").strip()
    if not configured_base_url:
        configured_base_url = str(
            GOOGLE_HUB_ENVS.get("endpoint") or GOOGLE_HUB_ENVS.get("base_url") or ""
        ).strip()

    resolved_urls = _split_google_hub_urls(configured_base_url)
    resolved_timeout = float(
        timeout
        if timeout is not None
        else os.getenv("BILI_GOOGLE_HUB_TIMEOUT")
        or GOOGLE_HUB_ENVS.get("timeout")
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


class ToolExecutor:
    """Executes tool calls from LLM responses.

    Dispatches tool calls to the appropriate handler and formats
    results for feeding back to the LLM.

    Usage:
        executor = ToolExecutor(search_service)
        result_message = executor.execute(tool_call)
    """

    def __init__(
        self,
        search_client,
        max_results: int = 15,
        google_client=None,
        transcript_client=None,
        verbose: bool = False,
    ):
        self.search_client = search_client
        self.google_client = google_client or create_google_search_client(
            verbose=verbose
        )
        self.transcript_client = transcript_client
        if self.transcript_client is None:
            candidate = getattr(search_client, "transcript_client", None)
            if candidate is not None and not isinstance(candidate, Mock):
                self.transcript_client = candidate
        self.max_results = max_results
        self.verbose = verbose
        # Cached Google availability: (is_available, timestamp)
        self._google_available: bool | None = None
        self._google_available_ts: float = 0.0
        self._GOOGLE_HEALTH_TTL = 120.0  # re-check every 120 seconds
        self._GOOGLE_HEALTH_TIMEOUT = 3.0  # fast health check timeout
        self._handlers = {
            "search_videos": self._search_videos,
            "get_video_transcript": self._get_video_transcript,
            "search_google": self._search_google,
            "search_owners": self._search_owners,
            "expand_query": self._expand_query,
            "read_spec": self._read_spec,
        }

    def _supports_relation_endpoint(self, name: str) -> bool:
        relation_endpoints = set(
            self.get_search_capabilities().get("relation_endpoints") or []
        )
        if relation_endpoints:
            return name in relation_endpoints
        method = getattr(self.search_client, name, None)
        if method is None or not callable(method):
            return False
        if isinstance(method, Mock):
            return (
                not isinstance(method.return_value, Mock)
                or method.side_effect is not None
            )
        return True

    def _search_client_method(self, name: str):
        method = getattr(self.search_client, name, None)
        if method is None or not callable(method):
            return None
        if isinstance(method, Mock):
            if isinstance(method.return_value, Mock) and method.side_effect is None:
                return None
        return method

    def _extract_video_lookup_request(self, args: dict) -> dict | None:
        normalized_args = coerce_search_video_lookup_arguments(args)
        if normalized_args is None:
            return None

        bvids: list[str] = []
        bvid_keys: set[str] = set()
        mids: list[str] = []
        for key in ("bv", "bvid"):
            for value in _normalize_seed_values(normalized_args.get(key)):
                match_key = normalize_bvid_key(value)
                if match_key and match_key not in bvid_keys:
                    bvid_keys.add(match_key)
                    bvids.append(value)
        for value in _normalize_seed_values(normalized_args.get("bvids")):
            match_key = normalize_bvid_key(value)
            if match_key and match_key not in bvid_keys:
                bvid_keys.add(match_key)
                bvids.append(value)

        for key in ("mid", "uid", "mids"):
            for value in _normalize_seed_values(normalized_args.get(key)):
                if value not in mids:
                    mids.append(value)

        date_window = str(normalized_args.get("date_window", "") or "").strip() or None
        if not (bvids or mids):
            return None

        return {
            "mode": "lookup",
            "lookup_by": "bvids" if bvids else "mids",
            "bvids": bvids,
            "mids": mids,
            "date_window": date_window,
            "exclude_bvids": _normalize_seed_values(
                normalized_args.get("exclude_bvids")
            ),
            "limit": int(
                normalized_args.get("limit") or normalized_args.get("size") or 10
            ),
        }

    def _format_lookup_video_result(
        self,
        result: dict,
        lookup_request: dict,
    ) -> dict:
        hits = result.get("hits") or []
        if hits and any(
            key in hits[0] for key in ("link", "pubdate_str", "pub_to_now_str")
        ):
            formatted_hits = hits[: self.max_results]
        else:
            formatted_hits = format_hits_for_llm(hits, max_hits=self.max_results)
        payload = {
            "mode": "lookup",
            "lookup_by": result.get("lookup_by")
            or lookup_request.get("lookup_by")
            or "unknown",
            "total_hits": int(result.get("total_hits", len(hits)) or 0),
            "hits": formatted_hits,
            "source_counts": result.get("source_counts") or {},
        }
        if lookup_request.get("bvids"):
            payload["bvids"] = lookup_request["bvids"]
        if lookup_request.get("mids"):
            payload["mids"] = lookup_request["mids"]
        if lookup_request.get("date_window"):
            payload["date_window"] = lookup_request["date_window"]
        if lookup_request.get("exclude_bvids"):
            payload["exclude_bvids"] = lookup_request["exclude_bvids"]
        return payload

    def _fallback_lookup_videos(self, lookup_request: dict) -> dict:
        queries: list[str] = []
        exclude_bvids = {
            normalize_bvid_key(value)
            for value in lookup_request.get("exclude_bvids") or []
            if str(value or "").strip()
        }
        date_window = str(lookup_request.get("date_window", "") or "").strip()

        for bvid in lookup_request.get("bvids") or []:
            query = f"bv={bvid}"
            if query not in queries:
                queries.append(query)
        for mid in lookup_request.get("mids") or []:
            query = f":uid={mid}"
            if date_window:
                query = f"{query} :date<={date_window}"
            if query not in queries:
                queries.append(query)

        if not queries:
            return {
                "mode": "lookup",
                "lookup_by": lookup_request.get("lookup_by") or "unknown",
                "hits": [],
                "total_hits": 0,
                "source_counts": {},
            }

        hits: list[dict] = []
        seen_bvids: set[str] = set()
        for query in queries:
            result = self._search_single_query(query)
            for hit in result.get("hits") or []:
                bvid = str(hit.get("bvid") or "").strip()
                bvid_key = normalize_bvid_key(bvid)
                if not bvid_key or bvid_key in seen_bvids or bvid_key in exclude_bvids:
                    continue
                seen_bvids.add(bvid_key)
                hits.append(hit)

        return {
            "mode": "lookup",
            "lookup_by": lookup_request.get("lookup_by") or "unknown",
            "hits": hits,
            "total_hits": len(hits),
            "source_counts": {"explore": len(hits)},
        }

    def _lookup_videos(self, lookup_request: dict) -> dict:
        method = self._search_client_method("lookup_videos")
        if method is not None:
            result = method(
                bvids=lookup_request.get("bvids") or None,
                mids=lookup_request.get("mids") or None,
                limit=lookup_request.get("limit") or 10,
                date_window=lookup_request.get("date_window"),
                exclude_bvids=lookup_request.get("exclude_bvids") or None,
                verbose=self.verbose,
            )
            if isinstance(result, dict) and not result.get("error"):
                return self._format_lookup_video_result(result, lookup_request)

        fallback = self._fallback_lookup_videos(lookup_request)
        return self._format_lookup_video_result(fallback, lookup_request)

    def _is_google_available(self) -> bool:
        """Check Google search hub availability with cached result."""
        if self.google_client is None:
            return False
        now = time.monotonic()
        if (
            self._google_available is not None
            and (now - self._google_available_ts) < self._GOOGLE_HEALTH_TTL
        ):
            return self._google_available
        try:
            resp = requests.get(
                f"{self.google_client.base_url}/health",
                timeout=self._GOOGLE_HEALTH_TIMEOUT,
            )
            self._google_available = resp.status_code == 200
        except Exception:
            self._google_available = False
        self._google_available_ts = now
        if self.verbose:
            logger.note(f"> Google search available: {self._google_available}")
        return self._google_available

    def get_search_capabilities(self, refresh: bool = False) -> dict:
        capability_getter = getattr(self.search_client, "capabilities", None)
        if callable(capability_getter):
            try:
                capabilities = capability_getter(refresh=refresh)
            except TypeError:
                capabilities = capability_getter()
            if isinstance(capabilities, dict):
                merged_capabilities = dict(capabilities)
                merged_capabilities["supports_google_search"] = (
                    self._is_google_available()
                )
                return merged_capabilities
        capabilities = dict(DEFAULT_SEARCH_CAPABILITIES)
        capabilities["supports_google_search"] = self._is_google_available()
        return capabilities

    def get_tool_definitions(
        self,
        include_read_spec: bool = False,
        include_internal: bool = False,
    ) -> list[dict]:
        tool_def_getter = getattr(self.search_client, "tool_definitions", None)
        if callable(tool_def_getter):
            try:
                tool_definitions = tool_def_getter(include_read_spec=include_read_spec)
            except TypeError:
                tool_definitions = tool_def_getter()
            if isinstance(tool_definitions, list):
                return tool_definitions
        return build_tool_definitions(
            self.get_search_capabilities(),
            include_read_spec=include_read_spec,
            include_internal=include_internal,
        )

    def execute_request(self, request: ToolCallRequest) -> dict:
        handler = self._handlers.get(canonical_tool_name(request.name))
        if handler is None:
            return {"error": f"Unknown tool: {request.name}"}
        return handler(request.arguments)

    def execute(self, tool_call: ToolCall) -> dict:
        """Execute a single tool call and return a tool result message.

        Args:
            tool_call: Parsed tool call from LLM response.

        Returns:
            Message dict with role="tool" for appending to conversation.
        """
        name = tool_call.name
        args = tool_call.parse_arguments()

        if self.verbose:
            logger.note(f"> Tool call: {name}({args})")

        request = ToolCallRequest(id=tool_call.id, name=name, arguments=args)
        result = self.execute_request(request)

        result_str = json.dumps(result, ensure_ascii=False)

        if self.verbose:
            preview_str = json.dumps(result, ensure_ascii=False, indent=2)
            preview = (
                preview_str[:200] + "..." if len(preview_str) > 200 else preview_str
            )
            logger.success(f"  Tool result ({len(result_str)} chars): {preview}")

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_str,
        }

    def _search_videos(self, args: dict) -> dict:
        """Execute the search_videos tool with multi-query support.

        Accepts either `queries` (array of strings) or a single `query` string.
        Each query is executed independently and results are merged.
        """
        lookup_request = self._extract_video_lookup_request(args)
        if lookup_request is not None:
            return self._lookup_videos(lookup_request)

        mode = str(args.get("mode", "auto") or "auto")
        bvids = _normalize_seed_values(args.get("bvids"))
        mids = _normalize_seed_values(args.get("mids"))
        discover_requested = mode == "discover"

        if discover_requested and bvids:
            if not self._supports_relation_endpoint("related_videos_by_videos"):
                return {
                    "error": "Video discovery unavailable",
                    "bvids": bvids,
                    "videos": [],
                }
            result = self.search_client.related_videos_by_videos(
                bvids=bvids,
                size=int(args.get("size", 10) or 10),
            )
            payload = self._format_relation_video_result(result, "bvids", bvids)
            payload["mode"] = "discover"
            return payload

        if discover_requested and mids:
            if not self._supports_relation_endpoint("related_videos_by_owners"):
                return {
                    "error": "Video discovery unavailable",
                    "mids": mids,
                    "videos": [],
                }
            result = self.search_client.related_videos_by_owners(
                mids=mids,
                size=int(args.get("size", 10) or 10),
            )
            payload = self._format_relation_video_result(result, "mids", mids)
            payload["mode"] = "discover"
            return payload

        queries = args.get("queries", [])
        if not queries:
            query = args.get("query", "")
            queries = [query] if query else []
        if not queries:
            return {"error": "Missing queries parameter", "results": []}

        max_workers = min(max(len(queries), 1), 4)
        if max_workers == 1:
            results = [self._search_single_query(query) for query in queries]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self._search_single_query, queries))

        # Keep single-query responses flat so downstream consumers do not need
        # to unwrap a one-item results list.
        if len(results) == 1:
            return results[0]

        return {"results": results}

    def _get_video_transcript(self, args: dict) -> dict:
        video_id = str(
            args.get("video_id") or args.get("bvid") or args.get("aid") or ""
        ).strip()
        if not video_id:
            return {"error": "Missing video_id parameter"}
        payload = {
            key: value
            for key, value in dict(args or {}).items()
            if key not in {"video_id", "bvid", "aid"} and value not in (None, "")
        }
        if self.transcript_client is not None:
            result = self.transcript_client.get_video_transcript(
                video_id,
                request=payload,
            )
        elif hasattr(self.search_client, "get_video_transcript"):
            result = self.search_client.get_video_transcript(
                video_id=video_id, **payload
            )
        else:
            result = {"error": "Transcript lookup unavailable"}
        if isinstance(result, dict) and not result.get("requested_video_id"):
            result = {"requested_video_id": video_id, **result}
        return result

    def _search_single_query(self, query: str) -> dict:
        """Execute a single search query via explore endpoint."""
        if not query:
            return {"query": query, "error": "Empty query", "hits": [], "total_hits": 0}

        def format_payload(query_text: str, hits: list[dict], total_hits: int) -> dict:
            filtered_hits = filter_relevant_hits_for_llm(hits)
            filtered_hits, theme_filter = _filter_hits_by_llm_theme(
                query_text,
                filtered_hits,
            )
            formatted_hits = format_hits_for_llm(
                filtered_hits,
                max_hits=self.max_results,
            )

            payload = {
                "query": query_text,
                "total_hits": total_hits,
                "hits": formatted_hits,
            }
            if theme_filter:
                payload["theme_filter"] = theme_filter
                if theme_filter.get("warning"):
                    payload["warning"] = theme_filter["warning"]
            return payload

        def run_query(query_text: str) -> dict:
            search_method = self._search_client_method("search")
            if search_method is not None:
                search_result = search_method(
                    query=query_text,
                    limit=self.max_results,
                    verbose=False,
                )
                if "error" not in search_result:
                    payload = format_payload(
                        query_text,
                        search_result.get("hits") or [],
                        int(search_result.get("total_hits") or 0),
                    )
                    if payload.get("hits"):
                        return payload
                elif self._search_client_method("explore") is None:
                    return {
                        "query": query_text,
                        "error": search_result["error"],
                        "hits": [],
                        "total_hits": 0,
                    }

            explore_method = self._search_client_method("explore")
            if explore_method is None:
                return {
                    "query": query_text,
                    "error": "Search explore unavailable",
                    "hits": [],
                    "total_hits": 0,
                }

            explore_result = explore_method(query=query_text)

            if "error" in explore_result:
                return {
                    "query": query_text,
                    "error": explore_result["error"],
                    "hits": [],
                    "total_hits": 0,
                }

            hits, total_hits = extract_explore_hits(explore_result)
            return format_payload(query_text, hits, total_hits)

        primary_result = run_query(query)
        if primary_result.get("error"):
            return primary_result
        if primary_result.get("hits"):
            return primary_result

        for fallback_query in _build_zero_hit_fallback_queries(query):
            fallback_result = run_query(fallback_query)
            if fallback_result.get("error"):
                continue
            if fallback_result.get("hits"):
                return {
                    "query": query,
                    "resolved_query": fallback_query,
                    "fallback_applied": True,
                    "total_hits": fallback_result.get("total_hits", 0),
                    "hits": fallback_result.get("hits", []),
                }

        return primary_result

    def _search_google(self, args: dict) -> dict:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "Missing query parameter", "results": []}
        if not self._is_google_available():
            return {"error": "Google search unavailable", "results": []}
        num = int(args.get("num", 5) or 5)
        lang = args.get("lang")
        result = self.google_client.search(query=query, num=num, lang=lang)
        if result.get("error"):
            return {
                "query": query,
                "error": result["error"],
                "result_count": 0,
                "results": [],
            }
        return {
            "query": query,
            "backend": result.get("backend", ""),
            "result_count": int(
                result.get("result_count", len(result.get("results", []))) or 0
            ),
            "results": format_google_results(result.get("results", []), max_hits=5),
        }

    @staticmethod
    def _owner_source_group(
        *,
        source: str,
        owners: list[dict],
        max_hits: int,
        text: str = "",
        error: str = "",
        query: str = "",
        google_results: list[dict] | None = None,
    ) -> dict:
        payload = {
            "source": source,
            "label": _OWNER_SOURCE_LABELS.get(source, source),
            "total_owners": len(owners),
            "owners": format_related_owners(owners, max_hits=max_hits),
        }
        if text:
            payload["text"] = text
        if error:
            payload["error"] = error
        if query:
            payload["query"] = query
        if google_results is not None:
            payload["google_results"] = google_results[:max_hits]
        return payload

    @staticmethod
    def _merge_owner_source_groups(
        source_owners: list[tuple[str, list[dict]]],
    ) -> list[dict]:
        merged: dict[str, dict] = {}
        for source, owners in source_owners:
            weight = _OWNER_SOURCE_WEIGHTS.get(source, 0.8)
            for rank, owner in enumerate(owners):
                key = _normalize_owner_identity_key(owner)
                if not key:
                    continue
                item = merged.setdefault(
                    key,
                    {
                        "mid": owner.get("mid"),
                        "name": owner.get("name", ""),
                        "score": owner.get("score", 0),
                        "sources": [],
                        "face": owner.get("face", ""),
                        "sample_title": owner.get("sample_title", ""),
                        "sample_bvid": owner.get("sample_bvid", ""),
                        "sample_pic": owner.get("sample_pic", ""),
                        "sample_view": owner.get("sample_view"),
                        "_fusion_score": 0.0,
                        "_best_rank": rank,
                        "_best_owner_score": float(owner.get("score", 0) or 0),
                    },
                )
                item["_fusion_score"] += weight / (rank + 1)
                item["_best_rank"] = min(int(item.get("_best_rank", rank)), rank)
                owner_score = float(owner.get("score", 0) or 0)
                item["_best_owner_score"] = max(
                    float(item.get("_best_owner_score", 0) or 0),
                    owner_score,
                )
                if source not in item["sources"]:
                    item["sources"].append(source)
                if not item.get("name") and owner.get("name"):
                    item["name"] = owner.get("name")
                if item.get("mid") in (None, "") and owner.get("mid"):
                    item["mid"] = owner.get("mid")
                for field in (
                    "face",
                    "sample_title",
                    "sample_bvid",
                    "sample_pic",
                    "sample_view",
                ):
                    if item.get(field) in (None, "") and owner.get(field) not in (
                        None,
                        "",
                    ):
                        item[field] = owner.get(field)

        merged_items = list(merged.values())
        merged_items.sort(
            key=lambda item: (
                -float(item.get("_fusion_score", 0) or 0),
                -len(item.get("sources") or []),
                -float(item.get("_best_owner_score", 0) or 0),
                int(item.get("_best_rank", 999) or 999),
                str(item.get("name", "")),
            )
        )
        for item in merged_items:
            item["score"] = round(float(item.get("_fusion_score", 0) or 0), 4)
            item.pop("_fusion_score", None)
            item.pop("_best_rank", None)
            item.pop("_best_owner_score", None)
        return merged_items

    def _search_owners(self, args: dict) -> dict:
        resolved_args = _coerce_owner_search_args(args)
        text = str(resolved_args.get("text", "")).strip()
        bvids = _normalize_seed_values(resolved_args.get("bvids"))
        mids = _normalize_seed_values(resolved_args.get("mids"))
        requested_mode = str(resolved_args.get("mode", "auto") or "auto")
        relation_requested = requested_mode == "relation" or (
            not text and (bvids or mids)
        )

        if relation_requested and bvids:
            if not self._supports_relation_endpoint("related_owners_by_videos"):
                return {
                    "error": "Owner discovery unavailable",
                    "bvids": bvids,
                    "owners": [],
                }
            result = self.search_client.related_owners_by_videos(
                bvids=bvids,
                size=int(resolved_args.get("size", 10) or 10),
            )
            payload = self._format_relation_owner_result(result, "bvids", bvids)
            payload["mode"] = "relation"
            return payload

        if relation_requested and mids:
            if not self._supports_relation_endpoint("related_owners_by_owners"):
                return {
                    "error": "Owner discovery unavailable",
                    "mids": mids,
                    "owners": [],
                }
            result = self.search_client.related_owners_by_owners(
                mids=mids,
                size=int(resolved_args.get("size", 10) or 10),
            )
            payload = self._format_relation_owner_result(result, "mids", mids)
            payload["mode"] = "relation"
            return payload

        if not text:
            return {"error": "Missing text parameter", "owners": []}
        size = int(resolved_args.get("size", 8) or 8)
        max_owner_hits = max(self.max_results, size)
        google_query = f"{text} site:space.bilibili.com"

        source_specs: dict[str, object] = {
            "name": lambda: self.search_client.search_owners(
                text=text,
                mode="name",
                size=size,
            ),
            "topic": lambda: self.search_client.search_owners(
                text=text,
                mode="topic",
                size=size,
            ),
            "relation": lambda: self.search_client.search_owners(
                text=text,
                mode="relation",
                size=size,
            ),
        }
        if self._supports_relation_endpoint("related_owners_by_tokens"):
            source_specs["related_tokens"] = (
                lambda: self.search_client.related_owners_by_tokens(
                    text=text,
                    size=size,
                )
            )
        if self._is_google_available():
            source_specs["google_space"] = lambda: self.google_client.search(
                query=google_query,
                num=max(5, min(size, 10)),
                lang="zh-CN",
            )

        source_groups: list[dict] = []
        merged_sources: list[tuple[str, list[dict]]] = []
        google_results: list[dict] = []
        errors: list[str] = []

        max_workers = min(max(len(source_specs), 1), 5)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                source: executor.submit(fetcher)
                for source, fetcher in source_specs.items()
            }
            for source, future in futures.items():
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"error": str(exc)}
                if not isinstance(result, dict):
                    result = {}
                if source == "google_space":
                    if result.get("error"):
                        errors.append(str(result["error"]))
                        source_groups.append(
                            self._owner_source_group(
                                source=source,
                                owners=[],
                                max_hits=max_owner_hits,
                                text=text,
                                error=str(result["error"]),
                                query=google_query,
                                google_results=[],
                            )
                        )
                        continue
                    google_results = format_google_results(
                        result.get("results", []),
                        max_hits=max(5, size),
                    )
                    google_owners = []
                    for item in google_results:
                        if item.get("site_kind") != "space" or not item.get("mid"):
                            continue
                        google_owners.append(
                            {
                                "mid": item.get("mid"),
                                "name": _extract_google_space_candidate_name(
                                    item.get("title", "")
                                )
                                or f"B站用户{item.get('mid')}",
                                "score": result.get("result_count", 0),
                                "sample_title": item.get("title", ""),
                                "sources": [source],
                            }
                        )
                    source_groups.append(
                        self._owner_source_group(
                            source=source,
                            owners=google_owners,
                            max_hits=max_owner_hits,
                            text=text,
                            query=google_query,
                            google_results=google_results,
                        )
                    )
                    merged_sources.append((source, google_owners))
                    continue

                if result.get("error"):
                    errors.append(str(result["error"]))
                    source_groups.append(
                        self._owner_source_group(
                            source=source,
                            owners=[],
                            max_hits=max_owner_hits,
                            text=text,
                            error=str(result["error"]),
                        )
                    )
                    continue

                owners = list(result.get("owners") or [])
                source_groups.append(
                    self._owner_source_group(
                        source=source,
                        owners=owners,
                        max_hits=max_owner_hits,
                        text=text,
                    )
                )
                merged_sources.append((source, owners))

        owners = self._merge_owner_source_groups(merged_sources)
        payload = {
            "text": text,
            "mode": "aggregate",
            "requested_mode": requested_mode,
            "total_owners": len(owners),
            "owners": format_related_owners(owners, max_hits=max_owner_hits),
            "source_counts": {
                group["source"]: int(group.get("total_owners", 0) or 0)
                for group in source_groups
            },
            "source_groups": source_groups,
        }
        if google_results:
            payload["google_results"] = google_results[: max(5, size)]
            payload["google_query"] = google_query
        if errors and not owners and not google_results:
            payload["error"] = errors[0]
        return payload

    def _expand_query(self, args: dict) -> dict:
        text = str(args.get("text", "")).strip()
        if not text:
            return {"error": "Missing text parameter", "options": []}
        if not self._supports_relation_endpoint("related_tokens_by_tokens"):
            return {"text": text, "error": "Query expansion unavailable", "options": []}
        requested_mode = str(args.get("mode", "semantic") or "semantic")
        size = int(args.get("size", 8) or 8)
        normalized_text = rewrite_known_term_aliases(text).strip() or text
        result = self.search_client.related_tokens_by_tokens(
            text=normalized_text,
            mode=requested_mode,
            size=size,
        )
        if requested_mode == "semantic" and self._should_retry_expand_query_with_auto(
            result
        ):
            result = self.search_client.related_tokens_by_tokens(
                text=normalized_text,
                mode="auto",
                size=size,
            )
        if result.get("error"):
            return {"text": text, "error": result["error"], "options": []}
        options = result.get("options", [])
        payload = {
            "text": text,
            "mode": result.get("mode", requested_mode),
            "total_options": len(options),
            "options": format_related_token_options(options, max_hits=self.max_results),
        }
        if normalized_text != text:
            payload["normalized_text"] = normalized_text
        return payload

    @staticmethod
    def _should_retry_expand_query_with_auto(result: dict | None) -> bool:
        if not isinstance(result, dict):
            return False
        error = str(result.get("error", "") or "").lower()
        if not error:
            return False
        return all(
            marker in error for marker in _EXPAND_QUERY_UNSUPPORTED_SEMANTIC_MARKERS
        )

    def _format_relation_video_result(
        self, result: dict, seed_key: str, seed_values: list
    ) -> dict:
        if result.get("error"):
            return {seed_key: seed_values, "error": result["error"], "videos": []}
        videos = result.get("videos", [])
        return {
            "relation": result.get("relation", ""),
            seed_key: seed_values,
            "total_videos": len(videos),
            "videos": format_related_videos(videos, max_hits=self.max_results),
        }

    def _format_relation_owner_result(
        self, result: dict, seed_key: str, seed_values: list
    ) -> dict:
        if result.get("error"):
            return {seed_key: seed_values, "error": result["error"], "owners": []}
        owners = result.get("owners", [])
        return {
            "relation": result.get("relation", ""),
            seed_key: seed_values,
            "total_owners": len(owners),
            "owners": format_related_owners(owners, max_hits=self.max_results),
        }

    def _read_spec(self, args: dict) -> dict:
        """Return a whitelisted spec document by name.

        Only documents registered in SPEC_REGISTRY can be read.
        No filesystem access — all content is pre-loaded in memory.
        """
        name = args.get("name", "")
        if name not in SPEC_REGISTRY:
            available = list(SPEC_REGISTRY.keys())
            return {"error": f"Unknown spec: {name}", "available": available}
        return {"name": name, "content": SPEC_REGISTRY[name]}
