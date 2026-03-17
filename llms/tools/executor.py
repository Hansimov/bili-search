"""Tool executor for dispatching and executing tool calls."""

import json
import os
import requests

from tclogger import logger

from llms.llm_client import ToolCall
from llms.prompts.syntax import SEARCH_SYNTAX
from llms.tools.defs import DEFAULT_SEARCH_CAPABILITIES, build_tool_definitions
from llms.tools.utils import (
    extract_explore_hits,
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
        relations_client=None,
        verbose: bool = False,
    ):
        self.video_searcher = video_searcher
        self.video_explorer = video_explorer
        self.relations_client = relations_client
        self.verbose = verbose
        self._capabilities = self._build_capabilities()

    def _build_capabilities(self) -> dict:
        return {
            **DEFAULT_SEARCH_CAPABILITIES,
            "service_type": "local",
            "service_name": "integrated_search",
            "supports_author_check": False,
            "supports_multi_query": True,
            "supports_google_search": False,
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
                "/suggest",
                "/search",
                "/random",
                "/latest",
                "/doc",
                "/knn_search",
                "/hybrid_search",
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
            "available_endpoints": ["/explore", "/suggest", "/health", "/capabilities"],
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

    def suggest(self, query: str, limit: int = 25, verbose: bool = False) -> dict:
        payload = {
            "query": query,
            "limit": limit,
            "verbose": verbose,
        }
        return self._post("/suggest", payload)

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
    ):
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self.verbose = verbose

    def health(self) -> dict:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def search(self, query: str, num: int = 5, lang: str | None = None) -> dict:
        params = {"q": query, "num": num}
        if lang:
            params["lang"] = lang
        try:
            response = requests.get(
                f"{self.base_url}/search",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warn(f"× Google hub request error: {exc}")
            return {"success": False, "error": str(exc), "results": []}


def create_google_search_client(
    *,
    base_url: str | None = None,
    timeout: float | None = None,
    verbose: bool = False,
):
    resolved_base_url = (
        base_url or os.getenv("BILI_GOOGLE_HUB_BASE_URL", "http://127.0.0.1:18100")
    ).strip()
    resolved_timeout = float(
        timeout if timeout is not None else os.getenv("BILI_GOOGLE_HUB_TIMEOUT", "45")
    )
    if not resolved_base_url:
        return None
    return GoogleSearchClient(
        base_url=resolved_base_url,
        timeout=resolved_timeout,
        verbose=verbose,
    )


def create_search_service(
    *,
    video_searcher=None,
    video_explorer=None,
    relations_client=None,
    base_url: str | None = None,
    timeout: float = 30.0,
    verbose: bool = False,
):
    if base_url:
        return SearchServiceClient(base_url=base_url, timeout=timeout, verbose=verbose)
    return SearchService(
        video_searcher=video_searcher,
        video_explorer=video_explorer,
        relations_client=relations_client,
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
        verbose: bool = False,
    ):
        self.search_client = search_client
        self.google_client = google_client or create_google_search_client(
            verbose=verbose
        )
        self.max_results = max_results
        self.verbose = verbose
        self._handlers = {
            "search_videos": self._search_videos,
            "search_google": self._search_google,
            "related_tokens_by_tokens": self._related_tokens_by_tokens,
            "related_owners_by_tokens": self._related_owners_by_tokens,
            "related_videos_by_videos": self._related_videos_by_videos,
            "related_owners_by_videos": self._related_owners_by_videos,
            "related_videos_by_owners": self._related_videos_by_owners,
            "related_owners_by_owners": self._related_owners_by_owners,
            "read_spec": self._read_spec,
        }

    def get_search_capabilities(self, refresh: bool = False) -> dict:
        capability_getter = getattr(self.search_client, "capabilities", None)
        if callable(capability_getter):
            try:
                capabilities = capability_getter(refresh=refresh)
            except TypeError:
                capabilities = capability_getter()
            if isinstance(capabilities, dict):
                merged_capabilities = dict(capabilities)
                if self.google_client is not None:
                    merged_capabilities["supports_google_search"] = True
                return merged_capabilities
        capabilities = dict(DEFAULT_SEARCH_CAPABILITIES)
        if self.google_client is not None:
            capabilities["supports_google_search"] = True
        return capabilities

    def get_tool_definitions(self, include_read_spec: bool = False) -> list[dict]:
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
        )

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

        handler = self._handlers.get(name)
        if handler is None:
            logger.warn(f"× Unknown tool: {name}")
            result = {"error": f"Unknown tool: {name}"}
        else:
            result = handler(args)

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

        Accepts `queries` (array of strings) or legacy `query` (single string).
        Each query is executed independently and results are merged.
        """
        # Support both new `queries` (array) and legacy `query` (string)
        queries = args.get("queries", [])
        if not queries:
            query = args.get("query", "")
            queries = [query] if query else []
        if not queries:
            return {"error": "Missing queries parameter", "results": []}

        results = []
        for query in queries:
            result = self._search_single_query(query)
            results.append(result)

        # For single query, return flat result for backward compatibility
        if len(results) == 1:
            return results[0]

        return {"results": results}

    def _search_single_query(self, query: str) -> dict:
        """Execute a single search query via explore endpoint."""
        if not query:
            return {"query": query, "error": "Empty query", "hits": [], "total_hits": 0}

        explore_result = self.search_client.explore(query=query)

        if "error" in explore_result:
            return {
                "query": query,
                "error": explore_result["error"],
                "hits": [],
                "total_hits": 0,
            }

        hits, total_hits = extract_explore_hits(explore_result)
        formatted_hits = format_hits_for_llm(hits, max_hits=self.max_results)

        return {
            "query": query,
            "total_hits": total_hits,
            "hits": formatted_hits,
        }

    def _search_google(self, args: dict) -> dict:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "Missing query parameter", "results": []}
        if self.google_client is None:
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

    def _related_tokens_by_tokens(self, args: dict) -> dict:
        text = str(args.get("text", "")).strip()
        if not text:
            return {"error": "Missing text parameter", "options": []}
        result = self.search_client.related_tokens_by_tokens(
            text=text,
            mode=args.get("mode", "auto"),
            size=int(args.get("size", 8) or 8),
        )
        if result.get("error"):
            return {"text": text, "error": result["error"], "options": []}
        options = result.get("options", [])
        return {
            "text": text,
            "mode": result.get("mode", args.get("mode", "auto")),
            "total_options": len(options),
            "options": format_related_token_options(options, max_hits=self.max_results),
        }

    def _related_owners_by_tokens(self, args: dict) -> dict:
        text = str(args.get("text", "")).strip()
        if not text:
            return {"error": "Missing text parameter", "owners": []}
        result = self.search_client.related_owners_by_tokens(
            text=text,
            size=int(args.get("size", 8) or 8),
        )
        if result.get("error"):
            return {"text": text, "error": result["error"], "owners": []}
        owners = result.get("owners", [])
        return {
            "text": text,
            "total_owners": len(owners),
            "owners": format_related_owners(owners, max_hits=self.max_results),
        }

    def _related_videos_by_videos(self, args: dict) -> dict:
        bvids = args.get("bvids") or []
        if not bvids:
            return {"error": "Missing bvids parameter", "videos": []}
        result = self.search_client.related_videos_by_videos(
            bvids=bvids,
            size=int(args.get("size", 10) or 10),
        )
        return self._format_relation_video_result(result, "bvids", bvids)

    def _related_owners_by_videos(self, args: dict) -> dict:
        bvids = args.get("bvids") or []
        if not bvids:
            return {"error": "Missing bvids parameter", "owners": []}
        result = self.search_client.related_owners_by_videos(
            bvids=bvids,
            size=int(args.get("size", 10) or 10),
        )
        return self._format_relation_owner_result(result, "bvids", bvids)

    def _related_videos_by_owners(self, args: dict) -> dict:
        mids = args.get("mids") or []
        if not mids:
            return {"error": "Missing mids parameter", "videos": []}
        result = self.search_client.related_videos_by_owners(
            mids=mids,
            size=int(args.get("size", 10) or 10),
        )
        return self._format_relation_video_result(result, "mids", mids)

    def _related_owners_by_owners(self, args: dict) -> dict:
        mids = args.get("mids") or []
        if not mids:
            return {"error": "Missing mids parameter", "owners": []}
        result = self.search_client.related_owners_by_owners(
            mids=mids,
            size=int(args.get("size", 10) or 10),
        )
        return self._format_relation_owner_result(result, "mids", mids)

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
