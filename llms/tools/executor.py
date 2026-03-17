"""Tool executor for dispatching and executing tool calls.

Handles the execution of search_videos and check_author tools,
using a search service that provides explore() and suggest() methods.
"""

import json
import requests

from tclogger import logger

from llms.llm_client import ToolCall
from llms.prompts.syntax import SEARCH_SYNTAX
from llms.tools.defs import DEFAULT_SEARCH_CAPABILITIES, build_tool_definitions
from llms.tools.utils import (
    format_hits_for_llm,
    extract_explore_hits,
    analyze_suggest_for_authors,
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

    def __init__(self, video_searcher, video_explorer, verbose: bool = False):
        self.video_searcher = video_searcher
        self.video_explorer = video_explorer
        self.verbose = verbose
        self._capabilities = self._build_capabilities()

    def _build_capabilities(self) -> dict:
        return {
            **DEFAULT_SEARCH_CAPABILITIES,
            "service_type": "local",
            "service_name": "integrated_search",
            "supports_author_check": True,
            "supports_multi_query": True,
            "available_endpoints": [
                "/explore",
                "/suggest",
                "/search",
                "/random",
                "/latest",
                "/doc",
                "/knn_search",
                "/hybrid_search",
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
            "available_endpoints": ["/explore", "/suggest", "/health"],
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


def create_search_service(
    *,
    video_searcher=None,
    video_explorer=None,
    base_url: str | None = None,
    timeout: float = 30.0,
    verbose: bool = False,
):
    if base_url:
        return SearchServiceClient(base_url=base_url, timeout=timeout, verbose=verbose)
    return SearchService(
        video_searcher=video_searcher,
        video_explorer=video_explorer,
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
        verbose: bool = False,
    ):
        self.search_client = search_client
        self.max_results = max_results
        self.verbose = verbose
        self._handlers = {
            "search_videos": self._search_videos,
            "check_author": self._check_author,
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
                return capabilities
        return dict(DEFAULT_SEARCH_CAPABILITIES)

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

    def _check_author(self, args: dict) -> dict:
        """Execute the check_author tool."""
        name = args.get("name", "")
        if not name:
            return {"error": "Missing name parameter"}
        suggest_result = self.search_client.suggest(query=name)

        if "error" in suggest_result:
            return {
                "query": name,
                "error": suggest_result["error"],
                "total_hits": 0,
                "highlighted_keywords": {},
                "related_authors": {},
            }

        return analyze_suggest_for_authors(suggest_result, query=name)

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
