"""Tool executor for dispatching and executing tool calls.

Handles the execution of search_videos and check_author tools,
using a search service that provides explore() and suggest() methods.
"""

import json

from tclogger import logger

from llms.llm_client import ToolCall
from llms.prompts.syntax import SEARCH_SYNTAX
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

        if name == "search_videos":
            result = self._search_videos(args)
        elif name == "check_author":
            result = self._check_author(args)
        elif name == "read_spec":
            result = self._read_spec(args)
        else:
            logger.warn(f"× Unknown tool: {name}")
            result = {"error": f"Unknown tool: {name}"}

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
