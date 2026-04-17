from __future__ import annotations

import argparse
import json
import sys
import time

from llms.models import DEFAULT_LARGE_MODEL_CONFIG, DEFAULT_SMALL_MODEL_CONFIG
from llms.models import create_model_clients
from llms.orchestration.engine import ChatOrchestrator
from llms.tools.executor import SearchService, ToolExecutor
from llms.tools.transcripts import BiliStoreTranscriptClient


class NoopVideoSearcher:
    def suggest(self, *args, **kwargs):
        return {"hits": [], "total_hits": 0}


class NoopVideoExplorer:
    def unified_explore(self, *args, **kwargs):
        return {"error": "unexpected search_videos invocation", "data": []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default="BV1YXZPB1Erc")
    parser.add_argument(
        "--query",
        default="请总结 {video_id} 这个视频主要讲了什么，给我 5 条中文要点。",
    )
    args = parser.parse_args()

    model_registry, clients = create_model_clients(
        primary_large_config=DEFAULT_LARGE_MODEL_CONFIG,
        verbose=True,
    )
    transcript_client = BiliStoreTranscriptClient(verbose=True)
    search_service = SearchService(
        video_searcher=NoopVideoSearcher(),
        video_explorer=NoopVideoExplorer(),
        owner_searcher=None,
        relations_client=None,
        transcript_client=transcript_client,
        verbose=True,
    )
    tool_executor = ToolExecutor(search_client=search_service, verbose=True)
    tool_executor._google_available = False
    tool_executor._google_available_ts = time.monotonic()

    orchestrator = ChatOrchestrator(
        llm_client=clients[model_registry.primary_large_config],
        small_llm_client=clients[model_registry.primary_small_config],
        tool_executor=tool_executor,
        model_registry=model_registry,
        verbose=True,
    )

    query = args.query.format(video_id=args.video_id)
    result = orchestrator.run(
        messages=[{"role": "user", "content": query}],
        thinking=False,
    )

    tool_calls = []
    for event in result.tool_events:
        for call in event.get("calls") or []:
            tool_calls.append(call.get("type"))

    payload = {
        "query": query,
        "answer": result.content,
        "tool_calls": tool_calls,
        "usage_trace": result.usage_trace,
        "planner_model": (result.usage_trace.get("models") or {})
        .get("planner", {})
        .get("config"),
        "response_model": (result.usage_trace.get("models") or {})
        .get("response", {})
        .get("config"),
        "transcript_tool_used": "get_video_transcript" in tool_calls,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    planner_model = payload["planner_model"]
    return int(
        not (
            payload["answer"]
            and payload["transcript_tool_used"]
            and planner_model == DEFAULT_SMALL_MODEL_CONFIG
        )
    )


if __name__ == "__main__":
    sys.exit(main())
