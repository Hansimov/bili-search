from __future__ import annotations

import argparse
import json
import sys
import time

from pathlib import Path

from tclogger import logger

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


class FileTranscriptClient:
    def __init__(self, transcript_file: str, video_id: str):
        self.transcript_file = Path(transcript_file)
        self.video_id = video_id
        self.is_configured = True

    def get_video_transcript(self, video_id: str, request: dict | None = None) -> dict:
        text = self.transcript_file.read_text(encoding="utf-8")
        return {
            "ok": True,
            "requested_video_id": video_id,
            "bvid": video_id,
            "title": f"本地回放转写 {video_id}",
            "page_index": 1,
            "selection": {
                "selected_text_length": len(text),
                "full_text_length": len(text),
            },
            "transcript": {
                "text": text,
                "text_length": len(text),
                "segment_count": max(text.count("。"), 1),
            },
            "request": request or {},
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default="BV1YXZPB1Erc")
    parser.add_argument("--transcript-file", default="")
    parser.add_argument(
        "--query",
        default="请总结 {video_id} 这个视频主要讲了什么，给我 5 条中文要点。",
    )
    args = parser.parse_args()

    model_registry, clients = create_model_clients(
        primary_large_config=DEFAULT_LARGE_MODEL_CONFIG,
        verbose=True,
    )
    if args.transcript_file:
        transcript_client = FileTranscriptClient(args.transcript_file, args.video_id)
    else:
        live_transcript_client = BiliStoreTranscriptClient(verbose=True)
        transcript_client = (
            live_transcript_client if live_transcript_client.is_configured else None
        )
        if transcript_client is None:
            logger.warn(
                "× Transcript client is not configured; running with transcript capability disabled."
            )
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
        "transcript_capability_enabled": transcript_client is not None,
        "transcript_tool_used": "get_video_transcript" in tool_calls,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    planner_model = payload["planner_model"]
    if not payload["transcript_capability_enabled"]:
        return int(not (payload["answer"] and not payload["transcript_tool_used"]))
    return int(
        not (
            payload["answer"]
            and payload["transcript_tool_used"]
            and planner_model == DEFAULT_SMALL_MODEL_CONFIG
        )
    )


if __name__ == "__main__":
    sys.exit(main())
