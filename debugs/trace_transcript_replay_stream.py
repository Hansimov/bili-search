from __future__ import annotations

import argparse
import json

from llms.chat.handler import ChatHandler
from llms.models import DEFAULT_LARGE_MODEL_CONFIG, DEFAULT_SMALL_MODEL_CONFIG
from llms.models import create_model_clients
from llms.tools.executor import SearchService

from debugs.verify_transcript_orchestration import FileTranscriptClient
from debugs.verify_transcript_orchestration import NoopVideoExplorer
from debugs.verify_transcript_orchestration import NoopVideoSearcher


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace transcript replay SSE output")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--transcript-file", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--thinking", action="store_true", default=False)
    return parser.parse_args()


def preview(text: str, limit: int = 200) -> str:
    text = (text or "").strip().replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def main() -> int:
    args = build_args()
    model_registry, clients = create_model_clients(
        primary_large_config=DEFAULT_LARGE_MODEL_CONFIG,
        verbose=True,
    )
    search_service = SearchService(
        video_searcher=NoopVideoSearcher(),
        video_explorer=NoopVideoExplorer(),
        owner_searcher=None,
        relations_client=None,
        transcript_client=FileTranscriptClient(args.transcript_file, args.video_id),
        verbose=True,
    )
    handler = ChatHandler(
        llm_client=clients[model_registry.primary_large_config],
        small_llm_client=clients[model_registry.primary_small_config],
        search_client=search_service,
        model_registry=model_registry,
        verbose=True,
    )

    content_parts: list[str] = []

    for raw_chunk in handler.handle_stream(
        messages=[{"role": "user", "content": args.query}],
        thinking=args.thinking,
    ):
        if raw_chunk == "[DONE]":
            print("[DONE]")
            break
        chunk = json.loads(raw_chunk)
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}

        if delta.get("reset_reasoning"):
            print(
                "reasoning_reset="
                f"{delta.get('reasoning_phase', 'unknown')}#"
                f"{delta.get('reasoning_iteration', '?')}"
            )
        if delta.get("reasoning_content"):
            print(f"reasoning+= {preview(delta['reasoning_content'])}")
        if delta.get("content"):
            content_parts.append(delta["content"])
            print(f"content+= {preview(delta['content'])}")
        if delta.get("retract_content"):
            print("retract_content=true")

        for event in chunk.get("tool_events") or []:
            print(
                json.dumps(
                    {
                        "iteration": event.get("iteration"),
                        "tools": event.get("tools"),
                    },
                    ensure_ascii=False,
                )
            )

    print("final_content_preview=")
    print(preview("".join(content_parts), limit=1200) or "<empty>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
