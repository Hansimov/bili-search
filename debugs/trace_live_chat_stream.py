"""Trace live streaming chat output from a running bili-search instance.

Usage:
    conda run -n ai python debugs/trace_live_chat_stream.py \
      --base-url http://127.0.0.1:21001 \
      --query "08和月亮3最近都发了哪些视频？"
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import requests


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace live chat SSE output")
    parser.add_argument("--base-url", default="http://127.0.0.1:21001")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def preview(text: str, limit: int = 240) -> str:
    text = (text or "").strip().replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"type": type(result).__name__}
    if isinstance(result.get("hits"), list):
        return {
            "query": result.get("query", ""),
            "total_hits": result.get("total_hits", len(result.get("hits") or [])),
            "hit_titles": [
                (hit or {}).get("title", "") for hit in (result.get("hits") or [])[:3]
            ],
        }
    if isinstance(result.get("results"), list):
        nested = []
        for item in result.get("results")[:4]:
            if not isinstance(item, dict):
                nested.append({"type": type(item).__name__})
                continue
            nested.append(
                {
                    "query": item.get("query", ""),
                    "total_hits": item.get("total_hits", len(item.get("hits") or [])),
                    "hit_titles": [
                        (hit or {}).get("title", "")
                        for hit in (item.get("hits") or [])[:2]
                    ],
                    "keys": sorted(item.keys()),
                }
            )
        return {"results": nested}
    return result


def iter_sse_events(response: requests.Response):
    event_lines: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        line = raw_line or ""
        if line == "":
            if event_lines:
                yield "\n".join(event_lines)
                event_lines = []
            continue
        if line.startswith(":"):
            continue
        event_lines.append(line)
    if event_lines:
        yield "\n".join(event_lines)


def main() -> int:
    args = build_args()
    response = requests.post(
        f"{args.base_url.rstrip('/')}/chat/completions",
        json={
            "model": args.model,
            "messages": [{"role": "user", "content": args.query}],
            "stream": True,
            "thinking": args.thinking,
        },
        timeout=args.timeout,
        stream=True,
    )
    response.raise_for_status()

    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    for raw_event in iter_sse_events(response):
        data_lines = []
        for line in raw_event.split("\n"):
            if line.startswith("data: "):
                data_lines.append(line[6:])
            elif line.startswith("data:"):
                data_lines.append(line[5:])
        if not data_lines:
            continue
        data_str = "\n".join(data_lines).strip()
        if not data_str or data_str == "[DONE]":
            continue

        payload = json.loads(data_str)
        if payload.get("stream_id"):
            print(f"stream_id={payload['stream_id']}")
            continue

        choices = payload.get("choices") or []
        delta = (choices[0].get("delta") or {}) if choices else {}
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
            print(f"reasoning+= {preview(delta['reasoning_content'])}")
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("retract_content"):
            print("retract_content=true")

        for event in payload.get("tool_events") or []:
            print(
                f"tool_event iteration={event.get('iteration')} tools={event.get('tools')}"
            )
            for call in event.get("calls") or []:
                result = call.get("result")
                print(
                    json.dumps(
                        {
                            "type": call.get("type"),
                            "status": call.get("status"),
                            "args": call.get("args"),
                            "result": (
                                compact_result(result) if result is not None else None
                            ),
                        },
                        ensure_ascii=False,
                    )
                )

    print("final_reasoning_preview=")
    print(preview("".join(reasoning_parts), limit=1200) or "<empty>")
    print("final_content_preview=")
    print(preview("".join(content_parts), limit=800) or "<empty>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
