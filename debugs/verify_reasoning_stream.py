"""Verify that handle_stream emits reasoning_content for a real request.

Usage:
    conda run -n ai python debugs/verify_reasoning_stream.py \
      --llm-config deepseek \
      --elastic-index bili_videos_dev6 \
      --elastic-env-name elastic_dev \
      --query "央视新闻最近有什么新视频？"
"""

from __future__ import annotations

import argparse
import json

from llms.runtime import _create_handler


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify reasoning stream output")
    parser.add_argument("--llm-config", type=str, default="deepseek")
    parser.add_argument("--elastic-index", type=str, default="bili_videos_dev6")
    parser.add_argument("--elastic-env-name", type=str, default="elastic_dev")
    parser.add_argument("--search-base-url", type=str, default=None)
    parser.add_argument("--search-timeout", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


def preview(text: str, limit: int = 300) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def main() -> int:
    args = build_args()
    handler = _create_handler(args)
    model_name = getattr(handler.llm_client, "model", "<unknown>")

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_event_count = 0

    stream = handler.handle_stream(
        messages=[{"role": "user", "content": args.query}],
        temperature=args.temperature,
        thinking=args.thinking,
    )

    for raw_chunk in stream:
        if raw_chunk == "[DONE]":
            break
        chunk = json.loads(raw_chunk)
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            reasoning_delta = delta.get("reasoning_content") or ""
            content_delta = delta.get("content") or ""
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
            if content_delta:
                content_parts.append(content_delta)
        tool_events = chunk.get("tool_events") or []
        tool_event_count += len(tool_events)

    reasoning_text = "".join(reasoning_parts)
    content_text = "".join(content_parts)

    print(f"model={model_name}")
    print(f"thinking_flag={args.thinking}")
    print(f"reasoning_chars={len(reasoning_text)}")
    print(f"content_chars={len(content_text)}")
    print(f"tool_event_count={tool_event_count}")
    print("reasoning_preview:")
    print(preview(reasoning_text) or "<empty>")
    print("content_preview:")
    print(preview(content_text) or "<empty>")

    return 0 if reasoning_text else 1


if __name__ == "__main__":
    raise SystemExit(main())
