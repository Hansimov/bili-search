from __future__ import annotations

import argparse
import json
import time

import requests


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace live chat SSE timing and tool-event payloads"
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:21001/chat/completions",
        help="Chat completions endpoint",
    )
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Stop after printing this many SSE events (0 means unlimited)",
    )
    parser.add_argument(
        "--skip-reasoning",
        action="store_true",
        default=False,
        help="Hide reasoning_content deltas and only print tool/content milestones",
    )
    return parser.parse_args()


def preview(text: str, limit: int = 80) -> str:
    value = str(text or "").replace("\n", "\\n").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def main() -> int:
    args = build_args()
    payload = {
        "messages": [{"role": "user", "content": args.query}],
        "stream": True,
        "thinking": args.thinking,
    }
    start = time.perf_counter()
    last = start
    printed_count = 0

    with requests.post(
        args.url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data: "):
                continue

            now = time.perf_counter()
            since_start = (now - start) * 1000
            since_last = (now - last) * 1000
            last = now
            data_str = raw_line[6:].strip()
            if data_str == "[DONE]":
                print(
                    f"[{since_start:8.1f} ms | +{since_last:7.1f} ms] [DONE]",
                    flush=True,
                )
                printed_count += 1
                break

            try:
                payload = json.loads(data_str)
            except json.JSONDecodeError:
                print(
                    f"[{since_start:8.1f} ms | +{since_last:7.1f} ms] malformed="
                    f"{preview(data_str, limit=160)}",
                    flush=True,
                )
                printed_count += 1
                continue

            if payload.get("stream_id") and not payload.get("choices"):
                print(
                    f"[{since_start:8.1f} ms | +{since_last:7.1f} ms] stream_id="
                    f"{payload['stream_id']}",
                    flush=True,
                )
                printed_count += 1
                if args.max_lines and printed_count >= args.max_lines:
                    break
                continue

            choices = payload.get("choices") or []
            choice = choices[0] if choices else {}
            delta = choice.get("delta") or {}

            parts: list[str] = []
            if delta.get("role"):
                parts.append(f"role={delta['role']}")
            if delta.get("reset_reasoning"):
                phase = delta.get("reasoning_phase") or "unknown"
                parts.append(f"reset_reasoning={phase}")
            if delta.get("reasoning_content") and not args.skip_reasoning:
                parts.append(
                    "reasoning="
                    + preview(str(delta.get("reasoning_content") or ""), limit=48)
                )
            if delta.get("content"):
                parts.append(
                    "content=" + preview(str(delta.get("content") or ""), limit=48)
                )

            for event in payload.get("tool_events") or []:
                calls = event.get("calls") or []
                summarized_calls: list[str] = []
                for call in calls:
                    result = call.get("result") or {}
                    result_text = str(result.get("result") or "")
                    summarized_calls.append(
                        f"{call.get('type')}:{call.get('status')}"
                        + (
                            f":len={len(result_text)}"
                            if call.get("type") == "run_small_llm_task"
                            else ""
                        )
                    )
                if summarized_calls:
                    parts.append(
                        f"tool_event#{event.get('iteration')}="
                        + ", ".join(summarized_calls)
                    )
                else:
                    parts.append(
                        f"tool_event#{event.get('iteration')}="
                        + ",".join(event.get("tools") or [])
                    )

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                parts.append(f"finish={finish_reason}")

            if not parts and args.skip_reasoning:
                continue

            if not parts:
                parts.append("chunk")

            print(
                f"[{since_start:8.1f} ms | +{since_last:7.1f} ms] " + " | ".join(parts),
                flush=True,
            )
            printed_count += 1
            if args.max_lines and printed_count >= args.max_lines:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
