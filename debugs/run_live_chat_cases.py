"""Run selected live chat cases against a deployed bili-search service.

Usage:
    python debugs/run_live_chat_cases.py
    python debugs/run_live_chat_cases.py --base-url http://127.0.0.1:21001 --case hongjing_timeline
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


CASES = {
    "hongjing_timeline": {
        "messages": [
            {"role": "user", "content": "红警08最近发了什么视频？"},
        ],
    },
    "double_alias_recent_videos": {
        "messages": [
            {"role": "user", "content": "08和月亮3最近都发了哪些视频？"},
        ],
    },
    "similar_hardware_creators": {
        "messages": [
            {
                "role": "user",
                "content": "和影视飓风风格接近，但更偏硬件评测的作者有哪些？",
            },
        ],
    },
    "gemini_api_followup": {
        "messages": [
            {
                "role": "user",
                "content": "Gemini 2.5 最近官方更新里，和开发者 API 最相关的点有哪些，B站有没有偏 API 侧的解读？",
            },
        ],
    },
    "productivity_compare": {
        "messages": [
            {
                "role": "user",
                "content": "对比一下老番茄和红警08最近一个月谁更高产",
            },
        ],
    },
    "he_tongxue_accounts_then_works": {
        "messages": [
            {"role": "user", "content": "何同学有哪些关联账号？那他的代表作有哪些？"},
        ],
    },
    "comfyui_alias": {
        "messages": [
            {"role": "user", "content": "康夫UI 有什么入门教程？"},
        ],
    },
}


def flatten_tool_names(tool_events: list[dict] | None) -> list[str]:
    names: list[str] = []
    for event in tool_events or []:
        names.extend(event.get("tools") or [])
    return names


def post_chat(base_url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live chat cases")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:21001",
        help="Chat service base URL",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES.keys()),
        help="Case ID to run; repeat to run multiple",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=4,
        help="Maximum tool iterations",
    )
    args = parser.parse_args()

    case_ids = args.case or list(CASES.keys())
    for case_id in case_ids:
        case = CASES[case_id]
        payload = {
            "messages": case["messages"],
            "stream": False,
            "thinking": args.thinking,
            "max_iterations": args.max_iterations,
        }

        print(f"\n=== {case_id} ===")
        print("messages:")
        for message in case["messages"]:
            print(f"- {message['role']}: {message['content']}")

        started = time.perf_counter()
        try:
            result = post_chat(args.base_url, payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP {exc.code}: {body}")
            continue
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        elapsed = time.perf_counter() - started
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        tool_events = result.get("tool_events", [])
        usage = result.get("usage", {})
        print(f"elapsed: {elapsed:.2f}s")
        print(f"tools: {flatten_tool_names(tool_events)}")
        print(f"usage: {usage}")
        print("content:")
        print(content)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
