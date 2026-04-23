"""Run a reusable live-case matrix against the local bili-search service.

Usage:
    python debugs/run_live_case_matrix.py
    python debugs/run_live_case_matrix.py --case comfyui_alias_tutorial --skip-chat
    python debugs/run_live_case_matrix.py --output-json /tmp/live-matrix.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from configs.envs import SEARCH_APP_ENVS


DEFAULT_BASE_URL = f"http://127.0.0.1:{SEARCH_APP_ENVS.get('port', 21001)}"
DEFAULT_CASES_PATH = Path(__file__).with_name("live_case_matrix.json")


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError(f"Expected a list of cases in {path}")
    return cases


def post_json(
    base_url: str, endpoint: str, payload: dict[str, Any], timeout: int
) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def flatten_tool_names(tool_events: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for event in tool_events or []:
        names.extend(event.get("tools") or [])
    return names


def truncate_text(text: str, limit: int = 800) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def summarize_search(result: dict[str, Any], limit: int) -> dict[str, Any]:
    return {
        "semantic_rewrite_info": result.get("semantic_rewrite_info") or {},
        "intent_info": result.get("intent_info") or {},
        "retry_info": result.get("retry_info") or {},
        "total_hits": result.get("total_hits", 0),
        "top_hits": [
            {
                "title": hit.get("title"),
                "owner": (hit.get("owner") or {}).get("name"),
                "bvid": hit.get("bvid"),
            }
            for hit in (result.get("hits") or [])[:limit]
        ],
    }


def summarize_related(result: dict[str, Any], limit: int) -> dict[str, Any]:
    return {
        "mode": result.get("mode"),
        "options": [
            {
                "text": option.get("text"),
                "score": option.get("score"),
                "keywords": option.get("keywords"),
            }
            for option in (result.get("options") or [])[:limit]
        ],
    }


def summarize_chat(result: dict[str, Any]) -> dict[str, Any]:
    message = (result.get("choices") or [{}])[0].get("message") or {}
    return {
        "tools": flatten_tool_names(result.get("tool_events") or []),
        "usage": result.get("usage") or {},
        "content": truncate_text(message.get("content") or ""),
    }


def select_cases(
    cases: list[dict[str, Any]],
    case_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if not case_ids:
        return cases
    selected = []
    for case_id in case_ids:
        matched = next((case for case in cases if case.get("id") == case_id), None)
        if not matched:
            raise ValueError(f"Unknown case id: {case_id}")
        selected.append(matched)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live bili-search case matrix")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Service base URL")
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to the JSON case matrix",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Case ID to run; repeat to run multiple cases",
    )
    parser.add_argument("--limit", type=int, default=5, help="Top hits/options to show")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=4,
        help="Maximum tool iterations for chat completions",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode for chat completions",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip the /search probe",
    )
    parser.add_argument(
        "--skip-related",
        action="store_true",
        help="Skip the /related_tokens_by_tokens probe",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Skip the /chat/completions probe",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to save the summarized results as JSON",
    )
    args = parser.parse_args()

    cases = select_cases(load_cases(args.cases_path), args.case)
    matrix_results: list[dict[str, Any]] = []

    for case in cases:
        case_result: dict[str, Any] = {
            "id": case.get("id"),
            "tags": case.get("tags") or [],
            "search_query": case.get("search_query") or "",
        }
        print(f"\n=== {case_result['id']} ===")
        if case_result["tags"]:
            print(f"tags: {', '.join(case_result['tags'])}")
        print(f"search_query: {case_result['search_query']}")

        if not args.skip_search and case_result["search_query"]:
            started = time.perf_counter()
            try:
                search_result = post_json(
                    args.base_url,
                    "/search",
                    {"query": case_result["search_query"], "limit": args.limit},
                    timeout=args.timeout,
                )
                case_result["search"] = summarize_search(search_result, args.limit)
                case_result["search_elapsed_s"] = round(
                    time.perf_counter() - started,
                    2,
                )
                print("search:")
                print(json.dumps(case_result["search"], ensure_ascii=False, indent=2))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                case_result["search_error"] = f"HTTP {exc.code}: {body}"
                print(f"search_error: {case_result['search_error']}")
            except Exception as exc:
                case_result["search_error"] = str(exc)
                print(f"search_error: {case_result['search_error']}")

        if not args.skip_related and case_result["search_query"]:
            started = time.perf_counter()
            try:
                related_result = post_json(
                    args.base_url,
                    "/related_tokens_by_tokens",
                    {
                        "text": case_result["search_query"],
                        "mode": case.get("related_mode") or "auto",
                        "size": args.limit,
                        "scan_limit": 128,
                        "use_pinyin": True,
                    },
                    timeout=args.timeout,
                )
                case_result["related"] = summarize_related(related_result, args.limit)
                case_result["related_elapsed_s"] = round(
                    time.perf_counter() - started,
                    2,
                )
                print("related_tokens:")
                print(json.dumps(case_result["related"], ensure_ascii=False, indent=2))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                case_result["related_error"] = f"HTTP {exc.code}: {body}"
                print(f"related_error: {case_result['related_error']}")
            except Exception as exc:
                case_result["related_error"] = str(exc)
                print(f"related_error: {case_result['related_error']}")

        if not args.skip_chat and case.get("chat_messages"):
            started = time.perf_counter()
            try:
                chat_result = post_json(
                    args.base_url,
                    "/chat/completions",
                    {
                        "messages": case["chat_messages"],
                        "stream": False,
                        "thinking": args.thinking,
                        "max_iterations": args.max_iterations,
                    },
                    timeout=args.timeout,
                )
                case_result["chat"] = summarize_chat(chat_result)
                case_result["chat_elapsed_s"] = round(
                    time.perf_counter() - started,
                    2,
                )
                print("chat:")
                print(json.dumps(case_result["chat"], ensure_ascii=False, indent=2))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                case_result["chat_error"] = f"HTTP {exc.code}: {body}"
                print(f"chat_error: {case_result['chat_error']}")
            except Exception as exc:
                case_result["chat_error"] = str(exc)
                print(f"chat_error: {case_result['chat_error']}")

        matrix_results.append(case_result)

    if args.output_json:
        args.output_json.write_text(
            json.dumps(matrix_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved JSON report to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
