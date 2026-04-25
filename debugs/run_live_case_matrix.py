"""Run a reusable live-case matrix against the local bili-search service.

Usage:
    python debugs/run_live_case_matrix.py
    python debugs/run_live_case_matrix.py --case comfyui_alias_tutorial --skip-chat
    python debugs/run_live_case_matrix.py --output-json debugs/live_case_reports/live-matrix.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from configs.envs import SEARCH_APP_ENVS


DEFAULT_BASE_URL = f"http://127.0.0.1:{SEARCH_APP_ENVS.get('port', 21001)}"
DEFAULT_CASES_PATH = Path(__file__).with_name("live_case_matrix.json")
DEFAULT_REPORTS_DIR = Path(__file__).with_name("live_case_reports")


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


def summarize_explore(result: dict[str, Any], limit: int) -> dict[str, Any]:
    steps = result.get("data") or []
    first_output = (steps[0].get("output") or {}) if steps else {}
    group_output = {}
    for step in steps:
        if step.get("name") == "group_hits_by_owner":
            group_output = step.get("output") or {}
            break
    return {
        "intent_info": result.get("intent_info") or {},
        "retry_info": result.get("retry_info") or {},
        "qmod": first_output.get("qmod"),
        "filter_only": first_output.get("filter_only", False),
        "total_hits": first_output.get("total_hits", 0),
        "top_hits": [
            {
                "title": hit.get("title"),
                "owner": (hit.get("owner") or {}).get("name"),
                "bvid": hit.get("bvid"),
            }
            for hit in (first_output.get("hits") or [])[:limit]
        ],
        "authors": [
            {
                "name": author.get("name"),
                "mid": author.get("mid"),
                "count": author.get("count"),
            }
            for author in (group_output.get("authors") or [])[:limit]
        ],
        "perf": result.get("perf") or first_output.get("perf") or {},
    }


def summarize_chat(result: dict[str, Any]) -> dict[str, Any]:
    message = (result.get("choices") or [{}])[0].get("message") or {}
    return {
        "tools": flatten_tool_names(result.get("tool_events") or []),
        "usage": result.get("usage") or {},
        "content": truncate_text(message.get("content") or ""),
    }


def should_retry_chat_error(message: str) -> bool:
    normalized = str(message or "").lower()
    return "timed out" in normalized or normalized.startswith("http 5")


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


def resolve_output_path(path: Path | None) -> Path:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_REPORTS_DIR / f"live-case-matrix-{timestamp}.json"


def write_results(path: Path, results: list[dict[str, Any] | None]) -> None:
    completed = [result for result in results if result is not None]
    path.write_text(
        json.dumps(completed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_case(
    case: dict[str, Any],
    *,
    base_url: str,
    limit: int,
    timeout: int,
    max_iterations: int,
    thinking: bool,
    skip_search: bool,
    skip_related: bool,
    skip_explore: bool,
    skip_chat: bool,
    chat_retries: int,
    chat_semaphore: threading.Semaphore | None,
) -> tuple[dict[str, Any], str]:
    case_result: dict[str, Any] = {
        "id": case.get("id"),
        "tags": case.get("tags") or [],
        "search_query": case.get("search_query") or "",
    }
    log_lines = [f"\n=== {case_result['id']} ==="]
    if case_result["tags"]:
        log_lines.append(f"tags: {', '.join(case_result['tags'])}")
    log_lines.append(f"search_query: {case_result['search_query']}")

    if not skip_search and case_result["search_query"]:
        started = time.perf_counter()
        try:
            search_result = post_json(
                base_url,
                "/search",
                {"query": case_result["search_query"], "limit": limit},
                timeout=timeout,
            )
            case_result["search"] = summarize_search(search_result, limit)
            case_result["search_elapsed_s"] = round(time.perf_counter() - started, 2)
            log_lines.append("search:")
            log_lines.append(
                json.dumps(case_result["search"], ensure_ascii=False, indent=2)
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            case_result["search_error"] = f"HTTP {exc.code}: {body}"
            log_lines.append(f"search_error: {case_result['search_error']}")
        except Exception as exc:
            case_result["search_error"] = str(exc)
            log_lines.append(f"search_error: {case_result['search_error']}")

    if not skip_related and case_result["search_query"]:
        started = time.perf_counter()
        try:
            related_result = post_json(
                base_url,
                "/related_tokens_by_tokens",
                {
                    "text": case_result["search_query"],
                    "mode": case.get("related_mode") or "auto",
                    "size": limit,
                    "scan_limit": 128,
                    "use_pinyin": True,
                },
                timeout=timeout,
            )
            case_result["related"] = summarize_related(related_result, limit)
            case_result["related_elapsed_s"] = round(
                time.perf_counter() - started,
                2,
            )
            log_lines.append("related_tokens:")
            log_lines.append(
                json.dumps(case_result["related"], ensure_ascii=False, indent=2)
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            case_result["related_error"] = f"HTTP {exc.code}: {body}"
            log_lines.append(f"related_error: {case_result['related_error']}")
        except Exception as exc:
            case_result["related_error"] = str(exc)
            log_lines.append(f"related_error: {case_result['related_error']}")

    if not skip_explore and case_result["search_query"]:
        started = time.perf_counter()
        try:
            explore_payload: dict[str, Any] = {
                "query": case_result["search_query"],
                "verbose": False,
            }
            if case.get("qmod") is not None:
                explore_payload["qmod"] = case.get("qmod")
            explore_result = post_json(
                base_url,
                "/explore",
                explore_payload,
                timeout=timeout,
            )
            case_result["explore"] = summarize_explore(explore_result, limit)
            case_result["explore_elapsed_s"] = round(
                time.perf_counter() - started,
                2,
            )
            log_lines.append("explore:")
            log_lines.append(
                json.dumps(case_result["explore"], ensure_ascii=False, indent=2)
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            case_result["explore_error"] = f"HTTP {exc.code}: {body}"
            log_lines.append(f"explore_error: {case_result['explore_error']}")
        except Exception as exc:
            case_result["explore_error"] = str(exc)
            log_lines.append(f"explore_error: {case_result['explore_error']}")

    if not skip_chat and case.get("chat_messages"):
        started: float | None = None
        attempts = 0
        semaphore = chat_semaphore or threading.Semaphore(1)
        while True:
            attempts += 1
            try:
                with semaphore:
                    if started is None:
                        started = time.perf_counter()
                    chat_result = post_json(
                        base_url,
                        "/chat/completions",
                        {
                            "messages": case["chat_messages"],
                            "stream": False,
                            "thinking": thinking,
                            "max_iterations": max_iterations,
                        },
                        timeout=timeout,
                    )
                case_result["chat"] = summarize_chat(chat_result)
                case_result["chat_elapsed_s"] = round(
                    time.perf_counter() - (started or time.perf_counter()),
                    2,
                )
                if attempts > 1:
                    case_result["chat_attempts"] = attempts
                log_lines.append("chat:")
                log_lines.append(
                    json.dumps(case_result["chat"], ensure_ascii=False, indent=2)
                )
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                message = f"HTTP {exc.code}: {body}"
                if attempts <= chat_retries and should_retry_chat_error(message):
                    log_lines.append(
                        f"chat_retry[{attempts}/{chat_retries}] error: {message}"
                    )
                    time.sleep(min(2.0, 0.5 * attempts))
                    continue
                case_result["chat_error"] = message
                if attempts > 1:
                    case_result["chat_attempts"] = attempts
                log_lines.append(f"chat_error: {case_result['chat_error']}")
                break
            except Exception as exc:
                message = str(exc)
                if attempts <= chat_retries and should_retry_chat_error(message):
                    log_lines.append(
                        f"chat_retry[{attempts}/{chat_retries}] error: {message}"
                    )
                    time.sleep(min(2.0, 0.5 * attempts))
                    continue
                case_result["chat_error"] = message
                if attempts > 1:
                    case_result["chat_attempts"] = attempts
                log_lines.append(f"chat_error: {case_result['chat_error']}")
                break

    return case_result, "\n".join(log_lines)


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
        "--skip-explore",
        action="store_true",
        help="Skip the /explore probe",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Skip the /chat/completions probe",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to save the summarized results as JSON (defaults under debugs/live_case_reports)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of cases to run concurrently",
    )
    parser.add_argument(
        "--chat-workers",
        type=int,
        default=1,
        help="Cap for concurrent /chat/completions probes (default: 1 for stable live runs)",
    )
    parser.add_argument(
        "--chat-retries",
        type=int,
        default=1,
        help="Retry count for transient chat 5xx/timeout failures",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print progress lines, not full per-case JSON logs",
    )
    parser.add_argument(
        "--no-incremental-write",
        action="store_true",
        help="Only write the JSON report once all cases finish",
    )
    args = parser.parse_args()

    cases = select_cases(load_cases(args.cases_path), args.case)
    output_path = resolve_output_path(args.output_json)
    worker_count = max(1, args.workers)
    chat_worker_count = max(1, args.chat_workers)
    chat_semaphore = None if args.skip_chat else threading.Semaphore(chat_worker_count)
    matrix_results: list[dict[str, Any] | None] = [None] * len(cases)
    print(
        f"Running {len(cases)} cases with workers={worker_count}, chat_workers={chat_worker_count}; report -> {output_path}"
    )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(
                run_case,
                case,
                base_url=args.base_url,
                limit=args.limit,
                timeout=args.timeout,
                max_iterations=args.max_iterations,
                thinking=args.thinking,
                skip_search=args.skip_search,
                skip_related=args.skip_related,
                skip_explore=args.skip_explore,
                skip_chat=args.skip_chat,
                chat_retries=max(0, args.chat_retries),
                chat_semaphore=chat_semaphore,
            ): index
            for index, case in enumerate(cases)
        }

        for completed_count, future in enumerate(
            as_completed(future_to_index), start=1
        ):
            index = future_to_index[future]
            case_result, case_log = future.result()
            matrix_results[index] = case_result
            print(f"[{completed_count}/{len(cases)}] completed {case_result['id']}")
            if not args.quiet:
                print(case_log)
            if not args.no_incremental_write:
                write_results(output_path, matrix_results)

    write_results(output_path, matrix_results)
    print(f"\nSaved JSON report to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
