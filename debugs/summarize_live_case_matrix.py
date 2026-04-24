from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def expected_bvid(case_id: str) -> str:
    parts = str(case_id or "").rsplit("_", 1)
    if len(parts) != 2:
        return ""
    suffix = parts[1].strip()
    return suffix if suffix.startswith("BV") else ""


def load_case_index(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    return {
        str(case.get("id") or ""): case
        for case in payload
        if isinstance(case, dict) and case.get("id")
    }


def summarize_case(case: dict, case_index: dict[str, dict]) -> dict:
    case_id = case.get("id") or ""
    source_case = case_index.get(case_id) or {}
    expected = expected_bvid(case.get("id") or "")
    search = case.get("search") or {}
    related = case.get("related") or {}
    intent_info = search.get("intent_info") or {}
    has_search_probe = "search" in case or bool(case.get("search_error"))
    has_related_probe = "related" in case or bool(case.get("related_error"))
    top_hits = list(search.get("top_hits") or [])
    top_bvids = [str(hit.get("bvid") or "").strip() for hit in top_hits]
    seed = source_case.get("seed") or {}
    seed_owner = str(((seed.get("owner") or {}).get("name") or "")).strip()
    top_hit_owners = [str(hit.get("owner") or "").strip() for hit in top_hits]
    category = str(
        source_case.get("category")
        or ((case.get("tags") or [""])[:1] or [""])[0]
        or "unknown"
    )
    has_owner_filter = bool(intent_info.get("owner_filter"))
    has_owner_candidates = bool(intent_info.get("owners"))
    has_chat_probe = "chat" in case or bool(case.get("chat_error"))
    has_chat_case = bool(source_case.get("chat_messages"))
    top1_match = bool(expected and top_bvids[:1] == [expected])
    top3_match = bool(expected and expected in top_bvids[:3])

    success: bool | None = None
    if has_search_probe and category in {
        "title_exact",
        "title_tag_combo",
        "long_desc",
        "boilerplate_noise",
        "single_typo",
        "mixed_script",
    }:
        success = top3_match
    elif has_search_probe and category in {"owner_recent", "owner_topic"}:
        success = bool(seed_owner and seed_owner in top_hit_owners)
    elif has_search_probe and category == "tag_only":
        related_ok = True
        if has_related_probe:
            related_ok = len(list(related.get("options") or [])) > 0
        success = bool(top_hits) and not has_owner_filter and related_ok
    elif has_search_probe and category == "topic_fragment":
        success = bool(top_hits) and not has_owner_filter
    elif has_search_probe:
        success = top3_match

    return {
        "id": case_id,
        "category": category,
        "seed_owner": seed_owner,
        "expected_bvid": expected,
        "has_search_probe": has_search_probe,
        "has_related_probe": has_related_probe,
        "top1_match": top1_match,
        "top3_match": top3_match,
        "success": success,
        "search_empty": len(top_hits) == 0 if has_search_probe else None,
        "related_empty": (
            len(list(related.get("options") or [])) == 0 if has_related_probe else None
        ),
        "search_error": bool(case.get("search_error")),
        "related_error": bool(case.get("related_error")),
        "chat_error": bool(case.get("chat_error")),
        "has_chat_probe": has_chat_probe,
        "has_chat_case": has_chat_case,
        "chat_tools": list((case.get("chat") or {}).get("tools") or []),
        "has_owner_filter": has_owner_filter,
        "has_owner_candidates": has_owner_candidates,
        "search_query": case.get("search_query") or "",
        "top_hits": top_hits,
    }


def print_overall(summary_cases: list[dict]) -> None:
    total = len(summary_cases)
    search_cases = [case for case in summary_cases if case["has_search_probe"]]
    related_cases = [case for case in summary_cases if case["has_related_probe"]]
    success_cases = [case for case in summary_cases if case["success"] is not None]
    print(f"total={total}")
    print(f"search_cases={len(search_cases)}")
    print(f"related_cases={len(related_cases)}")
    print(f"success={sum(1 for case in success_cases if case['success'])}")
    print(f"top1={sum(1 for case in search_cases if case['top1_match'])}")
    print(f"top3={sum(1 for case in search_cases if case['top3_match'])}")
    print(f"search_empty={sum(1 for case in search_cases if case['search_empty'])}")
    print(f"related_empty={sum(1 for case in related_cases if case['related_empty'])}")
    print(f"owner_filter={sum(1 for case in search_cases if case['has_owner_filter'])}")
    print(f"search_error={sum(1 for case in summary_cases if case['search_error'])}")
    print(f"related_error={sum(1 for case in summary_cases if case['related_error'])}")
    print(f"chat_error={sum(1 for case in summary_cases if case['chat_error'])}")


def print_by_category(summary_cases: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in summary_cases:
        grouped[case["category"]].append(case)

    for category in sorted(grouped):
        cases = grouped[category]
        search_cases = [case for case in cases if case["has_search_probe"]]
        related_cases = [case for case in cases if case["has_related_probe"]]
        success_cases = [case for case in cases if case["success"] is not None]
        print(
            "{} total={} success={} top1={} top3={} owner_filter={} search_empty={} related_empty={} chat_error={}".format(
                category,
                len(cases),
                sum(1 for case in success_cases if case["success"]),
                sum(1 for case in search_cases if case["top1_match"]),
                sum(1 for case in search_cases if case["top3_match"]),
                sum(1 for case in search_cases if case["has_owner_filter"]),
                sum(1 for case in search_cases if case["search_empty"]),
                sum(1 for case in related_cases if case["related_empty"]),
                sum(1 for case in cases if case["chat_error"]),
            )
        )


def print_failures(summary_cases: list[dict], limit: int) -> None:
    failures = [
        case
        for case in summary_cases
        if case["has_search_probe"]
        and (case["success"] is False or case["search_empty"] or case["related_empty"])
    ]
    failures.sort(
        key=lambda case: (
            case["category"],
            case["success"],
            case["search_empty"],
            case["related_empty"],
            case["id"],
        )
    )
    for case in failures[:limit]:
        print(
            f"FAIL {case['id']} category={case['category']} "
            f"success={case['success']} "
            f"top1={case['top1_match']} top3={case['top3_match']} "
            f"owner_filter={case['has_owner_filter']} search_empty={case['search_empty']} related_empty={case['related_empty']} "
            f"query={case['search_query']}"
        )
        if case["top_hits"]:
            top_hit = case["top_hits"][0]
            print(
                f"  top1_hit title={top_hit.get('title')} owner={top_hit.get('owner')} bvid={top_hit.get('bvid')}"
            )


def print_chat_summary(summary_cases: list[dict], limit: int) -> None:
    chat_cases = [case for case in summary_cases if case["has_chat_probe"]]
    if not chat_cases:
        return

    tool_counter: Counter[str] = Counter()
    search_video_cases = 0
    no_tool_cases: list[dict] = []
    no_search_video_cases: list[dict] = []
    for case in chat_cases:
        tools = list(case["chat_tools"])
        tool_counter.update(tools)
        if "search_videos" in tools:
            search_video_cases += 1
        elif not case["chat_error"]:
            no_search_video_cases.append(case)
        if not tools and not case["chat_error"]:
            no_tool_cases.append(case)

    print(f"chat_cases={len(chat_cases)}")
    print(f"chat_search_videos={search_video_cases}")
    print(f"chat_no_tools={len(no_tool_cases)}")
    print(f"chat_no_search_videos={len(no_search_video_cases)}")
    for tool_name, count in sorted(tool_counter.items()):
        print(f"chat_tool {tool_name}={count}")

    for case in no_search_video_cases[:limit]:
        print(
            f"CHAT {case['id']} category={case['category']} tools={case['chat_tools']} query={case['search_query']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a live case matrix JSON report"
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--cases-path", type=Path)
    parser.add_argument("--fail-limit", type=int, default=30)
    args = parser.parse_args()

    cases = json.loads(args.report.read_text(encoding="utf-8"))
    case_index = load_case_index(args.cases_path)
    summary_cases = [summarize_case(case, case_index) for case in cases]
    print_overall(summary_cases)
    print_by_category(summary_cases)
    print_failures(summary_cases, args.fail_limit)
    print_chat_summary(summary_cases, args.fail_limit)


if __name__ == "__main__":
    main()
