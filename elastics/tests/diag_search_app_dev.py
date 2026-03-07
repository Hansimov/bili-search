import argparse
import json
import urllib.request
import urllib.error

from llms.tools.utils import extract_explore_authors, extract_explore_hits


DEFAULT_CHAT_SUITE = [
    {
        "name": "owners_general",
        "query": "推荐几个做黑神话悟空内容的UP主",
    },
    {
        "name": "owners_walkthrough",
        "query": "想找专门做黑神话悟空攻略的UP主",
    },
    {
        "name": "owners_story",
        "query": "有没有偏剧情解析的黑神话悟空创作者",
    },
    {
        "name": "owners_funny",
        "query": "推荐几个做黑神话悟空整活搞笑向的UP主",
    },
]

DEFAULT_CHAT_CONTRAST_SUITE = [
    {
        "name": "videos_hot",
        "query": "推荐几条高播放的黑神话悟空视频",
        "require_any": ["search_videos"],
        "forbid": ["search_owners"],
    },
    {
        "name": "author_recent",
        "query": "影视飓风最近有什么新视频",
        "require_any": ["check_author", "search_videos"],
        "forbid": ["search_owners"],
    },
    {
        "name": "keyword_search",
        "query": "找几条黑神话悟空剧情解析视频",
        "require_any": ["search_videos"],
        "forbid": ["search_owners"],
    },
]


def post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize_chat_result(
    name: str, query: str, result: dict, preview_chars: int
) -> dict:
    tool_events = result.get("tool_events") or []
    tool_names = []
    for event in tool_events:
        tool_names.extend(event.get("tools") or [])

    content = (
        (result.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    )
    return {
        "name": name,
        "query": query,
        "tool_events": tool_events,
        "tool_names": tool_names,
        "used_search_owners": "search_owners" in tool_names,
        "answer_preview": content[:preview_chars],
    }


def evaluate_tool_expectations(
    scenario: dict,
    result_summary: dict,
) -> dict:
    tool_names = result_summary["tool_names"]
    required_any = scenario.get("require_any") or []
    forbidden = scenario.get("forbid") or []
    missing_required = required_any and not any(
        tool in tool_names for tool in required_any
    )
    hit_forbidden = [tool for tool in forbidden if tool in tool_names]
    return {
        **result_summary,
        "require_any": required_any,
        "forbid": forbidden,
        "missing_required": bool(missing_required),
        "hit_forbidden": hit_forbidden,
        "passed": (not missing_required) and (not hit_forbidden),
    }


def main(args: argparse.Namespace):
    base_url = args.base_url.rstrip("/")
    if args.mode == "health":
        print(
            json.dumps(
                get_json(f"{base_url}/health", timeout=args.request_timeout),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.mode == "explore":
        payload = {"query": args.query, "verbose": False}
        result = post_json(f"{base_url}/explore", payload, timeout=args.request_timeout)
        authors = extract_explore_authors(result)
        hits, total_hits = extract_explore_hits(result)
        output = {
            "query": args.query,
            "total_hits": total_hits,
            "hits_count": len(hits),
            "authors_count": len(authors),
            "authors": authors[: args.limit],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    if args.mode == "explore_raw":
        payload = {"query": args.query, "verbose": False}
        result = post_json(f"{base_url}/explore", payload, timeout=args.request_timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mode == "chat":
        payload = {
            "messages": [{"role": "user", "content": args.query}],
            "stream": False,
            "thinking": args.thinking,
            "max_iterations": args.max_iterations,
        }
        result = post_json(
            f"{base_url}/chat/completions",
            payload,
            timeout=args.request_timeout,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mode == "chat_suite":
        scenarios = []
        for item in DEFAULT_CHAT_SUITE:
            payload = {
                "messages": [{"role": "user", "content": item["query"]}],
                "stream": False,
                "thinking": args.thinking,
                "max_iterations": args.max_iterations,
            }
            try:
                result = post_json(
                    f"{base_url}/chat/completions",
                    payload,
                    timeout=args.request_timeout,
                )
                scenarios.append(
                    summarize_chat_result(
                        item["name"],
                        item["query"],
                        result,
                        preview_chars=args.preview_chars,
                    )
                )
            except Exception as exc:
                scenarios.append(
                    {
                        "name": item["name"],
                        "query": item["query"],
                        "tool_events": [],
                        "tool_names": [],
                        "used_search_owners": False,
                        "answer_preview": "",
                        "error": str(exc),
                    }
                )

        missed = [item["name"] for item in scenarios if not item["used_search_owners"]]
        output = {
            "base_url": base_url,
            "scenario_count": len(scenarios),
            "used_search_owners_count": sum(
                1 for item in scenarios if item["used_search_owners"]
            ),
            "missed": missed,
            "scenarios": scenarios,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if args.fail_on_miss and missed:
            raise SystemExit(1)
        return

    if args.mode == "chat_contrast_suite":
        scenarios = []
        for item in DEFAULT_CHAT_CONTRAST_SUITE:
            payload = {
                "messages": [{"role": "user", "content": item["query"]}],
                "stream": False,
                "thinking": args.thinking,
                "max_iterations": args.max_iterations,
            }
            try:
                result = post_json(
                    f"{base_url}/chat/completions",
                    payload,
                    timeout=args.request_timeout,
                )
                summary = summarize_chat_result(
                    item["name"],
                    item["query"],
                    result,
                    preview_chars=args.preview_chars,
                )
                scenarios.append(evaluate_tool_expectations(item, summary))
            except Exception as exc:
                scenarios.append(
                    {
                        "name": item["name"],
                        "query": item["query"],
                        "tool_events": [],
                        "tool_names": [],
                        "used_search_owners": False,
                        "answer_preview": "",
                        "require_any": item.get("require_any") or [],
                        "forbid": item.get("forbid") or [],
                        "missing_required": True,
                        "hit_forbidden": [],
                        "passed": False,
                        "error": str(exc),
                    }
                )

        failed = [item["name"] for item in scenarios if not item["passed"]]
        output = {
            "base_url": base_url,
            "scenario_count": len(scenarios),
            "passed_count": sum(1 for item in scenarios if item["passed"]),
            "failed": failed,
            "scenarios": scenarios,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if args.fail_on_miss and failed:
            raise SystemExit(1)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--base-url", default="http://127.0.0.1:21001")
    parser.add_argument(
        "-m",
        "--mode",
        choices=[
            "health",
            "explore",
            "explore_raw",
            "chat",
            "chat_suite",
            "chat_contrast_suite",
        ],
        default="health",
    )
    parser.add_argument("-q", "--query", default="黑神话悟空")
    parser.add_argument("-l", "--limit", type=int, default=5)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--preview-chars", type=int, default=160)
    parser.add_argument("--fail-on-miss", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=60)
    main(parser.parse_args())
