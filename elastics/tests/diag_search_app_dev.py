import argparse
import json
import urllib.request

from llms.tools.utils import extract_explore_authors, extract_explore_hits


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main(args: argparse.Namespace):
    base_url = args.base_url.rstrip("/")
    if args.mode == "health":
        print(json.dumps(get_json(f"{base_url}/health"), ensure_ascii=False, indent=2))
        return

    if args.mode == "explore":
        payload = {"query": args.query, "verbose": False}
        result = post_json(f"{base_url}/explore", payload)
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
        result = post_json(f"{base_url}/explore", payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mode == "chat":
        payload = {
            "messages": [{"role": "user", "content": args.query}],
            "stream": False,
            "thinking": args.thinking,
            "max_iterations": args.max_iterations,
        }
        result = post_json(f"{base_url}/chat/completions", payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--base-url", default="http://127.0.0.1:21001")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["health", "explore", "explore_raw", "chat"],
        default="health",
    )
    parser.add_argument("-q", "--query", default="黑神话悟空")
    parser.add_argument("-l", "--limit", type=int, default=5)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=3)
    main(parser.parse_args())
