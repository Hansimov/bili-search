from __future__ import annotations

import argparse
import json

from elastics.videos.constants import ELASTIC_DEV, ELASTIC_VIDEOS_DEV_INDEX
from elastics.videos.searcher_v2 import VideoSearcherV2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--index", default=ELASTIC_VIDEOS_DEV_INDEX)
    parser.add_argument("--elastic-env", default=ELASTIC_DEV)
    parser.add_argument("--show-intent", action="store_true")
    parser.add_argument("--show-search-body", action="store_true")
    parser.add_argument("--show-debug-info", action="store_true")
    args = parser.parse_args()

    searcher = VideoSearcherV2(
        index_name=args.index,
        elastic_env_name=args.elastic_env,
    )
    result = searcher.search(
        args.query,
        source_fields=["bvid", "title", "owner", "stat.view"],
        add_highlights_info=False,
        limit=args.limit,
        timeout=5,
        verbose=False,
    )

    print(f"QUERY: {args.query}")
    print(
        f"total_hits={result.get('total_hits', 0)} "
        f"return_hits={result.get('return_hits', 0)}"
    )
    if args.show_intent:
        print("intent_info=")
        print(json.dumps(result.get("intent_info") or {}, ensure_ascii=False, indent=2))
    if args.show_debug_info:
        print("query_focus_info=")
        print(
            json.dumps(
                result.get("query_focus_info") or {}, ensure_ascii=False, indent=2
            )
        )
        print("semantic_rewrite_info=")
        print(
            json.dumps(
                result.get("semantic_rewrite_info") or {},
                ensure_ascii=False,
                indent=2,
            )
        )
        print("title_rerank_info=")
        print(
            json.dumps(
                result.get("title_rerank_info") or {}, ensure_ascii=False, indent=2
            )
        )
    if args.show_search_body:
        print("search_body=")
        print(json.dumps(result.get("search_body") or {}, ensure_ascii=False, indent=2))
    for index, hit in enumerate(result.get("hits", [])[: args.limit], start=1):
        owner = hit.get("owner") or {}
        print(
            f"  {index}. {hit.get('title')}"
            f" | owner={owner.get('name')}"
            f" | bvid={hit.get('bvid')}"
            f" | view={((hit.get('stat') or {}).get('view'))}"
            f" | score={hit.get('score')}"
        )


if __name__ == "__main__":
    main()
