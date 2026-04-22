from __future__ import annotations

import argparse
import json

from elastics.owners.searcher import OwnerSearcher
from elastics.videos.constants import ELASTIC_DEV, ELASTIC_VIDEOS_DEV_INDEX


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--index", default=ELASTIC_VIDEOS_DEV_INDEX)
    parser.add_argument("--elastic-env", default=ELASTIC_DEV)
    parser.add_argument("--debug-buckets", action="store_true")
    args = parser.parse_args()

    searcher = OwnerSearcher(
        index_name=args.index,
        elastic_env_name=args.elastic_env,
    )
    if args.debug_buckets:
        query = (args.text or "").strip()
        prepared_query = searcher._prepare_query(query, args.mode)
        name_hits = []
        topic_hits = []
        relation_hits = []

        if args.mode in {"auto", "name", "relation"}:
            name_hits = searcher._search_name_candidates(
                prepared_query,
                size=max(args.size * 2, 12),
            )
        if args.mode in {"auto", "topic"}:
            topic_hits = searcher._search_topic_candidates(
                prepared_query,
                size=max(args.size * 2, 10),
            )
        if args.mode == "relation":
            relation_hits = searcher._search_relation_candidates(
                prepared_query,
                name_hits=name_hits,
                size=max(args.size * 2, 10),
            )
            if not relation_hits:
                topic_hits = searcher._search_topic_candidates(
                    prepared_query,
                    size=max(args.size * 2, 10),
                )

        result = {
            "text": query,
            "mode": args.mode,
            "prepared_query": prepared_query,
            "name_hits": name_hits,
            "topic_hits": topic_hits,
            "relation_hits": relation_hits,
            "merged": searcher.search(
                text=args.text,
                mode=args.mode,
                size=args.size,
            ),
        }
    else:
        result = searcher.search(
            text=args.text,
            mode=args.mode,
            size=args.size,
        )
    if args.summary:
        owners = []
        if args.debug_buckets:
            owners = ((result.get("merged") or {}).get("owners") or [])[: args.top]
        else:
            owners = (result.get("owners") or [])[: args.top]
        print(f"QUERY: {args.text} | mode={args.mode}")
        for index, owner in enumerate(owners, start=1):
            print(
                f"  {index}. {owner.get('name')}"
                f" | score={owner.get('score')}"
                f" | sources={owner.get('sources')}"
            )
        return

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
