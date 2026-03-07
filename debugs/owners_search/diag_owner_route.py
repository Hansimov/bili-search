import argparse
import json

from elastics.owners.searcher import OwnerSearcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", required=True)
    parser.add_argument("-ev", "--elastic-env", default="elastic_dev")
    parser.add_argument("-q", "--query", action="append", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    searcher = OwnerSearcher(
        index_name=args.index,
        elastic_env_name=args.elastic_env,
    )

    queries = args.query if isinstance(args.query, list) else [args.query]
    payload = {"index": args.index, "queries": []}
    for query in queries:
        exact_body = {
            "query": {"term": {"name.keyword": {"value": query}}},
            "_source": ["mid", "name"],
            "size": 5,
        }
        exact_raw = searcher.es.client.search(index=args.index, body=exact_body).body
        exact_hits = [
            hit.get("_source") or {}
            for hit in ((exact_raw.get("hits") or {}).get("hits") or [])
        ]
        payload["queries"].append(
            {
                "query": query,
                "exact_hit": searcher._has_exact_name_hit(query),
                "query_route": searcher._detect_query_route(query),
                "exact_total": ((exact_raw.get("hits") or {}).get("total") or {}).get(
                    "value"
                ),
                "exact_hits": exact_hits,
                "domain_result": searcher.search_by_domain(
                    query,
                    sort_by="influence",
                    limit=5,
                    compact=True,
                ),
                "search_result": searcher.search(
                    query,
                    sort_by="influence",
                    limit=5,
                    compact=True,
                ),
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
