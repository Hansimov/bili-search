import argparse
import json

from elastics.owners.searcher import OwnerSearcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", required=True)
    parser.add_argument("-ev", "--elastic-env", default="elastic_dev")
    parser.add_argument("-q", "--domain-query", default="黑神话悟空")
    parser.add_argument("-l", "--limit", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    searcher = OwnerSearcher(
        index_name=args.index,
        elastic_env_name=args.elastic_env,
    )
    searcher.es.client.indices.refresh(index=args.index)

    count = searcher.es.client.count(index=args.index).body.get("count", 0)
    domain_result = searcher.search_by_domain(
        args.domain_query,
        sort_by="influence",
        limit=args.limit,
        compact=True,
    )
    payload = {
        "index": args.index,
        "count": count,
        "domain_query": args.domain_query,
        "domain_total": domain_result.get("total"),
        "domain_hits": [
            {
                "mid": hit.get("mid"),
                "name": hit.get("name"),
                "top_tags": hit.get("top_tags"),
                "influence_score": hit.get("influence_score"),
            }
            for hit in (domain_result.get("hits") or [])[: args.limit]
        ],
    }

    if domain_result.get("hits"):
        top_name = domain_result["hits"][0].get("name")
        name_result = searcher.search_by_name(top_name, limit=3, compact=True)
        payload["name_query"] = top_name
        payload["name_total"] = name_result.get("total")
        payload["name_hits"] = [
            {
                "mid": hit.get("mid"),
                "name": hit.get("name"),
                "top_tags": hit.get("top_tags"),
            }
            for hit in (name_result.get("hits") or [])[:3]
        ]

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
