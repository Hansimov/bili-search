import argparse

from pprint import pprint

from elastics.owners.constants import ELASTIC_DEV, ELASTIC_OWNERS_DEV_INDEX
from elastics.owners.searcher import OwnerSearcher


def main(args: argparse.Namespace):
    searcher = OwnerSearcher(
        index_name=args.elastic_index,
        elastic_env_name=args.elastic_env_name,
    )

    if args.mode == "count":
        result = searcher.es.client.count(index=args.elastic_index)
        print({"index": args.elastic_index, "count": result.body.get("count", 0)})
        return

    result = searcher.search(
        query=args.query,
        sort_by=args.sort_by,
        limit=args.limit,
    )
    pprint(
        {
            "query": args.query,
            "sort_by": args.sort_by,
            "total": result.get("total"),
            "max_score": result.get("max_score"),
            "query_type": result.get("query_type"),
            "hits": result.get("hits", [])[: args.limit],
        },
        sort_dicts=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ei", "--elastic-index", default=ELASTIC_OWNERS_DEV_INDEX)
    parser.add_argument("-ev", "--elastic-env-name", default=ELASTIC_DEV)
    parser.add_argument("-m", "--mode", choices=["search", "count"], default="search")
    parser.add_argument("-q", "--query", default="影视飓风")
    parser.add_argument("-s", "--sort-by", default="relevance")
    parser.add_argument("-l", "--limit", type=int, default=5)
    main(parser.parse_args())
