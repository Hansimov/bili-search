from __future__ import annotations

import json

from elastics.videos.constants import ELASTIC_DEV
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX
from elastics.videos.searcher_v2 import VideoSearcherV2


OWNER_MIDS = [3546588246968975, 502925577, 374010007]
QUERIES = [
    "袁启",
    "袁启 采访",
    "袁启 访谈",
    "袁启 专访",
]


def summarize(result: dict, limit: int = 12) -> dict:
    query_text = (
        result.get("search_body", {})
        .get("query", {})
        .get("bool", {})
        .get("must", {})
        .get("es_tok_query_string", {})
        .get("query")
    )
    return {
        "retry_info": result.get("retry_info"),
        "query_dsl": query_text,
        "total_hits": result.get("total_hits"),
        "hits": [
            {
                "owner": (hit.get("owner") or {}).get("name"),
                "title": hit.get("title"),
            }
            for hit in (result.get("hits") or [])[:limit]
        ],
    }


def main() -> None:
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX,
        elastic_env_name=ELASTIC_DEV,
    )
    owner_filter = [{"terms": {"owner.mid": OWNER_MIDS}}]

    print(f"OWNER_MIDS={OWNER_MIDS}")
    for query in QUERIES:
        result = searcher.search(
            query=query,
            extra_filters=owner_filter,
            limit=12,
            verbose=False,
        )
        print(f"\n=== {query} ===")
        print(json.dumps(summarize(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
