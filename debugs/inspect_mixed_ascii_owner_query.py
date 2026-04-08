"""Inspect mixed CJK+ASCII owner-name matches on the live dev index.

Usage:
    conda run -n ai python debugs/inspect_mixed_ascii_owner_query.py
"""

from __future__ import annotations

from tclogger import logger

from elastics.videos.constants import ELASTIC_DEV, ELASTIC_VIDEOS_DEV_INDEX
from elastics.videos.searcher_v2 import VideoSearcherV2

TARGET_OWNER = "红警HBK08"
QUERIES = [
    "HBK08",
    "红警HBK08",
    "+红警HBK08",
]
LIMITS = [20, 50, 100, 200]


def make_searcher() -> VideoSearcherV2:
    return VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX,
        elastic_env_name=ELASTIC_DEV,
    )


def owner_names(hits: list[dict]) -> list[str]:
    names = []
    for hit in hits:
        owner = hit.get("owner") or {}
        name = owner.get("name")
        if name:
            names.append(name)
    return names


def main() -> None:
    searcher = make_searcher()

    for query in QUERIES:
        logger.note(f"\nQUERY: {query}")
        for limit in LIMITS:
            res = searcher.search(
                query,
                source_fields=["bvid", "title", "owner"],
                add_highlights_info=False,
                limit=limit,
                timeout=5,
                verbose=False,
            )
            names = owner_names(res.get("hits", []))
            found = TARGET_OWNER in set(names)
            preview = names[:10]
            logger.mesg(
                f"  limit={limit:<3} found={found} total_hits={res.get('total_hits', 0)} "
                f"preview={preview}"
            )


if __name__ == "__main__":
    main()
