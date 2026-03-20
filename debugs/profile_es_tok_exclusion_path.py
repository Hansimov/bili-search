"""Profile the slow es_tok_query_string exclusion path on the live dev index.

Compares end-to-end and raw ES costs for the same query under a few safe
parameter variants so we can see whether the latency is dominated by:
1. the broad negative match itself,
2. script_score ranking, or
3. track_total_hits bookkeeping.

Usage:
    conda run -n ai python debugs/profile_es_tok_exclusion_path.py
"""

from __future__ import annotations

import time
from copy import deepcopy

from tclogger import logger

from elastics.structure import construct_boosted_fields
from elastics.videos.constants import (
    ELASTIC_DEV,
    ELASTIC_VIDEOS_DEV_INDEX,
    EXPLORE_BOOSTED_FIELDS,
    SEARCH_MATCH_FIELDS,
    SEARCH_MATCH_TYPE,
)
from elastics.videos.searcher_v2 import VideoSearcherV2

SOURCE_FIELDS = ["bvid", "title"]
QUERIES = [
    "游戏音乐",
    "游戏音乐 +若生命将于明日落幕",
    "游戏音乐 -若生命将于明日落幕",
]
VARIANTS = [
    {
        "label": "default",
        "use_script_score": True,
        "track_total_hits": True,
    },
    {
        "label": "no_script_score",
        "use_script_score": False,
        "track_total_hits": True,
    },
    {
        "label": "no_track_total_hits",
        "use_script_score": True,
        "track_total_hits": False,
    },
    {
        "label": "no_script_score_no_track_total_hits",
        "use_script_score": False,
        "track_total_hits": False,
    },
]


def get_searcher() -> VideoSearcherV2:
    return VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX,
        elastic_env_name=ELASTIC_DEV,
    )


def build_query_context(
    searcher: VideoSearcherV2, query: str
) -> tuple[dict, dict, dict, list[str]]:
    boosted_match_fields, boosted_date_fields = construct_boosted_fields(
        match_fields=SEARCH_MATCH_FIELDS,
        boost=True,
        boosted_fields=EXPLORE_BOOSTED_FIELDS,
    )
    query_info, rewrite_info, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
        query=query,
        suggest_info={},
        boosted_match_fields=boosted_match_fields,
        boosted_date_fields=boosted_date_fields,
        match_type=SEARCH_MATCH_TYPE,
        extra_filters=[],
    )
    return query_info, rewrite_info, query_dsl_dict, boosted_match_fields


def build_search_body(
    searcher: VideoSearcherV2,
    query_dsl_dict: dict,
    boosted_match_fields: list[str],
    *,
    use_script_score: bool,
    track_total_hits: bool,
) -> dict:
    body = searcher.construct_search_body(
        query_dsl_dict=query_dsl_dict,
        match_fields=boosted_match_fields,
        source_fields=SOURCE_FIELDS,
        drop_no_highlights=False,
        is_explain=False,
        is_profile=True,
        is_highlight=False,
        use_script_score=use_script_score,
        score_threshold=None,
        limit=10,
        terminate_after=2000000,
        timeout=5,
    )
    body = deepcopy(body)
    body["track_total_hits"] = track_total_hits
    return body


def extract_profile_summary(res_dict: dict) -> list[str]:
    profile = res_dict.get("profile", {})
    shards = profile.get("shards", [])
    top_nodes = []
    for shard in shards:
        for search in shard.get("searches", []):
            for node in search.get("query", []):
                top_nodes.append(
                    (
                        node.get("time_in_nanos", 0),
                        node.get("type", "?"),
                        node.get("description", ""),
                    )
                )
    top_nodes.sort(reverse=True)
    lines = []
    for nanos, node_type, description in top_nodes[:5]:
        short_desc = description.replace("\n", " ")
        if len(short_desc) > 120:
            short_desc = short_desc[:117] + "..."
        lines.append(f"{nanos/1_000_000:.1f}ms | {node_type} | {short_desc}")
    return lines


def run_raw_es_profile(searcher: VideoSearcherV2, body: dict) -> dict:
    start = time.perf_counter()
    res_dict = searcher.submit_to_es(body, context="profile_es_tok_exclusion_path")
    wall_ms = (time.perf_counter() - start) * 1000
    hits = res_dict.get("hits", {}).get("hits", [])
    bvids = [hit.get("_source", {}).get("bvid") for hit in hits if hit.get("_source")]
    return {
        "wall_ms": wall_ms,
        "es_took_ms": res_dict.get("took", 0),
        "total_hits": res_dict.get("hits", {}).get("total", {}),
        "top_bvids": bvids[:5],
        "profile_lines": extract_profile_summary(res_dict),
    }


def run_end_to_end(
    searcher: VideoSearcherV2, query: str, *, use_script_score: bool
) -> dict:
    start = time.perf_counter()
    res = searcher.search(
        query,
        source_fields=SOURCE_FIELDS,
        add_highlights_info=False,
        is_highlight=False,
        use_script_score=use_script_score,
        limit=10,
        timeout=5,
        verbose=False,
    )
    wall_ms = (time.perf_counter() - start) * 1000
    hits = res.get("hits", [])
    bvids = [hit.get("bvid") for hit in hits if hit.get("bvid")]
    return {
        "wall_ms": wall_ms,
        "top_bvids": bvids[:5],
        "hit_count": len(hits),
    }


def main() -> None:
    searcher = get_searcher()
    logger.note(
        "> Profiling es_tok_query_string exclusion path on the live dev index..."
    )

    for query in QUERIES:
        logger.file(f"\n{'=' * 88}")
        logger.file(f"QUERY: {query}")
        logger.file(f"{'=' * 88}")

        query_info, _, query_dsl_dict, boosted_match_fields = build_query_context(
            searcher, query
        )
        logger.mesg(f"keywords_body={query_info.get('keywords_body', [])}")
        logger.mesg(f"constraint_texts={query_info.get('constraint_texts', [])}")
        logger.mesg(f"constraint_filter={query_info.get('constraint_filter', {})}")

        for variant in VARIANTS:
            logger.note(f"\n[{variant['label']}]")
            body = build_search_body(
                searcher,
                query_dsl_dict,
                boosted_match_fields,
                use_script_score=variant["use_script_score"],
                track_total_hits=variant["track_total_hits"],
            )
            raw = run_raw_es_profile(searcher, body)
            logger.mesg(
                f"raw_es wall={raw['wall_ms']:.1f}ms took={raw['es_took_ms']}ms total={raw['total_hits']} top_bvids={raw['top_bvids']}"
            )
            for line in raw["profile_lines"]:
                logger.mesg(f"  {line}")

            if variant["track_total_hits"]:
                end_to_end = run_end_to_end(
                    searcher,
                    query,
                    use_script_score=variant["use_script_score"],
                )
                logger.mesg(
                    f"end_to_end wall={end_to_end['wall_ms']:.1f}ms hits={end_to_end['hit_count']} top_bvids={end_to_end['top_bvids']}"
                )


if __name__ == "__main__":
    main()
