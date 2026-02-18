"""Test auto-constraint building and integration with explore pipeline."""

import json
from tclogger import logger
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.explorer import VideoExplorer
from elastics.structure import (
    analyze_tokens,
    select_covering_tokens,
    build_auto_constraint_filter,
)

FIELDS = ["title.words", "tags.words"]

_explorer = None


def get_explorer():
    global _explorer
    if _explorer is None:
        _explorer = VideoExplorer(
            index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
        )
    return _explorer


def format_hit(hit, idx):
    title = hit.get("title", "")
    score = hit.get("score", hit.get("rank_score", 0))
    owner = hit.get("owner", {})
    name = owner.get("name", "") if isinstance(owner, dict) else ""
    return f"  [{idx+1:2d}] score={score:.4f} | {title} | @{name}"


def run_explore(explorer, query, mode="v", auto_constraint=True, label=""):
    full_query = f"{query} q={mode}"
    logger.note(f"\n>>> Explore: [{full_query}] {label}")

    res = explorer.unified_explore(
        query=full_query,
        auto_constraint=auto_constraint,
        verbose=False,
    )

    # Extract hits from main search step
    hits = []
    step_name = ""
    for step in res.get("data", []):
        name = step.get("name", "")
        if name in ("knn_search", "hybrid_search", "most_relevant_search"):
            output = step.get("output", {})
            hits = output.get("hits", [])
            step_name = name
            break

    total = 0
    for step in res.get("data", []):
        output = step.get("output", {})
        if "total_hits" in output:
            total = output["total_hits"]
            break

    logger.success(f"  [{step_name}] total_hits={total}, returned={len(hits)}")
    for i, hit in enumerate(hits[:8]):
        logger.mesg(format_hit(hit, i))
    if len(hits) > 8:
        logger.hint(f"  ... and {len(hits) - 8} more")
    return hits


def test_auto_constraint_integration():
    """Compare explore results with and without auto-constraint."""
    explorer = get_explorer()
    queries = ["飓风营救", "通义实验室", "红警08", "小红书推荐系统", "吴恩达大模型"]

    for q in queries:
        # Show auto-generated covering tokens
        tokens = analyze_tokens(explorer.es.client, ELASTIC_VIDEOS_DEV_INDEX, q)
        covering = select_covering_tokens(tokens, q)
        logger.file(f"\n{'='*60}")
        logger.file(f"EXPLORE: {q}  →  tokens: {covering}")
        logger.file(f"{'='*60}")

        # Without auto-constraint
        hits_off = run_explore(
            explorer, q, mode="v", auto_constraint=False, label="[no constraint]"
        )

        # With auto-constraint (default)
        hits_on = run_explore(
            explorer, q, mode="v", auto_constraint=True, label="[auto-constraint]"
        )

        logger.file(f"\n  SUMMARY: no_constraint={len(hits_off)} auto={len(hits_on)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "knn":
        # Quick KNN-only test (no explore pipeline)
        from elastics.videos.searcher_v2 import VideoSearcherV2

        s = VideoSearcherV2(ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV)
        queries = [
            "飓风营救",
            "通义实验室",
            "红警08",
            "小红书推荐系统",
            "吴恩达大模型",
        ]
        for q in queries:
            covering = select_covering_tokens(
                analyze_tokens(s.es.client, ELASTIC_VIDEOS_DEV_INDEX, q), q
            )
            cf = build_auto_constraint_filter(
                s.es.client, ELASTIC_VIDEOS_DEV_INDEX, q, fields=FIELDS
            )
            logger.file(f"\n{q}: covering={covering}")
            logger.mesg(f"  filter: {json.dumps(cf, ensure_ascii=False)}")
    else:
        test_auto_constraint_integration()
