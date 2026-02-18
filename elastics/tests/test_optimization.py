"""
Optimization Test Suite

Tests for the 5 target queries to measure recall & rank quality.
Uses ELASTIC_DEV environment.

Target queries:
1. 通义实验室
2. 红警08
3. 小红书推荐系统
4. 吴恩达大模型
5. chatgpt
"""

from tclogger import logger, logstr, dict_to_str, brk

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.explorer import VideoExplorer
from ranks.constants import EXPLORE_RANK_TOP_K


def analyze_results(query: str, result: dict, top_n: int = 20) -> dict:
    """Analyze explore results and print quality metrics."""
    data = result.get("data", [])
    perf = result.get("perf", {})

    # Find the search step
    search_step = None
    for step in data:
        if step.get("name") in ("most_relevant_search", "knn_search", "hybrid_search"):
            search_step = step
            break

    if not search_step:
        logger.warn(f"  No search step found for [{query}]")
        return {}

    output = search_step.get("output", {})
    hits = output.get("hits", [])
    total_hits = output.get("total_hits", 0)
    return_hits = output.get("return_hits", len(hits))

    # Basic stats
    logger.note(f"\n{'='*70}")
    logger.note(f"Query: [{query}]")
    logger.note(f"{'='*70}")
    logger.mesg(f"  Total hits: {total_hits}, Returned: {return_hits}")
    logger.mesg(f"  Perf: {perf}")

    # Recall info
    recall_info = output.get("recall_info", {})
    if recall_info:
        for lane, info in recall_info.items():
            if lane.startswith("_"):
                logger.mesg(f"  {lane}: {info}")
            else:
                logger.mesg(f"  Lane '{lane}': {info}")

    # Dimension distribution
    dim_dist = output.get("dimension_distribution", {})
    if dim_dist:
        logger.mesg(f"  Dimension distribution: {dim_dist}")

    # Show top-N results
    logger.note(f"\n  Top {min(top_n, len(hits))} results:")
    query_lower = query.lower()

    title_match_count = 0
    owner_match_count = 0

    for i, hit in enumerate(hits[:top_n]):
        title = hit.get("title", "")
        owner_name = (
            hit.get("owner", {}).get("name", "")
            if isinstance(hit.get("owner"), dict)
            else ""
        )
        tags = hit.get("tags", "")
        bvid = hit.get("bvid", "")
        score = hit.get("score", 0)
        rank_score = hit.get("rank_score", 0)
        relevance_score = hit.get("relevance_score", 0)
        quality_score = hit.get("quality_score", 0)
        recency_score = hit.get("recency_score", 0)
        popularity_score = hit.get("popularity_score", 0)
        headline_score = hit.get("headline_score", 0)
        slot_dim = hit.get("_slot_dimension", "")
        title_matched = hit.get("_title_matched", False)
        owner_matched = hit.get("_owner_matched", False)
        stat = hit.get("stat", {})
        views = stat.get("view", 0) if isinstance(stat, dict) else 0
        duration = hit.get("duration", 0)

        # Check title match
        if query_lower in (title or "").lower():
            title_match_count += 1
        if query_lower in (owner_name or "").lower():
            owner_match_count += 1

        # Format
        slot_str = f"[{slot_dim}]" if slot_dim else ""
        tm_str = " ★TM" if title_matched else ""
        om_str = " ★OM" if owner_matched else ""
        views_str = f"{views:,}" if views else "0"

        logger.mesg(
            f"  {i+1:>3}. {slot_str:<12} rel={relevance_score:.3f} "
            f"q={quality_score:.3f} rec={recency_score:.3f} "
            f"pop={popularity_score:.3f} "
            f"views={views_str:>12}{tm_str}{om_str}"
        )
        logger.file(f"       [{bvid}] {title[:60]}")
        if owner_name:
            logger.hint(
                f"       UP: {owner_name} | dur: {duration}s | tags: {(tags or '')[:40]}"
            )

    # Summary
    logger.note(f"\n  Summary for [{query}]:")
    logger.mesg(
        f"    Title match in top-{top_n}: {title_match_count}/{min(top_n, len(hits))}"
    )
    logger.mesg(
        f"    Owner match in top-{top_n}: {owner_match_count}/{min(top_n, len(hits))}"
    )

    return {
        "total_hits": total_hits,
        "return_hits": return_hits,
        "top_n_title_match": title_match_count,
        "top_n_owner_match": owner_match_count,
    }


def test_optimization_baseline():
    """Run baseline test for all 5 target queries."""
    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    test_queries = [
        "通义实验室",
        "红警08",
        "小红书推荐系统",
        "吴恩达大模型",
        "chatgpt",
    ]

    results = {}
    for query in test_queries:
        logger.note(f"\n> Running explore for [{query}]...")
        try:
            result = explorer.unified_explore(
                query=query,
                rank_top_k=EXPLORE_RANK_TOP_K,
                group_owner_limit=10,
                verbose=False,
            )
            analysis = analyze_results(query, result, top_n=15)
            results[query] = analysis
        except Exception as e:
            logger.warn(f"  Error for [{query}]: {e}")
            import traceback

            traceback.print_exc()
            results[query] = {"error": str(e)}

    # Final summary
    logger.note(f"\n{'='*70}")
    logger.note(f"OPTIMIZATION BASELINE SUMMARY")
    logger.note(f"{'='*70}")
    for query, analysis in results.items():
        if "error" in analysis:
            logger.warn(f"  [{query}]: ERROR - {analysis['error']}")
        else:
            logger.mesg(
                f"  [{query}]: returned={analysis.get('return_hits', 0)}, "
                f"title_match_top15={analysis.get('top_n_title_match', 0)}"
            )


if __name__ == "__main__":
    # python -m elastics.tests.test_optimization
    test_optimization_baseline()
