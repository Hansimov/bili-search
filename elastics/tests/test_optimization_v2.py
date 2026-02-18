"""
Optimization Test Suite V2

Tests for the 7 target queries to measure recall & rank quality.
Uses ELASTIC_DEV environment.

Target queries:
1. 红警08 - owner intent (红警HBK08)
2. 小红书推荐系统 - technical topic
3. 吴恩达大模型 - person + topic
4. chatgpt - short English keyword (short-text noise issue)
5. gta - short English keyword (short-text noise issue)
6. 米娜 - common name (owner over-recall issue)
7. 蝴蝶刀 - topic keyword (negative stat robustness)
"""

from tclogger import logger

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.explorer import VideoExplorer
from ranks.constants import EXPLORE_RANK_TOP_K


def analyze_results(query: str, result: dict, top_n: int = 20) -> dict:
    """Analyze explore results and print quality metrics."""
    data = result.get("data", [])
    perf = result.get("perf", {})

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

    print(f"\n{'='*80}")
    print(f"Query: [{query}]")
    print(f"{'='*80}")
    print(f"  Total hits: {total_hits}, Returned: {return_hits}")
    print(f"  Perf: {perf}")

    # Recall info
    recall_info = output.get("recall_info", {})
    if recall_info:
        for lane, info in recall_info.items():
            print(f"  {lane}: {info}")

    # Dimension distribution
    dim_dist = output.get("dimension_distribution", {})
    if dim_dist:
        print(f"  Dim dist: {dim_dist}")

    # Show top-N results
    print(f"\n  Top {min(top_n, len(hits))} results:")
    query_lower = query.lower()

    title_match_count = 0
    owner_match_count = 0
    short_title_count = 0

    for i, hit in enumerate(hits[:top_n]):
        title = hit.get("title", "") or ""
        owner = hit.get("owner", {})
        owner_name = owner.get("name", "") if isinstance(owner, dict) else ""
        tags = hit.get("tags", "") or ""
        stat = hit.get("stat", {})
        views = stat.get("view", 0) if isinstance(stat, dict) else 0
        dur = hit.get("duration", 0) or 0
        rel = hit.get("relevance_score", 0)
        qual = hit.get("quality_score", 0)
        rec = hit.get("recency_score", 0)
        pop = hit.get("popularity_score", 0)
        tm = " TM" if hit.get("_title_matched") else ""
        om = " OM" if hit.get("_owner_matched") else ""
        slot = hit.get("_slot_dimension", "")

        if query_lower in title.lower():
            title_match_count += 1
        if len(title) < 25:
            short_title_count += 1
        if hit.get("_owner_matched"):
            owner_match_count += 1

        print(
            f"  {i+1:>3}. [{slot:<10}] r={rel:.3f} q={qual:.3f} "
            f"rec={rec:.3f} pop={pop:.3f} "
            f"v={views:>10,} d={dur:>5}s{tm}{om}"
        )
        print(f"       title: {title[:65]}")
        print(f"       UP: {owner_name} | tags: {tags[:50]}")

    # Summary
    print(f"\n  Summary for [{query}]:")
    print(f"    Returned docs: {return_hits}")
    print(
        f"    Title match in top-{top_n}: {title_match_count}/{min(top_n, len(hits))}"
    )
    print(
        f"    Owner match in top-{top_n}: {owner_match_count}/{min(top_n, len(hits))}"
    )
    print(
        f"    Short titles in top-{top_n}: {short_title_count}/{min(top_n, len(hits))}"
    )

    return {
        "total_hits": total_hits,
        "return_hits": return_hits,
        "top_n_title_match": title_match_count,
        "top_n_owner_match": owner_match_count,
        "top_n_short_title": short_title_count,
    }


def test_all_queries():
    """Run test for all 6 target queries."""
    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    test_queries = [
        "红警08",
        "小红书推荐系统",
        "吴恩达大模型",
        "chatgpt",
        "gta",
        "米娜",
        "蝴蝶刀",
    ]

    results = {}
    for query in test_queries:
        try:
            result = explorer.unified_explore(
                query=query,
                rank_top_k=EXPLORE_RANK_TOP_K,
                group_owner_limit=10,
                verbose=True,
            )
            analysis = analyze_results(query, result, top_n=20)
            results[query] = analysis
        except Exception as e:
            import traceback

            traceback.print_exc()
            results[query] = {"error": str(e)}

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    for query, analysis in results.items():
        if "error" in analysis:
            print(f"  [{query}]: ERROR - {analysis['error']}")
        else:
            print(
                f"  [{query}]: returned={analysis.get('return_hits', 0)}, "
                f"tm_top20={analysis.get('top_n_title_match', 0)}, "
                f"om_top20={analysis.get('top_n_owner_match', 0)}, "
                f"short_top20={analysis.get('top_n_short_title', 0)}"
            )


if __name__ == "__main__":
    # python -m elastics.tests.test_optimization_v2
    test_all_queries()
