"""Diagnose owner-recalled doc scoring for 红警08."""

from tclogger import logger

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.explorer import VideoExplorer
from ranks.constants import EXPLORE_RANK_TOP_K


def diag():
    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    query = "红警08"

    # Run recall manually to see intermediate state
    recall_pool = explorer.recall_manager.recall(
        searcher=explorer,
        query=query,
        mode="word",
        extra_filters=[],
        timeout=10,
        verbose=True,
    )

    # Analyze the recall pool
    om_hits = [h for h in recall_pool.hits if h.get("_owner_matched")]
    non_om_hits = [h for h in recall_pool.hits if not h.get("_owner_matched")]

    print(f"\n=== Recall Pool Stats ===")
    print(f"Total hits: {len(recall_pool.hits)}")
    print(f"Owner-matched: {len(om_hits)}")
    print(f"Non-owner-matched: {len(non_om_hits)}")

    # Score distributions
    all_scores = sorted(
        [h.get("score", 0) or 0 for h in recall_pool.hits], reverse=True
    )
    om_scores = sorted([h.get("score", 0) or 0 for h in om_hits], reverse=True)
    non_om_scores = sorted([h.get("score", 0) or 0 for h in non_om_hits], reverse=True)

    print(f"\nAll scores (top 10): {[round(s, 2) for s in all_scores[:10]]}")
    print(f"All scores (bottom 10): {[round(s, 2) for s in all_scores[-10:]]}")
    print(f"\nOM scores (top 10): {[round(s, 2) for s in om_scores[:10]]}")
    print(f"OM scores (bottom 5): {[round(s, 2) for s in om_scores[-5:]]}")
    print(f"\nNon-OM scores (top 10): {[round(s, 2) for s in non_om_scores[:10]]}")
    print(f"Non-OM scores (bottom 5): {[round(s, 2) for s in non_om_scores[-5:]]}")

    if all_scores:
        max_score = max(all_scores)
        p25_idx = min(int(len(all_scores) * 0.25), len(all_scores) - 1)
        p50_idx = min(int(len(all_scores) * 0.50), len(all_scores) - 1)
        print(f"\nMax score: {max_score:.2f}")
        print(f"P25 score: {all_scores[p25_idx]:.2f}")
        print(f"P50 score: {all_scores[p50_idx]:.2f}")

    # Check owner lanes
    om_lanes = set()
    for h in om_hits:
        bvid = h.get("bvid", "")
        lanes = recall_pool.lane_tags.get(bvid, set())
        om_lanes.update(lanes)
    print(f"\nOM doc lanes: {om_lanes}")

    # Sample OM hits with their details
    print(f"\n=== Sample OM Hits ===")
    for h in om_hits[:5]:
        title = (h.get("title") or "")[:50]
        owner = h.get("owner", {})
        oname = owner.get("name", "") if isinstance(owner, dict) else ""
        score = h.get("score", 0) or 0
        bvid = h.get("bvid", "")
        lanes = recall_pool.lane_tags.get(bvid, set())
        print(f"  score={score:.2f} lanes={lanes} UP:{oname} | {title}")

    # Now run full explore to see final rankings
    print(f"\n=== Full Explore Top 20 ===")
    result = explorer.unified_explore(
        query=query,
        rank_top_k=EXPLORE_RANK_TOP_K,
        group_owner_limit=10,
        verbose=False,
    )
    data = result.get("data", [])
    search_step = None
    for step in data:
        if step.get("name") in ("most_relevant_search", "knn_search", "hybrid_search"):
            search_step = step
            break

    if not search_step:
        print("No search step")
        return

    hits = search_step.get("output", {}).get("hits", [])

    # Check: OM flags on ranked hits
    om_count_full = sum(1 for h in hits if h.get("_owner_matched"))
    print(f"  OM docs in ranked {len(hits)}: {om_count_full}")

    # Check first few OM docs in ranked hits
    om_ranked = [(i, h) for i, h in enumerate(hits) if h.get("_owner_matched")]
    print(f"  First 5 OM docs in ranked hits:")
    for i, h in om_ranked[:5]:
        title = (h.get("title") or "")[:50]
        owner = h.get("owner", {})
        oname = owner.get("name", "") if isinstance(owner, dict) else ""
        r = h.get("relevance_score", 0)
        q = h.get("quality_score", 0)
        rec = h.get("recency_score", 0)
        pop = h.get("popularity_score", 0)
        slot = h.get("_slot_dimension", "")
        print(
            f"    pos={i+1} [{slot}] r={r:.3f} q={q:.3f} rec={rec:.3f} pop={pop:.3f} UP:{oname} | {title}"
        )

    # Debug: compute owner_intent_strength manually
    import re

    query_lower = "红警08"
    query_cjk = re.sub(r"[^\u4e00-\u9fff]", "", query_lower)
    q_terms = [t for t in re.split(r"[\s\-_,，。、/\\|]+", query_lower) if t]
    from ranks.diversified import DiversifiedRanker

    strength = DiversifiedRanker._analyze_owner_intent_strength(
        hits, query_lower, query_cjk, q_terms
    )
    print(f"  owner_intent_strength = {strength:.4f}")

    om_in_top = 0
    for i, h in enumerate(hits[:20]):
        title = (h.get("title") or "")[:50]
        owner = h.get("owner", {})
        oname = owner.get("name", "") if isinstance(owner, dict) else ""
        om = " OM" if h.get("_owner_matched") else ""
        if h.get("_owner_matched"):
            om_in_top += 1
        r = h.get("relevance_score", 0)
        q = h.get("quality_score", 0)
        slot = h.get("_slot_dimension", "")
        print(f"  {i+1:>2}. [{slot:<10}] r={r:.3f} q={q:.3f}{om}  UP:{oname}")
        print(f"      {title}")

    print(f"\n  OM in top-20: {om_in_top}")

    # Check: what positions are ALL OM docs in?
    om_positions = []
    for i, h in enumerate(hits):
        if h.get("_owner_matched"):
            om_positions.append(i + 1)
    print(f"  OM positions in full {len(hits)} results: {om_positions[:20]}")


if __name__ == "__main__":
    diag()
