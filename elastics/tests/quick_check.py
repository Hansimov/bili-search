"""Quick check for owner intent behavior on key queries."""

from tclogger import logger

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.explorer import VideoExplorer
from ranks.constants import EXPLORE_RANK_TOP_K


def quick_check():
    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    for query in ["红警08", "米娜", "蝴蝶刀", "吴恩达大模型"]:
        result = explorer.unified_explore(
            query=query,
            rank_top_k=EXPLORE_RANK_TOP_K,
            group_owner_limit=10,
            verbose=False,
        )
        data = result.get("data", [])
        search_step = None
        for step in data:
            if step.get("name") in (
                "most_relevant_search",
                "knn_search",
                "hybrid_search",
            ):
                search_step = step
                break
        if not search_step:
            print(f"No search step for [{query}]")
            continue
        hits = search_step.get("output", {}).get("hits", [])
        print(f"\n{'='*70}")
        print(f"[{query}] Top 15")
        print(f"{'='*70}")

        om_count = 0
        for i, h in enumerate(hits[:15]):
            title = (h.get("title") or "")[:55]
            owner = h.get("owner", {})
            oname = owner.get("name", "") if isinstance(owner, dict) else ""
            om = " OM" if h.get("_owner_matched") else ""
            if h.get("_owner_matched"):
                om_count += 1
            r = h.get("relevance_score", 0)
            q = h.get("quality_score", 0)
            slot = h.get("_slot_dimension", "")
            views = (
                h.get("stat", {}).get("view", 0)
                if isinstance(h.get("stat"), dict)
                else 0
            )
            print(
                f"  {i+1:>2}. [{slot:<10}] r={r:.3f} q={q:.3f} "
                f"v={views:>10,}{om}  UP:{oname}"
            )
            print(f"      {title}")

        print(f"  Owner-matched in top-15: {om_count}")


if __name__ == "__main__":
    quick_check()

    # python -m elastics.tests.quick_check
