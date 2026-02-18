"""Compare our search results with official Bilibili results using blux."""

import time
from blux.search import BiliSearcher, SearchType, SearchOrder
from tclogger import logger

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.explorer import VideoExplorer
from ranks.constants import EXPLORE_RANK_TOP_K


QUERIES = [
    "红警08",
    "小红书推荐系统",
    "吴恩达大模型",
    "chatgpt",
    "gta",
    "米娜",
    "蝴蝶刀",
]


def compare():
    # Our system
    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    # Official Bilibili
    bili = BiliSearcher()

    for query in QUERIES:
        print(f"\n{'='*80}")
        print(f"Query: [{query}]")
        print(f"{'='*80}")

        # --- Our results ---
        our_result = explorer.unified_explore(
            query=query,
            rank_top_k=EXPLORE_RANK_TOP_K,
            group_owner_limit=10,
            verbose=False,
        )
        data = our_result.get("data", [])
        step = next(
            (
                s
                for s in data
                if s.get("name")
                in ("most_relevant_search", "knn_search", "hybrid_search")
            ),
            None,
        )
        our_hits = step.get("output", {}).get("hits", []) if step else []

        # --- Official results ---
        try:
            official = bili.search_by_type(
                keyword=query, search_type=SearchType.VIDEO, page=1
            )
            official_videos = official.items if official.ok else []
        except Exception as e:
            logger.warn(f"  Official search failed: {e}")
            official_videos = []

        # Compare top 10
        print(f"\n  {'OURS (top 10)':>50}  |  {'OFFICIAL (top 10)':<50}")
        print(f"  {'-'*50}  |  {'-'*50}")

        our_bvids = set()
        off_bvids = set()
        max_rows = max(min(len(our_hits), 10), min(len(official_videos), 10))

        for i in range(max_rows):
            # Our result
            if i < len(our_hits):
                h = our_hits[i]
                title = (h.get("title") or "")[:30]
                owner = h.get("owner", {})
                oname = (owner.get("name", "") if isinstance(owner, dict) else "")[:8]
                bvid = h.get("bvid", "")
                our_bvids.add(bvid)
                our_str = f"{i+1:>2}. {title}  [{oname}]"
            else:
                our_str = ""

            # Official result
            if i < len(official_videos):
                v = official_videos[i]
                off_title = v.title[:30]
                off_author = v.author[:8]
                off_bvids.add(v.bvid)
                off_str = f"{i+1:>2}. {off_title}  [{off_author}]"
            else:
                off_str = ""

            print(f"  {our_str:>50}  |  {off_str:<50}")

        # Overlap analysis
        our_top20 = {h.get("bvid") for h in our_hits[:20] if h.get("bvid")}
        off_top20 = {v.bvid for v in official_videos[:20] if v.bvid}
        overlap = our_top20 & off_top20
        print(f"\n  Top-20 overlap: {len(overlap)} / {len(off_top20)} official")
        if overlap:
            print(f"  Shared bvids: {overlap}")

        # Rate limit: avoid hitting API too fast
        time.sleep(1.5)


if __name__ == "__main__":
    compare()
