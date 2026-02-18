"""Quick single-query diagnostic."""

import sys
from tclogger import logger
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.explorer import VideoExplorer
from ranks.constants import EXPLORE_RANK_TOP_K

query = sys.argv[1] if len(sys.argv) > 1 else "小红书推荐系统"
explorer = VideoExplorer(
    index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
)
result = explorer.unified_explore(
    query=query, rank_top_k=EXPLORE_RANK_TOP_K, group_owner_limit=10, verbose=False
)
data = result.get("data", [])
step = next(
    (
        s
        for s in data
        if s.get("name") in ("most_relevant_search", "knn_search", "hybrid_search")
    ),
    None,
)
if step:
    hits = step.get("output", {}).get("hits", [])
    for i, h in enumerate(hits[:20]):
        t = (h.get("title") or "")[:60]
        o = h.get("owner", {})
        on = o.get("name", "") if isinstance(o, dict) else ""
        r = h.get("relevance_score", 0)
        q = h.get("quality_score", 0)
        s = h.get("_slot_dimension", "")
        om = " OM" if h.get("_owner_matched") else ""
        tm = " TM" if h.get("_title_matched") else ""
        v = (h.get("stat", {}) or {}).get("view", 0)
        print(f"{i+1:>2}. [{s:<10}] r={r:.3f} q={q:.3f} v={v:>10,}{om}{tm}  UP:{on}")
        print(f"    {t}")
