"""Diagnostic script: compare q=w, q=v, q=vr search results to analyze KNN recall."""

from tclogger import logger, dict_to_str

from elastics.videos.explorer import VideoExplorer
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV


def diagnose_query(query: str):
    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # 1. Word search (q=w)
    logger.note(f"\n=== q=w (Word Search) for [{query}] ===")
    res_w = explorer.unified_explore(query=f"{query} q=w", verbose=False)
    w_hits = []
    for step in res_w.get("data", []):
        if step["name"] == "most_relevant_search":
            output = step.get("output", {})
            w_hits = output.get("hits", [])
            logger.success(
                f"  total_hits: {output.get('total_hits', 0)}, "
                f"return_hits: {output.get('return_hits', 0)}"
            )
            for h in w_hits[:5]:
                logger.mesg(f"    [{h.get('score',0):.2f}] {h.get('title','')}")

    # 2. Vector search via unified_explore (q=v, now with word recall + rerank)
    logger.note(f"\n=== q=v (Vector + Word Recall + Rerank) for [{query}] ===")
    res_v = explorer.unified_explore(query=f"{query} q=v", verbose=False)
    v_hits = []
    for step in res_v.get("data", []):
        if step["name"] == "word_recall_supplement":
            output = step.get("output", {})
            logger.hint(f"  word recall: {output}")
        if step["name"] == "knn_search":
            output = step.get("output", {})
            v_hits = output.get("hits", [])
            logger.success(
                f"  total_hits: {output.get('total_hits', 0)}, "
                f"return_hits: {output.get('return_hits', 0)}"
            )
            for h in v_hits[:5]:
                cosine = h.get("cosine_similarity", h.get("score", 0))
                rerank = h.get("rerank_score", 0)
                logger.mesg(
                    f"    [cos={cosine:.4f} rr={rerank:.4f}] {h.get('title','')}"
                )

    # 3. Overlap analysis
    w_bvids = {h.get("bvid") for h in w_hits}
    v_bvids = {h.get("bvid") for h in v_hits}
    overlap = w_bvids & v_bvids
    logger.note(f"\n=== Overlap Analysis ===")
    logger.mesg(f"  q=w hits: {len(w_bvids)}")
    logger.mesg(f"  q=v hits: {len(v_bvids)}")
    logger.mesg(f"  overlap:  {len(overlap)}")
    if w_bvids:
        logger.mesg(
            f"  overlap rate (of word results): {len(overlap)/len(w_bvids)*100:.1f}%"
        )

    # 4. Score distribution
    logger.note(f"\n=== q=v Score Distribution ===")
    if v_hits:
        scores = [h.get("rerank_score", h.get("score", 0)) for h in v_hits]
        logger.mesg(f"  count: {len(scores)}")
        logger.mesg(f"  max:   {max(scores):.6f}")
        logger.mesg(f"  min:   {min(scores):.6f}")

    # 5. Performance
    logger.note(f"\n=== Performance ===")
    logger.mesg(f"  q=v perf: {res_v.get('perf', {})}")


test_queries = [
    "影视飓风",
    "deepseek",
]

if __name__ == "__main__":
    for q in test_queries:
        diagnose_query(q)
