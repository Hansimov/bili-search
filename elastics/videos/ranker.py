import heapq
from tclogger import dict_get

from elastics.videos.constants import RANK_TOP_K

# RRF weights of fields
RRF_WEIGHTS = {
    "pubdate": 3.0,  # publish date timestamp
    "stat.view": 1.0,
    "stat.favorite": 1.0,
    "stat.coin": 1.0,
    "score": 3.0,  # relevance score calculated by ES
}
# RRF constant (k)
RRF_K = 60.0
# speed up rank, by only covering top hits for each metric
RRF_HEAP_SIZE = 2000
# heap_size = max(input heap_size, top_k * RRF_HEAP_RATIO)
RRF_HEAP_RATIO = 5


class VideoHitsRanker:
    def __init__(self):
        pass

    def tops(self, hits_info: dict, top_k: int = RANK_TOP_K) -> dict:
        """Get first k hits without rank"""
        hits = hits_info.get("hits", [])
        if top_k and len(hits) > top_k:
            hits = hits[:top_k]
            hits_info["hits"] = hits
            hits_info["return_hits"] = len(hits)
        return hits_info

    def rrf_rank(
        self,
        hits_info: dict,
        top_k: int = RANK_TOP_K,
        rrf_weights: dict = RRF_WEIGHTS,
        rrf_k: int = RRF_K,
        heap_size: int = RRF_HEAP_SIZE,
        heap_ratio: int = RRF_HEAP_RATIO,
    ) -> dict:
        """Get top k hits by RRF (Reciprocal Rank Fusion) rank with metrics and weights.
        Format of hits_info: * LINK: elastics/videos/hits.py
        """
        hits: list[dict] = hits_info.get("hits", [])
        hits_num = len(hits)
        top_k = min(top_k, hits_num)
        heap_size = min(max(heap_size, top_k * heap_ratio), hits_num)

        # get values for each metric
        metric_keys = list(rrf_weights.keys())
        metric_vals: dict[str, list[int, float]] = {}
        for key in metric_keys:
            mvals = [dict_get(hit, key, 0) for hit in hits]
            metric_vals[key] = mvals

        # calc ranks for each metric with heap
        metric_rank_dict: dict[str, dict[int, int]] = {}
        for mkey, mvals in metric_vals.items():
            top_idxs = heapq.nlargest(
                heap_size, range(hits_num), key=lambda i: mvals[i]
            )
            metric_rank_dict[mkey] = {
                idx: rank + 1 for rank, idx in enumerate(top_idxs)
            }

        # calc RRF scores by ranks of all metrics
        rrf_scores: list[float] = [0.0] * hits_num
        last_rank = heap_size + 1
        for mkey, mvals in metric_vals.items():
            w = float(rrf_weights.get(mkey, 1.0) or 0.0)
            mrank_dict = metric_rank_dict.get(mkey, {})
            for i in range(hits_num):
                rank = mrank_dict.get(i, last_rank)
                rrf_scores[i] += w / (rrf_k + rank)

        # get top_k hits by RRF scores
        for i, hit in enumerate(hits):
            hit["rrf_rank_score"] = round(rrf_scores[i], 6)
        ranked_hits = heapq.nlargest(top_k, hits, key=lambda x: x["rrf_rank_score"])
        hits_info["hits"] = ranked_hits
        hits_info["return_hits"] = len(ranked_hits)

        return hits_info
