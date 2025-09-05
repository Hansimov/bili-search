from elastics.videos.constants import RANK_TOP_K


class VideoHitsRanker:
    def __init__(self):
        pass

    def tops(self, hits_info: dict, top_k: int = RANK_TOP_K) -> dict:
        """
        Format of hits_info:
        * LINK: elastics/videos/hits.py
        """
        hits = hits_info.get("hits", [])
        if top_k and len(hits) > top_k:
            hits = hits[:top_k]
            hits_info["hits"] = hits
            hits_info["return_hits"] = len(hits)
        return hits_info

    def rank(self, hits_info: dict, top_k: int = RANK_TOP_K) -> dict:
        hits = hits_info.get("hits", [])
        if top_k and len(hits) > top_k:
            hits = hits[:top_k]
            hits_info["hits"] = hits
            hits_info["return_hits"] = len(hits)

        return hits_info
