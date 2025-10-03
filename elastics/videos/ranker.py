import heapq
import math

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


# 2010-01-01 00:00:00, the beginning of most videos in Bilibili
PUBDATE_BASE = 1262275200
# seconds per day
SECONDS_PER_DAY = 86400
# score for videos published most recently
ZERO_DAY_SCORE = 4.0
# score for videos published before base (infinity days ago)
INFT_DAY_SCORE = 0.25
# pubdate score interpolation points
PUBDATE_SCORE_POINTS = [(0, 4.0), (7, 1.0), (30, 0.6), (365, 0.3)]
# fields of stats
STAT_FIELDS = ["view", "favorite", "coin", "reply", "share", "danmaku"]
# offsets for log(x+offset) of stats
STAT_LOGX_OFFSETS = {
    "view": 10,
    "favorite": 2,
    "coin": 2,
    "reply": 2,
    "share": 2,
    "danmaku": 2,
}
# relate score power
RELATE_POWER = 4.0
# relate score power offset
RELATE_OFFSET = 1.0


def log_x(x: int, base: float = 10.0, offset: int = 10) -> float:
    x = max(x, 0)
    return math.log(x + offset, base)


class StatsScorer:
    def __init__(
        self,
        stat_fields: list = STAT_FIELDS,
        stat_logx_offsets: dict = STAT_LOGX_OFFSETS,
    ):
        self.stat_fields = stat_fields
        self.stat_logx_offsets = stat_logx_offsets

    def calc_stats_score_by_prod_logx(self, stats: dict) -> float:
        """Product of log(x+offset) of stats fields"""
        return math.prod(
            log_x(
                x=stats.get(field, 0),
                base=10,
                offset=self.stat_logx_offsets.get(field, 2),
            )
            for field in self.stat_fields
        )

    def calc(self, stats: dict) -> float:
        stats_score = self.calc_stats_score_by_prod_logx(stats)
        return stats_score


class PubdateScorer:
    def __init__(
        self,
        day_score_points: list[tuple[float, float]] = PUBDATE_SCORE_POINTS,
        zero_day_score: float = ZERO_DAY_SCORE,
        inft_day_score: float = INFT_DAY_SCORE,
    ):
        """
        - day_score_points: (days, score) pairs
            - (0, 4.0): score 4.0 for videos published most recently
            - (7, 1.0): score 1.0 for videos published 7 days ago
            - (30, 0.6): score 0.6 for videos published 30 days ago
            - (365, 0.3): score 0.3 for videos published 1 year ago
        - zero_day_score: score for videos published after now (default 4.0)
        - inft_day_score: score for videos published before base (default 0.25)
        """
        self.day_score_points = sorted(day_score_points)
        self.zero_day_score = zero_day_score
        self.inft_day_score = inft_day_score
        self.slope_offsets = self.pre_calc_slope_offsets(day_score_points)

    def pre_calc_slope_offsets(self, points: list[tuple[float, float]]):
        slope_offsets = []
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            slope = (y2 - y1) / (x2 - x1)
            offset = y1 - slope * x1
            slope_offsets.append((slope, offset))
        return slope_offsets

    def calc_pass_days(self, pubdate: int) -> float:
        return (pubdate - PUBDATE_BASE) / SECONDS_PER_DAY

    def calc_pubdate_score_by_slope_offsets(self, pubdate: int) -> float:
        """Segmented linear functions by slopes and offsets"""
        pass_days = self.calc_pass_days(pubdate)
        points = self.day_score_points
        if pass_days <= points[0][0]:
            return self.zero_day_score
        if pass_days >= points[-1][0]:
            return self.inft_day_score
        for i in range(1, len(points)):
            if pass_days <= points[i][0]:
                slope, offset = self.slope_offsets[i - 1]
                score = slope * pass_days + offset
                return score
        return self.inft_day_score

    def calc(self, pubdate: int) -> float:
        pubdate_score = self.calc_pubdate_score_by_slope_offsets(pubdate)
        return pubdate_score


class RelateScorer:
    def calc(self, relate: float) -> float:
        return (relate - 8) ** 4


class ScoreFuser:
    def calc_fuse_score_by_prod(
        self, stats_score: float, pubdate_score: float, relate_score: float
    ) -> float:
        return round(stats_score * pubdate_score * relate_score, 6)

    def fuse(
        self, stats_score: float, pubdate_score: float, relate_score: float
    ) -> float:
        fuse_score = self.calc_fuse_score_by_prod(
            stats_score=stats_score,
            pubdate_score=pubdate_score,
            relate_score=relate_score,
        )
        return fuse_score


class VideoHitsRanker:
    def __init__(self):
        self.stats_scorer = StatsScorer()
        self.pubdate_scorer = PubdateScorer()
        self.relate_scorer = RelateScorer()
        self.score_fuser = ScoreFuser()

    def heads(self, hits_info: dict, top_k: int = RANK_TOP_K) -> dict:
        """Get first k hits without rank"""
        hits = hits_info.get("hits", [])
        if top_k and len(hits) > top_k:
            hits = hits[:top_k]
            hits_info["hits"] = hits
            hits_info["return_hits"] = len(hits)
            hits_info["rank_method"] = "tops"
        return hits_info

    def get_top_hits(
        self, hits: list[dict], top_k: int, sort_field: str = "rank_score"
    ) -> list[dict]:
        return heapq.nlargest(top_k, hits, key=lambda x: x[sort_field])

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
            hit["rank_score"] = round(rrf_scores[i], 6)
        top_hits = self.get_top_hits(hits, top_k=top_k, sort_field="rank_score")
        hits_info["hits"] = top_hits
        hits_info["return_hits"] = len(top_hits)
        hits_info["rank_method"] = "rrf"

        return hits_info

    def stats_rank(self, hits_info: dict, top_k: int = RANK_TOP_K):
        hits: list[dict] = hits_info.get("hits", [])
        hits_num = len(hits)
        top_k = min(top_k, hits_num)
        for hit in hits:
            stats = dict_get(hit, "stat", {})
            pubdate = dict_get(hit, "pubdate", 0)
            relate = dict_get(hit, "score", 0.0)
            stats_score = self.stats_scorer.calc(stats)
            pubdate_score = self.pubdate_scorer.calc(pubdate)
            relate_score = self.relate_scorer.calc(relate)
            rank_score = self.score_fuser.fuse(
                stats_score=stats_score,
                pubdate_score=pubdate_score,
                relate_score=relate_score,
            )
            hit["rank_score"] = rank_score
        top_hits = self.get_top_hits(hits, top_k=top_k, sort_field="rank_score")
        hits_info["hits"] = top_hits
        hits_info["return_hits"] = len(top_hits)
        hits_info["rank_method"] = "stats"
        return hits_info
