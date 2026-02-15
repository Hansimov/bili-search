"""
Base Types for Recall System

Provides data classes for recall results and merged recall pools.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RecallResult:
    """Result from a single recall lane.

    Attributes:
        hits: List of candidate documents (dicts with at least 'bvid').
        lane: Name of the recall lane (e.g., 'relevance', 'popularity').
        total_hits: Estimated total matching documents in ES.
        took_ms: Time taken for this recall in milliseconds.
        timed_out: Whether the ES query timed out.
    """

    hits: list[dict]
    lane: str
    total_hits: int = 0
    took_ms: float = 0.0
    timed_out: bool = False


@dataclass
class RecallPool:
    """Merged, deduplicated pool from multiple recall lanes.

    Attributes:
        hits: Deduplicated list of candidate documents.
        lanes_info: Dict mapping lane name to RecallResult metadata.
        total_hits: Max total_hits from any lane (best estimate).
        took_ms: Total wall-clock time for all recalls.
        timed_out: Whether any lane timed out.
        lane_tags: Dict mapping bvid -> set of lane names it came from.
    """

    hits: list[dict] = field(default_factory=list)
    lanes_info: dict = field(default_factory=dict)
    total_hits: int = 0
    took_ms: float = 0.0
    timed_out: bool = False
    lane_tags: dict = field(default_factory=dict)

    @staticmethod
    def merge(*results: RecallResult) -> "RecallPool":
        """Merge multiple RecallResults into a deduplicated RecallPool.

        Documents appearing in multiple lanes get tagged with all their
        source lanes. This information is used by the diversified ranker
        to understand which dimensions each candidate covers.

        Args:
            *results: RecallResult instances to merge.

        Returns:
            Merged RecallPool with deduplicated hits.
        """
        seen_bvids: dict[str, int] = {}  # bvid -> index in hits
        merged_hits: list[dict] = []
        lane_tags: dict[str, set] = {}  # bvid -> set of lane names
        lanes_info: dict[str, dict] = {}
        total_hits = 0
        max_took = 0.0
        any_timeout = False

        for result in results:
            lane_name = result.lane
            lanes_info[lane_name] = {
                "hit_count": len(result.hits),
                "total_hits": result.total_hits,
                "took_ms": result.took_ms,
                "timed_out": result.timed_out,
            }
            total_hits = max(total_hits, result.total_hits)
            max_took = max(max_took, result.took_ms)
            any_timeout = any_timeout or result.timed_out

            for rank, hit in enumerate(result.hits):
                bvid = hit.get("bvid")
                if not bvid:
                    continue

                # Tag lane rank for this hit
                rank_key = f"{lane_name}_rank"
                hit[rank_key] = rank

                if bvid not in seen_bvids:
                    # New document
                    seen_bvids[bvid] = len(merged_hits)
                    merged_hits.append(hit)
                    lane_tags[bvid] = {lane_name}
                else:
                    # Already seen - merge lane tag and rank info
                    idx = seen_bvids[bvid]
                    merged_hits[idx][rank_key] = rank
                    lane_tags[bvid].add(lane_name)

        # Store lane tags in hits for downstream use
        for hit in merged_hits:
            bvid = hit.get("bvid")
            if bvid and bvid in lane_tags:
                hit["_recall_lanes"] = lane_tags[bvid]

        return RecallPool(
            hits=merged_hits,
            lanes_info=lanes_info,
            total_hits=total_hits,
            took_ms=max_took,
            timed_out=any_timeout,
            lane_tags=lane_tags,
        )
