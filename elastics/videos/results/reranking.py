from __future__ import annotations

from dataclasses import dataclass
import re


_NORMALIZE_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)


def _normalize_title_match_text(text: str) -> str:
    return _NORMALIZE_RE.sub("", str(text or "").lower())


def _shared_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _shared_suffix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[-(index + 1)] == right[-(index + 1)]:
        index += 1
    return index


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current_row[right_index - 1] + 1
            delete_cost = previous_row[right_index] + 1
            replace_cost = previous_row[right_index - 1] + (left_char != right_char)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


@dataclass(frozen=True, slots=True)
class FocusedTitleRerankInfo:
    applied: bool
    query: str = ""
    top_title: str = ""
    top_score: float = 0.0
    top_distance: int = -1

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "query": self.query,
            "top_title": self.top_title,
            "top_score": round(self.top_score, 6),
            "top_distance": self.top_distance,
        }


def rerank_focused_title_hits(
    hits: list[dict],
    *,
    query: str,
    focus_applied: bool,
    relation_rewritten: bool,
) -> tuple[list[dict], FocusedTitleRerankInfo]:
    normalized_query = _normalize_title_match_text(query)
    if (
        not focus_applied
        or relation_rewritten
        or len(normalized_query) < 6
        or len(hits) < 2
    ):
        return hits, FocusedTitleRerankInfo(applied=False)

    scored_hits: list[tuple[tuple[int, int, int, float, float], dict]] = []
    for hit in hits:
        normalized_title = _normalize_title_match_text(hit.get("title") or "")
        if len(normalized_title) < 4:
            score_key = (
                len(normalized_query),
                0,
                0,
                0.0,
                -float(hit.get("score") or 0.0),
            )
        else:
            distance = _levenshtein_distance(normalized_query, normalized_title)
            prefix_len = _shared_prefix_len(normalized_query, normalized_title)
            suffix_len = _shared_suffix_len(normalized_query, normalized_title)
            score_key = (
                distance,
                -suffix_len,
                -prefix_len,
                -(
                    1.0
                    - (distance / max(len(normalized_query), len(normalized_title), 1))
                ),
                -float(hit.get("score") or 0.0),
            )
        scored_hits.append((score_key, hit))

    ranked_pairs = sorted(scored_hits, key=lambda item: item[0])
    reranked = [hit for _, hit in ranked_pairs]
    if reranked == hits:
        return hits, FocusedTitleRerankInfo(applied=False)

    top_distance = ranked_pairs[0][0][0]
    normalized_top_title = _normalize_title_match_text(reranked[0].get("title") or "")
    max_distance = max(1, min(4, len(normalized_query) // 6))
    if top_distance > max_distance:
        return hits, FocusedTitleRerankInfo(applied=False)

    top_score = 1.0 - (
        top_distance / max(len(normalized_query), len(normalized_top_title), 1)
    )

    return reranked, FocusedTitleRerankInfo(
        applied=True,
        query=query,
        top_title=str(reranked[0].get("title") or ""),
        top_score=top_score,
        top_distance=top_distance,
    )


__all__ = ["FocusedTitleRerankInfo", "rerank_focused_title_hits"]
