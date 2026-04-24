from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from tclogger import logger

from elastics.videos.constants import EXPLORE_TIMEOUT
from ranks.constants import EXPLORE_GROUP_OWNER_LIMIT, EXPLORE_RANK_TOP_K
from recalls.base import RecallPool


OWNER_INTENT_RECALL_MIN = 8
OWNER_INTENT_RECALL_MAX = 16
OWNER_INTENT_BLEND_WINDOW = 12
OWNER_INTENT_BLEND_VISIBLE_HITS = 2
OWNER_INTENT_BLEND_SLOTS = (1, 4)


@dataclass(slots=True)
class ExploreOwnerIntentCoordinator:
    search_fn: Callable[..., dict]
    get_user_docs_fn: Callable[[list[str]], Any]

    @staticmethod
    def owner_intent_recall_limit(rank_top_k: int) -> int:
        if rank_top_k <= 0:
            return OWNER_INTENT_RECALL_MIN
        return min(
            max(rank_top_k // 20, OWNER_INTENT_RECALL_MIN),
            OWNER_INTENT_RECALL_MAX,
        )

    @staticmethod
    def get_owner_intent_candidates(owner_intent_info: dict | None) -> list[dict]:
        if not owner_intent_info:
            return []

        owners = [
            owner
            for owner in (owner_intent_info.get("owners") or [])
            if owner.get("mid") and owner.get("name")
        ]
        if owners:
            return owners

        owner = owner_intent_info.get("owner") or {}
        if owner.get("mid") and owner.get("name"):
            return [owner]
        return []

    @classmethod
    def resolve_owner_intent_supplement_filters(
        cls,
        owner_intent_info: dict | None,
    ) -> list[dict]:
        owner_filter = list((owner_intent_info or {}).get("owner_filter") or [])
        if owner_filter:
            return owner_filter

        if not (owner_intent_info or {}).get("source_query"):
            return []

        owner_candidates = cls.get_owner_intent_candidates(owner_intent_info)
        owner_mids = []
        for candidate in owner_candidates[:OWNER_INTENT_RECALL_MAX]:
            owner_mid = candidate.get("mid")
            if not owner_mid:
                continue
            owner_mids.append(int(owner_mid))

        if not owner_mids:
            return []

        if len(owner_mids) == 1:
            return [{"term": {"owner.mid": owner_mids[0]}}]
        return [{"terms": {"owner.mid": owner_mids}}]

    @staticmethod
    def get_spaced_owner_context_terms(owner_intent_info: dict | None) -> list[str]:
        source_query = str((owner_intent_info or {}).get("source_query") or "").strip()
        anchor_query = str((owner_intent_info or {}).get("query") or "").strip()
        if not source_query or not anchor_query:
            return []

        source_terms = [term.strip() for term in source_query.split() if term.strip()]
        anchor_terms = [term.strip() for term in anchor_query.split() if term.strip()]
        if not source_terms or not anchor_terms:
            return []
        if source_terms[: len(anchor_terms)] != anchor_terms:
            return []

        return [term for term in source_terms[len(anchor_terms) :] if len(term) >= 2]

    @staticmethod
    def hit_matches_owner_context(hit: dict, context_terms: list[str]) -> bool:
        if not context_terms:
            return True

        searchable_parts = [
            str(hit.get("title") or ""),
            str(hit.get("tags") or ""),
            str(hit.get("desc") or ""),
        ]
        searchable_text = "\n".join(part for part in searchable_parts if part)
        if not searchable_text:
            return False

        return any(term in searchable_text for term in context_terms)

    def ensure_owner_intent_author_groups(
        self,
        authors_list: list[dict],
        owner_intent_info: dict | None,
    ) -> list[dict]:
        owner_candidates = self.get_owner_intent_candidates(owner_intent_info)
        if not owner_candidates:
            return authors_list

        existing_mids = {
            str(author.get("mid") or "") for author in authors_list if author.get("mid")
        }
        missing_candidates = [
            candidate
            for candidate in owner_candidates
            if str(candidate.get("mid") or "") not in existing_mids
        ]
        if not missing_candidates:
            return authors_list

        user_docs = self.get_user_docs_fn(
            [candidate.get("mid") for candidate in missing_candidates]
        )
        if isinstance(user_docs, dict):
            user_docs_iter = user_docs.values()
        else:
            user_docs_iter = user_docs or []
        user_docs_by_mid = {
            str(doc.get("mid") or ""): doc
            for doc in user_docs_iter
            if isinstance(doc, dict) and doc.get("mid")
        }

        first_appear_base = (
            max(
                [int(author.get("first_appear_order") or 0) for author in authors_list]
                or [0]
            )
            + 1
        )
        extended_authors = list(authors_list)
        for offset, candidate in enumerate(missing_candidates):
            candidate_mid = str(candidate.get("mid") or "")
            user_doc = user_docs_by_mid.get(candidate_mid, {})
            extended_authors.append(
                {
                    "mid": candidate.get("mid"),
                    "name": candidate.get("name"),
                    "latest_pubdate": 0,
                    "sum_view": 0,
                    "sum_sort_score": 0,
                    "sum_rank_score": 0,
                    "top_rank_score": 0,
                    "first_appear_order": first_appear_base + offset,
                    "sum_count": 0,
                    "hits": [],
                    "face": user_doc.get("face", ""),
                }
            )

        return extended_authors

    def supplement_with_owner_intent_hits(
        self,
        pool: RecallPool,
        query: str,
        owner_intent_info: dict | None,
        source_fields: list[str] = None,
        extra_filters: list[dict] = None,
        timeout: float = EXPLORE_TIMEOUT,
        rank_top_k: int = EXPLORE_RANK_TOP_K,
        verbose: bool = False,
    ) -> RecallPool:
        owner_filter = self.resolve_owner_intent_supplement_filters(owner_intent_info)
        if not owner_filter:
            return pool

        effective_source_fields = list(
            source_fields
            or [
                "bvid",
                "title",
                "tags",
                "desc",
                "owner",
                "stat",
                "pubdate",
                "duration",
                "stat_score",
            ]
        )
        if "owner" not in effective_source_fields:
            effective_source_fields.append("owner")

        owner_info = owner_intent_info.get("owner", {}) if owner_intent_info else {}
        owner_candidates = self.get_owner_intent_candidates(owner_intent_info)
        owner_label = owner_info.get("name") or ", ".join(
            candidate.get("name") or "" for candidate in owner_candidates[:3]
        )
        owner_query = str((owner_intent_info or {}).get("query") or query or "").strip()
        owner_limit = self.owner_intent_recall_limit(rank_top_k)
        if verbose:
            logger.hint(
                f"> Supplement owner-intent recall: {owner_label}"
                f" ({owner_info.get('mid')}) limit={owner_limit}",
                verbose=verbose,
            )

        start = time.perf_counter()
        owner_res = self.search_fn(
            query=owner_query,
            source_fields=effective_source_fields,
            extra_filters=list(extra_filters or []) + owner_filter,
            parse_hits=True,
            add_region_info=False,
            add_highlights_info=False,
            is_highlight=False,
            boost=True,
            rank_method="heads",
            limit=owner_limit,
            rank_top_k=owner_limit,
            timeout=timeout,
            verbose=False,
        )
        owner_hits = owner_res.get("hits", [])
        context_terms = self.get_spaced_owner_context_terms(owner_intent_info)
        if context_terms:
            owner_hits = [
                hit
                for hit in owner_hits
                if self.hit_matches_owner_context(hit, context_terms)
            ]
        if not owner_hits:
            return pool

        took_ms = round((time.perf_counter() - start) * 1000, 2)
        merged_hits = list(pool.hits)
        merged_tags = {bvid: set(tags) for bvid, tags in (pool.lane_tags or {}).items()}
        seen_bvids = {
            hit.get("bvid"): idx
            for idx, hit in enumerate(merged_hits)
            if hit.get("bvid")
        }
        new_hits = 0

        for rank, hit in enumerate(owner_hits):
            bvid = hit.get("bvid")
            if not bvid:
                continue

            hit["owner_intent_rank"] = rank
            if bvid in seen_bvids:
                merged_hit = merged_hits[seen_bvids[bvid]]
                merged_hit["owner_intent_rank"] = rank
                merged_tags.setdefault(bvid, set()).add("owner_intent")
                existing_score = merged_hit.get("score", 0) or 0
                new_score = hit.get("score", 0) or 0
                if new_score > existing_score:
                    merged_hit["score"] = new_score
            else:
                merged_hits.append(hit)
                seen_bvids[bvid] = len(merged_hits) - 1
                merged_tags[bvid] = {"owner_intent"}
                new_hits += 1

        for hit in merged_hits:
            bvid = hit.get("bvid")
            if bvid and bvid in merged_tags:
                hit["_recall_lanes"] = merged_tags[bvid]

        return RecallPool(
            hits=merged_hits,
            lanes_info={
                **pool.lanes_info,
                "owner_intent": {
                    "hit_count": len(owner_hits),
                    "new_hits": new_hits,
                    "owner_mid": owner_info.get("mid"),
                    "owner_name": owner_info.get("name"),
                    "owner_count": len(owner_candidates),
                    "took_ms": took_ms,
                },
            },
            total_hits=max(pool.total_hits, owner_res.get("total_hits", 0)),
            took_ms=pool.took_ms,
            timed_out=pool.timed_out or owner_res.get("timed_out", False),
            lane_tags=merged_tags,
            pool_hints=pool.pool_hints,
        )

    @staticmethod
    def blend_owner_intent_hits(
        search_res: dict,
        owner_intent_info: dict | None,
    ) -> dict:
        hits = list(search_res.get("hits") or [])
        owner_mid = (owner_intent_info or {}).get("owner", {}).get("mid")
        if not hits or not owner_mid:
            return search_res

        owner_mid = str(owner_mid)

        def _is_owner_hit(hit: dict) -> bool:
            return str((hit.get("owner") or {}).get("mid") or "") == owner_mid

        owner_candidates = []
        visible_owner_hits = 0
        for idx, hit in enumerate(hits):
            if not _is_owner_hit(hit):
                continue
            hit["intent_match_video"] = True
            if idx < OWNER_INTENT_BLEND_WINDOW:
                visible_owner_hits += 1
                continue
            owner_candidates.append((idx, hit))

        if (
            visible_owner_hits >= OWNER_INTENT_BLEND_VISIBLE_HITS
            or not owner_candidates
        ):
            return search_res

        owner_candidates.sort(
            key=lambda item: (
                item[1].get("owner_intent_rank", 10**9),
                item[0],
            )
        )
        needed_hits = min(
            OWNER_INTENT_BLEND_VISIBLE_HITS - visible_owner_hits,
            len(OWNER_INTENT_BLEND_SLOTS),
            len(owner_candidates),
        )
        if needed_hits <= 0:
            return search_res

        selected_hits = [hit for _, hit in owner_candidates[:needed_hits]]
        selected_bvids = {hit.get("bvid") for hit in selected_hits if hit.get("bvid")}
        if not selected_bvids:
            return search_res

        base_hits = [hit for hit in hits if hit.get("bvid") not in selected_bvids]
        visible_owner_hits = sum(
            1 for hit in base_hits[:OWNER_INTENT_BLEND_WINDOW] if _is_owner_hit(hit)
        )

        inserted_hits = 0
        remaining_selected = list(selected_hits)
        for slot in OWNER_INTENT_BLEND_SLOTS:
            if visible_owner_hits >= OWNER_INTENT_BLEND_VISIBLE_HITS:
                break
            if not remaining_selected:
                break

            bounded_slot = min(slot, len(base_hits))
            if bounded_slot < len(base_hits) and _is_owner_hit(base_hits[bounded_slot]):
                visible_owner_hits += 1
                continue

            hit = remaining_selected.pop(0)
            hit["owner_intent_blended"] = True
            base_hits.insert(bounded_slot, hit)
            visible_owner_hits += 1
            inserted_hits += 1

        if inserted_hits:
            total = max(len(base_hits), 1)
            for idx, hit in enumerate(base_hits):
                hit["rank_score"] = round(1.0 - (idx / total), 6)
            search_res["hits"] = base_hits
            search_res["owner_intent_blend"] = {
                "owner_mid": owner_mid,
                "inserted_hits": inserted_hits,
                "visible_owner_hits": visible_owner_hits,
            }

        return search_res

    @classmethod
    def promote_owner_intent_author_group(
        cls,
        authors_list: list[dict],
        owner_intent_info: dict | None,
    ) -> list[dict]:
        owner_candidates = cls.get_owner_intent_candidates(owner_intent_info)
        if not owner_candidates or not authors_list:
            return authors_list

        has_spaced_source_query = bool((owner_intent_info or {}).get("source_query"))

        candidate_by_mid = {
            str(candidate.get("mid")): {
                **candidate,
                "intent_owner_rank": index,
            }
            for index, candidate in enumerate(owner_candidates)
        }

        reranked_authors: list[dict] = []
        for author in authors_list:
            author_mid = str(author.get("mid") or "")
            candidate = candidate_by_mid.get(author_mid)
            intent_author_score = (
                float(author.get("sum_rank_score") or 0.0)
                + min(float(author.get("sum_count") or 0.0), 20.0) * 0.2
                + float(author.get("top_rank_score") or 0.0) * 2.0
            )
            if candidate:
                author["intent_match"] = True
                author["intent_owner_score"] = float(candidate.get("score") or 0.0)
                author["intent_owner_rank"] = candidate["intent_owner_rank"]
                intent_author_score += author["intent_owner_score"] / 80.0
                if has_spaced_source_query and not float(
                    author.get("sum_count") or 0.0
                ):
                    intent_author_score = max(
                        intent_author_score,
                        4.0 - float(candidate["intent_owner_rank"]) * 0.05,
                    )
            author["intent_author_score"] = round(intent_author_score, 4)
            reranked_authors.append(author)

        reranked_authors.sort(
            key=lambda author: (
                float(author.get("intent_author_score") or 0.0),
                float(author.get("sum_rank_score") or 0.0),
                float(author.get("sum_count") or 0.0),
                float(author.get("top_rank_score") or 0.0),
                -(int(author.get("first_appear_order") or 0)),
            ),
            reverse=True,
        )
        return reranked_authors


__all__ = ["ExploreOwnerIntentCoordinator"]
