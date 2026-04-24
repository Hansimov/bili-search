from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tclogger import logger

from dsl.fields.word import is_short_han_segment
from elastics.videos.policies.owner import get_owner_intent_policy


OWNER_INTENT_POLICY = get_owner_intent_policy()


@dataclass(slots=True)
class OwnerIntentResolution:
    info: dict = field(default_factory=dict)
    filters: list[dict] = field(default_factory=list)
    context_query: str = ""


@dataclass(slots=True)
class OwnerQueryIntentResolver:
    owner_searcher: Any

    @classmethod
    def _is_compact_owner_prefix_candidate(
        cls,
        token: str,
        owner: dict,
        top_score: float,
    ) -> bool:
        normalized_token = str(token or "").strip()
        owner_name = str((owner or {}).get("name") or "").strip()
        if not normalized_token or not owner_name:
            return False

        score = float((owner or {}).get("score") or 0.0)
        if score < OWNER_INTENT_POLICY.score_min:
            return False
        if top_score - score > OWNER_INTENT_POLICY.multi_owner_score_margin:
            return False
        if not owner_name.startswith(normalized_token):
            return False

        max_owner_name_len = (
            len(normalized_token) + OWNER_INTENT_POLICY.prefix_candidate_max_extra_chars
        )
        return len(owner_name) <= max_owner_name_len

    def _resolve_vector_auto_constraint_query(self, query: str) -> str:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return normalized_query
        if OWNER_INTENT_POLICY.has_blocked_marker(
            normalized_query,
            allow_whitespace=False,
        ):
            return normalized_query

        segments = [
            segment.strip() for segment in normalized_query.split() if segment.strip()
        ]
        if len(segments) < 2:
            return normalized_query

        anchor = segments[0]
        if not is_short_han_segment(anchor):
            return normalized_query

        try:
            owners_result = self.owner_searcher.search(
                anchor,
                mode="name",
                size=OWNER_INTENT_POLICY.resolve_size,
            )
        except Exception as exc:
            logger.warn(
                f"> Failed to resolve vector auto-constraint owner anchor: {exc}"
            )
            return normalized_query

        owners = owners_result.get("owners") or []
        if not owners:
            return normalized_query

        top_score = float((owners[0] or {}).get("score") or 0.0)
        if top_score < OWNER_INTENT_POLICY.score_min:
            return normalized_query

        for owner in owners[: OWNER_INTENT_POLICY.max_candidates]:
            if self._is_compact_owner_prefix_candidate(anchor, owner, top_score):
                return anchor

        return normalized_query

    def _resolve_spaced_owner_intent_info(self, query: str) -> dict:
        normalized_query = str(query or "").strip()
        if not normalized_query or not any(char.isspace() for char in normalized_query):
            return {}

        anchor_query = self._resolve_vector_auto_constraint_query(normalized_query)
        if not anchor_query or anchor_query == normalized_query:
            return {}

        owner_intent_info = self._resolve_owner_intent_info(anchor_query)
        if owner_intent_info:
            owner_intent_info = dict(owner_intent_info)
            owner_intent_info["owners"] = self._rerank_spaced_owner_intent_candidates(
                owner_intent_info.get("owners") or []
            )
            owner_intent_info.setdefault("source_query", normalized_query)
            owner_intent_info["query"] = anchor_query
        return owner_intent_info

    @staticmethod
    def _build_spaced_owner_context_query(owner_intent_info: dict | None) -> str:
        source_query = str((owner_intent_info or {}).get("source_query") or "").strip()
        anchor_query = str((owner_intent_info or {}).get("query") or "").strip()
        if not source_query or not anchor_query:
            return ""

        source_terms = [term.strip() for term in source_query.split() if term.strip()]
        anchor_terms = [term.strip() for term in anchor_query.split() if term.strip()]
        if not source_terms or not anchor_terms:
            return ""
        if source_terms[: len(anchor_terms)] != anchor_terms:
            return ""

        return " ".join(
            term for term in source_terms[len(anchor_terms) :] if len(term) >= 2
        )

    @staticmethod
    def _resolve_owner_intent_search_filters(
        owner_intent_info: dict | None,
    ) -> list[dict]:
        owner_filter = list((owner_intent_info or {}).get("owner_filter") or [])
        if owner_filter:
            return owner_filter

        if not (owner_intent_info or {}).get("source_query"):
            return []

        owner_mids: list[int] = []
        for owner in (owner_intent_info or {}).get("owners") or []:
            owner_mid = owner.get("mid")
            if not owner_mid:
                continue
            owner_mids.append(int(owner_mid))
            if len(owner_mids) >= OWNER_INTENT_POLICY.max_candidates:
                break

        if not owner_mids:
            return []
        if len(owner_mids) == 1:
            return [{"term": {"owner.mid": owner_mids[0]}}]
        return [{"terms": {"owner.mid": owner_mids}}]

    @staticmethod
    def _should_suppress_title_like_owner_filter(
        query: str,
        candidate: dict | None,
    ) -> bool:
        if not candidate:
            return False
        if not OWNER_INTENT_POLICY.looks_like_title_query(query):
            return False
        if not OWNER_INTENT_POLICY.supports_name_sources(candidate):
            return True

        owner_name = str((candidate or {}).get("name") or "").strip()
        return not OWNER_INTENT_POLICY.is_title_like_owner_query(query, owner_name)

    @staticmethod
    def _should_suppress_short_query_owner_filter(
        query: str,
        candidate: dict | None,
    ) -> bool:
        if not candidate:
            return False
        if not OWNER_INTENT_POLICY.is_short_query(query):
            return False
        if not OWNER_INTENT_POLICY.supports_name_sources(candidate):
            return True

        owner_name = str((candidate or {}).get("name") or "").strip()
        return not OWNER_INTENT_POLICY.is_short_query_owner_name_match(
            query,
            owner_name,
        )

    def _resolve_owner_intent_info(self, query: str) -> dict:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return {}
        if OWNER_INTENT_POLICY.has_blocked_marker(
            normalized_query,
            allow_whitespace=True,
        ):
            return {}
        if any(char.isspace() for char in normalized_query):
            return {}
        if not OWNER_INTENT_POLICY.query_length_ok(normalized_query):
            return {}
        if self._looks_like_model_code_query(normalized_query):
            return {}

        owners_result = self.owner_searcher.search(
            normalized_query,
            mode="name",
            size=OWNER_INTENT_POLICY.resolve_size,
        )
        owners = owners_result.get("owners") or []
        owner_candidates = self._select_owner_intent_candidates(owners)
        candidate = self._select_confident_owner_intent_candidate(
            owners,
            candidates=owner_candidates,
        )
        if not owner_candidates and not candidate:
            return {}

        owner_intent_info = {
            "query": normalized_query,
            "owners": owner_candidates,
        }
        if candidate:
            owner_intent_info["owner"] = candidate
            if self._should_suppress_title_like_owner_filter(
                normalized_query,
                candidate,
            ):
                owner_intent_info["filter_suppressed_reason"] = "title_like_query"
            elif self._should_suppress_short_query_owner_filter(
                normalized_query,
                candidate,
            ):
                owner_intent_info["filter_suppressed_reason"] = "broad_short_query"
            else:
                owner_intent_info["owner_filter"] = self._build_owner_filter(candidate)
        return owner_intent_info

    def resolve_owner_intent(self, query: str) -> OwnerIntentResolution:
        owner_intent_info = self._resolve_owner_intent_info(query)
        if not owner_intent_info:
            owner_intent_info = self._resolve_spaced_owner_intent_info(query)
        if not owner_intent_info:
            return OwnerIntentResolution()

        owner_filters = self._resolve_owner_intent_search_filters(owner_intent_info)
        context_query = ""
        if owner_filters:
            context_query = self._build_spaced_owner_context_query(owner_intent_info)
        return OwnerIntentResolution(
            info=owner_intent_info,
            filters=owner_filters,
            context_query=context_query,
        )

    @staticmethod
    def _looks_like_model_code_query(text: str) -> bool:
        return OWNER_INTENT_POLICY.looks_like_model_code_query(text)

    @staticmethod
    def _build_owner_filter(owner: dict) -> list[dict]:
        owner_mid = owner.get("mid")
        if not owner_mid:
            return []
        return [{"term": {"owner.mid": int(owner_mid)}}]

    @staticmethod
    def _normalize_owner_intent_candidate(candidate: dict) -> dict | None:
        owner_mid = candidate.get("mid")
        owner_name = str(candidate.get("name") or "").strip()
        if not owner_mid or not owner_name:
            return None
        return {
            "mid": int(owner_mid),
            "name": owner_name,
            "score": float(candidate.get("score") or 0.0),
            "sample_view": int(candidate.get("sample_view") or 0),
            "sources": list(candidate.get("sources") or []),
        }

    @classmethod
    def _rerank_spaced_owner_intent_candidates(
        cls,
        candidates: list[dict],
    ) -> list[dict]:
        normalized_candidates = [
            dict(candidate or {})
            for candidate in candidates
            if candidate and candidate.get("mid") and candidate.get("name")
        ]
        if len(normalized_candidates) <= 1:
            return normalized_candidates

        top_score = float(normalized_candidates[0].get("score") or 0.0)

        def _sort_key(candidate: dict) -> tuple[int, float, int, int]:
            score = float(candidate.get("score") or 0.0)
            score_gap = top_score - score
            near_top = (
                0 if score_gap <= OWNER_INTENT_POLICY.spaced_rerank_score_gap else 1
            )
            sample_view = int(candidate.get("sample_view") or 0)
            return (
                near_top,
                -sample_view,
                -score,
                int(candidate.get("mid") or 0),
            )

        return sorted(normalized_candidates, key=_sort_key)

    @staticmethod
    def _candidate_supports_multi_owner_intent(candidate: dict) -> bool:
        return OWNER_INTENT_POLICY.supports_multi_owner_sources(candidate)

    @classmethod
    def _select_owner_intent_candidates(cls, owners: list[dict]) -> list[dict]:
        if not owners:
            return []

        top_score = float((owners[0] or {}).get("score") or 0.0)
        if top_score < OWNER_INTENT_POLICY.score_min:
            return []

        candidates: list[dict] = []
        for index, owner in enumerate(owners):
            normalized = cls._normalize_owner_intent_candidate(owner)
            if not normalized:
                continue

            score = normalized["score"]
            if score < OWNER_INTENT_POLICY.score_min:
                continue

            if index > 0:
                score_gap = top_score - score
                if score_gap > OWNER_INTENT_POLICY.multi_owner_score_margin:
                    continue
                if (
                    not cls._candidate_supports_multi_owner_intent(normalized)
                    and score_gap > OWNER_INTENT_POLICY.multi_owner_name_margin
                ):
                    continue

            candidates.append(normalized)
            if len(candidates) >= OWNER_INTENT_POLICY.max_candidates:
                break

        return candidates

    @classmethod
    def _select_confident_owner_intent_candidate(
        cls,
        owners: list[dict],
        candidates: list[dict] | None = None,
    ) -> dict | None:
        if not owners:
            return None

        candidates = (
            candidates
            if candidates is not None
            else cls._select_owner_intent_candidates(owners)
        )
        if len(candidates) != 1:
            return None

        top = cls._normalize_owner_intent_candidate(owners[0] or {})
        if not top:
            return None

        top_score = float(top.get("score") or 0.0)
        second_score = (
            float((owners[1] or {}).get("score") or 0.0) if len(owners) > 1 else 0.0
        )
        gap = top_score - second_score
        if (
            top_score < OWNER_INTENT_POLICY.score_min
            or gap < OWNER_INTENT_POLICY.filter_gap_min
        ):
            return None
        if top["mid"] != candidates[0]["mid"]:
            return None
        return top


__all__ = ["OwnerIntentResolution", "OwnerQueryIntentResolver"]
