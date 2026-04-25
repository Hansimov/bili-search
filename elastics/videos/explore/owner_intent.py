from __future__ import annotations

from elastics.videos.constants import EXPLORE_TIMEOUT
from elastics.videos.intent.explore_owner_intent import ExploreOwnerIntentCoordinator
from ranks.constants import EXPLORE_RANK_TOP_K
from recalls.base import RecallPool


class ExploreOwnerIntentMixin:
    def _build_owner_intent_coordinator(self) -> ExploreOwnerIntentCoordinator:
        return ExploreOwnerIntentCoordinator(
            search_fn=self.search,
            get_user_docs_fn=self.get_user_docs,
        )

    @staticmethod
    def _owner_intent_recall_limit(rank_top_k: int) -> int:
        return ExploreOwnerIntentCoordinator.owner_intent_recall_limit(rank_top_k)

    @classmethod
    def _resolve_owner_intent_supplement_filters(
        cls,
        owner_intent_info: dict | None,
    ) -> list[dict]:
        return ExploreOwnerIntentCoordinator.resolve_owner_intent_supplement_filters(
            owner_intent_info
        )

    @staticmethod
    def _get_spaced_owner_context_terms(
        owner_intent_info: dict | None,
    ) -> list[str]:
        return ExploreOwnerIntentCoordinator.get_spaced_owner_context_terms(
            owner_intent_info
        )

    @classmethod
    def _hit_matches_owner_context(
        cls,
        hit: dict,
        context_terms: list[str],
    ) -> bool:
        return ExploreOwnerIntentCoordinator.hit_matches_owner_context(
            hit,
            context_terms,
        )

    def _ensure_owner_intent_author_groups(
        self,
        authors_list: list[dict],
        owner_intent_info: dict | None,
    ) -> list[dict]:
        return self._build_owner_intent_coordinator().ensure_owner_intent_author_groups(
            authors_list,
            owner_intent_info,
        )

    def _supplement_with_owner_intent_hits(
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
        return self._build_owner_intent_coordinator().supplement_with_owner_intent_hits(
            pool=pool,
            query=query,
            owner_intent_info=owner_intent_info,
            source_fields=source_fields,
            extra_filters=extra_filters,
            timeout=timeout,
            rank_top_k=rank_top_k,
            verbose=verbose,
        )

    @staticmethod
    def _blend_owner_intent_hits(
        search_res: dict,
        owner_intent_info: dict | None,
    ) -> dict:
        return ExploreOwnerIntentCoordinator.blend_owner_intent_hits(
            search_res,
            owner_intent_info,
        )

    @staticmethod
    def _get_owner_intent_candidates(owner_intent_info: dict | None) -> list[dict]:
        return ExploreOwnerIntentCoordinator.get_owner_intent_candidates(
            owner_intent_info
        )

    @staticmethod
    def _promote_owner_intent_author_group(
        authors_list: list[dict], owner_intent_info: dict | None
    ) -> list[dict]:
        return ExploreOwnerIntentCoordinator.promote_owner_intent_author_group(
            authors_list,
            owner_intent_info,
        )
