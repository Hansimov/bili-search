from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from elastics.videos.constants import SEARCH_REQUEST_TYPE, SEARCH_REQUEST_TYPE_DEFAULT
from elastics.videos.intent.owner_query import (
    OwnerIntentResolution,
    OwnerQueryIntentResolver,
)
from elastics.videos.query.preparation import SearchPreparation, SearchQueryPreparer


@dataclass(slots=True)
class VideoQueryUnderstanding:
    query_rewriter: Any
    owner_searcher: Any
    relations_client: Any

    def _build_query_preparer(self) -> SearchQueryPreparer:
        return SearchQueryPreparer(
            query_rewriter=self.query_rewriter,
            relations_client=self.relations_client,
        )

    def _build_owner_intent_resolver(self) -> OwnerQueryIntentResolver:
        return OwnerQueryIntentResolver(owner_searcher=self.owner_searcher)

    def prepare_search_query(
        self,
        query: str,
        *,
        suggest_info: dict | None = None,
        request_type: SEARCH_REQUEST_TYPE = SEARCH_REQUEST_TYPE_DEFAULT,
    ) -> SearchPreparation:
        return self._build_query_preparer().prepare_search_query(
            query,
            suggest_info=suggest_info,
            request_type=request_type,
        )

    def resolve_owner_intent(self, query: str) -> OwnerIntentResolution:
        return self._build_owner_intent_resolver().resolve_owner_intent(query)


__all__ = [
    "OwnerIntentResolution",
    "SearchPreparation",
    "VideoQueryUnderstanding",
]
