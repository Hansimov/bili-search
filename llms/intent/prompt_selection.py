"""Prompt asset selection policy for the intent layer."""

from __future__ import annotations

from llms.contracts import IntentProfile


_BASE_ASSET_IDS = (
    "role.brief",
    "output.brief",
    "prompt_loading.brief",
    "result_isolation.brief",
    "response_style.brief",
    "routing_examples.brief",
)

_TARGET_ASSET_IDS = {
    "videos": (
        "route.videos.brief",
        "dsl.quickref.brief",
        "tool.search_videos.brief",
    ),
    "owners": (
        "route.owners.brief",
        "tool.search_owners.brief",
    ),
    "relations": (
        "route.relations.brief",
        "tool.search_owners.brief",
    ),
    "external": (
        "route.external.brief",
        "tool.search_google.brief",
    ),
    "mixed": (
        "route.mixed.brief",
        "dsl.quickref.brief",
        "tool.search_google.brief",
        "tool.search_videos.brief",
    ),
}

_SIGNAL_ASSET_IDS = {
    "needs_keyword_expansion": (
        "semantic.expansion.brief",
        "tool.expand_query.brief",
    ),
    "needs_term_normalization": (
        "tool.expand_query.detailed",
        "tool.expand_query.examples",
        "tool.search_videos.detailed",
        "tool.search_videos.examples",
    ),
    "needs_owner_resolution": (
        "tool.search_owners.brief",
        "tool.search_videos.detailed",
        "tool.search_videos.examples",
    ),
    "doc_signal_hints": ("facet.mapping.brief",),
}


def _extend_unique(asset_ids: list[str], additions: tuple[str, ...]) -> None:
    for asset_id in additions:
        if asset_id not in asset_ids:
            asset_ids.append(asset_id)


def select_prompt_asset_ids(intent: IntentProfile) -> list[str]:
    asset_ids = list(_BASE_ASSET_IDS)
    _extend_unique(asset_ids, _TARGET_ASSET_IDS.get(intent.final_target, ()))

    if intent.needs_keyword_expansion:
        _extend_unique(asset_ids, _SIGNAL_ASSET_IDS["needs_keyword_expansion"])
    if intent.needs_term_normalization:
        _extend_unique(asset_ids, _SIGNAL_ASSET_IDS["needs_term_normalization"])
    if intent.needs_owner_resolution:
        _extend_unique(asset_ids, _SIGNAL_ASSET_IDS["needs_owner_resolution"])
    if intent.doc_signal_hints:
        _extend_unique(asset_ids, _SIGNAL_ASSET_IDS["doc_signal_hints"])

    return asset_ids


__all__ = ["select_prompt_asset_ids"]
