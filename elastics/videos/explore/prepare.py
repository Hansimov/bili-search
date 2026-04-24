from __future__ import annotations

from dataclasses import dataclass

from tclogger import logger

from dsl.fields.qmod import has_rerank_qmod
from dsl.fields.qmod import is_hybrid_qmod
from dsl.fields.qmod import normalize_qmod
from dsl.fields.scope import get_scope_constraint_fields
from elastics.structure import build_auto_constraint_filter
from elastics.videos.constants import CONSTRAINT_FIELDS_DEFAULT


@dataclass(slots=True)
class UnifiedExploreRequest:
    qmod: list[str] | str | None
    owner_intent_info: dict | None
    constraint_filter: dict | None
    is_hybrid: bool
    has_word: bool
    has_vector: bool
    enable_rerank: bool


def _load_query_info(searcher, query: str, *, needed: bool) -> dict | None:
    if not needed:
        return None
    try:
        return searcher.query_rewriter.get_query_info(query)
    except Exception as exc:
        logger.warn(f"> Failed to parse query info for scope fallback: {exc}")
        return None


def _resolve_owner_intent_info(
    searcher, query: str, query_info: dict | None
) -> dict | None:
    query_understanding = searcher.query_understanding
    owner_intent_info = query_understanding.resolve_owner_intent(query).info
    if owner_intent_info or not query_info:
        return owner_intent_info

    keywords_body = query_info.get("keywords_body") or []
    body_query = " ".join(keywords_body).strip()
    if body_query and body_query != query:
        return query_understanding.resolve_owner_intent(body_query).info
    return owner_intent_info


def _apply_scope_fallback(
    qmod: list[str] | str | None,
    *,
    query_info: dict | None,
    is_hybrid: bool,
    has_word: bool,
    has_vector: bool,
    enable_rerank: bool,
) -> tuple[list[str] | str | None, bool, bool, bool, bool]:
    scope_fields = list((query_info or {}).get("scope_fields") or [])
    if not (scope_fields and (has_vector or enable_rerank)):
        return qmod, is_hybrid, has_word, has_vector, enable_rerank

    logger.warn(
        f"> Scope-limited search requested but qmod={qmod} uses shared "
        f"full-document embeddings; falling back to word-only mode"
    )
    return ["word"], False, True, False, False


def _apply_embed_fallback(
    searcher,
    qmod: list[str] | str | None,
    *,
    is_hybrid: bool,
    has_word: bool,
    has_vector: bool,
    enable_rerank: bool,
) -> tuple[list[str] | str | None, bool, bool, bool, bool]:
    if not has_vector or searcher.embed_client.is_available():
        return qmod, is_hybrid, has_word, has_vector, enable_rerank

    logger.warn(
        f"> Embed client unavailable, falling back from qmod={qmod} to word-only mode"
    )
    return ["word"], False, True, False, enable_rerank


def _build_constraint_filter(
    searcher,
    query: str,
    *,
    query_info: dict | None,
    constraint_filter: dict | None,
    auto_constraint: bool,
    has_vector: bool,
    verbose: bool,
    build_auto_constraint_filter_fn=build_auto_constraint_filter,
) -> dict | None:
    if constraint_filter is not None or not auto_constraint or not has_vector:
        return constraint_filter

    try:
        if query_info is None:
            query_info = searcher.query_rewriter.get_query_info(query)

        dsl_constraint = query_info.get("constraint_filter", {})
        if dsl_constraint:
            logger.hint(f"> DSL constraint: {dsl_constraint}", verbose=verbose)
            return dsl_constraint

        keywords_body = query_info.get("keywords_body", [])
        raw_query = " ".join(keywords_body) if keywords_body else ""
        if not raw_query:
            return None

        constraint_query = searcher._resolve_vector_auto_constraint_query(raw_query)
        if constraint_query != raw_query:
            logger.hint(
                f"> Relax auto-constraint to owner anchor: {constraint_query}",
                verbose=verbose,
            )
        resolved_constraint = build_auto_constraint_filter_fn(
            es_client=searcher.es.client,
            index_name=searcher.index_name,
            query=constraint_query,
            fields=get_scope_constraint_fields(
                query_info.get("scope_info"),
                CONSTRAINT_FIELDS_DEFAULT,
            ),
        )
        if resolved_constraint:
            logger.hint(f"> Auto-constraint: {resolved_constraint}", verbose=verbose)
        return resolved_constraint
    except Exception as exc:
        logger.warn(f"Auto-constraint build failed: {exc}")
        return constraint_filter


def prepare_unified_explore_request(
    searcher,
    *,
    query: str,
    qmod: list[str] | str | None,
    constraint_filter: dict | None,
    auto_constraint: bool,
    verbose: bool,
    build_auto_constraint_filter_fn=build_auto_constraint_filter,
) -> UnifiedExploreRequest:
    normalized_qmod = (
        searcher.get_qmod_from_query(query) if qmod is None else normalize_qmod(qmod)
    )
    logger.hint(f"> Query mode (qmod): {normalized_qmod}", verbose=verbose)

    is_hybrid = is_hybrid_qmod(normalized_qmod)
    has_word = "word" in normalized_qmod
    has_vector = "vector" in normalized_qmod
    enable_rerank = has_rerank_qmod(normalized_qmod)

    query_info = _load_query_info(
        searcher,
        query,
        needed=bool(has_vector or enable_rerank),
    )
    owner_intent_info = _resolve_owner_intent_info(searcher, query, query_info)
    normalized_qmod, is_hybrid, has_word, has_vector, enable_rerank = (
        _apply_scope_fallback(
            normalized_qmod,
            query_info=query_info,
            is_hybrid=is_hybrid,
            has_word=has_word,
            has_vector=has_vector,
            enable_rerank=enable_rerank,
        )
    )
    normalized_qmod, is_hybrid, has_word, has_vector, enable_rerank = (
        _apply_embed_fallback(
            searcher,
            normalized_qmod,
            is_hybrid=is_hybrid,
            has_word=has_word,
            has_vector=has_vector,
            enable_rerank=enable_rerank,
        )
    )
    resolved_constraint_filter = _build_constraint_filter(
        searcher,
        query,
        query_info=query_info,
        constraint_filter=constraint_filter,
        auto_constraint=auto_constraint,
        has_vector=has_vector,
        verbose=verbose,
        build_auto_constraint_filter_fn=build_auto_constraint_filter_fn,
    )

    return UnifiedExploreRequest(
        qmod=normalized_qmod,
        owner_intent_info=owner_intent_info,
        constraint_filter=resolved_constraint_filter,
        is_hybrid=is_hybrid,
        has_word=has_word,
        has_vector=has_vector,
        enable_rerank=enable_rerank,
    )
