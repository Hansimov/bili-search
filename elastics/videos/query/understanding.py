from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from tclogger import logger

from dsl.fields.word import is_short_han_segment
from elastics.videos.constants import SEARCH_REQUEST_TYPE, SEARCH_REQUEST_TYPE_DEFAULT
from elastics.videos.policies.owner import get_owner_intent_policy
from elastics.videos.policies.focus import get_search_focus_policy
from elastics.videos.policies.semantic import get_search_semantic_policy
from llms.intent.focus import compact_focus_key, rewrite_known_term_aliases


OWNER_INTENT_POLICY = get_owner_intent_policy()
SEARCH_FOCUS_POLICY = get_search_focus_policy()
SEARCH_SEMANTIC_POLICY = get_search_semantic_policy()


@dataclass(slots=True)
class SearchPreparation:
    effective_query: str
    suggest_info: dict = field(default_factory=dict)
    semantic_rewrite_info: dict = field(default_factory=dict)
    query_focus_info: dict = field(default_factory=dict)


@dataclass(slots=True)
class OwnerIntentResolution:
    info: dict = field(default_factory=dict)
    filters: list[dict] = field(default_factory=list)
    context_query: str = ""


@dataclass(slots=True)
class VideoQueryUnderstanding:
    query_rewriter: Any
    owner_searcher: Any
    relations_client: Any

    @staticmethod
    def _keyword_has_mixed_script(text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        has_cjk = any("\u4e00" <= char <= "\u9fff" for char in normalized)
        has_ascii_alnum = any(char.isascii() and char.isalnum() for char in normalized)
        return has_cjk and has_ascii_alnum

    @classmethod
    def _semantic_fallback_requires_auto(cls, result: dict | None) -> bool:
        if not isinstance(result, dict):
            return False
        error = str(result.get("error", "") or "").lower()
        if not error:
            return False
        return "mode must be" in error and "semantic" in error

    def _build_semantic_group_replaces_count(
        self,
        query: str,
        options: list[dict],
    ) -> tuple[list[list[object]], list[dict]]:
        query_info = self.query_rewriter.get_query_info(query)
        keywords = list(query_info.get("keywords_body") or [])
        if not keywords:
            return [], []

        accepted_groups: list[list[object]] = []
        accepted_options: list[dict] = []
        top_score = float((options[0] or {}).get("score") or 0.0) if options else 0.0
        for option in options:
            text = str((option or {}).get("text") or "").strip()
            if not text:
                continue
            if SEARCH_SEMANTIC_POLICY.has_candidate_blocked_marker(text):
                continue
            score = float((option or {}).get("score") or 0.0)
            if score < SEARCH_SEMANTIC_POLICY.min_option_score:
                continue
            if (
                top_score
                and score / top_score < SEARCH_SEMANTIC_POLICY.min_option_score_ratio
            ):
                continue

            option_query_info = self.query_rewriter.get_query_info(text)
            option_keywords = list(option_query_info.get("keywords_body") or [])
            if not option_keywords:
                continue
            if SEARCH_SEMANTIC_POLICY.require_same_keyword_count and len(
                option_keywords
            ) != len(keywords):
                continue

            group_replaces: list[str] = []
            for qword, hword in zip(keywords, option_keywords):
                if qword == hword:
                    continue
                group_replaces.extend([qword, hword])
            if not group_replaces:
                continue

            accepted_groups.append([group_replaces, int(round(score))])
            accepted_options.append(
                {
                    "text": text,
                    "score": score,
                    "keywords": option_keywords,
                }
            )
            if len(accepted_groups) >= SEARCH_SEMANTIC_POLICY.max_rewrite_options:
                break

        return accepted_groups, accepted_options

    def _resolve_search_semantic_rewrite(
        self,
        query: str,
        suggest_info: dict | None = None,
        request_type: SEARCH_REQUEST_TYPE = SEARCH_REQUEST_TYPE_DEFAULT,
        allow_relation_rewrite: bool = True,
    ) -> tuple[str, dict, dict]:
        semantic_rewrite_info = {
            "input_query": str(query or ""),
            "applied": False,
            "alias_rewritten": False,
            "relation_rewritten": False,
        }
        if (
            request_type != SEARCH_REQUEST_TYPE_DEFAULT
            or suggest_info
            or not SEARCH_SEMANTIC_POLICY.enabled
        ):
            semantic_rewrite_info["applied_query"] = str(query or "")
            return str(query or ""), {}, semantic_rewrite_info

        base_query = str(query or "").strip()
        if not base_query:
            semantic_rewrite_info["applied_query"] = base_query
            return base_query, {}, semantic_rewrite_info

        if SEARCH_SEMANTIC_POLICY.alias_rewrite_enabled:
            alias_query = rewrite_known_term_aliases(base_query).strip()
            if alias_query and alias_query != base_query:
                semantic_rewrite_info["alias_rewritten"] = True
                semantic_rewrite_info["alias_query"] = alias_query
                base_query = alias_query

        semantic_suggest_info: dict = {}
        query_info = self.query_rewriter.get_query_info(base_query)
        keywords = list(query_info.get("keywords_body") or [])
        should_attempt_relation_rewrite = (
            SEARCH_SEMANTIC_POLICY.relation_rewrite_enabled
            and allow_relation_rewrite
            and self.relations_client is not None
            and SEARCH_SEMANTIC_POLICY.query_length_ok(base_query)
            and SEARCH_SEMANTIC_POLICY.keyword_count_ok(keywords)
            and not SEARCH_SEMANTIC_POLICY.has_blocked_marker(base_query)
            and (
                (
                    SEARCH_SEMANTIC_POLICY.trigger_alias_rewritten_query
                    and semantic_rewrite_info["alias_rewritten"]
                )
                or (
                    SEARCH_SEMANTIC_POLICY.trigger_mixed_script_keyword
                    and any(
                        self._keyword_has_mixed_script(keyword) for keyword in keywords
                    )
                )
            )
        )
        if should_attempt_relation_rewrite:
            relation_result = self.relations_client.related_tokens_by_tokens(
                text=base_query,
                mode=SEARCH_SEMANTIC_POLICY.relation_mode,
                size=SEARCH_SEMANTIC_POLICY.relation_size,
                scan_limit=SEARCH_SEMANTIC_POLICY.relation_scan_limit,
                use_pinyin=True,
            )
            if self._semantic_fallback_requires_auto(relation_result):
                relation_result = self.relations_client.related_tokens_by_tokens(
                    text=base_query,
                    mode=SEARCH_SEMANTIC_POLICY.relation_fallback_mode,
                    size=SEARCH_SEMANTIC_POLICY.relation_size,
                    scan_limit=SEARCH_SEMANTIC_POLICY.relation_scan_limit,
                    use_pinyin=True,
                )

            options = list(relation_result.get("options") or [])
            group_replaces_count, accepted_options = (
                self._build_semantic_group_replaces_count(base_query, options)
            )
            if group_replaces_count:
                semantic_suggest_info = {"group_replaces_count": group_replaces_count}
                semantic_rewrite_info["relation_rewritten"] = True
                semantic_rewrite_info["relation_mode"] = relation_result.get(
                    "mode",
                    SEARCH_SEMANTIC_POLICY.relation_mode,
                )
                semantic_rewrite_info["accepted_options"] = accepted_options
            elif relation_result.get("error"):
                semantic_rewrite_info["relation_error"] = relation_result.get("error")

        semantic_rewrite_info["applied"] = bool(
            semantic_rewrite_info["alias_rewritten"]
            or semantic_rewrite_info["relation_rewritten"]
        )
        semantic_rewrite_info["applied_query"] = base_query
        return base_query, semantic_suggest_info, semantic_rewrite_info

    @staticmethod
    def _normalize_search_focus_segment(text: str) -> str:
        normalized = " ".join(str(text or "").split()).strip()
        normalized = normalized.strip("，。！？!?、|/:：<>\"'`~·#")
        return " ".join(normalized.split())

    @classmethod
    def _strip_search_focus_noise(cls, text: str) -> str:
        normalized = cls._normalize_search_focus_segment(text)
        previous = None
        while normalized and normalized != previous:
            previous = normalized
            for pattern in SEARCH_FOCUS_POLICY.prefix_strip_patterns:
                normalized = pattern.sub("", normalized).strip()
            for pattern in SEARCH_FOCUS_POLICY.suffix_strip_patterns:
                normalized = pattern.sub("", normalized).strip()
            normalized = cls._normalize_search_focus_segment(normalized)
        return normalized

    @classmethod
    def _resolve_search_focus_query(cls, query: str) -> tuple[str, dict]:
        normalized_query = str(query or "").strip()
        focus_info = {
            "input_query": normalized_query,
            "applied": False,
        }
        if not normalized_query or not SEARCH_FOCUS_POLICY.enabled:
            focus_info["applied_query"] = normalized_query
            return normalized_query, focus_info
        if not SEARCH_FOCUS_POLICY.should_focus(normalized_query):
            focus_info["applied_query"] = normalized_query
            return normalized_query, focus_info

        candidate_texts: list[str] = []

        bracket_match = SEARCH_FOCUS_POLICY.bracket_prefix_pattern.match(
            normalized_query
        )
        if bracket_match:
            prefix = cls._strip_search_focus_noise(bracket_match.group("prefix"))
            body = cls._strip_search_focus_noise(bracket_match.group("body"))
            if prefix and body:
                candidate_texts.append(f"{prefix} {body}")
            if body:
                candidate_texts.append(body)

        stripped_query = cls._strip_search_focus_noise(normalized_query)
        if stripped_query:
            candidate_texts.append(stripped_query)

        indexed_segments: list[tuple[int, str]] = []
        for index, raw_segment in enumerate(
            SEARCH_FOCUS_POLICY.segment_split_pattern.split(
                stripped_query or normalized_query
            )
        ):
            segment = cls._strip_search_focus_noise(raw_segment)
            if len(compact_focus_key(segment)) < SEARCH_FOCUS_POLICY.min_segment_chars:
                continue
            indexed_segments.append((index, segment))

        if indexed_segments:
            ranked_segments = sorted(
                indexed_segments,
                key=lambda item: (-len(compact_focus_key(item[1])), item[0]),
            )
            selected_segments = sorted(
                ranked_segments[: SEARCH_FOCUS_POLICY.max_segments],
                key=lambda item: item[0],
            )
            candidate_texts.append(
                " ".join(segment for _, segment in selected_segments).strip()
            )

        candidates: list[str] = []
        seen_keys: set[str] = set()
        for candidate in candidate_texts:
            normalized_candidate = cls._normalize_search_focus_segment(candidate)
            candidate_key = compact_focus_key(normalized_candidate)
            if len(candidate_key) < SEARCH_FOCUS_POLICY.min_segment_chars:
                continue
            if normalized_candidate == normalized_query:
                continue
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)
            candidates.append(normalized_candidate)

        if not candidates:
            focus_info["applied_query"] = normalized_query
            return normalized_query, focus_info

        def _score(candidate: str) -> tuple[int, int, int, str]:
            key_len = len(compact_focus_key(candidate))
            segment_count = len(candidate.split())
            return (
                1 if segment_count >= 2 else 0,
                key_len,
                -abs(len(candidate) - len(normalized_query)),
                candidate,
            )

        focused_query = sorted(candidates, key=_score, reverse=True)[0]
        focus_info["applied"] = focused_query != normalized_query
        focus_info["candidates"] = candidates
        focus_info["applied_query"] = focused_query
        return focused_query, focus_info

    def prepare_search_query(
        self,
        query: str,
        *,
        suggest_info: dict | None = None,
        request_type: SEARCH_REQUEST_TYPE = SEARCH_REQUEST_TYPE_DEFAULT,
    ) -> SearchPreparation:
        effective_query = str(query or "")
        effective_suggest_info = dict(suggest_info or {})
        semantic_rewrite_info: dict = {}
        query_focus_info: dict = {}
        if request_type == SEARCH_REQUEST_TYPE_DEFAULT:
            effective_query, query_focus_info = self._resolve_search_focus_query(
                effective_query
            )
            allow_relation_rewrite = not bool(query_focus_info.get("applied"))
            (
                effective_query,
                semantic_suggest_info,
                semantic_rewrite_info,
            ) = self._resolve_search_semantic_rewrite(
                query=effective_query,
                suggest_info=suggest_info,
                request_type=request_type,
                allow_relation_rewrite=allow_relation_rewrite,
            )
            if semantic_suggest_info and not effective_suggest_info:
                effective_suggest_info = semantic_suggest_info
        return SearchPreparation(
            effective_query=effective_query,
            suggest_info=effective_suggest_info,
            semantic_rewrite_info=semantic_rewrite_info,
            query_focus_info=query_focus_info,
        )

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
        except Exception as e:
            logger.warn(f"> Failed to resolve vector auto-constraint owner anchor: {e}")
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


__all__ = [
    "OwnerIntentResolution",
    "SearchPreparation",
    "VideoQueryUnderstanding",
]
