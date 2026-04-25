from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from elastics.videos.constants import SEARCH_REQUEST_TYPE, SEARCH_REQUEST_TYPE_DEFAULT
from elastics.videos.policies.focus import get_search_focus_policy
from elastics.videos.policies.semantic import get_search_semantic_policy
from llms.intent.focus import compact_focus_key, rewrite_known_term_aliases


SEARCH_FOCUS_POLICY = get_search_focus_policy()
SEARCH_SEMANTIC_POLICY = get_search_semantic_policy()
MODEL_CODE_RE = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9][a-z0-9._+-]{1,31}$", re.I)


@dataclass(slots=True)
class SearchPreparation:
    effective_query: str
    suggest_info: dict = field(default_factory=dict)
    semantic_rewrite_info: dict = field(default_factory=dict)
    query_focus_info: dict = field(default_factory=dict)


@dataclass(slots=True)
class SearchQueryPreparer:
    query_rewriter: Any
    relations_client: Any

    @staticmethod
    def _keyword_has_mixed_script(text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        has_cjk = any("\u4e00" <= char <= "\u9fff" for char in normalized)
        has_ascii_alnum = any(char.isascii() and char.isalnum() for char in normalized)
        return has_cjk and has_ascii_alnum

    @staticmethod
    def _keyword_is_model_code(text: str) -> bool:
        return bool(MODEL_CODE_RE.match(str(text or "").strip()))

    @staticmethod
    def _keyword_has_cjk(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))

    @classmethod
    def _keywords_have_model_code_attribute(cls, keywords: list[str]) -> bool:
        if len(keywords) < 2:
            return False
        return any(cls._keyword_is_model_code(keyword) for keyword in keywords) and any(
            cls._keyword_has_cjk(keyword) for keyword in keywords
        )

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
        relation_query = " ".join(str(keyword) for keyword in keywords).strip()
        should_attempt_relation_rewrite = (
            SEARCH_SEMANTIC_POLICY.relation_rewrite_enabled
            and allow_relation_rewrite
            and self.relations_client is not None
            and SEARCH_SEMANTIC_POLICY.query_length_ok(relation_query)
            and SEARCH_SEMANTIC_POLICY.keyword_count_ok(keywords)
            and not SEARCH_SEMANTIC_POLICY.has_blocked_marker(relation_query)
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
                or (
                    SEARCH_SEMANTIC_POLICY.trigger_model_code_attribute_keywords
                    and self._keywords_have_model_code_attribute(keywords)
                )
            )
        )
        if should_attempt_relation_rewrite:
            relation_result = self.relations_client.related_tokens_by_tokens(
                text=relation_query,
                mode=SEARCH_SEMANTIC_POLICY.relation_mode,
                size=SEARCH_SEMANTIC_POLICY.relation_size,
                scan_limit=SEARCH_SEMANTIC_POLICY.relation_scan_limit,
                use_pinyin=True,
            )
            if self._semantic_fallback_requires_auto(relation_result):
                relation_result = self.relations_client.related_tokens_by_tokens(
                    text=relation_query,
                    mode=SEARCH_SEMANTIC_POLICY.relation_fallback_mode,
                    size=SEARCH_SEMANTIC_POLICY.relation_size,
                    scan_limit=SEARCH_SEMANTIC_POLICY.relation_scan_limit,
                    use_pinyin=True,
                )

            options = list(relation_result.get("options") or [])
            group_replaces_count, accepted_options = (
                self._build_semantic_group_replaces_count(relation_query, options)
            )
            if group_replaces_count:
                semantic_suggest_info = {"group_replaces_count": group_replaces_count}
                semantic_rewrite_info["relation_rewritten"] = True
                semantic_rewrite_info["relation_query"] = relation_query
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


__all__ = ["SearchPreparation", "SearchQueryPreparer"]
