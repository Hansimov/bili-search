from __future__ import annotations

import math
import re


class OwnerResolutionMixin:
    @staticmethod
    def _normalize_name_key(text: str) -> str:
        return re.sub(r"\s+", "", (text or "").strip()).lower()

    @staticmethod
    def _extract_owner_text_parts(text: str) -> list[str]:
        return re.findall(r"[A-Za-z]+|\d+|[\u4e00-\u9fff]+", (text or ""))

    @staticmethod
    def _extract_cjk_tokens(text: str) -> list[str]:
        tokens = re.findall(r"[\u4e00-\u9fff]{2,4}", (text or ""))
        deduped: list[str] = []
        for token in tokens:
            if token not in deduped:
                deduped.append(token)
        return deduped

    @classmethod
    def _owner_name_matches_source(cls, source_text: str, owner_name: str) -> bool:
        parts = cls._extract_owner_text_parts(source_text)
        if not parts:
            return False
        normalized_name = cls._normalize_name_key(owner_name)
        cursor = 0
        for part in parts:
            normalized_part = cls._normalize_name_key(part)
            if not normalized_part:
                continue
            index = normalized_name.find(normalized_part, cursor)
            if index < 0:
                return False
            cursor = index + len(normalized_part)
        return True

    @classmethod
    def _extract_owner_context_tokens(
        cls, source_text: str, owner_name: str
    ) -> list[str]:
        parts = cls._extract_owner_text_parts(source_text)
        if not parts:
            return []

        search_start = 0
        first_index = None
        last_end = None
        for part in parts:
            index = owner_name.find(part, search_start)
            if index < 0:
                first_index = None
                last_end = None
                break
            if first_index is None:
                first_index = index
            last_end = index + len(part)
            search_start = last_end

        if first_index is not None and last_end is not None:
            prefix = owner_name[:first_index]
            suffix = owner_name[last_end:]
            tokens = cls._extract_cjk_tokens(prefix) + cls._extract_cjk_tokens(suffix)
            deduped: list[str] = []
            for token in tokens:
                if token not in deduped:
                    deduped.append(token)
            return deduped

        source_tokens = set(cls._extract_cjk_tokens(source_text))
        return [
            token
            for token in cls._extract_cjk_tokens(owner_name)
            if token not in source_tokens
        ]

    @classmethod
    def _is_short_ambiguous_owner_text(cls, text: str) -> bool:
        normalized = cls._normalize_name_key(text)
        if not normalized:
            return False
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", normalized))
        has_alnum = bool(re.search(r"[A-Za-z0-9]", normalized))
        return has_alnum and (not has_cjk or len(normalized) <= 2)

    @classmethod
    def _score_owner_candidate(
        cls,
        source_text: str,
        owner: dict,
        hint_tokens: list[str] | None = None,
    ) -> float:
        owner_name = str(owner.get("name") or "").strip()
        if not owner_name:
            return float("-inf")

        hint_tokens = hint_tokens or []
        score = float(owner.get("score") or 0.0)
        if cls._owner_name_matches_source(source_text, owner_name):
            score += 100.0

        context_tokens = cls._extract_owner_context_tokens(source_text, owner_name)
        score += 20.0 * len(context_tokens)

        sample_title = str(owner.get("sample_title") or "")
        for token in hint_tokens:
            if token and token in owner_name:
                score += 120.0
            elif token and token in sample_title:
                score += 40.0

        sample_view = owner.get("sample_view") or 0
        try:
            sample_view = int(sample_view)
        except (TypeError, ValueError):
            sample_view = 0
        if sample_view > 0:
            score += min(math.log10(sample_view + 1) * 8.0, 40.0)

        if cls._is_short_ambiguous_owner_text(source_text):
            has_cjk_phrase = bool(re.search(r"[\u4e00-\u9fff]{2,}", owner_name))
            if not has_cjk_phrase and not any(
                token in owner_name for token in hint_tokens
            ):
                score -= 120.0

        return score

    @classmethod
    def _select_best_owner_candidate(
        cls,
        source_text: str,
        owners: list[dict] | None,
        hint_tokens: list[str] | None = None,
    ) -> dict | None:
        candidates = owners or []
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda owner: cls._score_owner_candidate(
                source_text,
                owner,
                hint_tokens=hint_tokens,
            ),
        )

    @classmethod
    def _has_close_contextual_owner_competitor(
        cls,
        source_text: str,
        owner: dict | None,
        owners: list[dict] | None,
        hint_tokens: list[str] | None = None,
        score_gap: float = 12.0,
    ) -> bool:
        if not owner or not owners:
            return False

        hint_tokens = hint_tokens or []
        owner_name = str(owner.get("name") or "").strip()
        if not owner_name:
            return False

        owner_mid = owner.get("mid")
        selected_score = cls._score_owner_candidate(
            source_text,
            owner,
            hint_tokens=hint_tokens,
        )
        selected_has_context = bool(
            cls._extract_owner_context_tokens(source_text, owner_name)
            or any(token and token in owner_name for token in hint_tokens)
        )

        for other in owners or []:
            other_name = str(other.get("name") or "").strip()
            if not other_name:
                continue
            if other_name == owner_name and other.get("mid") == owner_mid:
                continue
            if not cls._owner_name_matches_source(source_text, other_name):
                continue

            other_score = cls._score_owner_candidate(
                source_text,
                other,
                hint_tokens=hint_tokens,
            )
            if selected_score - other_score > score_gap:
                continue

            other_has_context = bool(
                cls._extract_owner_context_tokens(source_text, other_name)
                or any(token and token in other_name for token in hint_tokens)
            )
            if other_has_context and not selected_has_context:
                return True

        return False

    @classmethod
    def _is_confident_owner_candidate(
        cls,
        source_text: str,
        owner: dict | None,
        hint_tokens: list[str] | None = None,
        owners: list[dict] | None = None,
    ) -> bool:
        if not owner:
            return False

        owner_name = str(owner.get("name") or "").strip()
        if not owner_name or not cls._owner_name_matches_source(
            source_text, owner_name
        ):
            return False

        hint_tokens = hint_tokens or []
        context_tokens = cls._extract_owner_context_tokens(source_text, owner_name)
        short_ambiguous = cls._is_short_ambiguous_owner_text(source_text)
        source_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", source_text or ""))

        if any(token and token in owner_name for token in hint_tokens):
            return True
        if short_ambiguous and not source_has_cjk:
            return False
        if cls._has_close_contextual_owner_competitor(
            source_text,
            owner,
            owners,
            hint_tokens=hint_tokens,
        ):
            return False
        if context_tokens:
            return True
        if not short_ambiguous:
            return True

        sample_view = owner.get("sample_view") or 0
        try:
            sample_view = int(sample_view)
        except (TypeError, ValueError):
            sample_view = 0
        return sample_view >= 50000

    @classmethod
    def _extract_resolved_owner_map(cls, results: list[dict] | None) -> dict[str, dict]:
        resolved: dict[str, dict] = {}
        for result_item in results or []:
            if result_item.get("type") != "search_owners":
                continue
            result = result_item.get("result") or {}
            source_text = str(
                result.get("text") or result_item.get("args", {}).get("text") or ""
            ).strip()
            owners = result.get("owners") or []
            if not source_text or not owners:
                continue
            top_owner = cls._select_best_owner_candidate(source_text, owners)
            if not cls._is_confident_owner_candidate(
                source_text,
                top_owner,
                owners=owners,
            ):
                continue
            key = cls._normalize_name_key(source_text)
            if not key:
                continue
            resolved[key] = {
                "source_text": source_text,
                "name": str(top_owner.get("name") or "").strip(),
                "mid": top_owner.get("mid"),
            }
        return resolved

    @classmethod
    def _filter_superseded_owner_results(
        cls,
        results: list[dict] | None,
    ) -> list[dict]:
        owner_results = [
            result_item
            for result_item in results or []
            if result_item.get("type") == "search_owners"
        ]
        effective: list[dict] = []
        source_texts = [
            str(
                (result_item.get("result") or {}).get("text")
                or (result_item.get("args") or {}).get("text")
                or ""
            ).strip()
            for result_item in owner_results
        ]
        for result_item, source_text in zip(owner_results, source_texts):
            source_key = cls._normalize_name_key(source_text)
            if not source_key:
                continue
            superseded = any(
                other_text
                and cls._normalize_name_key(other_text) != source_key
                and len(cls._normalize_name_key(other_text)) > len(source_key)
                and cls._owner_name_matches_source(source_text, other_text)
                for other_text in source_texts
            )
            if not superseded:
                effective.append(result_item)
        return effective

    @classmethod
    def _collect_owner_context_hints(cls, results: list[dict] | None) -> list[str]:
        hints: list[str] = []
        fallback_hints: list[str] = []
        for result_item in results or []:
            if result_item.get("type") != "search_owners":
                continue
            result = result_item.get("result") or {}
            source_text = str(
                result.get("text") or result_item.get("args", {}).get("text") or ""
            ).strip()
            owners = result.get("owners") or []
            candidate = cls._select_best_owner_candidate(source_text, owners)
            if cls._is_confident_owner_candidate(
                source_text,
                candidate,
                owners=owners,
            ):
                if cls._is_short_ambiguous_owner_text(source_text):
                    continue
                direct_tokens = cls._extract_owner_context_tokens(
                    source_text,
                    str((candidate or {}).get("name") or ""),
                )
                for token in direct_tokens:
                    if token not in hints:
                        hints.append(token)
                if direct_tokens:
                    continue

            if not re.search(r"[\u4e00-\u9fff]", source_text or ""):
                continue
            for token in cls._collect_near_top_owner_context_hints(source_text, owners):
                if token not in fallback_hints:
                    fallback_hints.append(token)
        return hints or fallback_hints

    @classmethod
    def _collect_near_top_owner_context_hints(
        cls,
        source_text: str,
        owners: list[dict] | None,
        score_gap: float = 12.0,
    ) -> list[str]:
        candidate_tokens: dict[str, float] = {}
        scored_candidates: list[tuple[float, list[str]]] = []
        for owner in owners or []:
            owner_name = str(owner.get("name") or "").strip()
            if not owner_name or not cls._owner_name_matches_source(
                source_text, owner_name
            ):
                continue
            context_tokens = cls._extract_owner_context_tokens(source_text, owner_name)
            if not context_tokens:
                continue
            scored_candidates.append(
                (cls._score_owner_candidate(source_text, owner), context_tokens)
            )

        if not scored_candidates:
            return []

        top_score = max(score for score, _ in scored_candidates)
        for score, context_tokens in scored_candidates:
            if top_score - score > score_gap:
                continue
            for token in context_tokens:
                previous = candidate_tokens.get(token)
                if previous is None or score > previous:
                    candidate_tokens[token] = score

        return sorted(
            candidate_tokens, key=lambda token: (-candidate_tokens[token], token)
        )
