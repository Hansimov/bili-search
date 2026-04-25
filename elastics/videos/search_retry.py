import re

from elastics.videos.constants import SEARCH_BOOSTED_FIELDS, SEARCH_LIMIT
from elastics.videos.intent.owner_query import OwnerQueryIntentResolver
from elastics.videos.policies.embedding_denoise import (
    get_search_embedding_denoise_policy,
)
from elastics.videos.policies.recall import get_search_recall_policy
from ranks.reranker import get_reranker


SHORT_HAN_QUERY_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,3}")
RELAXED_SHORT_HAN_RETRY_BOOST_OVERRIDES = {
    "title": 5.0,
    "title.words": 5.0,
    "owner.name": 4.0,
    "owner.name.words": 4.0,
    "tags": 1.2,
    "tags.words": 1.2,
    "desc": 0.02,
    "desc.words": 0.02,
}
SEARCH_RECALL_POLICY = get_search_recall_policy()
SEARCH_EMBEDDING_DENOISE_POLICY = get_search_embedding_denoise_policy()


class VideoSearchRetryMixin:
    @staticmethod
    def _get_relaxed_short_han_retry_boosted_fields(
        boosted_fields: dict | None,
    ) -> dict:
        retry_boosted_fields = dict(boosted_fields or SEARCH_BOOSTED_FIELDS)
        retry_boosted_fields.update(RELAXED_SHORT_HAN_RETRY_BOOST_OVERRIDES)
        return retry_boosted_fields

    @classmethod
    def _apply_model_code_score_cliff_filter(
        cls,
        result: dict,
        *,
        query: str,
        relation_rewritten: bool,
    ) -> dict:
        policy = SEARCH_EMBEDDING_DENOISE_POLICY
        if (
            relation_rewritten
            or not policy.enabled
            or not policy.score_cliff_filter_enabled
            or not cls._has_relaxable_model_code_attribute_terms(query)
        ):
            return {}
        hits = list(result.get("hits") or [])
        if len(hits) < 2:
            return {}
        top_score = float(hits[0].get("score") or 0.0)
        if top_score < policy.score_cliff_min_top_score:
            return {}
        threshold = top_score * policy.score_cliff_ratio
        kept = [hit for hit in hits if float(hit.get("score") or 0.0) >= threshold]
        min_keep = max(1, int(policy.score_cliff_min_keep or 1))
        if len(kept) < min_keep and len(hits) >= min_keep:
            kept = hits[:min_keep]
        if not kept or len(kept) >= len(hits):
            return {}
        result["hits"] = kept
        result["return_hits"] = len(kept)
        return {
            "applied": True,
            "query": query,
            "top_score": round(top_score, 6),
            "threshold": round(threshold, 6),
            "before": len(hits),
            "after": len(kept),
            "ratio": policy.score_cliff_ratio,
            "min_keep": min_keep,
        }

    @classmethod
    def _get_embedding_denoise_candidate_limit(cls, limit: int | None) -> int:
        policy = SEARCH_EMBEDDING_DENOISE_POLICY
        if not policy.enabled:
            return limit
        user_limit = int(limit or SEARCH_LIMIT)
        if user_limit <= 0:
            return user_limit
        return max(user_limit, policy.candidate_limit)

    @staticmethod
    def _get_embedding_denoise_reranker():
        return get_reranker()

    def _apply_embedding_denoise_to_retry_result(
        self,
        result: dict,
        denoise_query: str,
        original_limit: int | None,
        verbose: bool = False,
    ) -> dict:
        policy = SEARCH_EMBEDDING_DENOISE_POLICY
        if not policy.enabled or not isinstance(result, dict):
            return result

        hits = list(result.get("hits") or [])
        user_limit = int(original_limit or SEARCH_LIMIT)
        info = {
            "applied": False,
            "query": denoise_query,
            "candidate_count": len(hits),
            "returned_limit": user_limit,
        }

        def _finish_with_trim() -> dict:
            if user_limit > 0 and len(result.get("hits") or []) > user_limit:
                result["hits"] = list(result.get("hits") or [])[:user_limit]
                result["return_hits"] = len(result["hits"])
            return result

        if len(hits) < policy.min_hits:
            info["reason"] = "too_few_hits"
            result["embedding_denoise_info"] = info
            return _finish_with_trim()

        reranker = self._get_embedding_denoise_reranker()
        if not reranker.is_available():
            info["reason"] = "reranker_unavailable"
            result["embedding_denoise_info"] = info
            return _finish_with_trim()

        query_info = result.get("query_info") or {}
        keywords = list(query_info.get("keywords_body") or [])
        if not keywords:
            keywords = [
                token.strip()
                for token in str(denoise_query or "").split()
                if len(token.strip()) >= 2
            ]

        reranked_hits, perf = reranker.rerank(
            query=denoise_query,
            hits=hits,
            keywords=keywords,
            max_rerank=policy.max_rerank_hits,
            keyword_boost=policy.keyword_boost,
            title_keyword_boost=policy.title_keyword_boost,
            score_field=policy.score_field,
            verbose=verbose,
        )
        attribute_evidence_info = self._apply_attribute_evidence_gate(
            hits=reranked_hits,
            query_tokens=keywords,
            reranker=reranker,
            score_field=policy.score_field,
        )
        result["hits"] = reranked_hits
        info.update(
            {
                "applied": True,
                "reranked_count": perf.get("reranked_count", 0),
                "valid_passages": perf.get("valid_passages", 0),
                "perf": perf,
                "score_field": policy.score_field,
            }
        )
        if attribute_evidence_info:
            info["attribute_evidence"] = attribute_evidence_info
        result["embedding_denoise_info"] = info
        retry_info = dict(result.get("retry_info") or {})
        retry_info["embedding_denoise_applied"] = True
        result["retry_info"] = retry_info
        return _finish_with_trim()

    @classmethod
    def _split_model_and_attribute_tokens(
        cls, tokens: list[str] | tuple[str, ...]
    ) -> tuple[list[str], list[str]]:
        model_tokens: list[str] = []
        attribute_tokens: list[str] = []
        for token in tokens or []:
            text = str(token or "").strip()
            if not text:
                continue
            if cls._looks_like_model_code_query(text):
                model_tokens.append(text)
            else:
                attribute_tokens.append(text)
        return model_tokens, attribute_tokens

    @staticmethod
    def _hit_evidence_text(hit: dict) -> str:
        owner = hit.get("owner")
        owner_name = owner.get("name", "") if isinstance(owner, dict) else ""
        parts = [
            str(hit.get("title") or ""),
            str(hit.get("tags") or ""),
            str(hit.get("desc") or ""),
            str(owner_name or ""),
        ]
        return " ".join(part for part in parts if part)

    @classmethod
    def _extract_attribute_candidate_terms(
        cls,
        hits: list[dict],
        model_tokens: list[str],
    ) -> list[str]:
        policy = SEARCH_EMBEDDING_DENOISE_POLICY
        model_lowers = [token.lower() for token in model_tokens]
        seen: set[str] = set()
        terms: list[str] = []
        splitter = re.compile(r"[\s,，、;；|/\\:：#]+")
        token_re = re.compile(r"[A-Za-z][A-Za-z0-9.+_-]{1,31}|[\u4e00-\u9fff]{2,12}")

        def add_term(raw_term: str):
            term = str(raw_term or "").strip().strip("，。！？!?、|/:：<>\"'`~·#")
            if len(term) < 2 or len(term) > 32:
                return
            term_lower = term.lower()
            if any(
                model_lower and model_lower in term_lower
                for model_lower in model_lowers
            ):
                return
            if term_lower in seen:
                return
            seen.add(term_lower)
            terms.append(term)

        for hit in hits:
            tags = str(hit.get("tags") or "")
            for part in splitter.split(tags):
                add_term(part)
                if len(terms) >= policy.max_attribute_candidate_terms:
                    return terms

            title = str(hit.get("title") or "")
            for match in token_re.finditer(title):
                add_term(match.group(0))
                if len(terms) >= policy.max_attribute_candidate_terms:
                    return terms

        return terms

    def _build_attribute_evidence_terms(
        self,
        *,
        attribute_tokens: list[str],
        candidate_terms: list[str],
        reranker,
    ) -> tuple[dict[str, dict[str, float]], dict]:
        policy = SEARCH_EMBEDDING_DENOISE_POLICY
        evidence_by_attr: dict[str, dict[str, float]] = {}
        info = {
            "candidate_term_count": len(candidate_terms),
            "attributes": {},
        }
        embed_client = getattr(reranker, "embed_client", None)
        if embed_client is None:
            info["reason"] = "embed_client_unavailable"
            return evidence_by_attr, info

        for attr in attribute_tokens:
            attr_text = str(attr or "").strip()
            if not attr_text:
                continue
            evidence: dict[str, float] = {attr_text: 1.0}
            try:
                rankings = embed_client.rerank(attr_text, candidate_terms)
            except Exception as exc:
                info["reason"] = f"attribute_term_rerank_failed: {exc}"
                continue

            scored_terms: list[tuple[float, str]] = []
            for term, ranking in zip(candidate_terms, rankings or []):
                if not ranking:
                    continue
                _, score = ranking
                score = float(score or 0.0)
                min_score = policy.attribute_term_min_score
                if re.search(r"[\u4e00-\u9fff]", str(term or "")):
                    min_score = max(min_score, policy.attribute_cjk_term_min_score)
                if score >= min_score:
                    scored_terms.append((score, term))
            scored_terms.sort(reverse=True)
            for score, term in scored_terms[: policy.max_attribute_evidence_terms]:
                evidence[term] = score
            evidence_by_attr[attr_text] = evidence
            info["attributes"][attr_text] = [
                {"term": term, "score": round(score, 6)}
                for term, score in sorted(
                    evidence.items(), key=lambda item: item[1], reverse=True
                )
            ]

        return evidence_by_attr, info

    def _apply_attribute_evidence_gate(
        self,
        *,
        hits: list[dict],
        query_tokens: list[str],
        reranker,
        score_field: str,
    ) -> dict:
        policy = SEARCH_EMBEDDING_DENOISE_POLICY
        if not policy.attribute_evidence_gate_enabled or not hits:
            return {}

        model_tokens, attribute_tokens = self._split_model_and_attribute_tokens(
            query_tokens
        )
        if not model_tokens or not attribute_tokens:
            return {}

        candidate_terms = self._extract_attribute_candidate_terms(hits, model_tokens)
        evidence_by_attr, info = self._build_attribute_evidence_terms(
            attribute_tokens=attribute_tokens,
            candidate_terms=candidate_terms,
            reranker=reranker,
        )
        if not evidence_by_attr:
            return info

        boosted = 0
        penalized = 0
        for hit in hits:
            text = self._hit_evidence_text(hit).lower()
            matched_attr_count = 0
            matched_terms: list[str] = []
            max_evidence_score = 0.0

            for attr, evidence_terms in evidence_by_attr.items():
                matched_for_attr = False
                for term, term_score in evidence_terms.items():
                    if str(term or "").lower() in text:
                        matched_for_attr = True
                        matched_terms.append(term)
                        max_evidence_score = max(max_evidence_score, term_score)
                if matched_for_attr:
                    matched_attr_count += 1

            coverage = matched_attr_count / max(len(attribute_tokens), 1)
            old_score = float(hit.get(score_field) or hit.get("score") or 0.0)
            if coverage > 0:
                factor = 1.0 + policy.attribute_evidence_boost * coverage
                boosted += 1
            else:
                factor = policy.missing_attribute_penalty
                penalized += 1
            new_score = old_score * factor
            hit[score_field] = round(new_score, 6)
            hit["score"] = round(new_score, 6)
            hit["rank_score"] = round(new_score, 6)
            hit["attribute_evidence_factor"] = round(factor, 4)
            if matched_terms:
                hit["attribute_evidence_terms"] = sorted(set(matched_terms))[:8]

        hits.sort(key=lambda item: item.get(score_field, 0), reverse=True)
        info.update(
            {
                "model_tokens": model_tokens,
                "attribute_tokens": attribute_tokens,
                "boosted": boosted,
                "penalized": penalized,
                "missing_attribute_penalty": policy.missing_attribute_penalty,
                "attribute_evidence_boost": policy.attribute_evidence_boost,
            }
        )
        return info

    @staticmethod
    def _has_relaxable_short_han_terms(query: str) -> bool:
        normalized_query = str(query or "").strip()
        if not normalized_query or not any(char.isspace() for char in normalized_query):
            return False
        if any(
            marker in normalized_query
            for marker in [":", "=", '"', "'", "[", "]", "(", ")", "|", "&"]
        ):
            return False
        tokens = [token.strip() for token in normalized_query.split() if token.strip()]
        if len(tokens) < 2:
            return False
        return any(SHORT_HAN_QUERY_TOKEN_RE.fullmatch(token) for token in tokens)

    @classmethod
    def _has_relaxable_model_code_attribute_terms(cls, query: str) -> bool:
        if not SEARCH_RECALL_POLICY.model_code_attribute_retry:
            return False
        normalized_query = str(query or "").strip()
        if not normalized_query or not any(char.isspace() for char in normalized_query):
            return False
        if any(
            marker in normalized_query
            for marker in [":", "=", '"', "'", "[", "]", "(", ")", "|", "&"]
        ):
            return False
        tokens = [token.strip() for token in normalized_query.split() if token.strip()]
        if not (
            SEARCH_RECALL_POLICY.min_keyword_count
            <= len(tokens)
            <= SEARCH_RECALL_POLICY.max_keyword_count
        ):
            return False
        return any(cls._looks_like_model_code_query(token) for token in tokens)

    @classmethod
    def _should_retry_without_auto_exact_segments(
        cls,
        query: str,
        total_hits: int | None,
        allow_retry: bool = True,
    ) -> bool:
        if not allow_retry or not SEARCH_RECALL_POLICY.exact_relax_retry_enabled:
            return False
        if total_hits is None:
            return False
        if total_hits >= SEARCH_RECALL_POLICY.low_recall_total_hits:
            return False
        return cls._has_relaxable_model_code_attribute_terms(query)

    @classmethod
    def _should_retry_without_short_han_exact(
        cls,
        query: str,
        total_hits: int | None,
        allow_retry: bool = True,
    ) -> bool:
        if not allow_retry:
            return False
        if total_hits not in (None, 0):
            return False
        return cls._has_relaxable_short_han_terms(query)

    @classmethod
    def _is_compact_owner_prefix_candidate(
        cls,
        token: str,
        owner: dict,
        top_score: float,
    ) -> bool:
        return OwnerQueryIntentResolver._is_compact_owner_prefix_candidate(
            token,
            owner,
            top_score,
        )

    def _resolve_vector_auto_constraint_query(self, query: str) -> str:
        return self.owner_query_intent_resolver._resolve_vector_auto_constraint_query(
            query
        )

    def _resolve_spaced_owner_intent_info(self, query: str) -> dict:
        return self.owner_query_intent_resolver._resolve_spaced_owner_intent_info(query)

    @staticmethod
    def _build_spaced_owner_context_query(owner_intent_info: dict | None) -> str:
        return OwnerQueryIntentResolver._build_spaced_owner_context_query(
            owner_intent_info
        )

    @staticmethod
    def _resolve_owner_intent_search_filters(
        owner_intent_info: dict | None,
    ) -> list[dict]:
        return OwnerQueryIntentResolver._resolve_owner_intent_search_filters(
            owner_intent_info
        )

    @staticmethod
    def _should_suppress_title_like_owner_filter(
        query: str,
        candidate: dict | None,
    ) -> bool:
        return OwnerQueryIntentResolver._should_suppress_title_like_owner_filter(
            query,
            candidate,
        )

    @staticmethod
    def _should_suppress_short_query_owner_filter(
        query: str,
        candidate: dict | None,
    ) -> bool:
        return OwnerQueryIntentResolver._should_suppress_short_query_owner_filter(
            query,
            candidate,
        )

    def _resolve_owner_intent_info(self, query: str) -> dict:
        return self.owner_query_intent_resolver._resolve_owner_intent_info(query)

    @staticmethod
    def _looks_like_model_code_query(text: str) -> bool:
        return OwnerQueryIntentResolver._looks_like_model_code_query(text)

    @staticmethod
    def _build_owner_filter(owner: dict) -> list[dict]:
        return OwnerQueryIntentResolver._build_owner_filter(owner)

    @staticmethod
    def _normalize_owner_intent_candidate(candidate: dict) -> dict | None:
        return OwnerQueryIntentResolver._normalize_owner_intent_candidate(candidate)

    @classmethod
    def _rerank_spaced_owner_intent_candidates(
        cls,
        candidates: list[dict],
    ) -> list[dict]:
        return OwnerQueryIntentResolver._rerank_spaced_owner_intent_candidates(
            candidates
        )

    @staticmethod
    def _candidate_supports_multi_owner_intent(candidate: dict) -> bool:
        return OwnerQueryIntentResolver._candidate_supports_multi_owner_intent(
            candidate
        )

    @classmethod
    def _select_owner_intent_candidates(cls, owners: list[dict]) -> list[dict]:
        return OwnerQueryIntentResolver._select_owner_intent_candidates(owners)

    @classmethod
    def _select_confident_owner_intent_candidate(
        cls,
        owners: list[dict],
        candidates: list[dict] | None = None,
    ) -> dict | None:
        return OwnerQueryIntentResolver._select_confident_owner_intent_candidate(
            owners,
            candidates=candidates,
        )
