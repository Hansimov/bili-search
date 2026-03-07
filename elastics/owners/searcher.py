"""OwnerSearcher — core search service for the owners ES index.

Provides name search, domain search, relation search, suggest,
and top-owners queries against the independent owners index.
Independent of VideoSearcher — can be used standalone or as an
enhancement to the existing videos search pipeline.
"""

import re
from collections import Counter
from math import log1p

from sedb import ElasticOperator
from tclogger import logger

from configs.envs import SECRETS, ELASTIC_PRO_ENVS
from elastics.owners.constants import (
    ELASTIC_OWNERS_INDEX,
    SOURCE_FIELDS,
    SOURCE_FIELDS_COMPACT,
    NAME_MATCH_BOOSTS,
    DOMAIN_MATCH_BOOSTS,
    DOMAIN_STRICT_MATCH_BOOSTS,
    DOMAIN_PHRASE_MATCH_BOOSTS,
    DOMAIN_PHRASE_QUERY_MIN_CHARS,
    LOG_MAX_VIEW,
    LOG_MAX_VIDEOS,
    NAME_MATCH_NORM_DENOM,
    SORT_FIELD_TYPE,
    SORT_FIELD_DEFAULT,
    SORT_FIELD_MAP,
    SEARCH_LIMIT,
    SUGGEST_LIMIT,
    TOP_OWNERS_LIMIT,
    SEARCH_TIMEOUT,
    SUGGEST_TIMEOUT,
)
from elastics.owners.hits import OwnerHitsParser
from elastics.owners.scorer import (
    compute_owner_rank_score,
    detect_owner_query_type,
)


class OwnerSearcher:
    """Owner search service — queries the independent owners ES index.

    This is the primary interface for all owner search operations.
    It operates on the owners index (not videos), providing:

    - Name search: exact keyword + BM25 token match
    - Domain search: top_tags text match
    - Relation search: find owners by mentioned user names/mids
    - Suggest: prefix-based owner name completion
    - Top owners: rank owners by a specified metric

    Usage:
        searcher = OwnerSearcher(index_name="bili_owners_v1")
        result = searcher.search("影视飓风")
        result = searcher.search_by_domain("黑神话悟空", sort_by="influence")
        owner = searcher.get_owner(946974)
    """

    def __init__(
        self,
        index_name: str = ELASTIC_OWNERS_INDEX,
        elastic_env_name: str = None,
    ):
        self.index_name = index_name
        if elastic_env_name:
            elastic_envs = SECRETS[elastic_env_name]
        else:
            elastic_envs = ELASTIC_PRO_ENVS
        self.es = ElasticOperator(elastic_envs, connect_cls=self.__class__)
        self.hit_parser = OwnerHitsParser()
        self._latin_token_re = re.compile(r"[a-z0-9][a-z0-9_\-\.]{1,}")
        self._cjk_span_re = re.compile(r"[\u4e00-\u9fff]+")

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _submit(self, body: dict, context: str = "search") -> dict:
        """Submit a search request to ES and return the response body.

        Returns a safe error dict on failure so callers don't need try/except.
        """
        try:
            res = self.es.client.search(index=self.index_name, body=body)
            return res.body
        except Exception as e:
            logger.warn(f"× OwnerSearcher ES error [{context}]: {str(e)[:300]}")
            return {"hits": {"hits": [], "total": {"value": 0}}, "error": str(e)}

    def _build_name_query(
        self,
        query: str,
        boosts: dict = None,
    ) -> dict:
        """Build a multi-field bool/should name query.

        Uses keyword exact match (highest boost) + token BM25 match +
        optional tag/mention fields for auxiliary matching.
        """
        b = boosts or NAME_MATCH_BOOSTS
        return {
            "bool": {
                "should": [
                    {
                        "term": {
                            "name.keyword": {
                                "value": query,
                                "boost": b.get("name.keyword", 50.0),
                            }
                        }
                    },
                    {
                        "match": {
                            "name.words": {
                                "query": query,
                                "boost": b.get("name.words", 10.0),
                            }
                        }
                    },
                    {
                        "match": {
                            "top_tags.words": {
                                "query": query,
                                "boost": b.get("top_tags.words", 3.0),
                            }
                        }
                    },
                    {
                        "match": {
                            "topic_phrases.words": {
                                "query": query,
                                "boost": b.get("topic_phrases.words", 2.5),
                            }
                        }
                    },
                    {
                        "match": {
                            "domain_text.words": {
                                "query": query,
                                "boost": b.get("domain_text.words", 1.5),
                            }
                        }
                    },
                    {
                        "match": {
                            "mentioned_names.words": {
                                "query": query,
                                "boost": b.get("mentioned_names.words", 1.5),
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }

    def _build_name_strict_query(self, query: str) -> dict:
        return {
            "bool": {
                "should": [
                    {"term": {"name.keyword": {"value": query, "boost": 100.0}}},
                    {
                        "match": {
                            "name.words": {
                                "query": query,
                                "operator": "and",
                                "boost": 20.0,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }

    def _build_domain_query(self, query: str, boosts: dict = None) -> dict:
        """Build a domain/topic search query.

        Primarily matches top_tags.words with name.words as secondary signal.
        """
        b = boosts or DOMAIN_MATCH_BOOSTS
        if self._is_phrase_like_domain_query(query):
            strict_boosts = DOMAIN_STRICT_MATCH_BOOSTS
            phrase_boosts = DOMAIN_PHRASE_MATCH_BOOSTS
            strict_clauses = [
                {
                    "match_phrase": {
                        "topic_phrases.words": {
                            "query": query,
                            "boost": phrase_boosts.get("topic_phrases.words", 14.0),
                        }
                    }
                },
                {
                    "match_phrase": {
                        "domain_text.words": {
                            "query": query,
                            "boost": phrase_boosts.get("domain_text.words", 12.0),
                        }
                    }
                },
                {
                    "match": {
                        "topic_phrases.words": {
                            "query": query,
                            "operator": "and",
                            "boost": strict_boosts.get("topic_phrases.words", 7.0),
                        }
                    }
                },
                {
                    "match": {
                        "domain_text.words": {
                            "query": query,
                            "operator": "and",
                            "boost": strict_boosts.get("domain_text.words", 6.0),
                        }
                    }
                },
                {
                    "match": {
                        "semantic_terms.words": {
                            "query": query,
                            "operator": "and",
                            "boost": strict_boosts.get("semantic_terms.words", 5.0),
                        }
                    }
                },
                {
                    "match": {
                        "top_tags.words": {
                            "query": query,
                            "operator": "and",
                            "boost": strict_boosts.get("top_tags.words", 4.5),
                        }
                    }
                },
            ]
            return {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": strict_clauses,
                                "minimum_should_match": 1,
                            }
                        }
                    ],
                    "should": [
                        {
                            "match": {
                                "top_tags.words": {
                                    "query": query,
                                    "boost": b.get("top_tags.words", 4.5),
                                }
                            }
                        },
                        {
                            "match": {
                                "topic_phrases.words": {
                                    "query": query,
                                    "boost": b.get("topic_phrases.words", 4.0),
                                }
                            }
                        },
                        {
                            "match": {
                                "domain_text.words": {
                                    "query": query,
                                    "boost": b.get("domain_text.words", 3.0),
                                }
                            }
                        },
                        {
                            "match": {
                                "semantic_terms.words": {
                                    "query": query,
                                    "boost": b.get("semantic_terms.words", 2.5),
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 0,
                }
            }
        return {
            "bool": {
                "should": [
                    {
                        "match": {
                            "top_tags.words": {
                                "query": query,
                                "boost": b.get("top_tags.words", 4.5),
                            }
                        }
                    },
                    {
                        "match": {
                            "topic_phrases.words": {
                                "query": query,
                                "boost": b.get("topic_phrases.words", 4.0),
                            }
                        }
                    },
                    {
                        "match": {
                            "domain_text.words": {
                                "query": query,
                                "boost": b.get("domain_text.words", 3.0),
                            }
                        }
                    },
                    {
                        "match": {
                            "semantic_terms.words": {
                                "query": query,
                                "boost": b.get("semantic_terms.words", 2.5),
                            }
                        }
                    },
                    {
                        "match": {
                            "name.words": {
                                "query": query,
                                "boost": b.get("name.words", 1.5),
                            }
                        }
                    },
                    {
                        "match": {
                            "mentioned_names.words": {
                                "query": query,
                                "boost": b.get("mentioned_names.words", 1.0),
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }

    def _is_phrase_like_domain_query(self, query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            return False
        compact_len = len(re.sub(r"\s+", "", text))
        if compact_len < DOMAIN_PHRASE_QUERY_MIN_CHARS:
            return False
        latin_tokens = len(self._latin_token_re.findall(text))
        cjk_chars = sum(len(span) for span in self._cjk_span_re.findall(text))
        return latin_tokens >= 4 or cjk_chars >= DOMAIN_PHRASE_QUERY_MIN_CHARS

    def _has_exact_name_hit(self, query: str, timeout: float = SUGGEST_TIMEOUT) -> bool:
        body = {
            "query": {"term": {"name.keyword": {"value": query}}},
            "_source": ["mid", "name"],
            "size": 1,
            "timeout": f"{int(timeout * 1000)}ms",
        }
        raw = self._submit(body, context="route.exact_name")
        hits = ((raw or {}).get("hits") or {}).get("hits") or []
        if not hits:
            return False
        source = hits[0].get("_source") or {}
        return source.get("name") == query

    def _detect_query_route(self, query: str, exact_name_hit: bool = None) -> str:
        text = (query or "").strip().lower()
        if not text:
            return "domain"
        if self._is_phrase_like_domain_query(text):
            return "phrase"

        compact_len = len(re.sub(r"\s+", "", text))
        latin_tokens = len(self._latin_token_re.findall(text))
        cjk_chars = sum(len(span) for span in self._cjk_span_re.findall(text))
        owner_like_shape = (
            " " not in text
            and compact_len <= 24
            and ((2 <= cjk_chars <= 12) or (1 <= latin_tokens <= 3))
        )
        if not owner_like_shape:
            return "domain"
        if exact_name_hit is None:
            exact_name_hit = self._has_exact_name_hit(query)
        return "name" if exact_name_hit else "domain"

    def _build_combined_query(self, query: str) -> dict:
        """Build a combined owner query for non-relevance sorted searches."""
        return {
            "bool": {
                "should": [
                    self._build_name_query(query),
                    self._build_domain_query(query),
                ],
                "minimum_should_match": 1,
            }
        }

    def _build_phrase_fallback_query(self, query: str) -> dict:
        semantic_query = " ".join(self._extract_semantic_terms(query, max_terms=16))
        if not semantic_query:
            semantic_query = query
        return {
            "bool": {
                "should": [
                    {
                        "match": {
                            "semantic_terms.words": {
                                "query": semantic_query,
                                "minimum_should_match": "35%",
                                "boost": 6.0,
                            }
                        }
                    },
                    {
                        "match": {
                            "topic_phrases.words": {
                                "query": semantic_query,
                                "minimum_should_match": "35%",
                                "boost": 4.5,
                            }
                        }
                    },
                    {
                        "match": {
                            "domain_text.words": {
                                "query": semantic_query,
                                "minimum_should_match": "35%",
                                "boost": 3.5,
                            }
                        }
                    },
                    {
                        "match": {
                            "top_tags.words": {
                                "query": semantic_query,
                                "minimum_should_match": "25%",
                                "boost": 2.0,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }

    def _should_use_domain_semantic_rerank(self, query: str, sort_by: str) -> bool:
        if sort_by not in ("quality", "activity"):
            return False

        text = (query or "").strip().lower()
        if not text or self._is_phrase_like_domain_query(text):
            return False

        compact_len = len(re.sub(r"\s+", "", text))
        latin_tokens = len(self._latin_token_re.findall(text))
        cjk_chars = sum(len(span) for span in self._cjk_span_re.findall(text))
        semantic_terms = self._extract_semantic_terms(text, max_terms=12)
        return (
            compact_len >= 5
            and len(semantic_terms) >= 3
            and (cjk_chars >= 4 or latin_tokens >= 2)
        )

    def _score_to_unit(self, score: float | None) -> float:
        """Normalize a raw ES score into [0, 1] for fusion scoring."""
        if not score:
            return 0.0
        return min(score / NAME_MATCH_NORM_DENOM, 1.0)

    def _extract_semantic_terms(self, text: str, max_terms: int = 24) -> list[str]:
        content = (text or "").strip().lower()
        if not content:
            return []

        counter: Counter = Counter()
        for token in self._latin_token_re.findall(content):
            counter[token] += 2.0

        for span in self._cjk_span_re.findall(content):
            if not span:
                continue
            if len(span) == 1:
                counter[span] += 1.0
                continue

            counter[span] += 2.5 if len(span) <= 8 else 1.5
            max_n = 3 if len(span) >= 3 else 2
            for n in range(2, max_n + 1):
                for index in range(0, len(span) - n + 1):
                    counter[span[index : index + n]] += 1.0

        return [
            term
            for term, _ in sorted(
                counter.items(),
                key=lambda item: (-item[1], -len(item[0]), item[0]),
            )[:max_terms]
        ]

    def _build_hit_semantic_terms(self, hit: dict) -> set[str]:
        semantic_text = (hit.get("semantic_terms") or "").strip()
        if semantic_text:
            return set(term for term in semantic_text.split() if term)

        parts = [
            hit.get("name") or "",
            hit.get("top_tags") or "",
            hit.get("topic_phrases") or "",
            hit.get("domain_text") or "",
            hit.get("mentioned_names") or "",
        ]
        terms: set[str] = set()
        for part in parts:
            terms.update(self._extract_semantic_terms(part, max_terms=48))
        return terms

    def _compute_phrase_semantic_score(
        self, query_terms: list[str], hit: dict
    ) -> float:
        if not query_terms:
            return 0.0
        hit_terms = self._build_hit_semantic_terms(hit)
        if not hit_terms:
            return 0.0

        matched_terms = [term for term in query_terms if term in hit_terms]
        if not matched_terms:
            return 0.0

        coverage = len(matched_terms) / max(len(query_terms), 1)
        matched_weight = sum(min(len(term), 4) for term in matched_terms)
        total_weight = sum(min(len(term), 4) for term in query_terms) or 1
        weighted_coverage = matched_weight / total_weight
        return round((0.55 * coverage) + (0.45 * weighted_coverage), 4)

    def _normalize_sort_value(self, hit: dict, sort_by: str) -> float:
        if sort_by == "influence":
            return min(max(float(hit.get("influence_score") or 0.0), 0.0), 1.0)
        if sort_by == "quality":
            return min(max(float(hit.get("quality_score") or 0.0), 0.0), 1.0)
        if sort_by == "activity":
            return min(max(float(hit.get("activity_score") or 0.0), 0.0), 1.0)
        if sort_by == "total_view":
            return min(
                log1p(max(int(hit.get("total_view") or 0), 0)) / LOG_MAX_VIEW, 1.0
            )
        if sort_by == "total_videos":
            return min(
                log1p(max(int(hit.get("total_videos") or 0), 0)) / LOG_MAX_VIDEOS, 1.0
            )
        return self._score_to_unit(hit.get("_score"))

    def _trim_result(self, result: dict, compact: bool) -> dict:
        if not compact:
            return result

        allowed = set(SOURCE_FIELDS_COMPACT)
        trimmed_hits = []
        for hit in result.get("hits") or []:
            item = {key: value for key, value in hit.items() if key in allowed}
            if hit.get("_score") is not None:
                item["_score"] = hit.get("_score")
            trimmed_hits.append(item)

        return {
            **result,
            "hits": trimmed_hits,
        }

    def _rerank_phrase_hits(
        self,
        query: str,
        parsed: dict,
        sort_by: str,
        limit: int,
        compact: bool,
    ) -> dict:
        query_terms = self._extract_semantic_terms(query)
        hits = list(parsed.get("hits") or [])
        for hit in hits:
            semantic_score = self._compute_phrase_semantic_score(query_terms, hit)
            lexical_score = self._score_to_unit(hit.get("_score"))
            sort_score = self._normalize_sort_value(hit, sort_by)
            hybrid_score = (
                (0.62 * semantic_score) + (0.23 * lexical_score) + (0.15 * sort_score)
            )
            hit["_semantic_score"] = round(semantic_score, 4)
            hit["_score"] = round(hybrid_score, 4)

        hits.sort(
            key=lambda hit: (
                hit.get("_score", 0.0),
                hit.get("_semantic_score", 0.0),
                self._normalize_sort_value(hit, sort_by),
            ),
            reverse=True,
        )
        reranked = {
            "hits": hits[:limit],
            "total": parsed.get("total", len(hits)),
            "max_score": hits[0].get("_score") if hits else None,
        }
        return self._trim_result(reranked, compact=compact)

    def _search_phrase_candidates(
        self,
        query: str,
        es_query: dict,
        filters: list[dict],
        limit: int,
        timeout: float,
        context: str,
    ) -> tuple[dict, dict]:
        candidate_limit = max(limit * 8, 80)
        query_body = {
            "query": self._apply_filters(es_query, filters),
            "_source": SOURCE_FIELDS,
            "size": candidate_limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }
        raw = self._submit(query_body, context=context)
        parsed = self.hit_parser.parse_response(raw, compact=False)
        if parsed.get("total"):
            return parsed, query_body

        fallback_body = {
            "query": self._apply_filters(
                self._build_phrase_fallback_query(query),
                filters,
            ),
            "_source": SOURCE_FIELDS,
            "size": candidate_limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }
        fallback_raw = self._submit(fallback_body, context=f"{context}.fallback")
        fallback_parsed = self.hit_parser.parse_response(fallback_raw, compact=False)
        return fallback_parsed, fallback_body

    def _search_semantic_candidates(
        self,
        es_query: dict,
        filters: list[dict],
        limit: int,
        timeout: float,
        context: str,
    ) -> tuple[dict, dict]:
        candidate_limit = max(limit * 8, 80)
        body = {
            "query": self._apply_filters(es_query, filters),
            "_source": SOURCE_FIELDS,
            "size": candidate_limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }
        raw = self._submit(body, context=context)
        parsed = self.hit_parser.parse_response(raw, compact=False)
        return parsed, body

    def _rerank_domain_semantic_hits(
        self,
        query: str,
        parsed: dict,
        sort_by: str,
        limit: int,
        compact: bool,
    ) -> dict:
        query_terms = self._extract_semantic_terms(query)
        hits = list(parsed.get("hits") or [])
        for hit in hits:
            semantic_score = self._compute_phrase_semantic_score(query_terms, hit)
            lexical_score = self._score_to_unit(hit.get("_score"))
            sort_score = self._normalize_sort_value(hit, sort_by)
            hybrid_score = (
                (0.48 * semantic_score) + (0.18 * lexical_score) + (0.34 * sort_score)
            )
            hit["_semantic_score"] = round(semantic_score, 4)
            hit["_score"] = round(hybrid_score, 4)

        hits.sort(
            key=lambda hit: (
                hit.get("_score", 0.0),
                hit.get("_semantic_score", 0.0),
                self._normalize_sort_value(hit, sort_by),
                hit.get("influence_score", 0.0),
            ),
            reverse=True,
        )
        reranked = {
            "hits": hits[:limit],
            "total": parsed.get("total", len(hits)),
            "max_score": hits[0].get("_score") if hits else None,
        }
        return self._trim_result(reranked, compact=compact)

    def _merge_relevance_hits(
        self,
        query: str,
        name_hits: list[dict],
        domain_hits: list[dict],
        limit: int,
    ) -> dict:
        """Fuse name/domain owner hits with owner-level ranking signals."""
        merged: dict[int, dict] = {}
        query_type = detect_owner_query_type(query, name_hits, domain_hits)

        def upsert(hit: dict, score_key: str):
            mid = hit.get("mid")
            if mid is None:
                return

            item = merged.get(mid)
            if item is None:
                item = {k: v for k, v in hit.items() if k != "_score"}
                item["_name_score"] = 0.0
                item["_domain_score"] = 0.0
                merged[mid] = item
            else:
                for key, value in hit.items():
                    if key == "_score" or value is None:
                        continue
                    if key not in item or item.get(key) in (None, "", [], {}):
                        item[key] = value

            item[score_key] = max(item.get(score_key, 0.0), hit.get("_score") or 0.0)

        for hit in name_hits:
            upsert(hit, "_name_score")
        for hit in domain_hits:
            upsert(hit, "_domain_score")

        fused_hits = []
        for item in merged.values():
            final_score = compute_owner_rank_score(
                name_match_score=item.get("_name_score", 0.0),
                domain_score=self._score_to_unit(item.get("_domain_score", 0.0)),
                influence_score=item.get("influence_score") or 0.0,
                quality_score=item.get("quality_score") or 0.0,
                activity_score=item.get("activity_score") or 0.0,
                query_type=query_type,
            )
            item["_score"] = final_score
            fused_hits.append(item)

        fused_hits.sort(
            key=lambda hit: (
                hit.get("_score", 0.0),
                hit.get("_name_score", 0.0),
                hit.get("_domain_score", 0.0),
                hit.get("influence_score", 0.0),
            ),
            reverse=True,
        )
        fused_hits = fused_hits[:limit]

        max_score = fused_hits[0].get("_score") if fused_hits else None
        total = max(len(merged), len(name_hits), len(domain_hits))
        return {
            "hits": fused_hits,
            "total": total,
            "max_score": round(max_score, 4) if max_score is not None else None,
            "query_type": query_type,
        }

    def _apply_sort(self, body: dict, sort_by: str) -> dict:
        """Apply sort to the search body if sort_by is not relevance."""
        if sort_by and sort_by != "relevance":
            es_field = SORT_FIELD_MAP.get(sort_by)
            if es_field:
                body["sort"] = [{es_field: {"order": "desc"}}, "_score"]
        return body

    def _apply_filters(self, query: dict, filters: list[dict]) -> dict:
        """Wrap query in a bool+filter if extra filters are provided."""
        if not filters:
            return query
        return {
            "bool": {
                "must": [query],
                "filter": filters,
            }
        }

    # =========================================================================
    # Public search methods
    # =========================================================================

    def search(
        self,
        query: str,
        sort_by: SORT_FIELD_TYPE = SORT_FIELD_DEFAULT,
        filters: list[dict] = None,
        limit: int = SEARCH_LIMIT,
        timeout: float = SEARCH_TIMEOUT,
        compact: bool = False,
    ) -> dict:
        """Main search entry — returns owner list.

        Combines name match + domain match, auto-detecting query type.

        Args:
            query: Search query string.
            sort_by: Sort field (relevance/influence/quality/activity/total_view).
            filters: Additional ES filter clauses.
            limit: Max results to return.
            timeout: Search timeout in seconds.
            compact: If True, return compact source fields.

        Returns:
            Parsed result dict with "hits", "total", "max_score".
        """
        source = SOURCE_FIELDS_COMPACT if compact else SOURCE_FIELDS
        if sort_by != "relevance":
            query_route = self._detect_query_route(query)
            if query_route == "name":
                es_query = self._build_name_strict_query(query)
                context = "search.name_routed"
            elif query_route == "phrase":
                es_query = self._build_domain_query(query)
                context = "search.phrase_routed"
            else:
                es_query = self._build_combined_query(query)
                context = "search"
            if query_route == "phrase":
                parsed, body = self._search_phrase_candidates(
                    query=query,
                    es_query=es_query,
                    filters=filters,
                    limit=limit,
                    timeout=timeout,
                    context=context,
                )
                parsed = self._rerank_phrase_hits(
                    query=query,
                    parsed=parsed,
                    sort_by=sort_by,
                    limit=limit,
                    compact=compact,
                )
            elif query_route == "domain" and self._should_use_domain_semantic_rerank(
                query, sort_by
            ):
                parsed, body = self._search_semantic_candidates(
                    es_query=es_query,
                    filters=filters,
                    limit=limit,
                    timeout=timeout,
                    context="search.domain_semantic",
                )
                parsed = self._rerank_domain_semantic_hits(
                    query=query,
                    parsed=parsed,
                    sort_by=sort_by,
                    limit=limit,
                    compact=compact,
                )
            else:
                es_query = self._apply_filters(es_query, filters)
                body = {
                    "query": es_query,
                    "_source": source,
                    "size": limit,
                    "timeout": f"{int(timeout * 1000)}ms",
                }
                body = self._apply_sort(body, sort_by)
                raw = self._submit(body, context=context)
                parsed = self.hit_parser.parse_response(raw, compact=compact)
            parsed["query_route"] = query_route
            return parsed

        candidate_limit = max(limit * 3, 30)
        exact_name_hit = any(hit.get("name") == query for hit in [])
        name_query = self._apply_filters(self._build_name_query(query), filters)
        domain_query = self._apply_filters(self._build_domain_query(query), filters)

        name_body = {
            "query": name_query,
            "_source": source,
            "size": candidate_limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }
        domain_body = {
            "query": domain_query,
            "_source": source,
            "size": candidate_limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }

        raw_name = self._submit(name_body, context="search.name")
        raw_domain = self._submit(domain_body, context="search.domain")
        parsed_name = self.hit_parser.parse_response(raw_name, compact=compact)
        parsed_domain = self.hit_parser.parse_response(raw_domain, compact=compact)

        exact_name_hit = any(
            hit.get("name") == query for hit in parsed_name.get("hits", [])
        )
        merged = self._merge_relevance_hits(
            query=query,
            name_hits=parsed_name.get("hits", []),
            domain_hits=parsed_domain.get("hits", []),
            limit=limit,
        )
        merged["query_route"] = self._detect_query_route(
            query,
            exact_name_hit=exact_name_hit,
        )
        return merged

    def search_by_name(
        self,
        name: str,
        limit: int = SEARCH_LIMIT,
        timeout: float = SEARCH_TIMEOUT,
        compact: bool = False,
    ) -> dict:
        """Name-focused search — exact keyword + BM25 token match.

        Best for: "影视飓风", "红警08", etc.
        """
        es_query = self._build_name_query(name)
        source = SOURCE_FIELDS_COMPACT if compact else SOURCE_FIELDS
        body = {
            "query": es_query,
            "_source": source,
            "size": limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }

        raw = self._submit(body, context="search_by_name")
        return self.hit_parser.parse_response(raw, compact=compact)

    def search_by_domain(
        self,
        query: str,
        sort_by: SORT_FIELD_TYPE = "influence",
        filters: list[dict] = None,
        limit: int = SEARCH_LIMIT,
        timeout: float = SEARCH_TIMEOUT,
        compact: bool = False,
    ) -> dict:
        """Domain/topic search — find creators in a specific area.

        Uses top_tags.words as primary signal. Results are sorted by
        the specified metric (default: influence).

        Best for: "黑神话悟空", "科技数码", "游戏攻略", etc.
        """
        query_route = self._detect_query_route(query)
        if query_route == "name":
            es_query = self._build_name_strict_query(query)
            context = "search_by_domain.name_routed"
        else:
            es_query = self._build_domain_query(query)
            context = "search_by_domain"

        source = SOURCE_FIELDS_COMPACT if compact else SOURCE_FIELDS
        if query_route == "phrase":
            parsed, body = self._search_phrase_candidates(
                query=query,
                es_query=es_query,
                filters=filters,
                limit=limit,
                timeout=timeout,
                context=context,
            )
            parsed = self._rerank_phrase_hits(
                query=query,
                parsed=parsed,
                sort_by=sort_by,
                limit=limit,
                compact=compact,
            )
        elif query_route == "domain" and self._should_use_domain_semantic_rerank(
            query, sort_by
        ):
            parsed, body = self._search_semantic_candidates(
                es_query=es_query,
                filters=filters,
                limit=limit,
                timeout=timeout,
                context="search_by_domain.domain_semantic",
            )
            parsed = self._rerank_domain_semantic_hits(
                query=query,
                parsed=parsed,
                sort_by=sort_by,
                limit=limit,
                compact=compact,
            )
        else:
            es_query = self._apply_filters(es_query, filters)
            body = {
                "query": es_query,
                "_source": source,
                "size": limit,
                "timeout": f"{int(timeout * 1000)}ms",
            }
            body = self._apply_sort(body, sort_by)
            raw = self._submit(body, context=context)
            parsed = self.hit_parser.parse_response(raw, compact=compact)
        parsed["query_route"] = query_route
        return parsed

    def search_by_relation(
        self,
        query: str = None,
        mid: int = None,
        limit: int = SEARCH_LIMIT,
        timeout: float = SEARCH_TIMEOUT,
        compact: bool = False,
    ) -> dict:
        """Relation search — find owners connected to a target owner.

        Search by mentioned_names or mentioned_mids.
        Best for: "和影视飓风合作过的 UP 主", "经常提到XXX的 UP 主".

        Args:
            query: Name of the target owner to search mentions for.
            mid: Mid of the target owner to search mentions for.
            limit: Max results.
            timeout: Search timeout.
            compact: Compact source fields.
        """
        should_clauses = []
        if mid is not None:
            should_clauses.append({"term": {"mentioned_mids": mid}})
        if query:
            should_clauses.append(
                {"match": {"mentioned_names.words": {"query": query, "boost": 5.0}}}
            )
        if not should_clauses:
            return {"hits": [], "total": 0, "max_score": None}

        es_query = {"bool": {"should": should_clauses, "minimum_should_match": 1}}
        source = SOURCE_FIELDS_COMPACT if compact else SOURCE_FIELDS
        body = {
            "query": es_query,
            "_source": source,
            "size": limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }

        raw = self._submit(body, context="search_by_relation")
        return self.hit_parser.parse_response(raw, compact=compact)

    def get_owner(self, mid: int) -> dict:
        """Fetch a single owner document by mid.

        Returns empty dict if not found.
        """
        try:
            res = self.es.client.get(index=self.index_name, id=mid)
            return res.body.get("_source", {})
        except Exception:
            return {}

    def get_owners(self, mids: list[int]) -> dict[int, dict]:
        """Fetch multiple owner documents by mids.

        Returns dict mapping mid → owner_doc. Missing mids are omitted.
        """
        if not mids:
            return {}
        try:
            res = self.es.client.mget(
                index=self.index_name,
                body={"ids": [str(m) for m in mids]},
            )
            result = {}
            for doc in res.body.get("docs", []):
                if doc.get("found"):
                    source = doc.get("_source", {})
                    mid_val = source.get("mid")
                    if mid_val is not None:
                        result[mid_val] = source
            return result
        except Exception as e:
            logger.warn(f"× OwnerSearcher.get_owners error: {str(e)[:200]}")
            return {}

    def suggest(
        self,
        prefix: str,
        limit: int = SUGGEST_LIMIT,
        timeout: float = SUGGEST_TIMEOUT,
        compact: bool = True,
    ) -> dict:
        """Prefix-based owner name suggestion.

        Uses prefix query on name.keyword for fast typehead completion,
        falling back to name.words match for partial matches.
        """
        es_query = {
            "bool": {
                "should": [
                    {
                        "prefix": {
                            "name.keyword": {
                                "value": prefix,
                                "boost": 20.0,
                            }
                        }
                    },
                    {
                        "match": {
                            "name.words": {
                                "query": prefix,
                                "boost": 5.0,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }

        source = SOURCE_FIELDS_COMPACT if compact else SOURCE_FIELDS
        body = {
            "query": es_query,
            "_source": source,
            "size": limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }

        raw = self._submit(body, context="suggest")
        return self.hit_parser.parse_response(raw, compact=compact)

    def top_owners(
        self,
        sort_by: SORT_FIELD_TYPE = "influence",
        tid: int = None,
        limit: int = TOP_OWNERS_LIMIT,
        timeout: float = SEARCH_TIMEOUT,
        compact: bool = True,
    ) -> dict:
        """Top owners ranking — sorted by specified metric.

        Args:
            sort_by: Metric to sort by.
            tid: Optional filter by primary_tid (content category).
            limit: Max results.
            timeout: Timeout.
            compact: Compact fields.
        """
        es_query = {"match_all": {}}
        filters = []
        if tid is not None:
            filters.append({"term": {"primary_tid": tid}})
        # Filter out very low quality entries
        filters.append({"range": {"total_videos": {"gte": 3}}})

        es_query = self._apply_filters(es_query, filters)
        source = SOURCE_FIELDS_COMPACT if compact else SOURCE_FIELDS

        body = {
            "query": es_query,
            "_source": source,
            "size": limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }
        body = self._apply_sort(body, sort_by)

        raw = self._submit(body, context="top_owners")
        return self.hit_parser.parse_response(raw, compact=compact)
