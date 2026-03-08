"""OwnerSearcher — slim owner retrieval service.

The legacy owner-domain text features are intentionally disabled here.
Current live retrieval supports:

- name search: exact keyword + BM25 token match
- relation search: mentioned name / mid lookups
- suggest: prefix completion on owner names
- top owners: metric-sorted leaderboard

Future domain retrieval is reserved for CoreTagTokenizer / CoreTexTokenizer
token ids via the profile-token placeholder fields.
"""

import re

from sedb import ElasticOperator
from tclogger import logger

from configs.envs import ELASTIC_PRO_ENVS, SECRETS
from elastics.owners.constants import (
    DOMAIN_PHRASE_QUERY_MIN_CHARS,
    ELASTIC_OWNERS_INDEX,
    NAME_MATCH_BOOSTS,
    NAME_MATCH_NORM_DENOM,
    PROFILE_TOKEN_MATCH_BOOSTS,
    PROFILE_TOKEN_SCORE_DENOM,
    SEARCH_LIMIT,
    SEARCH_TIMEOUT,
    SORT_FIELD_DEFAULT,
    SORT_FIELD_MAP,
    SORT_FIELD_TYPE,
    SOURCE_FIELDS,
    SOURCE_FIELDS_COMPACT,
    SUGGEST_LIMIT,
    SUGGEST_TIMEOUT,
    TOP_OWNERS_LIMIT,
)
from elastics.owners.hits import OwnerHitsParser
from elastics.owners.scorer import compute_owner_rank_score, detect_owner_query_type


class OwnerSearcher:
    """Owner search service for the independent owners index."""

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

    def _submit(self, body: dict, context: str = "search") -> dict:
        try:
            res = self.es.client.search(index=self.index_name, body=body)
            return res.body
        except Exception as e:
            logger.warn(f"× OwnerSearcher ES error [{context}]: {str(e)[:300]}")
            return {"hits": {"hits": [], "total": {"value": 0}}, "error": str(e)}

    def _build_name_query(self, query: str, boosts: dict = None) -> dict:
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

    def _normalize_token_ids(self, token_ids: list[int] | None) -> list[int]:
        values = []
        for token_id in token_ids or []:
            try:
                value = int(token_id)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
        return sorted(set(values))

    def _has_query_tokens(
        self,
        tag_token_ids: list[int] | None = None,
        text_token_ids: list[int] | None = None,
    ) -> bool:
        return bool(
            self._normalize_token_ids(tag_token_ids)
            or self._normalize_token_ids(text_token_ids)
        )

    def _build_profile_token_query(
        self,
        tag_token_ids: list[int] | None = None,
        text_token_ids: list[int] | None = None,
        boosts: dict = None,
    ) -> dict:
        b = boosts or PROFILE_TOKEN_MATCH_BOOSTS
        tag_ids = self._normalize_token_ids(tag_token_ids)
        text_ids = self._normalize_token_ids(text_token_ids)
        should = []

        if tag_ids:
            should.append(
                {
                    "constant_score": {
                        "filter": {"terms": {"core_tag_token_ids": tag_ids}},
                        "boost": b.get("core_tag_token_ids", 4.0),
                    }
                }
            )
        if text_ids:
            should.append(
                {
                    "constant_score": {
                        "filter": {"terms": {"core_text_token_ids": text_ids}},
                        "boost": b.get("core_text_token_ids", 5.0),
                    }
                }
            )

        if not should:
            return {"match_none": {}}
        return {"bool": {"should": should, "minimum_should_match": 1}}

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

    def _score_to_unit(self, score: float | None) -> float:
        if not score:
            return 0.0
        return min(score / NAME_MATCH_NORM_DENOM, 1.0)

    def _domain_score_to_unit(self, score: float | None) -> float:
        if not score:
            return 0.0
        return min(score / PROFILE_TOKEN_SCORE_DENOM, 1.0)

    def _apply_sort(self, body: dict, sort_by: str) -> dict:
        if sort_by and sort_by != "relevance":
            es_field = SORT_FIELD_MAP.get(sort_by)
            if es_field:
                body["sort"] = [{es_field: {"order": "desc"}}, "_score"]
        return body

    def _apply_filters(self, query: dict, filters: list[dict]) -> dict:
        if not filters:
            return query
        return {"bool": {"must": [query], "filter": filters}}

    def _search_query(
        self,
        es_query: dict,
        *,
        compact: bool,
        limit: int,
        timeout: float,
        sort_by: str,
        filters: list[dict],
        context: str,
    ) -> dict:
        source = SOURCE_FIELDS_COMPACT if compact else SOURCE_FIELDS
        body = {
            "query": self._apply_filters(es_query, filters),
            "_source": source,
            "size": limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }
        body = self._apply_sort(body, sort_by)
        raw = self._submit(body, context=context)
        return self.hit_parser.parse_response(raw, compact=compact)

    def _empty_result(self, query_route: str, domain_status: str) -> dict:
        return {
            "hits": [],
            "total": 0,
            "max_score": None,
            "query_route": query_route,
            "query_type": "domain",
            "domain_status": domain_status,
        }

    def _merge_relevance_hits(
        self,
        query: str,
        name_hits: list[dict],
        domain_hits: list[dict],
        limit: int,
    ) -> dict:
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
            item["_score"] = compute_owner_rank_score(
                name_match_score=item.get("_name_score", 0.0),
                domain_score=self._domain_score_to_unit(item.get("_domain_score", 0.0)),
                influence_score=item.get("influence_score") or 0.0,
                quality_score=item.get("quality_score") or 0.0,
                activity_score=item.get("activity_score") or 0.0,
                query_type=query_type,
            )
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
        return {
            "hits": fused_hits,
            "total": max(len(merged), len(name_hits), len(domain_hits)),
            "max_score": round(max_score, 4) if max_score is not None else None,
            "query_type": query_type,
        }

    def search(
        self,
        query: str,
        sort_by: SORT_FIELD_TYPE = SORT_FIELD_DEFAULT,
        filters: list[dict] = None,
        limit: int = SEARCH_LIMIT,
        timeout: float = SEARCH_TIMEOUT,
        compact: bool = False,
        tag_token_ids: list[int] | None = None,
        text_token_ids: list[int] | None = None,
    ) -> dict:
        query_route = self._detect_query_route(query)
        if query_route == "name":
            es_query = (
                self._build_name_query(query)
                if sort_by == "relevance"
                else self._build_name_strict_query(query)
            )
            context = "search.name" if sort_by == "relevance" else "search.name_routed"
            parsed = self._search_query(
                es_query,
                compact=compact,
                limit=limit,
                timeout=timeout,
                sort_by=sort_by,
                filters=filters,
                context=context,
            )
            parsed["query_route"] = "name"
            parsed["query_type"] = "name"
            parsed["domain_status"] = "name_route"
            return parsed

        if not self._has_query_tokens(
            tag_token_ids=tag_token_ids, text_token_ids=text_token_ids
        ):
            return self._empty_result(
                query_route=query_route, domain_status="query_tokens_missing"
            )

        domain_query = self._build_profile_token_query(
            tag_token_ids=tag_token_ids,
            text_token_ids=text_token_ids,
        )
        domain_result = self._search_query(
            domain_query,
            compact=compact,
            limit=limit,
            timeout=timeout,
            sort_by=sort_by,
            filters=filters,
            context="search.domain_tokens",
        )
        if sort_by == "relevance":
            domain_result = self._merge_relevance_hits(
                query=query,
                name_hits=[],
                domain_hits=domain_result.get("hits", []),
                limit=limit,
            )
        domain_result["query_route"] = query_route
        domain_result["domain_status"] = "query_tokens_used"
        if sort_by == "relevance":
            domain_result.setdefault("query_type", "domain")
        return domain_result

    def search_by_name(
        self,
        name: str,
        limit: int = SEARCH_LIMIT,
        timeout: float = SEARCH_TIMEOUT,
        compact: bool = False,
    ) -> dict:
        parsed = self._search_query(
            self._build_name_query(name),
            compact=compact,
            limit=limit,
            timeout=timeout,
            sort_by="relevance",
            filters=None,
            context="search_by_name",
        )
        parsed["query_route"] = "name"
        parsed["query_type"] = "name"
        parsed["domain_status"] = "name_route"
        return parsed

    def search_by_domain(
        self,
        query: str,
        sort_by: SORT_FIELD_TYPE = "influence",
        filters: list[dict] = None,
        limit: int = SEARCH_LIMIT,
        timeout: float = SEARCH_TIMEOUT,
        compact: bool = False,
        tag_token_ids: list[int] | None = None,
        text_token_ids: list[int] | None = None,
    ) -> dict:
        query_route = self._detect_query_route(query)
        if query_route == "name":
            parsed = self._search_query(
                self._build_name_strict_query(query),
                compact=compact,
                limit=limit,
                timeout=timeout,
                sort_by=sort_by,
                filters=filters,
                context="search_by_domain.name_routed",
            )
            parsed["query_route"] = "name"
            parsed["query_type"] = "name"
            parsed["domain_status"] = "name_route"
            return parsed

        if not self._has_query_tokens(
            tag_token_ids=tag_token_ids, text_token_ids=text_token_ids
        ):
            return self._empty_result(
                query_route=query_route, domain_status="query_tokens_missing"
            )

        parsed = self._search_query(
            self._build_profile_token_query(
                tag_token_ids=tag_token_ids,
                text_token_ids=text_token_ids,
            ),
            compact=compact,
            limit=limit,
            timeout=timeout,
            sort_by=sort_by,
            filters=filters,
            context="search_by_domain.tokens",
        )
        parsed["query_route"] = query_route
        parsed["query_type"] = "domain"
        parsed["domain_status"] = "query_tokens_used"
        return parsed

    def search_by_relation(
        self,
        query: str = None,
        mid: int = None,
        limit: int = SEARCH_LIMIT,
        timeout: float = SEARCH_TIMEOUT,
        compact: bool = False,
    ) -> dict:
        should_clauses = []
        if mid is not None:
            should_clauses.append({"term": {"mentioned_mids": mid}})
        if query:
            should_clauses.append(
                {"match": {"mentioned_names.words": {"query": query, "boost": 5.0}}}
            )
        if not should_clauses:
            return {"hits": [], "total": 0, "max_score": None}

        parsed = self._search_query(
            {"bool": {"should": should_clauses, "minimum_should_match": 1}},
            compact=compact,
            limit=limit,
            timeout=timeout,
            sort_by="relevance",
            filters=None,
            context="search_by_relation",
        )
        parsed["query_route"] = "relation"
        parsed["query_type"] = "relation"
        parsed["domain_status"] = "relation_route"
        return parsed

    def get_owner(self, mid: int) -> dict:
        try:
            res = self.es.client.get(index=self.index_name, id=mid)
            return res.body.get("_source", {})
        except Exception:
            return {}

    def get_owners(self, mids: list[int]) -> dict[int, dict]:
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
        return self._search_query(
            es_query,
            compact=compact,
            limit=limit,
            timeout=timeout,
            sort_by="relevance",
            filters=None,
            context="suggest",
        )

    def top_owners(
        self,
        sort_by: SORT_FIELD_TYPE = "influence",
        tid: int = None,
        limit: int = TOP_OWNERS_LIMIT,
        timeout: float = SEARCH_TIMEOUT,
        compact: bool = True,
    ) -> dict:
        filters = [{"range": {"total_videos": {"gte": 3}}}]
        parsed = self._search_query(
            {"match_all": {}},
            compact=compact,
            limit=limit,
            timeout=timeout,
            sort_by=sort_by,
            filters=filters,
            context="top_owners",
        )
        if tid is not None:
            parsed["ignored_filters"] = ["tid"]
        return parsed
