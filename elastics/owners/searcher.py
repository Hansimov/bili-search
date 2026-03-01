"""OwnerSearcher — core search service for the owners ES index.

Provides name search, domain search, relation search, suggest,
and top-owners queries against the independent owners index.
Independent of VideoSearcher — can be used standalone or as an
enhancement to the existing videos search pipeline.
"""

from sedb import ElasticOperator
from tclogger import logger

from configs.envs import SECRETS, ELASTIC_PRO_ENVS
from elastics.owners.constants import (
    ELASTIC_OWNERS_INDEX,
    SOURCE_FIELDS,
    SOURCE_FIELDS_COMPACT,
    NAME_MATCH_BOOSTS,
    DOMAIN_MATCH_BOOSTS,
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

    def _build_domain_query(self, query: str, boosts: dict = None) -> dict:
        """Build a domain/topic search query.

        Primarily matches top_tags.words with name.words as secondary signal.
        """
        b = boosts or DOMAIN_MATCH_BOOSTS
        return {
            "bool": {
                "should": [
                    {
                        "match": {
                            "top_tags.words": {
                                "query": query,
                                "boost": b.get("top_tags.words", 5.0),
                            }
                        }
                    },
                    {
                        "match": {
                            "name.words": {
                                "query": query,
                                "boost": b.get("name.words", 2.0),
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
        es_query = self._build_name_query(query)
        es_query = self._apply_filters(es_query, filters)

        source = SOURCE_FIELDS_COMPACT if compact else SOURCE_FIELDS
        body = {
            "query": es_query,
            "_source": source,
            "size": limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }
        body = self._apply_sort(body, sort_by)

        raw = self._submit(body, context="search")
        parsed = self.hit_parser.parse_response(raw, compact=compact)
        return parsed

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
        es_query = self._build_domain_query(query)
        es_query = self._apply_filters(es_query, filters)

        source = SOURCE_FIELDS_COMPACT if compact else SOURCE_FIELDS
        body = {
            "query": es_query,
            "_source": source,
            "size": limit,
            "timeout": f"{int(timeout * 1000)}ms",
        }
        body = self._apply_sort(body, sort_by)

        raw = self._submit(body, context="search_by_domain")
        return self.hit_parser.parse_response(raw, compact=compact)

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
