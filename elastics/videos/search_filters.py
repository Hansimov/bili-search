from tclogger import dict_to_str, logger
from typing import Union

from dsl.fields.scope import filter_fields_by_scope
from elastics.structure import set_timeout
from elastics.videos.constants import (
    DATE_MATCH_FIELDS,
    MATCH_TYPE,
    SEARCH_LIMIT,
    SEARCH_MATCH_FIELDS,
    SEARCH_MATCH_TYPE,
    SEARCH_REQUEST_TYPE,
    SEARCH_REQUEST_TYPE_DEFAULT,
    SEARCH_TIMEOUT,
    SOURCE_FIELDS,
)
from ranks.constants import RANK_METHOD, RANK_METHOD_TYPE, RANK_TOP_K


class VideoSearchFilterMixin:
    def filter_only_search(
        self,
        query: str,
        source_fields: list[str] = SOURCE_FIELDS,
        extra_filters: list[dict] = [],
        parse_hits: bool = True,
        add_region_info: bool = True,
        add_highlights_info: bool = False,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD,
        limit: int = SEARCH_LIMIT,
        rank_top_k: int = RANK_TOP_K,
        timeout: Union[int, float, str] = SEARCH_TIMEOUT,
        verbose: bool = False,
    ) -> dict:
        """Search using only filters without text matching.

        This method is used when a query contains only filter expressions
        (like u=xxx, d>xxx, v>xxx) without any search keywords.
        It uses match_all with filters instead of text search or KNN search.

        Args:
            query: Query string (used to extract filters).
            source_fields: Fields to include in results.
            extra_filters: Additional filter clauses.
            parse_hits: Whether to parse hits.
            add_region_info: Whether to add region info.
            add_highlights_info: Whether to add highlight info.
            rank_method: Ranking method.
            limit: Maximum results.
            rank_top_k: Top-k for ranking.
            timeout: Search timeout.
            verbose: Enable verbose logging.

        Returns:
            Search results dict with hits.
        """
        logger.enter_quiet(not verbose)

        # Extract filters from query
        query_info, filter_clauses = self.get_filters_from_query(
            query=query,
            extra_filters=extra_filters,
        )

        # Check for narrow filters (user/bvid filters)
        # For narrow filters, we want to return ALL matching docs, not just rank_top_k
        has_narrow_filter = self.has_narrow_filters(filter_clauses)

        # Build search body with match_all + filters
        if filter_clauses:
            search_body = {
                "query": {"bool": {"filter": filter_clauses}},
                "_source": source_fields,
                "size": limit,
                "track_total_hits": True,
                # Sort by pubdate desc for filter-only queries (most recent first)
                "sort": [{"pubdate": {"order": "desc"}}],
            }
        else:
            # No filters at all, just match_all with pubdate sort
            search_body = {
                "query": {"match_all": {}},
                "_source": source_fields,
                "size": limit,
                "track_total_hits": True,
                "sort": [{"pubdate": {"order": "desc"}}],
            }

        search_body = set_timeout(search_body, timeout=timeout)

        logger.hint(
            f"> Filter-only search (no keywords, narrow_filter={has_narrow_filter})",
            verbose=verbose,
        )
        logger.mesg(
            dict_to_str(search_body, add_quotes=True), indent=2, verbose=verbose
        )

        # Submit to ES
        res_dict = self.submit_to_es(search_body, context="filter_only_search")

        # Parse results
        if parse_hits:
            parse_res = self.hit_parser.parse(
                query_info=query_info,
                match_fields=[],
                res_dict=res_dict,
                request_type="search",
                drop_no_highlights=False,
                add_region_info=add_region_info,
                add_highlights_info=add_highlights_info,
                match_type="cross_fields",
                match_operator="or",
                detail_level=-1,
                limit=limit,
                verbose=verbose,
            )
            # For filter-only search with narrow filters, return ALL results
            # (don't apply rank_top_k limit because users want all docs from the filter)
            # For filter-only search without narrow filters (e.g., d>xxx, v>xxx),
            # apply rank_top_k to limit results
            if has_narrow_filter:
                # Return all hits, just compute scores for display
                effective_top_k = len(parse_res.get("hits", []))
            else:
                effective_top_k = rank_top_k
            parse_res = self.hit_ranker.filter_only_rank(
                parse_res, top_k=effective_top_k
            )
        else:
            parse_res = res_dict

        parse_res["query_info"] = query_info
        parse_res["rewrite_info"] = {}  # Empty rewrite_info for filter-only search
        parse_res["filter_only"] = True
        parse_res["narrow_filter"] = has_narrow_filter

        # Remove non-jsonable items (like query_expr_tree) to avoid serialization errors
        parse_res = self.post_process_return_res(parse_res)

        logger.exit_quiet(not verbose)
        return parse_res

    def get_narrow_filter_owner_count(self, filter_clauses: list[dict]) -> int:
        """Count the number of distinct owners in narrow filter clauses.

        For user filters like `u="A|B|C"`, this returns 3.
        For single user like `u="A"`, this returns 1.

        Args:
            filter_clauses: List of filter clause dicts.

        Returns:
            Number of distinct owners, or 0 if no owner filter.
        """
        for clause in filter_clauses:
            # Check for terms filter (multiple owners)
            if "terms" in clause:
                field = next(iter(clause["terms"]), None)
                if field in ["owner.name.keyword", "owner.mid"]:
                    values = clause["terms"][field]
                    return len(values) if isinstance(values, list) else 1
            # Check for term filter (single owner)
            if "term" in clause:
                field = next(iter(clause["term"]), None)
                if field in ["owner.name.keyword", "owner.mid"]:
                    return 1
            # Check for bool.filter containing owner filters
            if "bool" in clause:
                bool_filter = clause["bool"].get("filter")
                if bool_filter:
                    if isinstance(bool_filter, list):
                        count = self.get_narrow_filter_owner_count(bool_filter)
                        if count > 0:
                            return count
                    elif isinstance(bool_filter, dict):
                        count = self.get_narrow_filter_owner_count([bool_filter])
                        if count > 0:
                            return count
        return 0

    def dual_sort_filter_search(
        self,
        query: str,
        source_fields: list[str] = SOURCE_FIELDS,
        extra_filters: list[dict] = [],
        parse_hits: bool = True,
        add_region_info: bool = True,
        add_highlights_info: bool = False,
        per_owner_recent_limit: int = 2000,
        per_owner_popular_limit: int = 2000,
        timeout: Union[int, float, str] = SEARCH_TIMEOUT,
        verbose: bool = False,
    ) -> dict:
        """Search using dual-sort approach for user/owner filters.

        This method is designed for narrow filters (user/bvid) where we want
        comprehensive coverage. It fetches:
        1. Top N most recent videos (sorted by pubdate desc)
        2. Top N most popular videos (sorted by stat.view desc)
        Then merges and deduplicates the results.

        This ensures we get both recent content and popular content from the
        filtered set, providing better coverage for semantic search.

        Args:
            query: Query string (used to extract filters).
            source_fields: Fields to include in results.
            extra_filters: Additional filter clauses.
            parse_hits: Whether to parse hits.
            add_region_info: Whether to add region info.
            add_highlights_info: Whether to add highlight info.
            per_owner_recent_limit: Max recent videos per owner (default 2000).
            per_owner_popular_limit: Max popular videos per owner (default 2000).
            timeout: Search timeout.
            verbose: Enable verbose logging.

        Returns:
            Search results dict with deduplicated hits from both sorts.
        """
        logger.enter_quiet(not verbose)

        # Extract filters from query
        query_info, filter_clauses = self.get_filters_from_query(
            query=query,
            extra_filters=extra_filters,
        )

        # Count number of owners in filter to scale the limits
        owner_count = self.get_narrow_filter_owner_count(filter_clauses)
        owner_count = max(owner_count, 1)  # At least 1

        recent_limit = per_owner_recent_limit * owner_count
        popular_limit = per_owner_popular_limit * owner_count

        logger.hint(
            f"> Dual-sort filter search (owners={owner_count}, "
            f"recent={recent_limit}, popular={popular_limit})",
            verbose=verbose,
        )

        # Build base query
        if filter_clauses:
            base_query = {"bool": {"filter": filter_clauses}}
        else:
            base_query = {"match_all": {}}

        # Search 1: Most recent (by pubdate desc)
        search_body_recent = {
            "query": base_query,
            "_source": source_fields,
            "size": recent_limit,
            "track_total_hits": True,
            "sort": [{"pubdate": {"order": "desc"}}],
        }
        search_body_recent = set_timeout(search_body_recent, timeout=timeout)

        logger.mesg("> Fetching most recent videos...", verbose=verbose)
        res_recent = self.submit_to_es(search_body_recent, context="dual_sort_recent")

        # Extract hits from recent search
        recent_hits_raw = res_recent.get("hits", {}).get("hits", [])

        # Get total hits count
        total_hits = res_recent.get("hits", {}).get("total", {})
        if isinstance(total_hits, dict):
            total_hits = total_hits.get("value", 0)

        # Optimization: If total docs <= recent_limit, we already have all docs
        # No need to do the second (popular) query
        if total_hits <= recent_limit:
            # All docs already fetched, skip popular query
            logger.mesg(
                f"  Recent: {len(recent_hits_raw)}, Total: {total_hits} "
                f"(all docs fetched, skipping popular query)",
                verbose=verbose,
            )
            popular_hits_raw = []
            merged_hits_raw = recent_hits_raw
            skipped_popular = True
        else:
            # Need both recent and popular queries to get comprehensive coverage
            # Search 2: Most popular (by stat.view desc)
            search_body_popular = {
                "query": base_query,
                "_source": source_fields,
                "size": popular_limit,
                "track_total_hits": True,
                "sort": [{"stat.view": {"order": "desc"}}],
            }
            search_body_popular = set_timeout(search_body_popular, timeout=timeout)

            logger.mesg("> Fetching most popular videos...", verbose=verbose)
            res_popular = self.submit_to_es(
                search_body_popular, context="dual_sort_popular"
            )

            popular_hits_raw = res_popular.get("hits", {}).get("hits", [])

            logger.mesg(
                f"  Recent: {len(recent_hits_raw)}, Popular: {len(popular_hits_raw)}, "
                f"Total in index: {total_hits}",
                verbose=verbose,
            )

            # Merge and deduplicate by bvid
            seen_bvids = set()
            merged_hits_raw = []

            # Add recent hits first (preserving recency priority)
            for hit in recent_hits_raw:
                bvid = hit.get("_source", {}).get("bvid")
                if bvid and bvid not in seen_bvids:
                    seen_bvids.add(bvid)
                    merged_hits_raw.append(hit)

            # Add popular hits that aren't already included
            for hit in popular_hits_raw:
                bvid = hit.get("_source", {}).get("bvid")
                if bvid and bvid not in seen_bvids:
                    seen_bvids.add(bvid)
                    merged_hits_raw.append(hit)

            skipped_popular = False

        logger.mesg(
            f"  After dedup: {len(merged_hits_raw)} unique videos",
            verbose=verbose,
        )

        # Build merged result dict in ES response format
        took_time = res_recent.get("took", 0)
        timed_out = res_recent.get("timed_out", False)
        if not skipped_popular:
            took_time += res_popular.get("took", 0)
            timed_out = timed_out or res_popular.get("timed_out", False)

        merged_res_dict = {
            "took": took_time,
            "timed_out": timed_out,
            "hits": {
                "total": {"value": total_hits, "relation": "eq"},
                "hits": merged_hits_raw,
            },
        }

        # Parse results
        if parse_hits:
            parse_res = self.hit_parser.parse(
                query_info=query_info,
                match_fields=[],
                res_dict=merged_res_dict,
                request_type="search",
                drop_no_highlights=False,
                add_region_info=add_region_info,
                add_highlights_info=add_highlights_info,
                match_type="cross_fields",
                match_operator="or",
                detail_level=-1,
                limit=len(merged_hits_raw),
                verbose=verbose,
            )
            # Use heads since there's no relevance score
            parse_res = self.hit_ranker.heads(parse_res, top_k=len(merged_hits_raw))
        else:
            parse_res = merged_res_dict

        parse_res["query_info"] = query_info
        parse_res["rewrite_info"] = {}
        parse_res["filter_only"] = True
        parse_res[
            "dual_sort_used"
        ] = not skipped_popular  # Only true if both queries ran
        parse_res["dual_sort_info"] = {
            "owner_count": owner_count,
            "recent_limit": recent_limit,
            "popular_limit": popular_limit,
            "recent_fetched": len(recent_hits_raw),
            "popular_fetched": len(popular_hits_raw),
            "popular_skipped": skipped_popular,
            "merged_unique": len(merged_hits_raw),
        }

        # Remove non-jsonable items
        parse_res = self.post_process_return_res(parse_res)

        logger.exit_quiet(not verbose)
        return parse_res

    def has_search_keywords(self, query: str) -> bool:
        """Check if query contains actual search keywords (not just filters).

        Args:
            query: Query string.

        Returns:
            True if query has search keywords, False if only filters.
        """
        query_info = self.query_rewriter.get_query_info(query)
        keywords_body = query_info.get("keywords_body", [])
        return len(keywords_body) > 0

    def rewrite_by_suggest(
        self,
        query_info: dict,
        suggest_info: dict = {},
        rewrite_info: dict = {},
        request_type: SEARCH_REQUEST_TYPE = SEARCH_REQUEST_TYPE_DEFAULT,
        return_res: dict = {},
    ) -> dict:
        """Example output:
        ```python
        {
            ... (input `return_res` key-vals),
            "suggest_info": dict,
            "rewrite_info": dict,
        }
        ```
        #ANCHOR[id=return_res]
        """
        if request_type == "suggest":
            suggest_info = self.suggest_parser.parse(
                qwords=query_info["keywords_body"],
                hits=return_res["hits"],
            )
            rewrite_info = self.query_rewriter.rewrite_query_info_by_suggest_info(
                query_info, suggest_info
            )
        else:
            suggest_info = suggest_info or {}
            rewrite_info = rewrite_info or {}
        return_res["suggest_info"] = suggest_info
        return_res["rewrite_info"] = rewrite_info
        return return_res

    def get_info_of_query_rewrite_dsl(
        self,
        query: str,
        suggest_info: dict = {},
        boosted_match_fields: list[str] = SEARCH_MATCH_FIELDS,
        boosted_date_fields: list[str] = DATE_MATCH_FIELDS,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        extra_filters: list[dict] = [],
    ) -> tuple[dict, dict, dict]:
        """return `(query_info, rewrite_info, query_dsl_dict)`"""
        # get query_info, and get rewrite_info with suggest_info
        query_info = self.query_rewriter.get_query_info(query)
        rewrite_info = self.query_rewriter.rewrite_query_info_by_suggest_info(
            query_info, suggest_info
        )
        scope_info = query_info.get("scope_info") or {}
        effective_match_fields = filter_fields_by_scope(
            boosted_match_fields,
            scope_info,
        )
        effective_date_match_fields = filter_fields_by_scope(
            boosted_date_fields,
            scope_info,
        )
        query_info["effective_match_fields"] = effective_match_fields
        query_info["effective_date_match_fields"] = effective_date_match_fields
        # get expr_tree, and construct query_dsl_dict from expr_tree
        rewrited_expr_trees = list(rewrite_info.get("rewrited_expr_trees") or [])
        expr_tree = rewrited_expr_trees[0] if rewrited_expr_trees else None
        expr_tree = expr_tree or query_info.get("query_expr_tree", None)
        self.elastic_converter.word_converter.switch_mode(
            match_fields=effective_match_fields,
            date_match_fields=effective_date_match_fields,
            match_type=match_type,
        )
        query_dsl_dict = self.elastic_converter.expr_tree_to_dict(expr_tree)
        # merge extra_filters to query_dsl_dict
        query_dsl_dict = self.filter_merger.merge(query_dsl_dict, extra_filters)
        # logger.hint("> query_dsl_dict:")
        # logger.mesg(dict_to_str(query_dsl_dict, add_quotes=True, align_list=False))
        # logger.hint("> expr_tree:")
        # logger.mesg(expr_tree.yaml())
        return query_info, rewrite_info, query_dsl_dict

    def get_filters_from_query(
        self,
        query: str,
        extra_filters: list[dict] = [],
    ) -> tuple[dict, list[dict]]:
        """Extract filter clauses from DSL query expression.

        This extracts non-word filters (date, stat, user, etc.) from the query
        and combines them with extra_filters for use in KNN search.

        Args:
            query: The query string (may contain DSL expressions like "d>2024-01-01").
            extra_filters: Additional filter clauses to include.

        Returns:
            Tuple of (query_info, filter_clauses).
        """
        query_info = self.query_rewriter.get_query_info(query)
        expr_tree = query_info.get("query_expr_tree", None)

        if expr_tree is None:
            return query_info, extra_filters

        # Extract filter atoms (non-word expressions) from expr_tree
        filter_expr_tree = expr_tree.filter_atoms_by_keys(exclude_keys=["word_expr"])

        # Convert filter expr_tree to elastic filter dict
        if filter_expr_tree and filter_expr_tree.children:
            self.elastic_converter.word_converter.switch_mode(
                match_fields=[],
                date_match_fields=[],
                match_type="cross_fields",
            )
            filter_dict = self.elastic_converter.expr_tree_to_dict(filter_expr_tree)

            # Extract all bool clauses (filter + must_not) from the dict
            # This ensures user exclusions (u!=[...]) are preserved in KNN pre-filters
            filter_clauses = (
                self.filter_merger.get_all_bool_clauses_from_query_dsl_dict(filter_dict)
            )
        else:
            filter_clauses = []

        # Combine with extra_filters
        all_filters = filter_clauses + list(extra_filters)

        # Also include exact-segment prefilters from +/- token expressions.
        # Without this, constraint-only queries (e.g., "+seedance +2.0")
        # would have no filtering in filter_only_search, returning all docs
        # instead of just those matching the required exact segments.
        constraint_filter = query_info.get("constraint_filter", {})
        if constraint_filter:
            all_filters.append(constraint_filter)

        return query_info, all_filters

    def has_narrow_filters(self, filter_clauses: list[dict]) -> bool:
        """Check if filter clauses contain narrow filters that severely limit results.

        Narrow filters include:
        - User filters (owner.name.keyword, owner.mid)
        - Specific bvid/aid filters

        When narrow filters are present, approximate KNN with filter may return
        very few results because it searches globally first then filters.
        For such cases, we should use a different approach (filter first, then KNN).

        Args:
            filter_clauses: List of filter clause dicts.

        Returns:
            True if narrow filters are detected.
        """
        NARROW_FILTER_FIELDS = [
            "owner.name.keyword",
            "owner.mid",
            "bvid.keyword",
            "aid",
        ]

        for clause in filter_clauses:
            # Check for term/terms filters
            for filter_type in ["term", "terms"]:
                if filter_type in clause:
                    field = next(iter(clause[filter_type]), None)
                    if field in NARROW_FILTER_FIELDS:
                        return True
            # Check for bool.filter containing narrow filters
            if "bool" in clause:
                bool_filter = clause["bool"].get("filter")
                if bool_filter:
                    if isinstance(bool_filter, list):
                        if self.has_narrow_filters(bool_filter):
                            return True
                    elif isinstance(bool_filter, dict):
                        if self.has_narrow_filters([bool_filter]):
                            return True
        return False
