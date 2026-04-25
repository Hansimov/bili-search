from concurrent.futures import ThreadPoolExecutor
from tclogger import dict_to_str, logger
from typing import Literal, Union

from dsl.fields.qmod import extract_qmod_from_expr_tree
from elastics.structure import construct_knn_query, construct_knn_search_body
from elastics.videos.constants import (
    IS_HIGHLIGHT,
    KNN_K,
    KNN_NUM_CANDIDATES,
    KNN_TEXT_EMB_FIELD,
    KNN_TIMEOUT,
    MATCH_TYPE,
    QMOD,
    SEARCH_BOOSTED_FIELDS,
    SEARCH_LIMIT,
    SEARCH_MATCH_FIELDS,
    SEARCH_MATCH_TYPE,
    SEARCH_TIMEOUT,
    SOURCE_FIELDS,
    TRACK_TOTAL_HITS,
    USE_SCRIPT_SCORE,
)
from ranks.constants import (
    HYBRID_RRF_K,
    HYBRID_VECTOR_WEIGHT,
    HYBRID_WORD_WEIGHT,
    RANK_METHOD,
    RANK_METHOD_TYPE,
    RANK_TOP_K,
)


class VideoSearchVectorMixin:
    def construct_knn_search_body(
        self,
        query_vector: list[int],
        filter_clauses: list[dict] = None,
        constraint_filter: dict = None,
        source_fields: list[str] = SOURCE_FIELDS,
        knn_field: str = KNN_TEXT_EMB_FIELD,
        k: int = KNN_K,
        num_candidates: int = KNN_NUM_CANDIDATES,
        similarity: float = None,
        limit: int = SEARCH_LIMIT,
        timeout: Union[int, float, str] = KNN_TIMEOUT,
        is_explain: bool = False,
    ) -> dict:
        """Construct KNN search body.

        Args:
            query_vector: Query vector as byte array (signed int8 list).
            filter_clauses: Filter clauses to apply during KNN search.
            constraint_filter: Optional query dict for exact-segment filtering
                via the es-tok plugin.
            source_fields: Fields to include in _source.
            knn_field: The dense_vector field to search.
            k: Number of nearest neighbors to return.
            num_candidates: Number of candidates to consider per shard.
            similarity: Minimum similarity threshold.
            limit: Maximum number of results.
            timeout: Search timeout.
            is_explain: Whether to include explanation.

        Returns:
            Complete search body dict.
        """
        knn_query = construct_knn_query(
            query_vector=query_vector,
            field=knn_field,
            k=k,
            num_candidates=num_candidates,
            filter_clauses=filter_clauses,
            constraint_filter=constraint_filter,
            similarity=similarity,
        )

        search_body = construct_knn_search_body(
            knn_query=knn_query,
            source_fields=source_fields,
            size=limit,
            timeout=timeout,
            track_total_hits=TRACK_TOTAL_HITS,
            is_explain=is_explain,
        )

        return search_body

    def knn_search(
        self,
        query: str,
        source_fields: list[str] = SOURCE_FIELDS,
        extra_filters: list[dict] = [],
        constraint_filter: dict = None,
        knn_field: str = KNN_TEXT_EMB_FIELD,
        k: int = KNN_K,
        num_candidates: int = KNN_NUM_CANDIDATES,
        similarity: float = None,
        parse_hits: bool = True,
        add_region_info: bool = True,
        is_explain: bool = False,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD,
        limit: int = SEARCH_LIMIT,
        rank_top_k: int = RANK_TOP_K,
        skip_ranking: bool = False,
        timeout: Union[int, float, str] = KNN_TIMEOUT,
        verbose: bool = False,
    ) -> Union[dict, list[dict]]:
        """Perform KNN search using text embeddings.

        This method:
        1. Extracts filter clauses from DSL query (date, stat, user filters)
        2. Extracts query words and converts them to embedding vector
        3. Performs KNN search on text_emb field with filters
        4. Parses and ranks the results

        Args:
            query: Query string (can include DSL expressions for filtering).
            source_fields: Fields to include in results.
            extra_filters: Additional filter clauses.
            constraint_filter: Optional query dict for exact-segment filtering
                via the es-tok plugin.
            knn_field: The dense_vector field to search.
            k: Number of nearest neighbors.
            num_candidates: Candidates per shard.
            similarity: Minimum similarity threshold.
            parse_hits: Whether to parse hits into structured format.
            add_region_info: Whether to add region info to results.
            is_explain: Whether to include ES explanation.
            rank_method: Ranking method ("heads", "rrf", "stats").
            limit: Maximum results to return.
            rank_top_k: Top-k for ranking.
            skip_ranking: If True, skip ranking step (caller will do reranking).
            timeout: Search timeout.
            verbose: Enable verbose logging.

        Returns:
            Search results dict with hits and metadata.
        """
        logger.enter_quiet(not verbose)

        # Extract query info and filter clauses from DSL query
        query_info, filter_clauses = self.get_filters_from_query(
            query=query,
            extra_filters=extra_filters,
        )

        # Check for narrow filters (user/bvid filters) that would cause
        # approximate KNN to return very few results
        has_narrow_filter = self.has_narrow_filters(filter_clauses)
        if has_narrow_filter:
            logger.hint(
                "> Narrow filters detected in KNN search, will use higher k",
                verbose=verbose,
            )

        # Get query words for embedding
        words_expr = query_info.get("words_expr", "")
        keywords_body = query_info.get("keywords_body", [])
        constraint_texts = query_info.get("constraint_texts", [])

        # Check if there are actual search keywords (scoring keywords).
        # Constraint texts (+token) don't produce BM25 scoring, so a query
        # with only constraints (e.g., "+seedance +2.0") has no scoring
        # keywords and should use filter-only search.
        if not keywords_body:
            logger.hint(
                "> No search keywords found, falling back to filter-only search",
                verbose=verbose,
            )
            logger.exit_quiet(not verbose)
            return self.filter_only_search(
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                parse_hits=parse_hits,
                add_region_info=add_region_info,
                add_highlights_info=False,
                rank_method=rank_method,
                limit=limit,
                rank_top_k=rank_top_k,
                timeout=timeout,
                verbose=verbose,
            )

        # Build text for embedding from scoring keywords + constraint texts.
        # Constraint texts are included because they add semantic context
        # (e.g., "+seedance +2.0 科幻" → embed "seedance 2.0 科幻" for
        # better semantic matching), even though they don't score via BM25.
        all_embed_parts = keywords_body + constraint_texts
        embed_text = " ".join(all_embed_parts)

        # Convert query text to embedding vector
        if not self.embed_client.is_available():
            logger.warn("× Embed client not available, KNN search cannot proceed")
            logger.exit_quiet(not verbose)
            return {
                "query": query,
                "request_type": "knn_search",
                "timed_out": True,
                "total_hits": 0,
                "return_hits": 0,
                "hits": [],
                "query_info": query_info,
                "error": "Embed client not available",
            }

        query_hex = self.embed_client.text_to_hex(embed_text)
        if not query_hex:
            logger.warn("× Failed to get embedding for query")
            logger.exit_quiet(not verbose)
            return {
                "query": query,
                "request_type": "knn_search",
                "timed_out": True,
                "total_hits": 0,
                "return_hits": 0,
                "hits": [],
                "query_info": query_info,
                "error": "Failed to compute embedding",
            }

        # Convert hex string to byte array for ES
        query_vector = self.embed_client.hex_to_byte_array(query_hex)

        # For narrow filters (user/bvid), increase k and num_candidates significantly
        # This ensures ES brute-forces through all matching documents
        # rather than finding top-k globally then filtering
        if has_narrow_filter:
            # Use very high values - ES will automatically switch to brute-force
            # when filter reduces the set to less than num_candidates
            k = max(k, 10000)
            num_candidates = max(num_candidates, 50000)
            # Also increase limit to get more results
            limit = max(limit, k)
            logger.hint(
                f"> Boosted KNN params for narrow filter: k={k}, num_candidates={num_candidates}",
                verbose=verbose,
            )

        # Construct KNN search body
        search_body = self.construct_knn_search_body(
            query_vector=query_vector,
            filter_clauses=filter_clauses if filter_clauses else None,
            constraint_filter=constraint_filter,
            source_fields=source_fields,
            knn_field=knn_field,
            k=k,
            num_candidates=num_candidates,
            similarity=similarity,
            limit=limit,
            timeout=timeout,
            is_explain=is_explain,
        )

        logger.mesg(
            dict_to_str(search_body, add_quotes=True), indent=2, verbose=verbose
        )

        # Submit to ES
        es_res_dict = self.submit_to_es(search_body, context="knn_search")

        # Parse results
        if parse_hits:
            parse_res = self.hit_parser.parse(
                query_info,
                match_fields=[],  # No highlight for KNN search
                res_dict=es_res_dict,
                request_type="knn_search",
                drop_no_highlights=False,
                add_region_info=add_region_info,
                add_highlights_info=False,  # KNN search doesn't use keyword highlights
                match_type="cross_fields",
                match_operator="or",
                detail_level=-1,
                limit=limit,
                verbose=verbose,
            )
            # Apply ranking - for KNN search, "relevance" is the default and preferred method
            # If skip_ranking is True, caller will do reranking later
            if not skip_ranking:
                if rank_method == "tiered":
                    # Tiered ranking with KNN score as relevance
                    parse_res = self.hit_ranker.tiered_rank(
                        parse_res, top_k=rank_top_k, relevance_field="score"
                    )
                elif rank_method == "relevance":
                    parse_res = self.hit_ranker.relevance_rank(
                        parse_res, top_k=rank_top_k
                    )
                elif rank_method == "rrf":
                    parse_res = self.hit_ranker.rrf_rank(parse_res, top_k=rank_top_k)
                elif rank_method == "stats":
                    parse_res = self.hit_ranker.stats_rank(parse_res, top_k=rank_top_k)
                else:  # "heads"
                    parse_res = self.hit_ranker.heads(parse_res, top_k=rank_top_k)
        else:
            parse_res = es_res_dict

        # Build return result
        return_res = {
            **(parse_res if isinstance(parse_res, dict) else {}),
            "query_info": query_info,
            "suggest_info": {},
            "rewrite_info": {},
            # Sanitize search_body to reduce network payload
            "search_body": self.sanitize_search_body_for_client(search_body),
        }

        # Clean up non-jsonable items
        return_res["query_info"].pop("query_expr_tree", None)

        logger.exit_quiet(not verbose)
        return return_res

    def get_qmod_from_query(self, query: str) -> list[str]:
        """Extract query mode (qmod) from query string.

        Parses the query for q=w/v/wv expression and returns the mode list.

        Args:
            query: Query string that may contain q=w/v/wv.

        Returns:
            List of query modes, e.g. ["word"], ["vector"], ["word", "vector"].
        """
        try:
            expr_tree = self.elastic_converter.construct_expr_tree(query)
            return extract_qmod_from_expr_tree(expr_tree)
        except Exception as e:
            logger.warn(f"× Failed to parse qmod: {e}")
            return QMOD.copy() if isinstance(QMOD, list) else [QMOD]

    def hybrid_search(
        self,
        query: str,
        source_fields: list[str] = SOURCE_FIELDS,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        extra_filters: list[dict] = [],
        constraint_filter: dict = None,
        suggest_info: dict = {},
        # Word search params
        boost: bool = True,
        boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
        use_script_score: bool = USE_SCRIPT_SCORE,
        # KNN params
        knn_field: str = KNN_TEXT_EMB_FIELD,
        knn_k: int = KNN_K,
        knn_num_candidates: int = KNN_NUM_CANDIDATES,
        # Hybrid fusion params
        word_weight: float = HYBRID_WORD_WEIGHT,
        vector_weight: float = HYBRID_VECTOR_WEIGHT,
        rrf_k: int = HYBRID_RRF_K,
        fusion_method: Literal["rrf", "weighted"] = "rrf",
        # Common params
        parse_hits: bool = True,
        add_region_info: bool = True,
        add_highlights_info: bool = True,
        is_highlight: bool = IS_HIGHLIGHT,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD,
        limit: int = SEARCH_LIMIT,
        rank_top_k: int = RANK_TOP_K,
        timeout: Union[int, float, str] = SEARCH_TIMEOUT,
        verbose: bool = False,
    ) -> dict:
        """Perform hybrid search combining word-based and vector-based retrieval.

        This method:
        1. Performs word-based search using traditional ES query
        2. Performs KNN vector search using text embeddings
        3. Fuses results using RRF or weighted combination

        Args:
            query: Query string (can include DSL expressions).
            source_fields: Fields to include in results.
            match_fields: Fields for word matching.
            match_type: Type of matching for word search.
            extra_filters: Additional filter clauses.
            constraint_filter: Optional es_tok_constraints query dict for
                token-level filtering via the es-tok plugin.
            suggest_info: Suggestion info from previous searches.
            boost: Whether to boost fields.
            boosted_fields: Field boost weights.
            use_script_score: Whether to use script scoring.
            knn_field: Dense vector field for KNN.
            knn_k: Number of KNN neighbors.
            knn_num_candidates: KNN candidates per shard.
            word_weight: Weight for word-based scores (weighted fusion).
            vector_weight: Weight for vector-based scores (weighted fusion).
            rrf_k: K parameter for RRF fusion.
            fusion_method: "rrf" or "weighted".
            parse_hits: Whether to parse hits.
            add_region_info: Whether to add region info.
            add_highlights_info: Whether to add highlight info.
            is_highlight: Whether to highlight matches.
            rank_method: Final ranking method.
            limit: Maximum results.
            rank_top_k: Top-k for ranking.
            timeout: Search timeout.
            verbose: Enable verbose logging.

        Returns:
            Search results with fused hits from both methods.
        """
        logger.enter_quiet(not verbose)

        # Get query info and check for keywords
        query_info = self.query_rewriter.get_query_info(query)
        keywords_body = query_info.get("keywords_body", [])

        # Check if there are actual search keywords
        # If no keywords, fall back to filter-only search
        if not keywords_body:
            logger.hint(
                "> No search keywords found, falling back to filter-only search",
                verbose=verbose,
            )
            logger.exit_quiet(not verbose)
            return self.filter_only_search(
                query=query,
                source_fields=source_fields,
                extra_filters=extra_filters,
                parse_hits=parse_hits,
                add_region_info=add_region_info,
                add_highlights_info=add_highlights_info,
                rank_method=rank_method,
                limit=limit,
                rank_top_k=rank_top_k,
                timeout=timeout,
                verbose=verbose,
            )

        # Execute word search and KNN search in parallel
        logger.hint("> Hybrid: Running word search and KNN search in parallel ...")

        # IMPORTANT: Search with higher limit than rank_top_k to ensure enough results
        # after fusion. Use at least 3x rank_top_k for each source to account for:
        # 1. Overlap between word and KNN results
        # 2. Filtering that may reduce results
        search_limit = max(limit, rank_top_k * 3, 1500)
        word_limit = min(search_limit, 2000)  # Cap word search at 2000
        word_search_params = {
            "query": query,
            "source_fields": source_fields,
            "match_fields": match_fields,
            "match_type": match_type,
            "extra_filters": extra_filters,
            "suggest_info": suggest_info,
            "boost": boost,
            "boosted_fields": boosted_fields,
            "use_script_score": use_script_score,
            "parse_hits": True,
            "add_region_info": add_region_info,
            "add_highlights_info": add_highlights_info,
            "is_highlight": is_highlight,
            "rank_method": "heads",  # Use heads, we'll re-rank after fusion
            "limit": word_limit,
            "rank_top_k": word_limit,
            "timeout": timeout,
            "verbose": verbose,
        }

        # KNN search: use same search_limit for consistency
        # k must not exceed num_candidates
        knn_limit = min(search_limit, knn_num_candidates)
        knn_search_params = {
            "query": query,
            "source_fields": source_fields,
            "extra_filters": extra_filters,
            "constraint_filter": constraint_filter,
            "knn_field": knn_field,
            "k": knn_limit,  # k must be <= num_candidates
            "num_candidates": knn_num_candidates,
            "parse_hits": True,
            "add_region_info": add_region_info,
            "rank_method": "heads",  # Use heads, we'll re-rank after fusion
            "limit": knn_limit,
            "rank_top_k": knn_limit,
            "timeout": timeout,
            "verbose": verbose,
        }

        # Use ThreadPoolExecutor to run both searches in parallel
        word_search_res = {}
        knn_search_res = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both search tasks
            word_future = executor.submit(self.search, **word_search_params)
            knn_future = executor.submit(self.knn_search, **knn_search_params)

            # Wait for both to complete
            word_search_res = word_future.result()
            knn_search_res = knn_future.result()

        logger.hint("> Hybrid: Both searches completed")

        # Fuse results
        logger.hint("> Hybrid: Fusing results ...")
        word_hits = word_search_res.get("hits", [])
        knn_hits = knn_search_res.get("hits", [])

        # fusion_limit: target number of results to return after fusion
        # Use the maximum of rank_top_k and limit to ensure enough results
        # for subsequent ranking. The fill-and-supplement strategy in _rrf_fusion
        # will ensure we return exactly this many results when both sources have data.
        fusion_limit = max(rank_top_k, limit)

        if fusion_method == "rrf":
            fused_hits = self._rrf_fusion(
                word_hits, knn_hits, k=rrf_k, limit=fusion_limit
            )
        else:
            fused_hits = self._weighted_fusion(
                word_hits,
                knn_hits,
                word_weight=word_weight,
                vector_weight=vector_weight,
                limit=fusion_limit,
            )

        # Build result dict
        parse_res = {
            "timed_out": word_search_res.get("timed_out", False)
            or knn_search_res.get("timed_out", False),
            "total_hits": max(
                word_search_res.get("total_hits", 0),
                knn_search_res.get("total_hits", 0),
            ),
            "return_hits": len(fused_hits),
            "hits": fused_hits,
            "word_hits_count": len(word_hits),
            "knn_hits_count": len(knn_hits),
            "fusion_method": fusion_method,
        }

        # Apply final ranking
        # For hybrid search, RRF fusion already produces a well-ranked list
        # Tiered ranking adds stats/recency tie-breaking within relevance tiers
        if rank_method == "tiered":
            # Tiered ranking: relevance-first with stats/recency tie-breaking
            parse_res = self.hit_ranker.tiered_rank(
                parse_res, top_k=rank_top_k, relevance_field="hybrid_score"
            )
        elif rank_method == "relevance":
            # For hybrid search with relevance ranking, just take top-k by hybrid_score
            # Don't apply min_score filtering because hybrid_score has different scale
            parse_res = self.hit_ranker.heads(parse_res, top_k=rank_top_k)
            parse_res["rank_method"] = "relevance"
        elif rank_method == "rrf":
            parse_res = self.hit_ranker.rrf_rank(parse_res, top_k=rank_top_k)
        elif rank_method == "stats":
            parse_res = self.hit_ranker.stats_rank(parse_res, top_k=rank_top_k)
        else:  # "heads"
            parse_res = self.hit_ranker.heads(parse_res, top_k=rank_top_k)

        # Build return result
        return_res = {
            **parse_res,
            "query_info": query_info,
            "suggest_info": word_search_res.get("suggest_info", {}),
            "rewrite_info": word_search_res.get("rewrite_info", {}),
        }

        # Clean up non-jsonable items
        return_res["query_info"].pop("query_expr_tree", None)

        logger.exit_quiet(not verbose)
        return return_res

    def _rrf_fusion(
        self,
        word_hits: list[dict],
        knn_hits: list[dict],
        k: int = HYBRID_RRF_K,
        limit: int = SEARCH_LIMIT,
    ) -> list[dict]:
        """Fuse results using fill-and-supplement approach with RRF scoring.

        Strategy (fill-and-supplement):
        1. Take top N results from word search (by word rank)
        2. Take top N results from KNN search (by knn rank)
        3. Merge with deduplication (overlapping items count once)
        4. Fill remaining slots from unused results by RRF fusion score

        This ensures we always return `limit` results when both sources have
        sufficient data, prioritizing top results from each source.

        Score composition:
        - word_score: Original BM25 score from word search (typically 1-30)
        - knn_score: Original Hamming similarity from KNN search (0-1)
        - hybrid_score: Combined score preserving word score magnitude
        - rrf_rank_score: Pure RRF score for ranking
        - selection_tier: 'word_top', 'knn_top', or 'fusion_fill'

        Args:
            word_hits: Hits from word-based search.
            knn_hits: Hits from KNN search.
            k: RRF k parameter (default 60).
            limit: Maximum results to return.

        Returns:
            Fused and sorted list of hits with exactly `limit` results
            (when both sources have sufficient data).
        """
        # Priority slots for each source (half of limit each)
        priority_slots = limit // 2  # 200 for word, 200 for knn

        # Build bvid -> hit mapping and calculate scores
        hit_map = {}  # bvid -> hit dict
        rrf_scores = {}  # bvid -> rrf score
        word_scores_raw = {}  # bvid -> original word score
        knn_scores_raw = {}  # bvid -> original knn score
        word_ranks = {}  # bvid -> word rank
        knn_ranks = {}  # bvid -> knn rank

        # Get max scores for normalization
        word_max_score = (
            max((h.get("score", 0) or 0 for h in word_hits), default=1) or 1
        )
        knn_max_score = max((h.get("score", 0) or 0 for h in knn_hits), default=1) or 1

        # Weight for each source (balanced 0.5:0.5)
        word_weight = 0.5
        knn_weight = 0.5

        # Process word hits
        for rank, hit in enumerate(word_hits, start=1):
            bvid = hit.get("bvid")
            if not bvid:
                continue
            hit_map[bvid] = hit
            rrf_scores[bvid] = rrf_scores.get(bvid, 0) + word_weight / (k + rank)
            word_ranks[bvid] = rank
            word_scores_raw[bvid] = hit.get("score", 0) or 0

        # Process KNN hits
        for rank, hit in enumerate(knn_hits, start=1):
            bvid = hit.get("bvid")
            if not bvid:
                continue
            if bvid not in hit_map:
                hit_map[bvid] = hit
            rrf_scores[bvid] = rrf_scores.get(bvid, 0) + knn_weight / (k + rank)
            knn_ranks[bvid] = rank
            knn_scores_raw[bvid] = hit.get("score", 0) or 0

        # === Fill-and-Supplement Strategy ===
        selected_bvids = set()
        selection_tier = {}  # bvid -> tier name

        # Step 1: Take top N from word search
        word_top_bvids = sorted(
            [bvid for bvid in word_ranks.keys()], key=lambda x: word_ranks[x]
        )[:priority_slots]
        for bvid in word_top_bvids:
            selected_bvids.add(bvid)
            selection_tier[bvid] = "word_top"

        # Step 2: Take top N from KNN search (may overlap with word)
        knn_top_bvids = sorted(
            [bvid for bvid in knn_ranks.keys()], key=lambda x: knn_ranks[x]
        )[:priority_slots]
        for bvid in knn_top_bvids:
            if bvid not in selected_bvids:
                selected_bvids.add(bvid)
                selection_tier[bvid] = "knn_top"
            else:
                # Already selected from word, mark as both
                selection_tier[bvid] = "word_knn_top"

        # Step 3: Fill remaining slots from unused results by RRF score
        remaining_slots = limit - len(selected_bvids)
        if remaining_slots > 0:
            # Get all bvids not yet selected, sorted by RRF score
            remaining_bvids = sorted(
                [bvid for bvid in rrf_scores.keys() if bvid not in selected_bvids],
                key=lambda x: rrf_scores[x],
                reverse=True,
            )
            for bvid in remaining_bvids[:remaining_slots]:
                selected_bvids.add(bvid)
                selection_tier[bvid] = "fusion_fill"

        # Sort all selected results by RRF score for final ordering
        sorted_selected = sorted(
            selected_bvids, key=lambda x: rrf_scores.get(x, 0), reverse=True
        )

        # Build result list with meaningful hybrid_score
        fused_hits = []
        for bvid in sorted_selected:
            hit = hit_map[bvid]

            # Store raw scores
            w_score_raw = word_scores_raw.get(bvid, 0)
            k_score_raw = knn_scores_raw.get(bvid, 0)

            # Normalize scores to 0-1 range
            w_score_norm = w_score_raw / word_max_score if word_max_score > 0 else 0
            k_score_norm = k_score_raw / knn_max_score if knn_max_score > 0 else 0

            # Calculate hybrid_score that preserves word score magnitude
            # Formula: (normalized_word * 0.5 + normalized_knn * 0.5) * word_max_score
            # This keeps the score in a similar range as word search scores
            hybrid_relevance = word_weight * w_score_norm + knn_weight * k_score_norm
            hybrid_score = hybrid_relevance * word_max_score

            hit["word_score"] = round(w_score_raw, 4)
            hit["knn_score"] = round(k_score_raw, 4)
            hit["word_score_norm"] = round(w_score_norm, 4)
            hit["knn_score_norm"] = round(k_score_norm, 4)
            hit["hybrid_score"] = round(hybrid_score, 4)
            hit["rrf_rank_score"] = round(rrf_scores[bvid], 6)
            hit["word_rank"] = word_ranks.get(bvid)
            hit["knn_rank"] = knn_ranks.get(bvid)
            hit["selection_tier"] = selection_tier[bvid]
            hit["fusion_method"] = "rrf_fill"
            fused_hits.append(hit)

        return fused_hits

    def _weighted_fusion(
        self,
        word_hits: list[dict],
        knn_hits: list[dict],
        word_weight: float = HYBRID_WORD_WEIGHT,
        vector_weight: float = HYBRID_VECTOR_WEIGHT,
        limit: int = SEARCH_LIMIT,
    ) -> list[dict]:
        """Fuse results using weighted score combination.

        Args:
            word_hits: Hits from word-based search.
            knn_hits: Hits from KNN search.
            word_weight: Weight for word scores.
            vector_weight: Weight for vector scores.
            limit: Maximum results.

        Returns:
            Fused and sorted list of hits.
        """
        # Normalize weights
        total_weight = word_weight + vector_weight
        word_weight = word_weight / total_weight
        vector_weight = vector_weight / total_weight

        # Get max scores for normalization
        word_max_score = (
            max((h.get("score", 0) or 0 for h in word_hits), default=1) or 1
        )
        knn_max_score = max((h.get("score", 0) or 0 for h in knn_hits), default=1) or 1

        # Build mappings
        hit_map = {}  # bvid -> hit
        word_scores = {}  # bvid -> normalized word score
        knn_scores = {}  # bvid -> normalized knn score

        for hit in word_hits:
            bvid = hit.get("bvid")
            if not bvid:
                continue
            hit_map[bvid] = hit
            word_scores[bvid] = (hit.get("score", 0) or 0) / word_max_score

        for hit in knn_hits:
            bvid = hit.get("bvid")
            if not bvid:
                continue
            if bvid not in hit_map:
                hit_map[bvid] = hit
            knn_scores[bvid] = (hit.get("score", 0) or 0) / knn_max_score

        # Calculate weighted scores
        weighted_scores = {}
        for bvid in hit_map:
            w_score = word_scores.get(bvid, 0)
            k_score = knn_scores.get(bvid, 0)
            weighted_scores[bvid] = word_weight * w_score + vector_weight * k_score

        # Sort by weighted score
        sorted_bvids = sorted(
            weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True
        )

        # Build result list
        fused_hits = []
        for bvid in sorted_bvids[:limit]:
            hit = hit_map[bvid]
            hit["hybrid_score"] = weighted_scores[bvid]
            hit["word_score_norm"] = word_scores.get(bvid, 0)
            hit["knn_score_norm"] = knn_scores.get(bvid, 0)
            hit["fusion_method"] = "weighted"
            fused_hits.append(hit)

        return fused_hits
