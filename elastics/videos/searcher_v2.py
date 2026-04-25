from copy import deepcopy
from sedb.elastic import ElasticOperator
from sedb.mongo import MongoOperator
from tclogger import logger, dict_to_str, get_now, tcdatetime
from typing import Union, Literal

from configs.envs import MONGO_ENVS, SECRETS, ELASTIC_PRO_ENVS
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from dsl.rewrite import DslExprRewriter
from dsl.elastic import DslExprToElasticConverter
from dsl.filter import QueryDslDictFilterMerger
from elastics.owners.searcher import OwnerSearcher
from elastics.relations import RelationsClient
from elastics.structure import get_highlight_settings, construct_boosted_fields
from elastics.structure import set_min_score, set_terminate_after
from elastics.structure import set_timeout, set_profile
from elastics.videos.constants import ELASTIC_VIDEOS_PRO_INDEX
from elastics.videos.intent.owner_query import OwnerQueryIntentResolver
from elastics.videos.query.understanding import VideoQueryUnderstanding
from elastics.videos.results.reranking import rerank_focused_title_hits
from elastics.videos.search_basic import VideoSearchBasicMixin
from elastics.videos.search_filters import VideoSearchFilterMixin
from elastics.videos.search_lookup import VideoSearchLookupMixin
from elastics.videos.search_retry import VideoSearchRetryMixin
from elastics.videos.search_vector import VideoSearchVectorMixin
from elastics.videos.constants import SEARCH_REQUEST_TYPE, SEARCH_REQUEST_TYPE_DEFAULT
from elastics.videos.constants import SOURCE_FIELDS, DOC_EXCLUDED_SOURCE_FIELDS
from elastics.videos.constants import SEARCH_MATCH_FIELDS, SEARCH_BOOSTED_FIELDS
from elastics.videos.constants import SUGGEST_MATCH_FIELDS, SUGGEST_BOOSTED_FIELDS
from elastics.videos.constants import DATE_MATCH_FIELDS
from elastics.videos.constants import MATCH_TYPE, MATCH_BOOL, MATCH_OPERATOR
from elastics.videos.constants import SEARCH_MATCH_TYPE, SUGGEST_MATCH_TYPE
from elastics.videos.constants import SEARCH_MATCH_BOOL, SEARCH_MATCH_OPERATOR
from elastics.videos.constants import SUGGEST_MATCH_BOOL, SUGGEST_MATCH_OPERATOR
from elastics.videos.constants import SEARCH_DETAIL_LEVELS, SUGGEST_DETAIL_LEVELS
from elastics.videos.constants import SEARCH_LIMIT, SUGGEST_LIMIT, AGG_TOP_K
from elastics.videos.constants import SEARCH_TIMEOUT, SUGGEST_TIMEOUT
from elastics.videos.constants import NO_HIGHLIGHT_REDUNDANCE_RATIO
from elastics.videos.constants import USE_SCRIPT_SCORE
from elastics.videos.constants import TRACK_TOTAL_HITS, IS_HIGHLIGHT
from elastics.videos.constants import AGG_TIMEOUT, AGG_PERCENTS
from elastics.videos.constants import AGG_SORT_FIELD, AGG_SORT_ORDER
from elastics.videos.constants import TERMINATE_AFTER
from elastics.videos.results.parsing import VideoHitsParser, SuggestInfoParser
from elastics.es_logger import get_es_debug_logger
from converters.embed.embed_client import TextEmbedClient, get_embed_client
from dsl.fields.word import override_auto_require_short_han_exact

# Import from ranks module (use direct submodule imports)
from ranks.constants import (
    RANK_METHOD_TYPE,
    RANK_METHOD,
    RANK_TOP_K,
)
from ranks.ranker import VideoHitsRanker


class VideoSearcherV2(
    VideoSearchBasicMixin,
    VideoSearchLookupMixin,
    VideoSearchFilterMixin,
    VideoSearchRetryMixin,
    VideoSearchVectorMixin,
):
    def __init__(
        self,
        index_name: str = ELASTIC_VIDEOS_PRO_INDEX,
        elastic_env_name: str = None,
        mongo_env_name: str = None,
        owner_searcher: OwnerSearcher | None = None,
        relations_client: RelationsClient | None = None,
    ):
        """
        - index_name:
            name of elastic index for videos
            - example: "bili_videos_dev4"
        - elastic_env_name:
            name of elastic envs in secrets.json
            - example: "elastic", "elastic_dev"
        - mongo_env_name:
            name of mongo envs in secrets.json
            - example: "mongo"
        """
        self.index_name = index_name
        self.elastic_env_name = elastic_env_name
        self.mongo_env_name = mongo_env_name
        self._owner_searcher = owner_searcher
        self._relations_client = relations_client
        self.init_processors()

    def init_processors(self):
        if self.elastic_env_name:
            elastic_envs = SECRETS[self.elastic_env_name]
        else:
            elastic_envs = ELASTIC_PRO_ENVS
        if self.mongo_env_name:
            mongo_envs = SECRETS[self.mongo_env_name]
        else:
            mongo_envs = MONGO_ENVS
        self.es = ElasticOperator(elastic_envs, connect_cls=self.__class__)
        self.mongo = MongoOperator(mongo_envs, connect_cls=self.__class__)
        self.hit_parser = VideoHitsParser()
        self.hit_ranker = VideoHitsRanker()
        self.query_rewriter = DslExprRewriter()
        self.elastic_converter = DslExprToElasticConverter()
        self.filter_merger = QueryDslDictFilterMerger()
        self.suggest_parser = SuggestInfoParser("v2")
        # Lazy-initialized embed client for KNN search
        self._embed_client = None
        self._query_understanding = None
        self._owner_query_intent_resolver = None

    @property
    def embed_client(self) -> TextEmbedClient:
        """Lazy-initialized embed client for KNN search.

        Uses the singleton instance so that the app-level warmup and
        keepalive are shared, avoiding redundant 60s health checks.
        """
        if self._embed_client is None:
            self._embed_client = get_embed_client()
        return self._embed_client

    @property
    def owner_searcher(self) -> OwnerSearcher:
        owner_searcher = getattr(self, "_owner_searcher", None)
        if owner_searcher is None:
            self._owner_searcher = OwnerSearcher(
                index_name=self.index_name,
                elastic_env_name=self.elastic_env_name,
            )
        return self._owner_searcher

    @property
    def relations_client(self) -> RelationsClient:
        relations_client = getattr(self, "_relations_client", None)
        if relations_client is None:
            self._relations_client = RelationsClient(
                index_name=self.index_name,
                elastic_env_name=self.elastic_env_name,
            )
        return self._relations_client

    @property
    def query_understanding(self) -> VideoQueryUnderstanding:
        query_understanding = getattr(self, "_query_understanding", None)
        if query_understanding is None:
            relations_client = getattr(self, "_relations_client", None)
            if relations_client is None and hasattr(self, "elastic_env_name"):
                relations_client = self.relations_client
            query_understanding = VideoQueryUnderstanding(
                query_rewriter=self.query_rewriter,
                owner_searcher=self.owner_searcher,
                relations_client=relations_client,
            )
            self._query_understanding = query_understanding
        return query_understanding

    @property
    def owner_query_intent_resolver(self) -> OwnerQueryIntentResolver:
        resolver = getattr(self, "_owner_query_intent_resolver", None)
        if resolver is None:
            resolver = OwnerQueryIntentResolver(owner_searcher=self.owner_searcher)
            self._owner_query_intent_resolver = resolver
        return resolver

    def submit_to_es(self, body: dict, context: str = None) -> dict:
        try:
            res = self.es.client.search(index=self.index_name, body=body)
            res_dict = res.body
        except Exception as e:
            error_msg = str(e)
            logger.warn(f"× ES error [{context or 'unknown'}]: {error_msg}")
            # Log detailed error info to logs/es.log
            es_logger = get_es_debug_logger()
            es_logger.log_error(
                request_body=body,
                error=e,
                index_name=self.index_name,
                context=context,
            )
            # Return a structured error response instead of empty dict
            # so that hit_parser can distinguish ES errors from timeouts
            res_dict = {
                "took": 0,
                "timed_out": False,
                "_es_error": True,
                "_es_error_msg": error_msg,
                "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
            }
        return res_dict

    def search(
        self,
        query: str,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        source_fields: list[str] = SOURCE_FIELDS,
        match_type: MATCH_TYPE = SEARCH_MATCH_TYPE,
        match_bool: MATCH_BOOL = SEARCH_MATCH_BOOL,
        match_operator: MATCH_OPERATOR = SEARCH_MATCH_OPERATOR,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        request_type: SEARCH_REQUEST_TYPE = SEARCH_REQUEST_TYPE_DEFAULT,
        parse_hits: bool = True,
        drop_no_highlights: bool = False,
        add_region_info: bool = True,
        add_highlights_info: bool = True,
        is_explain: bool = False,
        is_profile: bool = False,
        is_highlight: bool = IS_HIGHLIGHT,
        boost: bool = True,
        boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
        combined_fields_list: list[list[str]] = [],
        use_script_score: bool = USE_SCRIPT_SCORE,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD,
        score_threshold: float = None,
        use_pinyin: bool = False,
        detail_level: int = -1,
        detail_levels: dict = SEARCH_DETAIL_LEVELS,
        limit: int = SEARCH_LIMIT,
        rank_top_k: int = RANK_TOP_K,
        terminate_after: int = TERMINATE_AFTER,
        timeout: Union[int, float, str] = SEARCH_TIMEOUT,
        verbose: bool = False,
        _allow_short_han_retry: bool = True,
        _allow_owner_context_retry: bool = True,
    ) -> Union[dict, list[dict]]:
        logger.enter_quiet(not verbose)
        effective_query = str(query or "")
        effective_suggest_info = dict(suggest_info or {})
        semantic_rewrite_info: dict = {}
        query_focus_info: dict = {}
        owner_context_query = ""
        query_understanding = self.query_understanding
        if request_type == SEARCH_REQUEST_TYPE_DEFAULT:
            prepared_query = query_understanding.prepare_search_query(
                effective_query,
                suggest_info=suggest_info,
                request_type=request_type,
            )
            effective_query = prepared_query.effective_query
            query_focus_info = prepared_query.query_focus_info
            semantic_rewrite_info = prepared_query.semantic_rewrite_info
            if prepared_query.suggest_info and not effective_suggest_info:
                effective_suggest_info = prepared_query.suggest_info

        effective_extra_filters = list(extra_filters)
        owner_intent_query = effective_query
        owner_resolution = query_understanding.resolve_owner_intent(owner_intent_query)
        owner_resolution_from_rewritten_query = True
        if (
            not owner_resolution.filters
            and semantic_rewrite_info.get("applied")
            and effective_query != str(query or "")
        ):
            fallback_owner_resolution = query_understanding.resolve_owner_intent(query)
            if fallback_owner_resolution.filters:
                owner_resolution = fallback_owner_resolution
                owner_resolution_from_rewritten_query = False
        owner_intent_info = owner_resolution.info
        owner_intent_filters = owner_resolution.filters
        if owner_intent_filters:
            effective_extra_filters.extend(owner_intent_filters)
            if _allow_owner_context_retry and owner_resolution_from_rewritten_query:
                owner_context_query = owner_resolution.context_query
                if owner_context_query:
                    effective_query = owner_context_query

        # Check if there are actual search keywords
        # If no keywords, fall back to filter-only search
        if not self.has_search_keywords(query):
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

        # init params by detail_level
        if detail_level in detail_levels:
            match_detail = detail_levels[detail_level]
            match_type = match_detail["match_type"]
            match_bool = match_detail["bool"]
            match_operator = match_detail.get("operator", "or")
            use_pinyin = match_detail.get("pinyin", use_pinyin)
            extra_filters = match_detail.get("filters", extra_filters)
            timeout = match_detail.get("timeout", timeout)
        # construct boosted fields
        boosted_match_fields, boosted_date_fields = construct_boosted_fields(
            match_fields=match_fields,
            boost=boost,
            boosted_fields=boosted_fields,
            use_pinyin=use_pinyin,
        )
        query_rewrite_dsl_params = {
            "query": effective_query,
            "suggest_info": effective_suggest_info,
            "boosted_match_fields": boosted_match_fields,
            "boosted_date_fields": boosted_date_fields,
            "match_type": match_type,
            "extra_filters": effective_extra_filters,
        }
        query_info, rewrite_info, query_dsl_dict = self.get_info_of_query_rewrite_dsl(
            **query_rewrite_dsl_params
        )
        # construct search_body
        search_body_params = {
            "query_dsl_dict": query_dsl_dict,
            "match_fields": boosted_match_fields,
            "source_fields": source_fields,
            "drop_no_highlights": drop_no_highlights,
            "is_explain": is_explain,
            "is_profile": is_profile,
            "is_highlight": is_highlight,
            "use_script_score": use_script_score,
            "score_threshold": score_threshold,
            "limit": limit,
            "terminate_after": terminate_after,
            "timeout": timeout,
        }
        search_body = self.construct_search_body(**search_body_params)
        logger.mesg(
            dict_to_str(search_body, add_quotes=True), indent=2, verbose=verbose
        )
        # submit search_body to es client
        es_res_dict = self.submit_to_es(search_body, context="search")
        # parse results
        if parse_hits:
            parse_res = self.hit_parser.parse(
                query_info,
                match_fields=match_fields,
                res_dict=es_res_dict,
                request_type=request_type,
                drop_no_highlights=drop_no_highlights,
                add_region_info=add_region_info,
                add_highlights_info=add_highlights_info,
                match_type=match_type,
                match_operator=match_operator,
                detail_level=detail_level,
                limit=limit,
                verbose=verbose,
            )
            if rank_method == "tiered":
                # Tiered ranking with word-search score as relevance
                parse_res = self.hit_ranker.tiered_rank(
                    parse_res, top_k=rank_top_k, relevance_field="score"
                )
            elif rank_method == "rrf":
                parse_res = self.hit_ranker.rrf_rank(parse_res, top_k=rank_top_k)
            elif rank_method == "stats":
                parse_res = self.hit_ranker.stats_rank(parse_res, top_k=rank_top_k)
            else:  # "heads"
                parse_res = self.hit_ranker.heads(parse_res, top_k=rank_top_k)
        else:
            parse_res = es_res_dict

        title_rerank_info: dict = {}
        if parse_hits and isinstance(parse_res, dict):
            reranked_hits, rerank_info = rerank_focused_title_hits(
                list(parse_res.get("hits") or []),
                query=query_focus_info.get("applied_query") or effective_query,
                focus_applied=bool(query_focus_info.get("applied")),
                relation_rewritten=bool(
                    semantic_rewrite_info.get("relation_rewritten")
                ),
            )
            if rerank_info.applied:
                parse_res["hits"] = reranked_hits
                title_rerank_info = rerank_info.to_dict()
        # rewrite_by_suggest, only apply for "suggest" request_type
        return_res = self.rewrite_by_suggest(
            query_info,
            suggest_info=effective_suggest_info,
            rewrite_info=rewrite_info,
            request_type=request_type,
            return_res=parse_res,
        )
        return_res = self.post_process_return_res(parse_res)
        # Sanitize search_body to reduce network payload (removes large terms arrays)
        return_res["search_body"] = self.sanitize_search_body_for_client(search_body)
        return_res["intent_info"] = owner_intent_info
        if query_focus_info:
            return_res["query_focus_info"] = query_focus_info
        if semantic_rewrite_info:
            semantic_rewrite_info = dict(semantic_rewrite_info)
            semantic_rewrite_info["input_query"] = str(query or "")
            if query_focus_info.get("applied"):
                semantic_rewrite_info["focus_query"] = query_focus_info.get(
                    "applied_query",
                    effective_query,
                )
            semantic_rewrite_info["applied_query"] = (
                list(rewrite_info.get("rewrited_word_exprs") or [effective_query])[0]
                if rewrite_info.get("rewrited_word_exprs")
                else semantic_rewrite_info.get("applied_query", effective_query)
            )
            return_res["semantic_rewrite_info"] = semantic_rewrite_info
        if title_rerank_info:
            return_res["title_rerank_info"] = title_rerank_info

        recall_retry_query = str(query_info.get("words_expr") or effective_query)
        score_cliff_info = self._apply_model_code_score_cliff_filter(
            return_res,
            query=recall_retry_query,
            relation_rewritten=bool(semantic_rewrite_info.get("relation_rewritten")),
        )
        if score_cliff_info:
            return_res["score_cliff_filter_info"] = score_cliff_info

        if (
            owner_context_query
            and owner_intent_filters
            and _allow_owner_context_retry
            and not return_res.get("total_hits", 0)
        ):
            logger.warn(
                "> Retry spaced owner-intent search with original query",
                verbose=verbose,
            )
            retry_res = self.search(
                query=query,
                match_fields=match_fields,
                source_fields=source_fields,
                match_type=match_type,
                match_bool=match_bool,
                match_operator=match_operator,
                extra_filters=extra_filters,
                suggest_info=suggest_info,
                request_type=request_type,
                parse_hits=parse_hits,
                drop_no_highlights=drop_no_highlights,
                add_region_info=add_region_info,
                add_highlights_info=add_highlights_info,
                is_explain=is_explain,
                is_profile=is_profile,
                is_highlight=is_highlight,
                boost=boost,
                boosted_fields=boosted_fields,
                combined_fields_list=combined_fields_list,
                use_script_score=use_script_score,
                rank_method=rank_method,
                score_threshold=score_threshold,
                use_pinyin=use_pinyin,
                detail_level=detail_level,
                detail_levels=detail_levels,
                limit=limit,
                rank_top_k=rank_top_k,
                terminate_after=terminate_after,
                timeout=timeout,
                verbose=verbose,
                _allow_short_han_retry=_allow_short_han_retry,
                _allow_owner_context_retry=False,
            )
            retry_info = dict(retry_res.get("retry_info") or {})
            retry_info["relaxed_spaced_owner_context"] = True
            retry_res["retry_info"] = retry_info
            logger.exit_quiet(not verbose)
            return retry_res

        if self._should_retry_without_auto_exact_segments(
            recall_retry_query,
            total_hits=return_res.get("total_hits", 0),
            allow_retry=_allow_short_han_retry,
        ):
            logger.warn(
                "> Retry word search with only model-code auto-exact segments",
                verbose=verbose,
            )
            retry_limit = self._get_embedding_denoise_candidate_limit(limit)
            retry_rank_top_k = max(rank_top_k, retry_limit)
            with override_auto_require_short_han_exact("model_code"):
                retry_res = self.search(
                    query=query,
                    match_fields=match_fields,
                    source_fields=source_fields,
                    match_type=match_type,
                    match_bool=match_bool,
                    match_operator=match_operator,
                    extra_filters=extra_filters,
                    suggest_info=suggest_info,
                    request_type=request_type,
                    parse_hits=parse_hits,
                    drop_no_highlights=drop_no_highlights,
                    add_region_info=add_region_info,
                    add_highlights_info=add_highlights_info,
                    is_explain=is_explain,
                    is_profile=is_profile,
                    is_highlight=is_highlight,
                    boost=boost,
                    boosted_fields=boosted_fields,
                    combined_fields_list=combined_fields_list,
                    use_script_score=use_script_score,
                    rank_method=rank_method,
                    score_threshold=score_threshold,
                    use_pinyin=use_pinyin,
                    detail_level=detail_level,
                    detail_levels=detail_levels,
                    limit=retry_limit,
                    rank_top_k=retry_rank_top_k,
                    terminate_after=terminate_after,
                    timeout=timeout,
                    verbose=verbose,
                    _allow_short_han_retry=False,
                    _allow_owner_context_retry=_allow_owner_context_retry,
                )
            retry_res = self._apply_embedding_denoise_to_retry_result(
                retry_res,
                denoise_query=recall_retry_query,
                original_limit=limit,
                verbose=verbose,
            )
            retry_info = dict(retry_res.get("retry_info") or {})
            retry_info["relaxed_auto_exact_segments"] = True
            retry_info["kept_model_code_exact"] = True
            retry_res["retry_info"] = retry_info
            score_cliff_info = self._apply_model_code_score_cliff_filter(
                retry_res,
                query=recall_retry_query,
                relation_rewritten=bool(
                    (retry_res.get("semantic_rewrite_info") or {}).get(
                        "relation_rewritten"
                    )
                ),
            )
            if score_cliff_info:
                retry_res["score_cliff_filter_info"] = score_cliff_info
            logger.exit_quiet(not verbose)
            return retry_res

        if self._should_retry_without_short_han_exact(
            query,
            total_hits=return_res.get("total_hits", 0),
            allow_retry=_allow_short_han_retry,
        ):
            logger.warn(
                "> Retry word search without auto-exact short Han segments",
                verbose=verbose,
            )
            with override_auto_require_short_han_exact("first"):
                retry_res = self.search(
                    query=query,
                    match_fields=match_fields,
                    source_fields=source_fields,
                    match_type=match_type,
                    match_bool=match_bool,
                    match_operator=match_operator,
                    extra_filters=extra_filters,
                    suggest_info=suggest_info,
                    request_type=request_type,
                    parse_hits=parse_hits,
                    drop_no_highlights=drop_no_highlights,
                    add_region_info=add_region_info,
                    add_highlights_info=add_highlights_info,
                    is_explain=is_explain,
                    is_profile=is_profile,
                    is_highlight=is_highlight,
                    boost=boost,
                    boosted_fields=self._get_relaxed_short_han_retry_boosted_fields(
                        boosted_fields
                    ),
                    combined_fields_list=combined_fields_list,
                    use_script_score=use_script_score,
                    rank_method=rank_method,
                    score_threshold=score_threshold,
                    use_pinyin=use_pinyin,
                    detail_level=detail_level,
                    detail_levels=detail_levels,
                    limit=limit,
                    rank_top_k=rank_top_k,
                    terminate_after=terminate_after,
                    timeout=timeout,
                    verbose=verbose,
                    _allow_short_han_retry=False,
                    _allow_owner_context_retry=_allow_owner_context_retry,
                )
            retry_info = dict(retry_res.get("retry_info") or {})
            retry_info["relaxed_short_han_exact"] = True
            retry_res["retry_info"] = retry_info
            logger.exit_quiet(not verbose)
            return retry_res

        logger.exit_quiet(not verbose)
        return return_res
