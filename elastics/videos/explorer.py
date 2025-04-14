from copy import deepcopy
from tclogger import dict_get, get_by_threshold, get_now_ts, dict_to_str, to_digits
from tclogger import logstr, logger, brk
from typing import Generator, Union, Literal

from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.dsl.filter import QueryDslDictFilterMerger
from elastics.videos.constants import SEARCH_MATCH_FIELDS, SEARCH_BOOSTED_FIELDS
from elastics.videos.constants import SEARCH_MATCH_TYPE
from elastics.videos.constants import AGG_TIMEOUT
from elastics.videos.searcher import VideoSearcherV2


class VideoExplorer(VideoSearcherV2):
    def get_total_hits(self, agg_result: dict):
        return dict_get(agg_result, "hits.total.value", None)

    def get_stat_filter_by_threshold(
        self,
        agg_result: dict,
        field: Literal["score", "pubdate", "view", "favorite"],
        threshold: float = None,
        max_doc_count: int = None,
    ) -> dict:
        total_hits = self.get_total_hits(agg_result)
        if total_hits is None:
            logger.warn(f"× Not found total_hits")
            return None
        if max_doc_count is None and threshold is None:
            return {}
        if max_doc_count is not None:
            if total_hits <= max_doc_count:
                return {}
            threshold = (1 - max_doc_count / total_hits) * 100
        else:
            threshold = threshold
        stat_agg_dict = dict_get(agg_result, f"aggregations.{field}_ps.values", None)
        if stat_agg_dict is None:
            logger.warn(f"× Not found aggregation: {field}_ps")
            return None
        stat_key, stat_value = get_by_threshold(
            stat_agg_dict,
            threshold=threshold,
            direction="upper_bound",
            target="key",
        )
        if stat_key is None and stat_value is None:
            logger.warn(f"× Not found threshold: {threshold}")
            return None
        stat_filter = {"range": {f"stat.{field}": {"gte": int(stat_value)}}}
        return stat_filter

    def explore(
        self,
        # `query_dsl_dict` related params
        query: str,
        query_dsl_dict: dict = None,
        match_fields: list[str] = SEARCH_MATCH_FIELDS,
        match_type: str = SEARCH_MATCH_TYPE,
        extra_filters: list[dict] = [],
        suggest_info: dict = {},
        boost: bool = True,
        boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
        # timeout and verbose
        timeout: Union[int, float, str] = AGG_TIMEOUT,
        verbose: bool = False,
        # `explore` related params
        relevance_percentile: float = 50.0,
        most_relevant_count: int = 10000,
        relevant_recall_count: int = 10,
    ) -> Generator[dict, None, None]:
        """Multi-step explorative search, yield result at each step.

        Steps:
        1. Aggregate: Get overall statistics (percentiles of score, view, pubdate, etc.)
        2. Top documents:
            - most-relevant
            - most-popular
            - most-recent

        yields:
            dict: 'step', 'name', 'input', 'output'
        """
        logger.enter_quiet(not verbose)

        step_idx = 0
        # Step 0: Construct query_dsl_dict
        logger.hint("> [step 1] Aggregating ...")
        if query_dsl_dict is None:
            boosted_fields_params = {
                "match_fields": match_fields,
                "boost": boost,
                "boosted_fields": boosted_fields,
            }
            boosted_match_fields, boosted_date_fields = self.construct_boosted_fields(
                **boosted_fields_params
            )
            query_rewrite_dsl_params = {
                "query": query,
                "suggest_info": suggest_info,
                "boosted_match_fields": boosted_match_fields,
                "boosted_date_fields": boosted_date_fields,
                "match_type": match_type,
                "extra_filters": extra_filters,
                "verbose": verbose,
            }
            _, _, query_dsl_dict = self.get_info_of_query_rewrite_dsl(
                **query_rewrite_dsl_params
            )
        query_dsl_dict_yield = {
            "step": step_idx,
            "name": "init_query_dsl_dict",
            "input": query_rewrite_dsl_params,
            "output": query_dsl_dict,
            "comment": "",
        }
        yield query_dsl_dict_yield

        # Step 1: Aggregation
        step_idx += 1
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(f"> {step_str} Aggregating ...")
        agg_result = self.agg(
            query_dsl_dict=query_dsl_dict, timeout=timeout, verbose=verbose
        )
        agg_yield = {
            "step": step_idx,
            "name": "aggregation",
            "input": query_dsl_dict,
            "output": agg_result,
            "comment": "",
        }
        yield agg_yield

        filter_merger = QueryDslDictFilterMerger()

        # Step 2: Most-relevant docs
        #   - with view filter
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(
            f"> {step_str} Top relevant docs based on relevance and view filter"
        )
        view_filter = self.get_stat_filter_by_threshold(
            agg_result,
            field="view",
            threshold=relevance_percentile,
            max_doc_count=most_relevant_count,
        )
        relevant_extra_filters = deepcopy(extra_filters)
        if view_filter is None:
            logger.warn("  × No view filter")
        else:
            logger.hint("  > View filter:")
            logger.okay(dict_to_str(view_filter), indent=2)
            if view_filter:
                relevant_extra_filters.append(view_filter)

        relevant_search_params = {
            "query": query,
            "suggest_info": suggest_info,
            "extra_filters": relevant_extra_filters,
            "use_script_score": True,
            "limit": 3,
            "verbose": verbose,
        }
        relevant_search_res = self.search(**relevant_search_params)
        relevant_search_yield = {
            "step": step_idx,
            "name": "most_relevant_search",
            "input": relevant_search_params,
            "output": relevant_search_res,
            "comment": "",
        }
        yield relevant_search_yield
