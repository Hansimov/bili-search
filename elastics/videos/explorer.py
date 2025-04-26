import json

from copy import deepcopy
from tclogger import dict_get, get_by_threshold, get_now_ts, dict_to_str, to_digits
from tclogger import logstr, logger, brk
from typing import Generator, Union, Literal

from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.dsl.filter import QueryDslDictFilterMerger
from elastics.videos.constants import SEARCH_MATCH_FIELDS, EXPLORE_BOOSTED_FIELDS
from elastics.videos.constants import SEARCH_MATCH_TYPE
from elastics.videos.constants import AGG_TIMEOUT, EXPLORE_TIMEOUT
from elastics.videos.searcher import VideoSearcherV2

STEP_ZH_NAMES = {
    "init": {
        "name_zh": "初始化",
    },
    "construct_query_dsl_dict": {
        "name_zh": "解析查询",
        "output_type": "info",
    },
    "aggregation": {
        "name_zh": "聚合",
        "output_type": "info",
    },
    "most_relevant_search": {
        "name_zh": "相关性排序",
        "output_type": "hits",
    },
    "most_popular_search": {
        "name_zh": "热门排序",
        "output_type": "hits",
    },
    "group_hits_by_owner": {
        "name_zh": "UP主聚合",
        "output_type": "info",
    },
}


class VideoExplorer(VideoSearcherV2):
    def get_total_hits(self, agg_result: dict):
        return dict_get(agg_result, "hits.total.value", None)

    def get_stat_filter_by_threshold(
        self,
        agg_result: dict,
        field: Literal["view", "like", "coin", "favorite"],
        threshold: float = None,
        max_doc_count: int = None,
        res_format: Literal["dict", "tuple"] = "dict",
    ) -> Union[dict, tuple]:
        total_hits = self.get_total_hits(agg_result)
        if total_hits is None:
            logger.warn(f"× Not found total_hits")
            return None
        if max_doc_count is None and threshold is None:
            return {}
        if threshold is None:
            threshold = 0
        if max_doc_count is not None:
            if total_hits <= max_doc_count:
                return {}
            threshold = max((1 - max_doc_count / total_hits) * 100, threshold)
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
        if res_format == "tuple":
            return stat_key, stat_value
        else:
            stat_filter = {"range": {f"stat.{field}": {"gte": int(stat_value)}}}
            return stat_filter

    def get_score_threshold_by_ratio(
        self,
        agg_result: dict,
        ratio: float = None,
        max_doc_count: int = None,
    ) -> float:
        """ratio: 0.0 - 1.0, means should be greater than ratio * max_score
        for example, if ratio=0.75, means should be greater than 75% of max_score
        """
        field = "score"
        total_hits = self.get_total_hits(agg_result)
        if total_hits is None:
            logger.warn(f"× Not found total_hits")
            return None
        if max_doc_count is None and ratio is None:
            return 0
        score_agg_dict = dict_get(agg_result, f"aggregations.{field}_ps.values", None)
        if score_agg_dict is None:
            logger.warn(f"× Not found aggregation: {field}_ps")
            return None
        max_score = max(score_agg_dict.values())
        min_score = min(score_agg_dict.values())
        # ratio could be min/max of the constraints by max_doc_count and ratio,
        # - if use min, could ensure there are enough-but-not-too-much candidate docs
        # - if use max, would get less candidates, especially useful when the first recall docs are too many
        if ratio is None:
            ratio = 0
        if max_doc_count is not None and max_doc_count < total_hits:
            doc_count_percentile = max_doc_count / total_hits * 100
            percent_by_count, _ = get_by_threshold(
                score_agg_dict,
                threshold=doc_count_percentile,
                direction="upper_bound",
                target="key",
            )
            ratio_by_count = percent_by_count / 100
            ratio = max(ratio, ratio_by_count)
        score_threshold = round(max(max_score * ratio, min_score), 4)
        return score_threshold

    def format_result(
        self, res: dict, res_format: Literal["json", "str"] = "json"
    ) -> Union[dict, str]:
        if res_format == "str":
            # used for EventSourceResponse
            return json.dumps(res)
        else:
            # used for normal response
            return res

    def update_step_output(self, step_yield: dict, step_output: dict = None) -> dict:
        step_yield["status"] = "finished"
        step_yield["output"] = step_output
        return step_yield

    def group_hits_by_owner(
        self,
        search_res: dict,
        sort_field: Literal[
            "total_view", "doc_count", "total_sort_score"
        ] = "total_sort_score",
        limit: int = 10,
    ) -> dict:
        group_res = {}
        for hit in search_res.get("hits", []):
            owner_name = dict_get(hit, "owner.name", None)
            owner_mid = dict_get(hit, "owner.mid", None)
            pubdate = dict_get(hit, "pubdate", 0)
            view = dict_get(hit, "stat.view", 0)
            sort_score = dict_get(hit, "sort_score", 0)
            if owner_mid is None or owner_name is None:
                continue
            item = group_res.get(owner_mid, None)
            if item is None:
                group_res[owner_mid] = {
                    "owner_name": owner_name,
                    "latest_pubdate": pubdate,
                    "total_view": view,
                    "total_sort_score": sort_score,
                    "doc_count": 0,
                    "hits": [],
                }
            else:
                latest_pubdate = group_res[owner_mid]["latest_pubdate"]
                if pubdate > latest_pubdate:
                    group_res[owner_mid]["latest_pubdate"] = pubdate
                    group_res[owner_mid]["owner_name"] = owner_name
                total_view = group_res[owner_mid]["total_view"]
                total_sort_score = group_res[owner_mid]["total_sort_score"]
                group_res[owner_mid]["total_view"] = total_view + view
                group_res[owner_mid]["total_sort_score"] = total_sort_score + sort_score
            group_res[owner_mid]["hits"].append(hit)
            group_res[owner_mid]["doc_count"] = len(group_res[owner_mid]["hits"])
        # sort by sort_field, and limit to top N
        sorted_items = sorted(
            group_res.items(), key=lambda item: item[1][sort_field], reverse=True
        )[:limit]
        group_res = dict(sorted_items)
        return group_res

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
        boosted_fields: dict = EXPLORE_BOOSTED_FIELDS,
        verbose: bool = False,
        # `explore` related params
        view_percent_threshold: float = 25.0,
        score_ratio_threshold: float = 0.5,
        max_count_by_view: int = 10000,
        max_count_by_score: int = 10000,
        relevant_search_limit: int = 400,
        group_owner_limit: int = 10,
        res_format: Literal["json", "str"] = "json",
    ) -> Generator[dict, None, None]:
        """Multi-step explorative search, yield result at each step.

        Steps:
        1. Aggregate: Get overall statistics (percentiles of score, view, pubdate, etc.)
        2. Top documents:
            - most-relevant
            - most-popular
            - most-recent

        yields:
            dict: 'step', 'name', 'name_zh', 'status', 'input', 'output_type', 'output', 'comment'
        """
        logger.enter_quiet(not verbose)

        # Step 0: Construct query_dsl_dict
        step_idx = 0
        step_name = "construct_query_dsl_dict"
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
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "finished",
            "input": query_rewrite_dsl_params,
            "output_type": STEP_ZH_NAMES[step_name]["output_type"],
            "output": query_dsl_dict,
            "comment": "",
        }
        yield self.format_result(query_dsl_dict_yield, res_format=res_format)

        # Step 1: Aggregation
        step_idx += 1
        step_name = "aggregation"
        agg_yield = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "running",
            "input": query_dsl_dict,
            "output": {},
            "output_type": "info",
            "comment": "",
        }
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(f"> {step_str} Aggregating ...")
        yield self.format_result(agg_yield, res_format=res_format)
        agg_result = self.agg(
            query_dsl_dict=query_dsl_dict, timeout=AGG_TIMEOUT, verbose=verbose
        )
        self.update_step_output(agg_yield, step_output=agg_result)
        yield self.format_result(agg_yield, res_format=res_format)

        # Step 2: Most-relevant docs
        #   - with view filter
        step_idx += 1
        step_name = "most_relevant_search"
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(
            f"> {step_str} Top relevant docs based on relevance and view filter"
        )
        view_filter = self.get_stat_filter_by_threshold(
            agg_result,
            field="view",
            threshold=view_percent_threshold,
            max_doc_count=max_count_by_view,
        )
        relevant_extra_filters = deepcopy(extra_filters)
        for stat_filter in []:
            if stat_filter is None:
                logger.warn(f"  × No stat_filter: {stat_filter}")
            else:
                logger.hint("  > stat_filter:")
                logger.okay(dict_to_str(stat_filter), indent=2)
                if stat_filter:
                    relevant_extra_filters.append(stat_filter)
        score_threshold = self.get_score_threshold_by_ratio(
            agg_result,
            ratio=score_ratio_threshold,
            max_doc_count=max_count_by_score,
        )
        relevant_search_params = {
            "query": query,
            "suggest_info": suggest_info,
            "extra_filters": relevant_extra_filters,
            "use_script_score": True,
            "score_threshold": score_threshold,
            "limit": relevant_search_limit,
            "timeout": EXPLORE_TIMEOUT,
            "verbose": verbose,
        }
        relevant_search_yield = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "running",
            "input": relevant_search_params,
            "output": {},
            "output_type": "hits",
            "comment": "",
        }
        yield self.format_result(relevant_search_yield, res_format=res_format)
        relevant_search_res = self.search(**relevant_search_params)
        self.update_step_output(relevant_search_yield, step_output=relevant_search_res)
        yield self.format_result(relevant_search_yield, res_format=res_format)

        # Step 3: Group hits by owner
        step_idx += 1
        step_name = "group_hits_by_owner"
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(f"> {step_str} Group hits by owner, sort by sum of view")
        group_hits_by_owner_params = {
            "search_res": relevant_search_res,
            "limit": group_owner_limit,
        }
        group_hits_by_owner_yield = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "running",
            "input": {"limit": group_owner_limit},
            "output": {},
            "output_type": STEP_ZH_NAMES[step_name]["output_type"],
            "comment": "",
        }
        yield self.format_result(group_hits_by_owner_yield, res_format=res_format)
        group_res = self.group_hits_by_owner(**group_hits_by_owner_params)
        self.update_step_output(group_hits_by_owner_yield, step_output=group_res)
        yield self.format_result(group_hits_by_owner_yield, res_format=res_format)
