import json

from copy import deepcopy
from tclogger import dict_get, get_by_threshold
from tclogger import logstr, logger, brk
from typing import Union, Literal

from converters.dsl.fields.bvid import bvids_to_filter
from elastics.videos.constants import SEARCH_MATCH_FIELDS, EXPLORE_BOOSTED_FIELDS
from elastics.videos.constants import SEARCH_MATCH_TYPE
from elastics.videos.constants import RANK_METHOD_TYPE, RANK_METHOD_DEFAULT
from elastics.videos.constants import AGG_TIMEOUT, EXPLORE_TIMEOUT
from elastics.videos.constants import TERMINATE_AFTER
from elastics.structure import construct_boosted_fields
from elastics.videos.searcher_v2 import VideoSearcherV2

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
        "name_zh": "搜索相关",
        "output_type": "hits",
    },
    "most_popular_search": {
        "name_zh": "搜索热门",
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
        try:
            max_score = max(score_agg_dict.values())
            min_score = min(score_agg_dict.values())
        except:
            return None
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
            # ratio could be min/max of the constraints by max_doc_count and ratio,
            # - if use min, could ensure there are enough-but-not-too-much candidate docs
            # - if use max, would get less candidates, especially useful when the first recall docs are too many
            ratio = max(ratio, ratio_by_count)
        score_threshold = round(max(max_score * ratio, min_score), 4)
        return score_threshold

    def set_total_hits(self, agg_res: dict, search_res: dict) -> None:
        agg_total_hits = self.get_total_hits(agg_res)
        search_total_hits = search_res.get("total_hits", 0)
        if agg_total_hits is not None and search_total_hits is not None:
            if search_total_hits == TERMINATE_AFTER:
                search_res["total_hits"] = agg_total_hits
        return search_res

    def format_result(
        self, res: dict, res_format: Literal["json", "str"] = "json"
    ) -> Union[dict, str]:
        if res_format == "str":
            # used for EventSourceResponse
            return json.dumps(res)
        else:
            # used for normal response
            return res

    def update_step_output(
        self, step_yield: dict, step_output: dict = None, field: str = None
    ) -> dict:
        if isinstance(step_output, dict) and step_output.get("timed_out", None) is True:
            step_yield["status"] = "timedout"
        else:
            step_yield["status"] = "finished"
        if field is not None:
            step_yield["output"][field] = step_output
        else:
            step_yield["output"] = step_output
        return step_yield

    def get_user_docs(self, mids: list[str]) -> dict:
        user_docs_list = self.mongo.get_docs(
            "users",
            ids=mids,
            id_field="mid",
            include_fields=["mid", "name", "face"],
        )
        user_docs_dict = {user_doc["mid"]: user_doc for user_doc in user_docs_list}
        return user_docs_dict

    def group_hits_by_owner(
        self,
        search_res: dict,
        sort_field: Literal[
            "sum_count", "sum_view", "sum_sort_score", "sum_rank_score"
        ] = "sum_rank_score",
        limit: int = 10,
    ) -> dict:
        group_res = {}
        for hit in search_res.get("hits", []):
            name = dict_get(hit, "owner.name", None)
            mid = dict_get(hit, "owner.mid", None)
            pubdate = dict_get(hit, "pubdate") or 0
            view = dict_get(hit, "stat.view") or 0
            sort_score = dict_get(hit, "sort_score") or 0
            rank_score = dict_get(hit, "rank_score") or 0
            if mid is None or name is None:
                continue
            item = group_res.get(mid, None)
            if item is None:
                group_res[mid] = {
                    "mid": mid,
                    "name": name,
                    "latest_pubdate": pubdate,
                    "sum_view": view,
                    "sum_sort_score": sort_score,
                    "sum_rank_score": rank_score,
                    "sum_count": 0,
                    "hits": [],
                }
            else:
                latest_pubdate = group_res[mid]["latest_pubdate"]
                if pubdate > latest_pubdate:
                    group_res[mid]["latest_pubdate"] = pubdate
                    group_res[mid]["name"] = name
                sum_view = group_res[mid]["sum_view"] or 0
                sum_sort_score = group_res[mid]["sum_sort_score"] or 0
                sum_rank_score = group_res[mid]["sum_rank_score"] or 0
                group_res[mid]["sum_view"] = sum_view + view
                group_res[mid]["sum_sort_score"] = sum_sort_score + sort_score
                group_res[mid]["sum_rank_score"] = sum_rank_score + rank_score
            group_res[mid]["hits"].append(hit)
            group_res[mid]["sum_count"] += len(group_res[mid]["hits"])
        # sort by sort_field, and limit to top N
        sorted_items = sorted(
            group_res.items(), key=lambda item: item[1][sort_field], reverse=True
        )[:limit]
        group_res = dict(sorted_items)
        # add user faces
        mids = list(group_res.keys())
        user_docs = self.get_user_docs(mids)
        for mid, user_doc in user_docs.items():
            group_res[mid]["face"] = user_doc.get("face", "")
        return group_res

    def is_status_timedout(self, result: dict) -> bool:
        return result.get("status", None) == "timedout"

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
        most_relevant_limit: int = 10000,
        rank_method: RANK_METHOD_TYPE = RANK_METHOD_DEFAULT,
        rank_top_k: int = 400,
        group_owner_limit: int = 20,
    ) -> dict:
        """Explore and return all step results in a single response.

        Returns:
            dict: {
                "query": str,
                "status": "finished" | "timedout",
                "data": list[dict]  # list of step results
            }
        """
        logger.enter_quiet(not verbose)

        step_results = []  # Collect all step results
        final_status = "finished"

        # Step 0: Construct query_dsl_dict
        step_idx = 0
        step_name = "construct_query_dsl_dict"
        logger.hint("> [step 1] Query constructing ...")
        if query_dsl_dict is None:
            boosted_fields_params = {
                "match_fields": match_fields,
                "boost": boost,
                "boosted_fields": boosted_fields,
            }
            boosted_match_fields, boosted_date_fields = construct_boosted_fields(
                **boosted_fields_params
            )
            query_rewrite_dsl_params = {
                "query": query,
                "suggest_info": suggest_info,
                "boosted_match_fields": boosted_match_fields,
                "boosted_date_fields": boosted_date_fields,
                "match_type": match_type,
                "extra_filters": extra_filters,
            }
            _, _, query_dsl_dict = self.get_info_of_query_rewrite_dsl(
                **query_rewrite_dsl_params
            )
        query_dsl_dict_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "finished",
            "input": query_rewrite_dsl_params,
            "output_type": STEP_ZH_NAMES[step_name]["output_type"],
            "output": query_dsl_dict,
            "comment": "",
        }
        step_results.append(query_dsl_dict_result)

        # Step 1: Most-relevant docs
        step_idx += 1
        step_name = "most_relevant_search"
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(f"> {step_str} Top relevant docs")
        relevant_search_params = {
            "query": query,
            "suggest_info": suggest_info,
            "source_fields": ["bvid", "stat", "pubdate", "duration"],  # reduce io
            "extra_filters": extra_filters,
            "use_script_score": False,  # speed up
            "rank_method": rank_method,  # better ranking
            "add_region_info": False,  # speed up
            "add_highlights_info": False,  # speed up
            "is_profile": False,
            "is_highlight": False,  # speed up
            "limit": most_relevant_limit,
            "rank_top_k": rank_top_k,  # final returned docs with partial fields
            "timeout": EXPLORE_TIMEOUT,
            "verbose": verbose,
        }
        relevant_search_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "running",
            "input": relevant_search_params,
            "output": {},
            "output_type": "hits",
            "comment": "",
        }
        relevant_search_res = self.search(**relevant_search_params)
        self.update_step_output(relevant_search_result, step_output=relevant_search_res)
        if self.is_status_timedout(relevant_search_result):
            final_status = "timedout"
            step_results.append(relevant_search_result)
            logger.exit_quiet(not verbose)
            return {"query": query, "status": final_status, "data": step_results}

        # Step 3: Fetch full-docs by return ranked ids
        full_doc_search_params = deepcopy(relevant_search_params)
        bvids = [hit.get("bvid", None) for hit in relevant_search_res.get("hits", [])]
        bvid_filter = bvids_to_filter(bvids)
        full_doc_search_params.pop("source_fields", None)
        full_doc_search_params.update(
            {
                "rank_method": rank_method,  # same with relevant search
                "is_highlight": True,
                "add_region_info": True,
                "add_highlights_info": True,
                "extra_filters": extra_filters + [bvid_filter],
                "limit": len(bvids),
            }
        )
        full_doc_search_res = self.search(**full_doc_search_params)
        full_doc_search_res["total_hits"] = relevant_search_res.get("total_hits", 0)
        self.update_step_output(relevant_search_result, step_output=full_doc_search_res)
        if self.is_status_timedout(relevant_search_result):
            final_status = "timedout"
            step_results.append(relevant_search_result)
            logger.exit_quiet(not verbose)
            return {"query": query, "status": final_status, "data": step_results}
        step_results.append(relevant_search_result)

        # Step 4: Group hits by owner
        step_idx += 1
        step_name = "group_hits_by_owner"
        step_str = logstr.note(brk(f"Step {step_idx}"))
        logger.hint(f"> {step_str} Group hits by owner, sort by sum of view")
        group_hits_by_owner_params = {
            "search_res": full_doc_search_res,
            "limit": group_owner_limit,
        }
        group_hits_by_owner_result = {
            "step": step_idx,
            "name": step_name,
            "name_zh": STEP_ZH_NAMES[step_name]["name_zh"],
            "status": "running",
            "input": {"limit": group_owner_limit},
            "output": {},
            "output_type": STEP_ZH_NAMES[step_name]["output_type"],
            "comment": "",
        }
        group_res = self.group_hits_by_owner(**group_hits_by_owner_params)
        self.update_step_output(
            group_hits_by_owner_result, step_output=group_res, field="authors"
        )
        step_results.append(group_hits_by_owner_result)

        logger.exit_quiet(not verbose)
        return {"query": query, "status": final_status, "data": step_results}
