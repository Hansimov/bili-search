from copy import deepcopy
from typing import Literal, Union, Any

MERGE_TYPE = Literal["query", "extra", "union", "intersect"]
GL_UI_KEY_OP_MAPS = {
    ("g", "u"): ("gt", "gte", min),  # (>=? lower_bound) gt, gte, union
    ("l", "u"): ("lt", "lte", max),  # (<=? upper_bound) lt, lte, union
    ("g", "i"): ("gt", "gte", max),  # (>=? lower_bound) gt, gte, intersect
    ("l", "i"): ("lt", "lte", min),  # (<=? upper_bound) lt, lte, intersect
}
GL_KEYS_MAPS = {
    "g": ["gt", "gte"],
    "l": ["lt", "lte"],
}
GL_TYPE = Literal["g", "l"]
UI_TYPE = Literal["u", "i"]


class QueryDslDictFilterMerger:
    def get_filter_type_field_value(self, filter_clause: dict) -> tuple[str, str, dict]:
        """Example of input:
        ```
        {"range": {"stat.view": {"gte": 1000}}}
        ```
        Example of output:
        ```
        ("range", "stat.view", {"gte": 1000})
        ```
        """
        filter_type = next(iter(filter_clause))
        filter_field = next(iter(filter_clause[filter_type]))
        filter_value = filter_clause[filter_type][filter_field]
        # unify "term" to "terms" for consistency in future processing
        if filter_type == "term":
            filter_type = "terms"
        return filter_type, filter_field, filter_value

    def get_query_filters_from_query_dsl_dict(self, query_dsl_dict: dict) -> list[dict]:
        query_filters = query_dsl_dict.get("bool", {}).get("filter", [])
        if isinstance(query_filters, dict):
            query_filters = [query_filters]
        return query_filters

    def set_filters_to_query_dsl_dict(
        self, query_dsl_dict: dict, filters: list[dict]
    ) -> dict:
        query_dsl_dict["bool"]["filter"] = filters
        return query_dsl_dict

    def get_gl_key_val(self, range_value: dict, gl: GL_TYPE) -> tuple:
        GL_KEYS = GL_KEYS_MAPS[gl]
        for key in GL_KEYS:
            if key in range_value:
                return key, range_value[key]
        return None, None

    def get_key_val_from_range_values(
        self, value1: dict, value2: dict, gl: GL_TYPE, ui: UI_TYPE
    ) -> tuple[Union[str, None], Any]:
        """gl: 'g' for gt/gte,'l' for lt/lte;
        ui: 'u' for union, 'i' for intersect"""
        _, _, GL_OP = GL_UI_KEY_OP_MAPS[(gl, ui)]
        key1, val1 = self.get_gl_key_val(value1, gl=gl)
        key2, val2 = self.get_gl_key_val(value2, gl=gl)
        if key1 and key2:
            res_val = GL_OP(val1, val2)
            if res_val == val1:
                res_key = key1
            else:
                res_key = key2
        elif key1:
            res_key, res_val = key1, val1
        elif key2:
            res_key, res_val = key2, val2
        else:
            res_key, res_val = None, None
        return res_key, res_val

    def merge_range_values(self, value1: dict, value2: dict, ui: UI_TYPE) -> dict:
        res = {}
        for gl in ["g", "l"]:
            gl_key, gl_val = self.get_key_val_from_range_values(
                value1, value2, gl=gl, ui=ui
            )
            if gl_key:
                res[gl_key] = gl_val
        return res

    def merge_term_values(
        self, value1: Union[str, list], value2: Union[str, list], ui: UI_TYPE
    ) -> list:
        if isinstance(value1, str):
            value1 = [value1]
        if isinstance(value2, str):
            value2 = [value2]
        if ui == "u":
            res = list(set(value1 + value2))
        else:
            res = list(set(value1) & set(value2))
        return res

    def merge_query_and_extra_filter_maps(
        self,
        query_filter_maps: dict[tuple[str, str], dict],
        extra_filter_maps: dict[tuple[str, str], dict],
        merge_type: MERGE_TYPE = "query",
    ) -> dict:
        res_filter_maps: dict[tuple[str, str], dict] = deepcopy(query_filter_maps)
        for extra_type_field, extra_value in extra_filter_maps.items():
            extra_type, extra_field = extra_type_field
            if extra_type_field in query_filter_maps:
                query_value = query_filter_maps[extra_type_field]
                if merge_type == "query":
                    res_filter_maps[extra_type_field] = query_value
                elif merge_type == "extra":
                    res_filter_maps[extra_type_field] = extra_value
                elif merge_type in ["union", "intersect"]:
                    ui = "u" if merge_type == "union" else "i"
                    if extra_type == "range":
                        res_filter_maps[extra_type_field] = self.merge_range_values(
                            query_value, extra_value, ui=ui
                        )
                    elif extra_type == "terms":
                        res_filter_maps[extra_type_field] = self.merge_term_values(
                            query_value, extra_value, ui=ui
                        )
                    else:
                        res_filter_maps[extra_type_field] = query_value
                else:
                    res_filter_maps[extra_type_field] = query_value
            else:
                res_filter_maps[extra_type_field] = extra_value
        return res_filter_maps

    def filter_maps_to_list(
        self, filter_maps: dict[tuple[str, str], dict]
    ) -> list[dict]:
        res = []
        for filter_type_field, filter_value in filter_maps.items():
            filter_type, filter_field = filter_type_field
            if filter_value in [None, {}, []]:
                continue
            if filter_type == "range":
                res.append({filter_type: {filter_field: filter_value}})
            elif filter_type == "terms":
                if isinstance(filter_value, list):
                    if len(filter_value) > 1:
                        res.append({filter_type: {filter_field: filter_value}})
                    else:
                        res.append({"term": filter_value[0]})
                else:
                    res.append({"term": {filter_field: filter_value}})
            else:
                res.append({filter_type: {filter_field: filter_value}})
        return res

    def merge(
        self,
        query_dsl_dict: dict,
        extra_filters: list[dict],
        merge_type: MERGE_TYPE = "query",
    ) -> dict:
        """Merge extra_filters to query_dsl_dict.

        Example of extra_filters:
        ```
        [
            {"range": {"stat.view": {"gte": 100}}},
            {"term": {"owner.name": "红警HBK08"}},
        ]
        ```

        Example of query_dsl_dict:
        ```
        {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": "hongjing 2024",
                        "type"   : "phrase_prefix",
                        "fields" : ["title.words^3", "tags.words^2.5", "owner.name.words^2"]
                    }
                },
                "filter": [
                    {"range": {"stat.view": {"gte": 1000}}},
                ],
            }
        }
        ```

        If filter type ("range" or "term") and field are the same, then `prefer` is used to decide which one to keep:
            - "query" means to use the filter in query_dsl_dict
            - "extra" means to use the filter in extra_filters
            - "union" means to use the looser bound of the two
            - "intersect" means to use the tighter bound of the two
        """
        if not extra_filters:
            return query_dsl_dict
        query_filters = self.get_query_filters_from_query_dsl_dict(query_dsl_dict)
        if not query_filters:
            self.set_filters_to_query_dsl_dict(query_dsl_dict, extra_filters)
            return query_dsl_dict
        query_filter_maps: dict[tuple[str, str], dict] = {}
        extra_filter_maps: dict[tuple[str, str], dict] = {}
        for query_filter in query_filters:
            filter_type, filter_field, filter_value = self.get_filter_type_field_value(
                query_filter
            )
            query_filter_maps[(filter_type, filter_field)] = filter_value
        for extra_filter in extra_filters:
            filter_type, filter_field, filter_value = self.get_filter_type_field_value(
                extra_filter
            )
            extra_filter_maps[(filter_type, filter_field)] = filter_value
        merged_filter_maps = self.merge_query_and_extra_filter_maps(
            query_filter_maps, extra_filter_maps, merge_type=merge_type
        )
        filter_dict = self.filter_maps_to_list(merged_filter_maps)
        self.set_filters_to_query_dsl_dict(query_dsl_dict, filter_dict)
        return query_dsl_dict


def test_merge_range_values():
    from tclogger import logger, logstr, dict_to_str, brk

    merger = QueryDslDictFilterMerger()
    value1 = {"gt": 1000, "lte": 2000}
    value2 = {"gte": 1500, "lt": 2500}
    logger.mesg(value1)
    logger.mesg(value2)
    for ui in ["u", "i"]:
        res = merger.merge_range_values(value1, value2, ui=ui)
        logger.note(f"res: {logstr.file(brk(ui))}", end=" ")
        logger.okay(res)


def test_merge_term_values():
    from tclogger import logger, logstr, dict_to_str, brk

    merger = QueryDslDictFilterMerger()
    value1 = ["红警HBK08", "红警月亮3"]
    value2 = "红警HBK08"
    logger.mesg(value1)
    logger.mesg(value2)
    for ui in ["u", "i"]:
        res = merger.merge_term_values(value1, value2, ui=ui)
        logger.note(f"res: {logstr.file(brk(ui))}", end=" ")
        logger.okay(res)


if __name__ == "__main__":
    test_merge_range_values()
    test_merge_term_values()

    # python -m converters.dsl.filter
