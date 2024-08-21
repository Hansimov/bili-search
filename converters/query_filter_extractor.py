import re

from tclogger import logger
from pprint import pformat


class QueryFilterExtractor:
    STAT_FIELDS = ["view", "like", "coin", "favorite", "reply", "danmaku", "share"]
    STAT_ALIASES = {
        "view": ["播放", "bf", "bofang"],
        "like": ["点赞", "dz", "dianzan"],
        "coin": ["投币", "tb", "toubi"],
        "favorite": ["收藏", "sc", "shoucang"],
        "reply": ["评论", "pl", "pinglun"],
        "danmaku": ["弹幕", "dm", "danmu"],
        "share": ["分享", "fx", "fenxiang"],
    }
    OPERATOR_MAPS = {
        ">": "gt",
        "<": "lt",
        ">=": "gte",
        "<=": "lte",
    }
    BRACKET_MAPS = {
        "(": "gt",
        ")": "lt",
        "[": "gte",
        "]": "lte",
    }

    def map_key_to_stat_field(self, key: str) -> str:
        for stat_field, aliases in self.STAT_ALIASES.items():
            if (
                stat_field.startswith(key)
                or f"stat.{stat_field}".startswith(key)
                or any(alias.startswith(key) for alias in aliases)
            ):
                return f"stat.{stat_field}"

        logger.warn(f"× No matching stat field: {key}")
        return None

    def map_val_to_range_dict(self, val: str) -> dict:
        """
        Examples:
            - ">1000" -> {"gt": 1000}
            - "<=1000" -> {"lte": 1000}
            - "=1000" -> {"gte": 1000, "lte": 1000}
            - "=[1000,2000)" -> {"gte": 1000, "lt": 2000}
            - "=[,2000)" -> {"lt": 2000}
            - "=(1000,)" -> {"gt": 1000}
            - "=[,)" -> {}
        """

        # use regex to parse the val into range_dict
        range_dict = {}
        range_patterns = [
            r"([<>]=?)(\d+)",
            r"(=)(\d+)",
            r"(=)([\[\(])(\d*),(\d*)([\]\)])",
        ]
        for pattern in range_patterns:
            match = re.match(pattern, val)
            if match:
                op = match.group(1)
                if op in self.OPERATOR_MAPS.keys():
                    op_str = self.OPERATOR_MAPS[op]
                    num = int(match.group(2))
                    range_dict[op_str] = num
                elif op == "=" and len(match.groups()) == 2:
                    num = int(match.group(2))
                    range_dict["gte"] = num
                    range_dict["lte"] = num
                elif op == "=" and len(match.groups()) == 5:
                    left_bracket = match.group(2)
                    lower_num_str = match.group(3)
                    upper_num_str = match.group(4)
                    right_bracket = match.group(5)

                    left_op_str = self.BRACKET_MAPS[left_bracket]
                    right_op_str = self.BRACKET_MAPS[right_bracket]

                    if lower_num_str != "" and upper_num_str != "":
                        lower_num_int = int(lower_num_str)
                        upper_num_int = int(upper_num_str)
                        if upper_num_int >= lower_num_int:
                            range_dict[left_op_str] = lower_num_int
                            range_dict[right_op_str] = upper_num_int
                        else:
                            logger.warn(f"× Invalid lower and upper bounds: {val}")
                    elif lower_num_str != "":
                        lower_num_int = int(lower_num_str)
                        range_dict[left_op_str] = lower_num_int
                    elif upper_num_str != "":
                        upper_num_int = int(upper_num_str)
                        range_dict[right_op_str] = upper_num_int
                    else:
                        logger.mesg(f"* No range bounds: {val}")
                else:
                    logger.warn(f"× Error when parsing: {val}")

                break

        if not range_dict:
            logger.warn(f"× No matching patterns: {val}")

        return range_dict

    def split_stat_filter_key_and_val(self, keyword: str) -> tuple[str, dict]:
        """
        Syntaxes:
            - input_keys: stat_fields, or stat_fields with matched prefix
                - Eg: view, vi, v, stat.view, stat.vi ... all maps to "view"
            - operators: <, >, =, <=, >=
            - values: int, float, range
                - int, float
                - range: [a,b), (a,b], (a,b), [a,b], and a or b could be empty

        Examples:
            - ":view>1000" -> ("view", ">1000")
            - ":fav<=1000" -> ("favorite", "<=1000")
            - ":coin=[1000,2000)" -> ("coin", "[1000,2000)")
            - ":danm=[,2000)" -> ("danmaku", "[,2000)")
        """

        filter_key = None
        filter_val = None

        # use regex to split the keyword into filter_key and filter_val
        match = re.match(r":(\w+)(.*)", keyword)
        if match:
            matched_key = match.group(1)
            filter_key = self.map_key_to_stat_field(matched_key)
            matched_val = match.group(2)
            filter_val = self.map_val_to_range_dict(matched_val)

        if filter_key and filter_val:
            return filter_key, filter_val
        else:
            return None, None

    def merge_filter(self, filter_item: dict[str, dict], filters: dict[str, dict]):
        """
        Examples:
            - {"gte": 1000} + {"gt": 1000} -> {"gt": 1000}
            - {"gt": 1000} + {"lte": 2000} -> {"gt": 1000, "lte": 2000}
            - {"gt": 2000} + {"lt": 1000} -> {"gt":2000, "lt": 1000}
                - this is invalid, but still keep it as is
        """
        for key, val in filter_item.items():
            if key in filters.keys():
                for sub_key, sub_val in val.items():
                    if sub_key in filters[key].keys():
                        if sub_key in ["gt", "gte"]:
                            filters[key][sub_key] = max(filters[key][sub_key], sub_val)
                        elif sub_key in ["lt", "lte"]:
                            filters[key][sub_key] = min(filters[key][sub_key], sub_val)
                        else:
                            logger.warn(f"× Invalid sub_key: {sub_key}")
                    else:
                        filters[key][sub_key] = sub_val
            else:
                filters.update(filter_item)

            if "gte" in filters[key].keys() and "gt" in filters[key].keys():
                if filters[key]["gte"] > filters[key]["gt"]:
                    filters[key].pop("gt")
                else:
                    filters[key].pop("gte")

            if "lte" in filters[key].keys() and "lt" in filters[key].keys():
                if filters[key]["lte"] < filters[key]["lt"]:
                    filters[key].pop("lt")
                else:
                    filters[key].pop("lte")

    def categorize_filter(self, keyword: str) -> dict[str, dict]:
        filter_item = {}
        filter_key, filter_val = self.split_stat_filter_key_and_val(keyword)
        if filter_key and filter_val:
            filter_item[filter_key] = filter_val
        return filter_item

    def extract(self, query: str) -> tuple[dict, list]:
        query_keywords = query.split()
        query_keywords_without_filters = []
        filters = {}
        for keyword in query_keywords:
            if keyword.startswith(":"):
                filter_item = self.categorize_filter(keyword)
                self.merge_filter(filter_item=filter_item, filters=filters)
            else:
                query_keywords_without_filters.append(keyword)

        return filters, query_keywords_without_filters


if __name__ == "__main__":
    extractor = QueryFilterExtractor()
    queries = [
        "影视飓风 :view>1000 :fav<=1000 :coin=[1000,2000)",
        "影视飓风 :view>1000 :view<2000 :播放<=2000 :fav=[,2000)",
        "黑神话 :bf>1000 :view=(2000,3000] :view>=1000 :view<2500",
        "黑神话 :bf=1000 :date=2024-08",
    ]
    for query in queries:
        logger.line(f"{query}")
        filters, query_keywords_without_filters = extractor.extract(query)
        logger.note("  - Filters:")
        logger.success(pformat(filters, sort_dicts=False, compact=False), indent=2)
        logger.note("  - Keywords:", end=" ")
        logger.mesg(f"{query_keywords_without_filters}")

    # python -m converters.query_filter_extractor
