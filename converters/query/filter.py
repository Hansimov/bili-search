import re

from pprint import pformat
from tclogger import logger

from converters.field.date import DateFieldConverter
from converters.field.stat import StatFieldConverter
from converters.field.user import UserFieldConverter, UidFieldConverter
from converters.field.operators import RE_COMMA


class QueryFilterExtractor:
    OP_MAPS = {">": "gt", "<": "lt", ">=": "gte", "<=": "lte"}
    BRACKET_MAPS = {"(": "gt", ")": "lt", "[": "gte", "]": "lte"}

    RE_LB = r"(\[|\(|【|（)"
    RE_RB = r"(\]|\)|】|）)"
    RE_OP = r"(<=?|>=?|=|《=?|》=?)"
    RE_RANGE_SEP = RE_COMMA

    RE_KEYWORD = r"[^:：\n\s\.]+"
    RE_FILTER_SEP = r"[:：]*"

    REP_DATE_FIELD = DateFieldConverter.REP_DATE_FIELD
    REP_STAT_FIELD = StatFieldConverter.REP_STAT_FIELD
    REP_UID_FIELD = UidFieldConverter.REP_UID_FIELD
    REP_USER_FIELD = UserFieldConverter.REP_USER_FIELD
    RE_USER_SYM = "@"

    RE_DATE_VAL = DateFieldConverter.RE_DATE_VAL
    RE_STAT_VAL = StatFieldConverter.RE_STAT_VAL
    RE_UID_VAL = UidFieldConverter.RE_UID_VAL
    RE_USER_VAL = UserFieldConverter.RE_USER_VAL

    REP_DATE_FILTER = (
        rf"(?P<date_filter>{RE_FILTER_SEP}\s*{REP_DATE_FIELD}\s*"
        rf"(?P<date_op>{RE_OP})\s*((?P<date_val>{RE_DATE_VAL})|"
        rf"(?P<date_lb>{RE_LB})\s*(?P<date_lval>{RE_DATE_VAL}*)\s*{RE_RANGE_SEP}\s*(?P<date_rval>{RE_DATE_VAL}*)\s*(?P<date_rb>{RE_RB})))"
    )
    REP_STAT_FILTER = (
        rf"(?P<stat_filter>{RE_FILTER_SEP}\s*{REP_STAT_FIELD}\s*"
        rf"(?P<stat_op>{RE_OP})\s*((?P<stat_val>{RE_STAT_VAL})|"
        rf"(?P<stat_lb>{RE_LB})\s*(?P<stat_lval>{RE_STAT_VAL}*)\s*{RE_RANGE_SEP}\s*(?P<stat_rval>{RE_STAT_VAL}*)\s*(?P<stat_rb>{RE_RB})))"
    )
    REP_UID_FILTER = (
        rf"(?P<uid_filter>{RE_FILTER_SEP}\s*{REP_UID_FIELD}\s*"
        rf"(?P<uid_op>{RE_OP})\s*"
        rf"((?:{RE_LB}?)\s*(?P<uid_vals>{RE_UID_VAL}(?:\s*{RE_COMMA}\s*{RE_UID_VAL})*)\s*(?:{RE_RB}?)))"
    )
    REP_USER_FILTER = (
        rf"(?P<user_filter>"
        rf"({RE_FILTER_SEP}\s*{REP_USER_FIELD}\s*(?P<user_op>{RE_OP})\s*|"
        rf"\s*(?P<user_sym>{RE_USER_SYM})\s*)"
        rf"((?:{RE_LB}?)\s*(?P<user_vals>{RE_USER_VAL}(?:\s*{RE_COMMA}\s*{RE_USER_VAL})*)\s*(?:{RE_RB}?)))"
    )
    REP_KEYWORD = rf"(?P<keyword>{RE_KEYWORD})"

    QUERY_PATTERN = rf"({REP_DATE_FILTER}|{REP_STAT_FILTER}|{REP_UID_FILTER}|{REP_USER_FILTER}|{REP_KEYWORD})"

    def split_keyword_and_filter_expr(self, query: str) -> dict:
        """use regex to split keywords and filter exprs, which allows spaces in filter exprs
        Examples:
            - "影视飓风 :view>1000 :fav<=1000 :coin=[1000,2000)"
                -> ["影视飓风"]
                -> [":view>1000", ":fav<=1000", ":coin=[1000,2000)"]
            - "黑神话 2024 :bf >1000 :view = (2000, 3000] :view< 2500"
                -> ["黑神话", "2024"]
                -> [":bf>1000", ":view=(2000,3000]", ":view<2500"]
            - "黑神话 ::bf >1000 :view< 2500 2024-10-11"
                -> ["黑神话", "2024-10-11"]
                -> [":bf>1000", ":view<2500"]
            - "黑神话 ::bf >1000 :view<2500 map"
                -> ["黑神话", "map"]
                -> [":bf>1000", ":view<2500"]
            - "黑神话 :view>1000 :date=2024-08-20"
                -> ["黑神话"]
                -> [":view>1000]
                -> [":date=2024-08-20"]
            - "黑神话 :view>1000 :date=[08-10,08-20)"
                -> ["黑神话"]
                -> [":view>1000]
                -> [":date=[08-10,08-20)"]
            - "黑神话 :view>1000 :date<=7d"
                -> ["黑神话"]
                -> [":date<=7d"]
        """

        matches = re.finditer(self.QUERY_PATTERN, query)
        res = {}
        keywords = []
        stat_filter_exprs = []
        date_filter_exprs = []
        uid_filter_exprs = []
        user_filter_exprs = []
        filters_str = ""
        for match in matches:
            keyword = match.group("keyword")
            stat_filter_expr = match.group("stat_filter")
            date_filter_expr = match.group("date_filter")
            uid_filter_expr = match.group("uid_filter")
            user_filter_expr = match.group("user_filter")
            if keyword:
                keywords.append(keyword.strip())
            if stat_filter_expr:
                stat_filter_exprs.append(stat_filter_expr.strip())
                filters_str += f" {stat_filter_expr.strip()}"
            if date_filter_expr:
                date_filter_exprs.append(date_filter_expr.strip())
                filters_str += f" {date_filter_expr.strip()}"
            if uid_filter_expr:
                uid_filter_exprs.append(uid_filter_expr.strip())
                filters_str += f" {uid_filter_expr.strip()}"
            if user_filter_expr:
                user_filter_exprs.append(user_filter_expr.strip())
                filters_str += f" {user_filter_expr.strip()}"
        res = {
            "keywords": keywords,
            "stat_filter_exprs": stat_filter_exprs,
            "date_filter_exprs": date_filter_exprs,
            "uid_filter_exprs": uid_filter_exprs,
            "user_filter_exprs": user_filter_exprs,
            "filters": filters_str.strip(),
        }
        return res

    def map_key_to_field(self, key: str) -> str:
        if re.match(self.REP_DATE_FIELD, key):
            return "date"
        if re.match(self.REP_UID_FIELD, key):
            return "uid"
        if re.match(self.REP_USER_FIELD, key):
            return "user"
        match = re.match(self.REP_STAT_FIELD, key)
        if match:
            for k, v in match.groupdict().items():
                if k != "stat_field" and v:
                    return k
        else:
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
                if op in self.OP_MAPS.keys():
                    op_str = self.OP_MAPS[op]
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

    def get_filter_field_and_val_from_expr(self, keyword: str) -> dict:
        """
        Examples:
            - ":view>1000"
                {'field':'view', 'field_type':'stat', 'op':'>', 'val':'1000','val_type':'value'}
            - ":fav<=1000"
                {'field':'favorite', 'field_type':'stat', 'op':'<=', 'val':'1000', 'val_type':'value'}
            - ":coin=[1000,2000)"
                {'field':'coin', 'field_type':'stat', 'op':'[', 'lb':'[', 'lval':'1000', 'rval':'2000', 'rb':')', 'val_type':'range'}
            - ":danm=[,2000)"
                {'field':'danmaku', 'field_type':'stat', 'op':'=', 'lb':'[', 'lval':'', 'rval':'2000', 'rb':')', 'val_type':'range'}
        """
        res = {}
        if re.match(self.REP_STAT_FILTER, keyword):
            field_type = "stat"
            match = re.match(self.REP_STAT_FILTER, keyword)
        elif re.match(self.REP_DATE_FILTER, keyword):
            field_type = "date"
            match = re.match(self.REP_DATE_FILTER, keyword)
        elif re.match(self.REP_UID_FILTER, keyword):
            field_type = "uid"
            match = re.match(self.REP_UID_FILTER, keyword)
        elif re.match(self.REP_USER_FILTER, keyword):
            field_type = "user"
            match = re.match(self.REP_USER_FILTER, keyword)
        else:
            logger.warn(f"× No matched filter field: {keyword}")
            return None, None

        res["field_type"] = field_type

        if field_type == "user" and match.groupdict().get("user_sym"):
            res["field"] = "user"
            res["op"] = "="
        else:
            res["field"] = self.map_key_to_field(match.group(f"{field_type}_field"))
            res["op"] = match.group(f"{field_type}_op")

        if field_type in ["uid", "user"]:
            res["vals"] = match.group(f"{field_type}_vals")
            res["val_type"] = "list"
        else:
            if match.group(f"{field_type}_val"):
                res["val"] = match.group(f"{field_type}_val")
                res["val_type"] = "value"
            elif match.group(f"{field_type}_lb"):
                res["lb"] = match.group(f"{field_type}_lb")
                res["lval"] = match.group(f"{field_type}_lval")
                res["rval"] = match.group(f"{field_type}_rval")
                res["val_type"] = "range"
                res["rb"] = match.group(f"{field_type}_rb")
            else:
                logger.warn(f"× No matched stat_val: {keyword}")
        logger.mesg(pformat(res, sort_dicts=False, compact=False), indent=4)

        return res

    def merge_range_filter(
        self, filter_item: dict[str, dict], filters: dict[str, dict]
    ):
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

    def merge_term_filter(self, filter_item: dict[str, dict], filters: dict[str, dict]):
        for key, val in filter_item.items():
            if key in filters.keys():
                if isinstance(filters[key], list):
                    if isinstance(val, list):
                        filters[key] += val
                    else:
                        filters[key].append(val)
                else:
                    if isinstance(val, list):
                        filters[key] = [filters[key]] + val
                    else:
                        filters[key] = [filters[key], val]
            else:
                if isinstance(val, list) and len(val) == 1:
                    filters[key] = val[0]
                else:
                    filters.update(filter_item)

    def filter_expr_to_dict(
        self, filter_expr: str, use_date_str: bool = False, verbose: bool = False
    ) -> dict[str, dict]:
        logger.enter_quiet(not verbose)
        res = {}
        filter_dict = self.get_filter_field_and_val_from_expr(filter_expr)
        if filter_dict["field_type"] == "stat":
            converter = StatFieldConverter()
            res = converter.filter_dict_to_es_dict(filter_dict)
        elif filter_dict["field_type"] == "date":
            converter = DateFieldConverter()
            res = converter.filter_dict_to_es_dict(
                filter_dict, use_date_str=use_date_str
            )
        elif filter_dict["field_type"] == "uid":
            converter = UidFieldConverter()
            res = converter.filter_dict_to_es_dict(filter_dict)
        elif filter_dict["field_type"] == "user":
            converter = UserFieldConverter()
            res = converter.filter_dict_to_es_dict(filter_dict)
        else:
            logger.warn(f"× No matching field type: {filter_dict['field_type']}")

        if res:
            logger.success(pformat(res, sort_dicts=False, compact=False), indent=4)
        logger.exit_quiet(not verbose)

        return res

    def extract(
        self, query: str, use_date_str: bool = False
    ) -> tuple[list[str], dict, dict]:
        range_filters = {}
        split_res = self.split_keyword_and_filter_expr(query)
        keywords = split_res["keywords"]
        range_filter_exprs = (
            split_res["stat_filter_exprs"] + split_res["date_filter_exprs"]
        )
        for filter_expr in range_filter_exprs:
            filter_item = self.filter_expr_to_dict(
                filter_expr, use_date_str=use_date_str
            )
            self.merge_range_filter(filter_item=filter_item, filters=range_filters)

        term_filters = {}
        term_filter_exprs = (
            split_res["uid_filter_exprs"] + split_res["user_filter_exprs"]
        )
        for filter_expr in term_filter_exprs:
            filter_item = self.filter_expr_to_dict(filter_expr)
            self.merge_term_filter(filter_item=filter_item, filters=term_filters)

        return keywords, range_filters, term_filters

    def construct(
        self, query: str, use_date_str: bool = False
    ) -> tuple[list, list[str]]:
        keywords, range_filters, term_filters = self.extract(
            query, use_date_str=use_date_str
        )
        filters = []
        if range_filters:
            for k, v in range_filters.items():
                filters.append({"range": {k: v}})
        if term_filters:
            for k, v in term_filters.items():
                if isinstance(v, list):
                    filters.append({"terms": {k: v}})
                else:
                    filters.append({"term": {k: v}})
        return keywords, filters


if __name__ == "__main__":
    extractor = QueryFilterExtractor()
    queries = [
        # "影视飓风 :view>1000 :fav<=1000 :coin=[1000,2000)",
        # "影视飓风 :view>1000 :view<2000 :播放<=2000 :fav=[,2000)",
        # "黑神话 :bf>1000 :view=(2000,3000] :view>=1000 :view<2500",
        # "黑神话 :bf=1000 :date>=2024-08-21 rank",
        # "黑神话 :bf=1000 :date=[2020-08-20, 08-22] 2024",
        # "黑神话 ：bf >1000K :view<2百w 2024-10-11",
        # "黑神话 :bf=【200,1000) :date=[2020-08-20,） 2024",
        # "黑神话 :bf=【200,1000) :date=2020-08-20 2024",
        # "黑神话 ::bf >1000 map :view<2500 :date<=7days",
        # "黑神话 :view>1000 :date=2024-08-20",
        # "黑神话 :view>1000 :date=[08-10,08-20)",
        # "黑神话 :view>1000 :date=[08.10, 08/20)",
        # "黑神话 ::date=[7d,]",
        # "黑神话 :date>7d",
        # "黑神话 :date<=7d :vw>100w :coin>2k :star>1k",
        # "黑神话 :date<=7d :vw>100w :uid=642389251",
        # "黑神话 :date<=7d :vw>100w :uid=[946974]",
        # "黑神话 :date<=7d :vw>100w :mid=[642389251,946974]",
        "黑神话 :date<=7d :vw>100w :mid=642389251，946974",
        # "黑神话 :date<=7d :vw>100w :uid=[]",
        # ":date<=7d :user=影视飓风",
        ":date<=7d :user = 影视飓风, 亿点点不一样",
        ":date<=7d @ 影视飓风, 亿点点不一样",
        ":date<=7d :up=[影视飓风，飓多多StormCrew， 亿点点不一样]",
        # "黑神话 :view>1000 :date=[7d,1d]",
        # "黑神话 :view>1000 :date <= 3 天",
        # "黑神话 :view>1000 :date <= past_hour 1小时",
        # "黑神话 :view>1000 :d = 1小时学完",
    ]
    for query in queries:
        logger.line(f"{query}")
        res = extractor.split_keyword_and_filter_expr(query)
        logger.note("  * Parsed:")
        logger.success(pformat(res, sort_dicts=False, compact=False), indent=4)
        # keywords = res["keywords"]
        # filter_exprs = res["stat_filter_exprs"] + res["date_filter_exprs"]
        logger.note("  * Extracted:")
        keywords, range_filter_dicts, term_filter_dicts = extractor.extract(query)
        logger.success(range_filter_dicts, indent=4)
        logger.success(term_filter_dicts, indent=4)
        logger.note("  * Constructed:")
        keywords, filter_dicts = extractor.construct(query)
        logger.mesg(keywords, indent=4)
        logger.success(filter_dicts, indent=4)

    # python -m converters.query.filter
