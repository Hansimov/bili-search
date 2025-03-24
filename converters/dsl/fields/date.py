from calendar import monthrange
from datetime import timedelta
from tclogger import logger, tcdatetime, ts_to_str
from typing import Union, Literal

from converters.dsl.node import DslExprNode

TIME_DELTA_1US = timedelta(microseconds=1)


class DateExprElasticConverter:
    DATE_FIELD = "pubdate"

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def calc_ts_range_of_ymdh_dict(self, expr_key: str, ymdh_dict: dict) -> dict:
        now = tcdatetime.now()
        ymdh = list(map(lambda x: ymdh_dict.get(x, None), ["yyyy", "mm", "dd", "hh"]))
        year, month, day, hour = list(
            map(lambda x: int(x) if x is not None else None, ymdh)
        )
        if expr_key == "yyyymmddhh":
            start_dt = tcdatetime(year, month, day, hour)
            end_dt = start_dt + timedelta(hours=1) - TIME_DELTA_1US
        elif expr_key == "yyyymmdd":
            start_dt = tcdatetime(year, month, day)
            end_dt = start_dt + timedelta(days=1) - TIME_DELTA_1US
        elif expr_key == "yyyymm":
            start_dt = tcdatetime(year, month, 1)
            _, month_days = monthrange(year, month)
            end_dt = start_dt + timedelta(days=month_days) - TIME_DELTA_1US
        elif expr_key == "mmddhh":
            start_dt = tcdatetime(now.year, month, day, hour)
            if start_dt.timestamp() > now.timestamp():
                start_dt = tcdatetime(now.year - 1, month, day, hour)
            end_dt = start_dt + timedelta(hours=1) - TIME_DELTA_1US
        elif expr_key == "yyyy":
            start_dt = tcdatetime(year, 1, 1)
            end_dt = tcdatetime(year, 12, 31, 23, 59, 59)
        elif expr_key == "mmdd":
            start_dt = tcdatetime(now.year, month, day)
            if start_dt.timestamp() > now.timestamp():
                start_dt = tcdatetime(now.year - 1, month, day)
            end_dt = start_dt + timedelta(days=1) - TIME_DELTA_1US
        else:
            logger.warn(f"× Invalid date val key: {expr_key}")
            start_dt = None
            end_dt = None

        if start_dt and end_dt:
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
            start_str = ts_to_str(start_ts)
            end_str = ts_to_str(end_ts)
        else:
            start_ts = 0
            end_ts = 0
            start_str = ""
            end_str = ""

        res = {
            # "range_dt": [start_dt, end_dt],
            "range_ts": [start_ts, end_ts],
            "range_str": [start_str, end_str],
        }

        return res

    def calc_ts_range_of_date_this(
        self, unit: Literal["year", "month", "week", "day", "hour"]
    ) -> dict:
        now = tcdatetime.now()
        if unit == "year":
            start_dt = tcdatetime(now.year, 1, 1)
        elif unit == "month":
            start_dt = tcdatetime(now.year, now.month, 1)
        elif unit == "week":
            start_dt = now - timedelta(days=now.weekday())
            start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif unit == "day":
            start_dt = tcdatetime(now.year, now.month, now.day)
        elif unit == "hour":
            start_dt = tcdatetime(now.year, now.month, now.day, now.hour)
        else:
            logger.warn(f"× No match unit of date_this: {unit}")
            start_dt = None

        if start_dt:
            start_ts = int(start_dt.timestamp())
            end_ts = int(now.timestamp())
            start_str = ts_to_str(start_ts)
            end_str = ts_to_str(end_ts)
        else:
            start_ts = 0
            end_ts = 0
            start_str = ""
            end_str = ""

        res = {
            "range_ts": [start_ts, end_ts],
            "range_str": [start_str, end_str],
        }

        return res

    def calc_ts_range_of_date_last(
        self, unit: Literal["year", "month", "week", "day", "hour"]
    ) -> dict:
        now = tcdatetime.now()
        if unit == "year":
            start_dt = tcdatetime(now.year - 1, 1, 1)
            end_dt = tcdatetime(now.year - 1, 12, 31, 23, 59, 59)
        elif unit == "month":
            if now.month == 1:
                start_dt = tcdatetime(now.year - 1, 12, 1)
            else:
                start_dt = tcdatetime(now.year, now.month - 1, 1)
            _, month_days = monthrange(start_dt.year, start_dt.month)
            end_dt = start_dt + timedelta(days=month_days) - TIME_DELTA_1US
        elif unit == "week":
            start_dt = now - timedelta(days=now.weekday() + 7)
            start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt = start_dt + timedelta(days=7) - TIME_DELTA_1US
        elif unit == "day":
            start_dt = now - timedelta(days=1)
            start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt = start_dt + timedelta(days=1) - TIME_DELTA_1US
        elif unit == "hour":
            start_dt = now - timedelta(hours=1)
            start_dt = start_dt.replace(minute=0, second=0, microsecond=0)
            end_dt = start_dt + timedelta(hours=1) - TIME_DELTA_1US
        else:
            logger.warn(f"× No match unit of date_last: {unit}")
            start_dt = None

        if start_dt:
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
            start_str = ts_to_str(start_ts)
            end_str = ts_to_str(end_ts)
        else:
            start_ts = 0
            end_ts = 0
            start_str = ""
            end_str = ""

        res = {
            "range_ts": [start_ts, end_ts],
            "range_str": [start_str, end_str],
        }

        return res

    def calc_ts_range_of_date_num_unit(self, n: int, unit: str) -> dict:
        now = tcdatetime.now()
        if unit == "year":
            start_dt = tcdatetime(
                now.year - n, now.month, now.day, now.hour, now.minute, now.second
            )
        elif unit == "month":
            start_year = now.year - (n + 12 - now.month) // 12
            start_month = ((now.month - n) % 12) or 12
            start_dt = tcdatetime(
                start_year, start_month, now.day, now.hour, now.minute, now.second
            )
        elif unit == "week":
            start_dt = now - timedelta(weeks=n)
        elif unit == "day":
            start_dt = now - timedelta(days=n)
        elif unit == "hour":
            start_dt = now - timedelta(hours=n)
        else:
            logger.warn(f"× No match unit of date_num_unit: {unit}")
            start_dt = None

        if start_dt:
            start_ts = int(start_dt.timestamp())
            end_ts = int(start_dt.timestamp())
            start_str = ts_to_str(start_ts)
            end_str = ts_to_str(end_ts)
        else:
            start_ts = 0
            end_ts = 0
            start_str = ""
            end_str = ""

        res = {
            "range_ts": [start_ts, end_ts],
            "range_str": [start_str, end_str],
        }

        return res

    def get_info_of_date_iso(self, node: DslExprNode) -> dict:
        expr_node = node.find_child_with_key(
            ["yyyymmddhh", "yyyymmdd", "yyyymm", "mmddhh", "yyyy", "mmdd"]
        )
        ymdh_dict = expr_node.get_value_dict_by_keys(["yyyy", "mm", "dd", "hh"])
        range_dict = self.calc_ts_range_of_ymdh_dict(expr_node.key, ymdh_dict)
        return {
            "date_type": "date_iso",
            "value_format": expr_node.key,
            "ymdh": ymdh_dict,
            **range_dict,
        }

    def get_info_of_date_recent(self, node: DslExprNode) -> dict:
        expr_node = node.find_child_with_key(["date_this", "date_last"])
        date_unit_node = expr_node.find_child_with_key("date_unit")
        unit_value_node = date_unit_node.find_child_with_key("date_unit_", use_re=True)
        unit = unit_value_node.get_deepest_node_key().split("_")[-1]
        if expr_node.is_key("date_this"):
            range_dict = self.calc_ts_range_of_date_this(unit)
        elif expr_node.is_key("date_last"):
            range_dict = self.calc_ts_range_of_date_last(unit)
        else:
            logger.warn(f"× Invalid date_recent key: {expr_node.key}")
            range_dict = None

        return {
            "date_type": "date_recent",
            "value_format": expr_node.key,
            **range_dict,
        }

    def get_info_of_date_num_unit(self, node: DslExprNode) -> dict:
        num_node = node.find_child_with_key("date_num")
        num = int(num_node.get_deepest_node_value())
        unit_node = node.find_child_with_key("date_unit")
        unit = unit_node.get_deepest_node_key().split("_")[-1]
        range_dict = self.calc_ts_range_of_date_num_unit(num, unit)
        return {
            "date_type": "date_num_unit",
            "value_format": f"{num}_{unit}",
            **range_dict,
        }

    def convert_single_info_to_elastic_dict(self, op_key: str, info: dict):
        date_type = info["date_type"]
        start_ts, end_ts = info["range_ts"]
        if date_type == "date_num_unit":
            op_range = {
                "eqs": {"gte": start_ts},
                "gt": {"lt": start_ts},
                "lt": {"gt": end_ts},
                "geqs": {"lte": start_ts},
                "leqs": {"gte": end_ts},
            }
        else:
            op_range = {
                "eqs": {"gte": start_ts, "lte": end_ts},
                "gt": {"gt": end_ts},
                "lt": {"lt": start_ts},
                "geqs": {"gte": start_ts},
                "leqs": {"lte": end_ts},
            }

        return {"range": {self.DATE_FIELD: op_range.get(op_key, {})}}

    def convert_list_info_to_elastic_dict(
        self,
        op_key: str,
        info_list: list[dict],
        lb_key: Literal["lk", "lp"] = None,
        rb_key: Literal["rk", "rp"] = None,
    ):
        info_l, info_r = info_list
        if info_l:
            l_date_type = info_l["date_type"]
            l_ts_beg, l_ts_end = info_l["range_ts"]
        else:
            l_date_type = None
            l_ts_beg, l_ts_end = None, None

        if info_r:
            r_date_type = info_r["date_type"]
            r_ts_beg, r_ts_end = info_r["range_ts"]
        else:
            r_date_type = None
            r_ts_beg, r_ts_end = None, None

        if l_date_type == "date_num_unit" and r_date_type == "date_num_unit":
            # swap l_ts and r_ts for date_num_unit
            # for example, [3d,1d) should be (1d,3d]
            if l_ts_beg < r_ts_beg:
                l_ts_beg, r_ts_beg = r_ts_beg, l_ts_beg
                l_ts_end, r_ts_end = r_ts_end, l_ts_end
                lb_key = "l" + rb_key[-1]
                rb_key = "r" + lb_key[-1]

        es_op_val = {}

        if l_ts_beg:
            if l_date_type == "date_num_unit":
                if lb_key == "lk":
                    # e.g.: "[1d" means "<= end of 1d", which contains 1d itself
                    es_op_val["lte"] = l_ts_end
                else:  # lb_key == "lp"
                    # e.g.: "(1d" means "< beg of 1d", which does not contain 1d itself
                    es_op_val["lt"] = l_ts_beg
            else:
                if lb_key == "lk":
                    es_op_val["gte"] = l_ts_beg
                else:  # lb_key == "lp"
                    es_op_val["gt"] = l_ts_end

        if r_ts_beg:
            if r_date_type == "date_num_unit":
                if rb_key == "rk":
                    # e.g.: "3d]" means ">= beg of 3d", which contains 3d itself
                    es_op_val["gte"] = r_ts_beg
                else:
                    # e.g.: "3d)" means "> end of 3d", which does not contain 3d itself
                    es_op_val["gt"] = r_ts_end
            else:
                if rb_key == "rk":
                    es_op_val["lte"] = r_ts_end
                else:
                    es_op_val["lt"] = r_ts_beg

        return {"range": {self.DATE_FIELD: es_op_val}}

    def convert_single(self, node: DslExprNode) -> dict:
        val_single = node.find_child_with_key("date_val_single")
        val_node = val_single.find_child_with_key(
            ["date_num_unit", "date_iso", "date_recent"]
        )
        if val_node.is_key("date_iso"):
            info = self.get_info_of_date_iso(val_node)
        elif val_node.is_key("date_recent"):
            info = self.get_info_of_date_recent(val_node)
        elif val_node.is_key("date_num_unit"):
            info = self.get_info_of_date_num_unit(val_node)
        else:
            logger.warn(f"× Invalid date_val_single key: {val_node.key}")
            info = None
        return info

    def convert_list(self, node: DslExprNode) -> list[dict]:
        val_list = node.find_child_with_key("date_val_list")
        val_left_node = val_list.find_child_with_key("date_val_left")
        val_right_node = val_list.find_child_with_key("date_val_right")
        if val_left_node:
            info_left = self.convert_single(val_left_node)
        else:
            info_left = None
        if val_right_node:
            info_right = self.convert_single(val_right_node)
        else:
            info_right = None
        return [info_left, info_right]

    def convert(self, node: DslExprNode) -> dict:
        """node key is `date_expr`"""
        op_val_node = node.find_child_with_key(
            ["date_op_val_single", "date_op_val_list"]
        )
        if not op_val_node:
            return None
        if op_val_node.is_key("date_op_val_single"):
            info = self.convert_single(op_val_node)
            op_key = op_val_node.find_child_with_key(
                "date_op_single"
            ).get_deepest_node_key()
            elastic_dict = self.convert_single_info_to_elastic_dict(op_key, info)
        elif op_val_node.is_key("date_op_val_list"):
            info_list = self.convert_list(op_val_node)
            op_key = op_val_node.find_child_with_key(
                "date_op_list"
            ).get_deepest_node_key()
            lb_node = op_val_node.find_child_with_key("lb")
            if lb_node:
                lb_key = lb_node.get_deepest_node_key()
            else:
                lb_key = None
            rb_node = op_val_node.find_child_with_key("rb")
            if rb_node:
                rb_key = rb_node.get_deepest_node_key()
            else:
                rb_key = None
            elastic_dict = self.convert_list_info_to_elastic_dict(
                op_key, info_list, lb_key, rb_key
            )
        else:
            logger.warn(f"× Invalid date_op_val key: {op_val_node.key}")
            return None
        is_or_node_parent = bool(node.find_parent_with_key("or"))
        if not is_or_node_parent:
            elastic_dict = {"bool": {"filter": elastic_dict}}
        return elastic_dict
