import re

from tclogger import logger
from typing import Literal


class StatFieldConverter:
    RE_VIEW = r"(播放|bofang|bof|bf|view|vw|v)"
    RE_LIKE = r"(点赞|dianzan|dianz|dz|like|lk|l)"
    RE_COIN = r"(投币|toubi|toub|tb|coin|cn|c)"
    RE_FAVORITE = r"(收藏|shouchang|shouc|sc|favorite|fav|fv|star|f)"
    RE_REPLY = r"(评论|回复|pinglun|pingl|huifu|huif|pl|hf|reply|rp|r)"
    RE_DANMAKU = r"(弹幕|danmu|danm|dm|danmaku|m)"
    RE_SHARE = r"(分享|转发|fenxiang|fenx|zhuanfa|zhuanf|fx|zf|share|sh|s)"
    RE_STAT_FIELD = rf"({RE_VIEW}|{RE_LIKE}|{RE_COIN}|{RE_FAVORITE}|{RE_REPLY}|{RE_DANMAKU}|{RE_SHARE})"

    REP_VIEW = rf"(?P<view>{RE_VIEW})"
    REP_LIKE = rf"(?P<like>{RE_LIKE})"
    REP_COIN = rf"(?P<coin>{RE_COIN})"
    REP_FAVORITE = rf"(?P<favorite>{RE_FAVORITE})"
    REP_REPLY = rf"(?P<reply>{RE_REPLY})"
    REP_DANMAKU = rf"(?P<danmaku>{RE_DANMAKU})"
    REP_SHARE = rf"(?P<share>{RE_SHARE})"
    REP_STAT_FIELD = rf"(?P<stat_field>{REP_VIEW}|{REP_LIKE}|{REP_COIN}|{REP_FAVORITE}|{REP_REPLY}|{REP_DANMAKU}|{REP_SHARE})"

    RE_UNIT = r"[百千kK万wWmM亿]*"
    RE_STAT_VAL = rf"(\d+\s*{RE_UNIT})"
    REP_STAT_VAL = rf"((?P<num>\d+)\s*(?P<unit>{RE_UNIT}))"

    UNIT_MAPS = {"百": 100, "千kK": 1000, "万wW": 10000, "mM": 1000000, "亿": 100000000}
    OP_EN_MAPS = {">": "gt", "<": "lt", ">=": "gte", "<=": "lte"}
    OP_ZH_MAPS = {"》": "gt", "《": "lt", "》=": "gte", "《=": "lte"}
    OP_MAPS = {**OP_EN_MAPS, **OP_ZH_MAPS}
    BRACKET_EN_MAPS = {"(": "gt", ")": "lt", "[": "gte", "]": "lte"}
    BRACKET_ZH_MAPS = {"（": "gt", "）": "lt", "【": "gte", "】": "lte"}
    BRACKET_MAPS = {**BRACKET_EN_MAPS, **BRACKET_ZH_MAPS}

    def val_unit_to_int(self, val: str) -> int:
        if val == "":
            return None

        match = re.match(self.REP_STAT_VAL, val)
        if match:
            num = int(match.group("num"))
            unit = match.group("unit")
            for ch in unit:
                for k, v in self.UNIT_MAPS.items():
                    if ch in k:
                        num *= v
                        break
            return int(num)
        else:
            logger.warn(f"× No matching stat val: {val}")
            return None

    def val_to_elastic_dict(
        self,
        field: str,
        val: str,
        op: Literal["=", "<", ">", "<=", ">="],
    ) -> dict:
        """
        Examples:
        - {'field':'view', 'field_type':'stat', 'op':'>', 'val':'1000','val_type':'value'}
            -> {"stat.view": {"gt": 1000}}
        - {'field':'favorite', 'field_type':'stat', 'op':'<=', 'val':'1000', 'val_type':'value'}
            -> {"stat.favorite": {"lte": 1000}}
        """
        res = {}
        res_val = {}

        if val:
            val_int = self.val_unit_to_int(val)
        else:
            return {}

        if op == "=":
            res_val = {"gte": val_int, "lte": val_int}
        else:
            op_str = self.OP_MAPS[op]
            res_val[op_str] = val_int

        res = {f"stat.{field}": res_val}
        return res

    def range_to_elastic_dict(
        self,
        field: str,
        val: str,
        lb: str,
        lval: str,
        rval: str,
        rb: str,
        op: str = "=",
        field_type: Literal["stat", "date"] = "stat",
    ) -> dict:
        """
        - {'field':'coin', 'field_type':'stat', 'op':'=', 'lb':'[', 'lval':'1000', 'rval':'2000', 'rb':')', 'val_type':'range'}
            -> {"stat.coin": {"gte": 1000, "lt": 2000}}
        - {'field':'danmaku', 'field_type':'stat', 'op':'=', 'lb':'[', 'lval':'', 'rval':'2000', 'rb':')', 'val_type':'range'}
            -> {"stat.danmaku": {"lt": 2000}}
        """
        pass


if __name__ == "__main__":
    converter = StatFieldConverter()
    for val_unit in [
        "1000",
        "100k",
        "100K",
        "100kk",
        "100w",
        "100W",
        "100kW",
        "10m",
        "10M",
        "1百万亿",
    ]:
        logger.note(f"> {val_unit}:", end=" ")
        res = converter.val_unit_to_int(val_unit)
        logger.success(f"{res}")

    # python -m converters.stat_field_converter
