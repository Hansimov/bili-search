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

    def val_to_int(self, val: str) -> int:
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

    def op_val_to_es_dict(
        self,
        field: str,
        op: Literal["=", "<", ">", "<=", ">="],
        val: str,
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
            val_int = self.val_to_int(val)
        else:
            return {}

        if op == "=":
            res_val = {"gte": val_int, "lte": val_int}
        else:
            op_str = self.OP_MAPS[op]
            res_val[op_str] = val_int

        res = {f"stat.{field}": res_val}
        return res

    def range_val_to_es_dict(
        self,
        field: str,
        lb: str,
        lval: str,
        rval: str,
        rb: str,
    ) -> dict:
        """
        - {'field':'coin', 'field_type':'stat', 'op':'=', 'lb':'[', 'lval':'1000', 'rval':'2000', 'rb':')', 'val_type':'range'}
            -> {"stat.coin": {"gte": 1000, "lt": 2000}}
        - {'field':'danmaku', 'field_type':'stat', 'op':'=', 'lb':'[', 'lval':'', 'rval':'2000', 'rb':')', 'val_type':'range'}
            -> {"stat.danmaku": {"lt": 2000}}
        """
        res = {}
        res_val = {}

        if lval:
            lval_int = self.val_to_int(lval)
            lb_str = self.BRACKET_MAPS[lb]
        else:
            lval_int = None
            lb_str = None

        if rval:
            rval_int = self.val_to_int(rval)
            rb_str = self.BRACKET_MAPS[rb]
        else:
            rval_int = None
            rb_str = None

        if lval_int and lb_str:
            res_val[lb_str] = lval_int
        if rval_int and rb_str:
            res_val[rb_str] = rval_int

        res = {f"stat.{field}": res_val}
        return res

    def filter_dict_to_es_dict(self, filter_dict: dict) -> dict:
        """
        - {'field':'view', 'field_type':'stat', 'op':'>', 'val':'1000','val_type':'value'}
            -> {"stat.view": {"gt": 1000}}
        - {'field':'coin', 'field_type':'stat', 'op':'=', 'lb':'[', 'lval':'', 'rval':'2000', 'rb':')', 'val_type':'range'}
            -> {"stat.coin": {"lt": 2000}}
        - {'field':'reply', 'field_type':'stat', 'op':'=', 'lb':'[', 'lval':'100', 'rval':'2000', 'rb':']', 'val_type':'range'}
            -> {"stat.reply": {"gte": 100, "lte": 2000}}
        """
        res = {}
        if filter_dict["val_type"] == "value":
            res = self.op_val_to_es_dict(
                filter_dict["field"], filter_dict["op"], filter_dict["val"]
            )
        elif filter_dict["val_type"] == "range":
            res = self.range_val_to_es_dict(
                filter_dict["field"],
                filter_dict["lb"],
                filter_dict["lval"],
                filter_dict["rval"],
                filter_dict["rb"],
            )
        else:
            logger.warn(f"× No matching val type: {filter_dict['val_type']}")
        return res


if __name__ == "__main__":
    converter = StatFieldConverter()
    for field_op_val in [
        ("view", "<", "1000"),
        ("like", ">", "100k"),
        ("coin", "<=", "100K"),
        ("favorite", ">=", "100kk"),
        ("danmaku", "=", "100w"),
        ("reply", "<", "100W"),
        ("share", ">", "100kW"),
        ("view", "<=", "10m"),
        ("coin", ">=", "10M"),
        ("reply", "=", "1百万亿"),
    ]:
        field, op, val = field_op_val
        logger.note(f"> {field}{op}{val}:", end=" ")
        # res = converter.val_to_int(val_unit)
        res = converter.op_val_to_es_dict(field, op, val)
        logger.success(f"{res}")

    # python -m converters.stat_field_converter
