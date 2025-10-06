import re
from tclogger import logger
from converters.dsl.node import DslExprNode

DURA_UNIT_MAP = {"d": 86400, "h": 3600, "m": 60, "s": 1}
DURA_OP_MAP = {
    "gt": "gt",
    "lt": "lt",
    "geqs": "gte",
    "leqs": "lte",
    "eqs": "gte",
}
DURA_LR_OP_MAP = {
    "lk": "gte",
    "lp": "gt",
    "rk": "lte",
    "rp": "lt",
}


RE_DURA_DIGITS_UNIT = r"(\d+)([dhms])"
REP_DURA_DIGITS_UNIT = re.compile(RE_DURA_DIGITS_UNIT)


class DuraExprElasticConverter:
    DURA_FIELD = "duration"

    def dura_str_to_number(self, s: str) -> int:
        """Convert duration string to seconds number: \n
        '1h30m' -> 5400"""
        matches = REP_DURA_DIGITS_UNIT.findall(s)
        if not matches:
            return None
        number = 0
        for digits, unit in matches:
            number += int(digits) * DURA_UNIT_MAP.get(unit, 1)
        return number

    def val_node_to_number(self, val_node: DslExprNode) -> int:
        """node key is `dura_val_single`"""
        dura_digits_node = val_node.find_child_with_key("dura_digits")
        if dura_digits_node:
            dura_digits = dura_digits_node.get_deepest_node_value()
            number = int(dura_digits)
            return number

        dura_string_node = val_node.find_child_with_key("dura_string")
        if dura_string_node:
            dura_str = dura_string_node.get_deepest_node_value()
            number = self.dura_str_to_number(dura_str)
            return number
        return None

    def field_op_number_to_elastic_dict(self, op_key: str, number: int) -> dict:
        if op_key not in DURA_OP_MAP:
            op_key = "gte"
        es_op = DURA_OP_MAP[op_key]
        return {"range": {self.DURA_FIELD: {es_op: number}}}

    def get_lr_op_val_dict(self, val_node: DslExprNode) -> dict:
        """node key is `dura_val_list`.
        Return dict with keys `l_val`, `r_val`, `l_op`, `r_op`.
        """
        l_val, r_val = None, None
        l_op, r_op = None, None
        l_val_node = val_node.find_child_with_key("dura_val_left")
        r_val_node = val_node.find_child_with_key("dura_val_right")
        l_op_node = val_node.find_child_with_key("lb")
        r_op_node = val_node.find_child_with_key("rb")
        if l_val_node:
            l_node = l_val_node.find_child_with_key("dura_val_single")
            l_val = self.val_node_to_number(l_node)
        if r_val_node:
            r_node = r_val_node.find_child_with_key("dura_val_single")
            r_val = self.val_node_to_number(r_node)
        if l_op_node:
            l_op = l_op_node.get_deepest_node_key()
        if r_op_node:
            r_op = r_op_node.get_deepest_node_key()
        res = {"l_val": l_val, "r_val": r_val, "l_op": l_op, "r_op": r_op}
        return res

    def lr_op_val_dict_to_elastic_dict(self, op_val_dict: dict) -> dict:
        l_val = op_val_dict["l_val"]
        r_val = op_val_dict["r_val"]
        l_op = op_val_dict["l_op"]
        r_op = op_val_dict["r_op"]
        if l_val is not None and r_val is not None and l_val > r_val:
            l_val, r_val = r_val, l_val
            l_op = l_op.replace("l", "r")
            r_op = r_op.replace("r", "l")
        l_es_op = DURA_LR_OP_MAP.get(l_op, "gte")
        r_es_op = DURA_LR_OP_MAP.get(r_op, "lte")
        es_op_val_dict = {}
        if l_val is not None:
            es_op_val_dict[l_es_op] = l_val
        if r_val is not None:
            es_op_val_dict[r_es_op] = r_val
        return {"range": {self.DURA_FIELD: es_op_val_dict}}

    def convert(self, node: DslExprNode) -> dict:
        """node key is `dura_expr`"""
        op_node = node.find_child_with_key("dura_op")
        val_node = node.find_child_with_key(["dura_val_single", "dura_val_list"])
        if not op_node or not val_node:
            return None

        op_key = op_node.get_deepest_node_key()
        if val_node.is_key("dura_val_single"):
            number = self.val_node_to_number(val_node)
            elastic_dict = self.field_op_number_to_elastic_dict(op_key, number)
        elif val_node.is_key("dura_val_list"):
            lr_val_op_dict = self.get_lr_op_val_dict(val_node)
            elastic_dict = self.lr_op_val_dict_to_elastic_dict(lr_val_op_dict)
        else:
            logger.warn(f"× Invalid dura_val key: {val_node.key}")
            return None
        is_or_node_parent = bool(node.find_parent_with_key("or"))
        if not is_or_node_parent:
            elastic_dict = {"bool": {"filter": elastic_dict}}
        return elastic_dict
