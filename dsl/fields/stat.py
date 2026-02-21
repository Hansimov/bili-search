from tclogger import logger
from dsl.constants import STAT_KEYS
from dsl.node import DslExprNode
from typing import Union

STAT_UNIT_MAP = {"百": 100, "千kK": 1000, "万wW": 10000, "mM": 1000000, "亿": 100000000}
STAT_OP_MAP = {
    "gt": "gt",
    "lt": "lt",
    "geqs": "gte",
    "leqs": "lte",
    "eqs": "gte",
}
STAT_LR_OP_MAP = {
    "lk": "gte",
    "lp": "gt",
    "rk": "lte",
    "rp": "lt",
}


class StatExprElasticConverter:
    def get_stat_field(self, node: DslExprNode) -> str:
        """node key is `stat_expr`"""
        key_node = node.find_child_with_key("stat_key")
        stat_key = key_node.get_deepest_node_key()
        if stat_key:
            stat_field = stat_key.split("_")[0]
            return f"stat.{stat_field}"
        else:
            logger.warn("× No stat_key found")
            return None

    def unit_to_number(self, unit: str) -> int:
        res = 1
        for ch in unit:
            for k, v in STAT_UNIT_MAP.items():
                if ch in k:
                    res *= v
                    break
        return res

    def val_node_to_number(self, val_node: DslExprNode) -> int:
        """node key is `stat_val_single`"""
        stat_num_node = val_node.find_child_with_key("stat_num")
        if stat_num_node:
            stat_num = stat_num_node.get_deepest_node_value()
        else:
            stat_num = None

        stat_unit_node = val_node.find_child_with_key("stat_unit")
        if stat_unit_node:
            stat_unit = stat_unit_node.get_deepest_node_value()
        else:
            stat_unit = None

        if stat_num is None and stat_unit is None:
            return None
        if stat_num is None:
            stat_num = 1
        else:
            stat_num = int(stat_num)

        if stat_unit is None:
            stat_unit_number = 1
        else:
            stat_unit_number = self.unit_to_number(stat_unit)

        return stat_num * stat_unit_number

    def field_op_number_to_elastic_dict(
        self, stat_field: str, op_key: str, number: int
    ) -> dict:
        if op_key not in STAT_OP_MAP:
            op_key = "gte"
        es_op = STAT_OP_MAP[op_key]
        return {"range": {stat_field: {es_op: number}}}

    def get_lr_op_val_dict(self, op_val_node: DslExprNode) -> dict:
        """node key is `stat_op_val_list`.
        Return dict with keys `l_val`, `r_val`, `l_op`, `r_op`.
        """
        l_val, r_val = None, None
        l_op, r_op = None, None
        l_val_node = op_val_node.find_child_with_key("stat_val_left")
        r_val_node = op_val_node.find_child_with_key("stat_val_right")
        l_op_node = op_val_node.find_child_with_key("lb")
        r_op_node = op_val_node.find_child_with_key("rb")
        if l_val_node:
            l_node = l_val_node.find_child_with_key("stat_val_single")
            l_val = self.val_node_to_number(l_node)
        if r_val_node:
            r_node = r_val_node.find_child_with_key("stat_val_single")
            r_val = self.val_node_to_number(r_node)
        if l_op_node:
            l_op = l_op_node.get_deepest_node_key()
        if r_op_node:
            r_op = r_op_node.get_deepest_node_key()
        res = {"l_val": l_val, "r_val": r_val, "l_op": l_op, "r_op": r_op}
        return res

    def lr_op_val_dict_to_elastic_dict(
        self, stat_field: str, op_val_dict: dict
    ) -> dict:
        l_val = op_val_dict["l_val"]
        r_val = op_val_dict["r_val"]
        l_op = op_val_dict["l_op"]
        r_op = op_val_dict["r_op"]
        if l_val is not None and r_val is not None and l_val > r_val:
            l_val, r_val = r_val, l_val
            l_op = l_op.replace("l", "r")
            r_op = r_op.replace("r", "l")
        l_es_op = STAT_LR_OP_MAP.get(l_op, "gte")
        r_es_op = STAT_LR_OP_MAP.get(r_op, "lte")
        es_op_val_dict = {}
        if l_val is not None:
            es_op_val_dict[l_es_op] = l_val
        if r_val is not None:
            es_op_val_dict[r_es_op] = r_val
        return {"range": {stat_field: es_op_val_dict}}

    def convert(self, node: DslExprNode) -> dict:
        """node key is `stat_expr`"""
        stat_field = self.get_stat_field(node)
        op_val_node = node.find_child_with_key(
            ["stat_op_val_single", "stat_op_val_list"]
        )
        if not op_val_node:
            return None
        op_node = op_val_node.find_child_with_key("op_rel")
        op_key = op_node.get_deepest_node_key()
        if op_val_node.is_key("stat_op_val_single"):
            val_node = op_val_node.find_child_with_key("stat_val_single")
            number = self.val_node_to_number(val_node)
            elastic_dict = self.field_op_number_to_elastic_dict(
                stat_field, op_key, number
            )
        elif op_val_node.is_key("stat_op_val_list"):
            lr_val_op_dict = self.get_lr_op_val_dict(op_val_node)
            elastic_dict = self.lr_op_val_dict_to_elastic_dict(
                stat_field, lr_val_op_dict
            )
        else:
            logger.warn(f"× Invalid stat_op_val key: {op_val_node.key}")
            return None
        is_or_node_parent = bool(node.find_parent_with_key("or"))
        if not is_or_node_parent:
            elastic_dict = {"bool": {"filter": elastic_dict}}
        return elastic_dict
