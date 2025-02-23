from converters.dsl.node import DslExprNode
from typing import Union


class ExprElasticConverter:
    pass


class DateExprElasticConverter(ExprElasticConverter):
    def is_single(self, node: DslExprNode) -> bool:
        return node.key == "date_op_val_single"

    def is_list(self, node: DslExprNode) -> bool:
        return node.key == "date_op_val_list"

    def convert_single(self, node: DslExprNode) -> dict:
        op_single = node.find_child_with_key("date_op_single")
        val_single = node.find_child_with_key("date_val_single")
        op_key = op_single.get_deepest_node_key()
        val_node = val_single.find_child_with_key(
            ["date_num_unit", "date_iso", "date_recent", "date_past"]
        )
        if val_node.key == "date_iso":
            iso_expr_node = node.find_child_with_key(
                ["yyyymmddhh", "yyyymmdd", "yyyymm", "mmddhh", "mmdd", "yyyy"]
            )
            value_dict = iso_expr_node.get_value_dict_by_keys(
                ["yyyy", "mm", "dd", "hh"]
            )
            return value_dict

    def convert_list(self, node: DslExprNode) -> dict:
        pass

    def convert(self, node: DslExprNode) -> dict:
        """node key is `date_expr`"""
        op_val_node = node.find_child_with_key(
            ["date_op_val_single", "date_op_val_list"]
        )
        if not op_val_node:
            return None
        if self.is_single(op_val_node):
            return self.convert_single(op_val_node)
        elif self.is_list(op_val_node):
            return self.convert_list(op_val_node)
