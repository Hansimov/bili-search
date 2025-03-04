from converters.dsl.constants import USER_OPS
from converters.dsl.node import DslExprNode


class UserExprElasticConverter:
    USER_FIELD = "owner.name.keyword"

    def clean_user_val(self, val: str) -> str:
        return val.strip('" ')

    def convert_multi(self, nodes: list[DslExprNode]) -> dict:
        texts = []
        for node in nodes:
            text_node = node.find_child_with_key(["text_quoted", "text_strict"])
            text = text_node.get_deepest_node_value()
            if text:
                text = self.clean_user_val(text)
                texts.append(text)
        if len(texts) == 1:
            return {"term": {self.USER_FIELD: texts[0]}}
        else:
            return {"terms": {self.USER_FIELD: texts}}

    def convert(self, node: DslExprNode) -> dict:
        """node key is `user_expr`"""
        val_node = node.find_child_with_key("user_val")
        if not val_node:
            return {}
        key_op_node = node.find_child_with_key("user_key_op")
        single_nodes = val_node.find_all_childs_with_key("user_val_single")
        elastic_dict = self.convert_multi(single_nodes)

        op_node = key_op_node.find_child_with_key(USER_OPS)
        op = op_node.find_child_with_key(USER_OPS).get_deepest_node_key()
        if op in ["at_neq", "neq"]:
            elastic_dict = {"must_not": elastic_dict}

        return elastic_dict
