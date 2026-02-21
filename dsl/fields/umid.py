from dsl.node import DslExprNode

UID_FIELD = "owner.mid"


class UmidExprElasticConverter:
    def parse_uid_value(self, val: str) -> str:
        return int(val.strip('" '))

    def convert_multi(self, nodes: list[DslExprNode]) -> dict:
        vals = []
        for node in nodes:
            val_node = node.find_child_with_key(["digits"])
            if val_node:
                val = val_node.get_deepest_node_value()
                if val:
                    val = self.parse_uid_value(val)
                    vals.append(val)
        if len(vals) == 1:
            return {"term": {UID_FIELD: vals[0]}}
        else:
            return {"terms": {UID_FIELD: vals}}

    def convert(self, node: DslExprNode) -> dict:
        """node key is `uid_expr`"""
        val_node = node.find_child_with_key("uid_val")
        if not val_node:
            return {}
        single_nodes = val_node.find_all_childs_with_key("uid_val_single")
        elastic_dict = self.convert_multi(single_nodes)
        op_node = node.find_child_with_key("uid_op")
        if op_node:
            op = op_node.get_deepest_node_key()
        else:
            op = "eqs"
        if op in ["neq", "neqs"]:
            elastic_dict = {"bool": {"must_not": elastic_dict}}
        else:
            elastic_dict = {"bool": {"filter": elastic_dict}}

        return elastic_dict
