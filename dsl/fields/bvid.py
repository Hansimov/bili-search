import re

from dsl.node import DslExprNode
from collections import defaultdict


RE_BVID = re.compile(r"bv[0-9A-Za-z]+", re.IGNORECASE)
RE_AVID = re.compile(r"(av)?[0-9]+", re.IGNORECASE)

AVID_FIELD = "aid"
BVID_FIELD = "bvid.keyword"


class BvidExprElasticConverter:
    def parse_bvid_value(self, value: str) -> tuple[str, int]:
        value = value.strip('" ')
        if RE_AVID.match(value):
            return AVID_FIELD, int(value)
        else:
            return BVID_FIELD, value

    def convert_multi(self, nodes: list[DslExprNode]) -> dict:
        field_values = []
        for node in nodes:
            val_node = node.find_child_with_key(["bvid_str"])
            if val_node:
                value = val_node.get_deepest_node_value()
                if value:
                    field, value = self.parse_bvid_value(value)
                    field_values.append((field, value))
        if len(field_values) == 1:
            field, value = field_values[0]
            return {"term": {field: value}}
        else:
            vid_dict = defaultdict(list)
            for field, value in field_values:
                vid_dict[field].append(value)
            if AVID_FIELD in vid_dict and BVID_FIELD in vid_dict:
                res = {
                    "bool": {
                        "should": [
                            {"terms": {AVID_FIELD: vid_dict[AVID_FIELD]}},
                            {"terms": {BVID_FIELD: vid_dict[BVID_FIELD]}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            else:
                res = {"terms": dict(vid_dict)}
            return res

    def convert(self, node: DslExprNode) -> dict:
        """node key is `bvid_expr`"""
        val_node = node.find_child_with_key("bvid_val")
        if not val_node:
            return {}
        single_nodes = val_node.find_all_childs_with_key("bvid_val_single")
        elastic_dict = self.convert_multi(single_nodes)
        op_node = node.find_child_with_key("bvid_op")
        if op_node:
            op = op_node.get_deepest_node_key()
        else:
            op = "eq"
        if op in ["neq"]:
            elastic_dict = {"bool": {"must_not": elastic_dict}}
        else:
            elastic_dict = {"bool": {"filter": elastic_dict}}

        return elastic_dict


def bvids_to_filter(bvids: list[str]) -> dict:
    return {"terms": {BVID_FIELD: bvids}}
