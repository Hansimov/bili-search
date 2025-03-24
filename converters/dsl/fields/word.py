from typing import Literal, Union

from converters.dsl.node import DslExprNode
from converters.dsl.constants import WORD_OPS, TEXT_TYPES
from elastics.videos.constants import SEARCH_MATCH_FIELDS, SEARCH_BOOSTED_FIELDS
from elastics.videos.constants import SUGGEST_MATCH_FIELDS, SUGGEST_BOOSTED_FIELDS
from elastics.videos.constants import DATE_MATCH_FIELDS, DATE_BOOSTED_FIELDS
from elastics.videos.constants import QUERY_TYPE, QUERY_TYPE_DEFAULT
from elastics.videos.constants import MATCH_TYPE, MATCH_BOOL, MATCH_OPERATOR
from elastics.videos.constants import SEARCH_MATCH_TYPE, SUGGEST_MATCH_TYPE


class WordExprElasticConverter:
    def __init__(self, mode: Literal["search", "suggest"] = "search"):
        self.mode = mode
        self.switch_mode(mode)

    def switch_mode(
        self,
        mode: Literal["search", "suggest"] = None,
        match_fields: list[str] = None,
        date_match_fields: list[str] = None,
        match_type: MATCH_TYPE = None,
        query_type: QUERY_TYPE = None,
    ):
        if mode == "suggest":
            self.match_fields = match_fields or SUGGEST_MATCH_FIELDS
        else:
            self.match_fields = match_fields or SEARCH_MATCH_FIELDS
        self.date_match_fields = date_match_fields or DATE_MATCH_FIELDS
        self.match_type = match_type or SEARCH_MATCH_TYPE
        self.query_type = query_type or QUERY_TYPE_DEFAULT

    def clean_word_val(self, val: str) -> str:
        return val.strip('" ')

    def query_to_match_dict(self, query: str, is_date_format: bool = False) -> dict:
        if is_date_format:
            match_fields = self.date_match_fields
        else:
            match_fields = self.match_fields
        return {
            self.query_type: {
                "query": query,
                "type": self.match_type,
                "fields": match_fields,
            }
        }

    def node_to_match_dict(self, node: DslExprNode) -> dict:
        text_node = node.find_child_with_key(TEXT_TYPES)
        text = text_node.get_deepest_node_value()
        is_date_format = node.extras.get("is_date_format", False)
        if text:
            text = self.clean_word_val(text)
            return self.query_to_match_dict(text, is_date_format=is_date_format)
        else:
            return {}

    def convert_multi(self, nodes: list[DslExprNode]) -> Union[dict, list[dict]]:
        node_dicts = []
        for node in nodes:
            node_dict = self.node_to_match_dict(node)
            if node_dict:
                node_dicts.append(node_dict)
        if len(node_dicts) == 1:
            return node_dicts[0]
        else:
            return node_dicts

    def convert(self, node: DslExprNode) -> dict:
        """node key is `word_expr`"""
        val_node = node.find_child_with_key("word_op_val")
        if not val_node:
            return {}
        single_nodes = val_node.find_all_childs_with_key("word_val_single")
        match_dict = self.convert_multi(single_nodes)

        op_node = node.find_child_with_key(["word_op", "word_sp"])
        op = "eq"
        if op_node:
            op_node = op_node.find_child_with_key(WORD_OPS)
            op = op_node.find_child_with_key(WORD_OPS).get_deepest_node_key()

        if op == "eq":
            elastic_dict = {"bool": {"must": match_dict}}
        elif op == "neq":
            elastic_dict = {"bool": {"must_not": match_dict}}
        elif op == "qs":
            elastic_dict = {
                "bool": {
                    "should": match_dict,
                    "minimum_should_match": 0,
                }
            }
        else:
            elastic_dict = {"bool": {"must": match_dict}}

        return elastic_dict


class WordNodeToExprConstructor:
    def get_word_key_op_str(self, node: DslExprNode) -> tuple[str, str, str]:
        word_key_node = node.find_child_with_key("word_key")
        if word_key_node:
            word_key_str = word_key_node.get_deepest_node_value()
        else:
            word_key_str = ""
        word_op_node = node.find_child_with_key("word_op")
        if word_op_node:
            word_op_key = word_op_node.get_deepest_node_key()
        else:
            word_op_key = ""
        if word_op_key == "neq":
            word_op_str = "!="
        elif word_op_key == "eq":
            word_op_str = "="
        else:
            word_op_str = ""
        word_sp_node = node.find_child_with_key("word_sp")
        if word_sp_node:
            word_sp_str = "?"
        else:
            word_sp_str = ""

        return word_key_str, word_op_str, word_sp_str

    def construct(self, node: DslExprNode) -> str:
        """node key is `word_expr`.
        This would construct word expr str from a DslExprNode with key `word_expr`."""
        word_key_str, word_op_str, word_sp_str = self.get_word_key_op_str(node)
        word_val_single_nodes = node.find_all_childs_with_key("word_val_single")
        if len(word_val_single_nodes) == 1:
            word_val_joined_str = word_val_single_nodes[0].get_deepest_node_value()
        elif len(word_val_single_nodes) > 1:
            word_val_strs = [
                word_val_node.get_deepest_node_value()
                for word_val_node in word_val_single_nodes
            ]
            word_val_joined_str = "[" + ",".join(word_val_strs) + "]"
        else:
            word_val_joined_str = ""

        return f"{word_key_str}{word_op_str}{word_val_joined_str}{word_sp_str}"
