from converters.dsl.constants import TEXT_TYPES
from converters.dsl.node import DslExprNode
from converters.dsl.constants import WORD_OPS

TEXT_MATCH_TYPE = "multi_match"
MULTI_MATCH_PARAMS = {
    "type": "phrase_prefix",
    "fields": ["title^1.0", "tags^2.0"],
}


class WordExprElasticConverter:
    def clean_word_val(self, val: str) -> str:
        return val.strip('" ')

    def query_to_match_dict(self, query: str) -> dict:
        return {TEXT_MATCH_TYPE: {"query": query, **MULTI_MATCH_PARAMS}}

    def convert_multi(self, nodes: list[DslExprNode]) -> dict:
        texts = []
        for node in nodes:
            text_node = node.find_child_with_key(["text_quoted", "text_strict"])
            text = text_node.get_deepest_node_value()
            if text:
                text = self.clean_word_val(text)
                texts.append(text)
        if len(texts) == 1:
            return self.query_to_match_dict(texts[0])
        else:
            return [self.query_to_match_dict(text) for text in texts]

    def convert(self, node: DslExprNode) -> dict:
        """node key is `word_expr`"""
        val_node = node.find_child_with_key("word_val")
        if not val_node:
            return {}
        single_nodes = val_node.find_all_child_with_key("word_val_single")
        match_dict = self.convert_multi(single_nodes)

        key_op_node = node.find_child_with_key(["word_key_op", "word_sp"])
        if key_op_node:
            op_node = key_op_node.find_child_with_key(WORD_OPS)
            op = op_node.find_child_with_key(WORD_OPS).get_deepest_node_key()
            if op == "neq":
                bool_key = "must_not"
            elif op == "qs":
                bool_key = "should"
            else:
                bool_key = "must"
        else:
            bool_key = "must"

        if len(match_dict) == 1 and bool_key == "must":
            elastic_dict = match_dict
        else:
            elastic_dict = {"bool": {bool_key: match_dict}}

        return elastic_dict
