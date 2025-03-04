from tclogger import logger, dict_to_str

from converters.dsl.constants import BOOL_OPS, ATOMS, MSM
from converters.dsl.parse import DslLarkParser
from converters.dsl.node import DslExprNode, DslTreeBuilder, DslTreeExprGrouper
from converters.dsl.node import DslExprTreeFlatter
from converters.dsl.fields.date import DateExprElasticConverter
from converters.dsl.fields.user import UserExprElasticConverter
from converters.dsl.fields.word import WordExprElasticConverter
from converters.dsl.fields.bool import BoolElasticReducer


class DslExprToElasticConverter(DslTreeExprGrouper):
    def __init__(self):
        self.parser = DslLarkParser()
        self.builder = DslTreeBuilder()
        self.grouper = DslTreeExprGrouper()
        self.flatter = DslExprTreeFlatter()
        self.reducer = BoolElasticReducer()
        self.date_converter = DateExprElasticConverter()
        self.user_converter = UserExprElasticConverter()
        self.word_converter = WordExprElasticConverter()

    def atom_node_to_elastic_dict(self, node: DslExprNode) -> dict:
        expr_node = node.find_child_with_key(ATOMS)
        if expr_node.is_key("date_expr"):
            return self.date_converter.convert(node)
        elif expr_node.is_key("user_expr"):
            return self.user_converter.convert(node)
        elif expr_node.is_key("word_expr"):
            return self.word_converter.convert(node)
        else:
            logger.warn(f"× Unknown atom_node: <{expr_node.key}>")
            return {}

    def word_cluster_to_bool_clauses(self, node: DslExprNode) -> list[dict]:
        word_expr_nodes = node.find_all_childs_with_key("word_expr")
        bool_clauses = [
            self.word_converter.convert(word_expr_node)
            for word_expr_node in word_expr_nodes
        ]
        return bool_clauses

    def bool_node_to_elastic_dict(self, node: DslExprNode) -> dict:
        if node.is_key("pa"):
            return self.node_to_elastic_dict(node.children[0])
        elif node.is_key(["and", "co"]):
            bool_clauses = []
            if (
                node.is_all_bool_childs_are_co_and()
                and node.is_all_atom_childs_are_word_expr()
            ):
                bool_clauses = self.word_cluster_to_bool_clauses(node)
            else:
                for child in node.children:
                    bool_clauses.append(self.node_to_elastic_dict(child))
            return self.reducer.reduce_bool_clauses(bool_clauses)
        elif node.is_key("or"):
            shoulds = []
            for child in node.children:
                shoulds.append(self.node_to_elastic_dict(child))
            return {"bool": {"should": shoulds, MSM: 1}}
        else:
            logger.warn(f"× Unknown bool_node: <{node.key}>")
            return {}

    def node_to_elastic_dict(self, node: DslExprNode) -> dict:
        expr_node = node.find_child_with_key([*BOOL_OPS, "atom"])
        if expr_node.is_key(BOOL_OPS):
            res = self.bool_node_to_elastic_dict(expr_node)
            return self.reducer.reduce(res)
        elif expr_node.is_key("atom"):
            return self.atom_node_to_elastic_dict(expr_node)
        else:
            logger.warn(f"× Unknown node: <{expr_node.key}>")
            return {}

    def convert(self, expr: str) -> dict:
        lark_tree = self.parser.parse(expr)
        # logger.mesg(parser.str(lark_tree), verbose=lark_tree)
        dsl_tree = self.builder.build_dsl_tree_from_lark_tree(lark_tree)
        # logger.mesg(str(dsl_tree), verbose=dsl_tree)
        expr_tree = self.grouper.group_dsl_tree_to_expr_tree(dsl_tree)
        # logger.mesg(str(expr_tree), verbose=expr_tree)
        expr_tree = self.flatter.flatten(expr_tree)
        logger.mesg(str(expr_tree), verbose=expr_tree)
        elastic_dict = {"query": self.node_to_elastic_dict(expr_tree)}
        return elastic_dict


def test_converter():
    from converters.field.test import test_date_strs, test_date_list_strs
    from converters.field.test import test_user_strs
    from converters.field.test import test_text_strs
    from converters.field.test import test_comb_strs

    elastic_converter = DslExprToElasticConverter()
    test_strs = test_text_strs
    for test_str in test_strs:
        logger.note(f"{test_str}")
        elastic_dict = elastic_converter.convert(test_str)
        logger.mesg(dict_to_str(elastic_dict, add_quotes=True), indent=2)


if __name__ == "__main__":
    test_converter()

    # python -m converters.dsl.elastic
