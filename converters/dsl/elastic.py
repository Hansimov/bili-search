from tclogger import logger, dict_to_str, dict_get, dict_set

from converters.dsl.constants import BOOL_OPS, ITEM_EXPRS, FILTER_EXPRS
from converters.dsl.constants import MSM, BMM, BM_MAP
from converters.dsl.parse import DslLarkParser
from converters.dsl.node import DslExprNode, DslTreeBuilder, DslTreeExprGrouper
from converters.dsl.node import DslExprTreeFlatter
from converters.dsl.fields.bvid import BvidExprElasticConverter
from converters.dsl.fields.date import DateExprElasticConverter
from converters.dsl.fields.stat import StatExprElasticConverter
from converters.dsl.fields.user import UserExprElasticConverter
from converters.dsl.fields.word import WordExprElasticConverter
from converters.dsl.fields.word import WordNodeToExprConstructor
from converters.dsl.fields.bool import BoolElasticReducer
from elastics.videos.constants import SEARCH_MATCH_TYPE, QUERY_TYPE_DEFAULT

BMM = BM_MAP[QUERY_TYPE_DEFAULT]["BM"]
BMMQ = BM_MAP[QUERY_TYPE_DEFAULT]["BMQ"]


class DslExprToElasticConverter:
    def __init__(self, verbose: bool = False):
        self.parser = DslLarkParser()
        self.builder = DslTreeBuilder()
        self.grouper = DslTreeExprGrouper()
        self.flatter = DslExprTreeFlatter()
        self.reducer = BoolElasticReducer()
        self.bvid_converter = BvidExprElasticConverter()
        self.date_converter = DateExprElasticConverter()
        self.user_converter = UserExprElasticConverter()
        self.stat_converter = StatExprElasticConverter()
        self.word_converter = WordExprElasticConverter()
        self.verbose = verbose

    def atom_node_to_elastic_dict(self, node: DslExprNode) -> dict:
        expr_node = node.find_child_with_key(ITEM_EXPRS)
        if expr_node.is_key("bvid_expr"):
            return self.bvid_converter.convert(node)
        elif expr_node.is_key("date_expr"):
            return self.date_converter.convert(node)
        elif expr_node.is_key("stat_expr"):
            return self.stat_converter.convert(node)
        elif expr_node.is_key("user_expr"):
            return self.user_converter.convert(node)
        elif expr_node.is_key("word_expr"):
            return self.word_converter.convert(node)
        else:
            logger.warn(f"× Unknown atom_node: <{expr_node.key}>", verbose=self.verbose)
            return {}

    def merge_word_match_clauses(self, bool_clauses: list[dict]) -> list[dict]:
        """Merge multiple "multi_match"/"query_string" word_expr exprs into one query with "cross_fields"."""
        word_match_clauses = []
        other_bool_clauses = []
        for bool_clause in bool_clauses:
            if dict_get(bool_clause, BMM):
                word_match_clauses.append(bool_clause)
            else:
                other_bool_clauses.append(bool_clause)

        new_bool_clauses = other_bool_clauses
        if word_match_clauses:
            query_words = [
                dict_get(word_match_clause, BMMQ)
                for word_match_clause in word_match_clauses
            ]
            merged_query_words = " ".join(query_words)
            merged_word_match_clause = word_match_clauses[0]
            dict_set(merged_word_match_clause, BMMQ, merged_query_words)
            new_bool_clauses.insert(0, merged_word_match_clause)
        return new_bool_clauses

    def co_and_cluster_to_bool_clauses(self, node: DslExprNode) -> list[dict]:
        word_expr_nodes = node.find_all_childs_with_key("word_expr")
        word_clauses = [
            self.word_converter.convert(word_expr_node)
            for word_expr_node in word_expr_nodes
        ]
        if SEARCH_MATCH_TYPE == "cross_fields":
            word_clauses = self.merge_word_match_clauses(word_clauses)
        filter_expr_nodes = node.find_all_childs_with_key(FILTER_EXPRS)
        filter_clauses = [
            self.atom_node_to_elastic_dict(non_word_expr_node)
            for non_word_expr_node in filter_expr_nodes
        ]
        bool_clauses = word_clauses + filter_clauses
        return bool_clauses

    def bool_node_to_elastic_dict(self, node: DslExprNode) -> dict:
        if node.is_key("pa"):
            return self.node_to_elastic_dict(node.children[0])
        elif node.is_key(["and", "co"]):
            bool_clauses = []
            if node.all_bool_childs_are_co_and():
                bool_clauses = self.co_and_cluster_to_bool_clauses(node)
            else:
                for child in node.children:
                    bool_clauses.append(self.node_to_elastic_dict(child))
            return self.reducer.reduce_co_bool_clauses(bool_clauses)
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
        if not expr_node:
            return {}
        elif expr_node.is_key(BOOL_OPS):
            res = self.bool_node_to_elastic_dict(expr_node)
            return self.reducer.reduce(res)
        elif expr_node.is_key("atom"):
            return self.atom_node_to_elastic_dict(expr_node)
        else:
            logger.warn(f"× Unknown node: <{expr_node.key}>", verbose=self.verbose)
            return {}

    def construct_expr_tree(self, expr: str) -> DslExprNode:
        lark_tree = self.parser.parse(expr)
        dsl_tree = self.builder.build_dsl_tree_from_lark_tree(lark_tree)
        expr_tree = self.grouper.group_dsl_tree_to_expr_tree(dsl_tree)
        return expr_tree

    def expr_tree_to_dict(self, expr_tree: DslExprNode) -> dict:
        expr_tree = self.flatter.flatten(expr_tree)
        return self.node_to_elastic_dict(expr_tree)

    def expr_to_dict(self, expr: str) -> dict:
        expr_tree = self.construct_expr_tree(expr)
        return self.expr_tree_to_dict(expr_tree)


class DslTreeToExprConstructor(DslExprToElasticConverter):
    def __init__(self):
        super().__init__()
        self.word_constructor = WordNodeToExprConstructor()

    def co_node_to_expr(self, node: DslExprNode) -> str:
        return " ".join([self.construct(child) for child in node.children])

    def pa_node_to_expr(self, node: DslExprNode) -> str:
        return f"({self.construct(node.first_child)})"

    def or_node_to_expr(self, node: DslExprNode) -> str:
        return " || ".join([self.construct(child) for child in node.children])

    def and_node_to_expr(self, node: DslExprNode) -> str:
        return " && ".join([self.construct(child) for child in node.children])

    def atom_node_to_expr(self, node: DslExprNode) -> str:
        """node key is `atom`."""
        child = node.first_child
        if child.is_key("word_expr"):
            return self.word_constructor.construct(child)

    def construct(self, node: DslExprNode) -> str:
        """Construct expr from DslExprNode."""
        if node.is_key("co"):
            return self.co_node_to_expr(node)
        elif node.is_key("pa"):
            return self.pa_node_to_expr(node)
        elif node.is_key("or"):
            return self.or_node_to_expr(node)
        elif node.is_key("and"):
            return self.and_node_to_expr(node)
        elif node.is_key("atom"):
            return self.atom_node_to_expr(node)
        else:
            if node.children:
                return self.construct(node.first_child)
            else:
                logger.warn(f"× Unknown expr_tree: <{node.key}>", verbose=self.verbose)
                return ""


def test_converter():
    from converters.field.test import test_date_strs, test_date_list_strs
    from converters.field.test import test_user_strs
    from converters.field.test import test_text_strs
    from converters.field.test import test_comb_strs

    elastic_converter = DslExprToElasticConverter(verbose=True)
    test_strs = test_text_strs
    for test_str in test_strs:
        logger.note(f"{test_str}")
        elastic_dict = elastic_converter.expr_to_dict(test_str)
        logger.mesg(dict_to_str(elastic_dict, add_quotes=True), indent=2)


if __name__ == "__main__":
    test_converter()

    # python -m converters.dsl.elastic
