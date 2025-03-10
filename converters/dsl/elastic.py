from tclogger import logger, dict_to_str

from converters.dsl.constants import BOOL_OPS, ITEM_EXPRS, MSM
from converters.dsl.parse import DslLarkParser
from converters.dsl.node import DslExprNode, DslTreeBuilder, DslTreeExprGrouper
from converters.dsl.node import DslExprTreeFlatter
from converters.dsl.fields.date import DateExprElasticConverter
from converters.dsl.fields.user import UserExprElasticConverter
from converters.dsl.fields.word import WordExprElasticConverter
from converters.dsl.fields.word import WordNodeToExprConstructor
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
        expr_node = node.find_child_with_key(ITEM_EXPRS)
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
                node.all_bool_childs_are_co_and()
                and node.all_atom_childs_are_word_expr()
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
        if not expr_node:
            return {}
        elif expr_node.is_key(BOOL_OPS):
            res = self.bool_node_to_elastic_dict(expr_node)
            return self.reducer.reduce(res)
        elif expr_node.is_key("atom"):
            return self.atom_node_to_elastic_dict(expr_node)
        else:
            logger.warn(f"× Unknown node: <{expr_node.key}>")
            return {}

    def construct_expr_tree(self, expr: str) -> DslExprNode:
        lark_tree = self.parser.parse(expr)
        dsl_tree = self.builder.build_dsl_tree_from_lark_tree(lark_tree)
        expr_tree = self.grouper.group_dsl_tree_to_expr_tree(dsl_tree)
        return expr_tree

    def convert(self, expr: str) -> dict:
        expr_tree = self.construct_expr_tree(expr)
        expr_tree = self.flatter.flatten(expr_tree)
        logger.mesg(str(expr_tree), verbose=expr_tree)
        elastic_dict = {"query": self.node_to_elastic_dict(expr_tree)}
        return elastic_dict


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
                logger.warn(f"× Unknown expr_tree: <{node.key}>")
                return ""


class DslExprKeywordsFilterSplitter(DslExprToElasticConverter):
    def __init__(self):
        super().__init__()
        self.expr_constructor = DslTreeToExprConstructor()

    def get_keywords_info(self, expr_tree: DslExprNode) -> dict[str, list[str]]:
        keywords_expr_tree = expr_tree.filter_atoms_by_keys(["word_expr"])
        keywords_only_expr = self.expr_constructor.construct(keywords_expr_tree)
        return {"keywords": keywords_only_expr}

    def split_expr_to_keywords_and_filters(self, expr: str) -> dict:
        expr_tree = self.construct_expr_tree(expr)
        keywords_info = self.get_keywords_info(expr_tree)
        return {
            **keywords_info,
        }


def test_converter():
    from converters.field.test import test_date_strs, test_date_list_strs
    from converters.field.test import test_user_strs
    from converters.field.test import test_text_strs
    from converters.field.test import test_comb_strs

    elastic_converter = DslExprToElasticConverter()
    keyword_splitter = DslExprKeywordsFilterSplitter()
    test_strs = test_text_strs
    for test_str in test_strs:
        logger.note(f"{test_str}")
        # elastic_dict = elastic_converter.convert(test_str)
        # logger.mesg(dict_to_str(elastic_dict, add_quotes=True), indent=2)
        split_dict = keyword_splitter.split_expr_to_keywords_and_filters(test_str)
        logger.mesg(dict_to_str(split_dict, add_quotes=True), indent=2)


if __name__ == "__main__":
    test_converter()

    # python -m converters.dsl.elastic
