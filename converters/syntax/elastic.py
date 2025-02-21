import yaml

from lark import Lark, Token, Tree
from pathlib import Path
from tclogger import logger, dict_to_str
from typing import Union, Literal

from converters.syntax.node import DslNode, DslTreeBuilder


class DslToElasticConverter:
    def __init__(self, verbose: bool = False):
        self.dsl_lark = Path(__file__).parent / "dsl.lark"
        self.verbose = verbose
        self.init_parser()

    def init_parser(self):
        with open(self.dsl_lark, "r") as rf:
            syntax = rf.read()
        self.parser = Lark(syntax, parser="earley")

    def parse_as_tree(self, expr: str) -> Tree:
        try:
            tree = self.parser.parse(expr)
            logger.success(f"✓ {expr}", verbose=self.verbose)
            return tree
        except Exception as e:
            logger.warn(f"× {expr}", verbose=self.verbose)
            if self.verbose:
                raise e
            return None

    def log_node(
        self,
        node: Union[Tree, Token],
        node_type: Literal["tree", "token", "unknown"] = "unknown",
        level: int = 0,
    ):
        indent_str = " " * 2 * level
        if node_type == "tree":
            if level > 0:
                indent_str += "- "
            node_str = f"{indent_str}{node.data}:"
            logger.note(node_str, verbose=self.verbose)
        elif node_type == "token":
            node_str = f'{indent_str}{node.type}: "{node.value}"'
            logger.mesg(node_str, verbose=self.verbose)
        else:
            node_str = f"{indent_str}Unknown: None"
            logger.warn(node_str, verbose=self.verbose)
        self.tree_str += f"{node_str}\n"

    def traverse_node(
        self,
        node: Union[Tree, Token],
        level: int = 0,
        node_func: callable = log_node,
    ):
        if isinstance(node, Tree):
            node_func(node, node_type="tree", level=level)
            for child in node.children:
                self.traverse_node(child, level=level + 1, node_func=node_func)
        elif isinstance(node, Token):
            node_func(node, node_type="token", level=level)
        else:
            node_func(node, node_type="unknown", level=level)

    def construct(self, expr: str) -> dict:
        self.tree_dict = {}
        self.tree_str = ""
        tree = self.parse_as_tree(expr)
        self.traverse_node(tree, node_func=self.log_node)
        dsl_tree = DslTreeBuilder().build_dsl_tree_from_lark_tree(tree)
        logger.mesg(dsl_tree.str(), verbose=dsl_tree)
        # self.tree_dict = yaml.safe_load(self.tree_str)
        return self.tree_dict


def test_dsl_to_elastic_converter():
    from converters.syntax.test import queries

    converter = DslToElasticConverter(verbose=True)

    for query in queries[:1]:
        tree_dict = converter.construct(query)
        # logger.mesg(dict_to_str(tree_dict))


if __name__ == "__main__":
    test_dsl_to_elastic_converter()

    # python -m converters.syntax.elastic
