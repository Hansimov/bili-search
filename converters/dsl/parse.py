from lark import Lark, Token, Tree
from pathlib import Path
from tclogger import logger, logstr
from typing import Union, Literal


class DslLarkParser:
    def __init__(self, verbose: bool = False):
        self.dsl_lark = Path(__file__).parent / "syntax.lark"
        self.verbose = verbose
        self.init_parser()

    def init_parser(self):
        with open(self.dsl_lark, "r") as rf:
            syntax = rf.read()
        self.parser = Lark(syntax, parser="earley")

    def parse(self, expr: str) -> Tree:
        try:
            return self.parser.parse(expr)
        except Exception as e:
            logger.warn(f"× {expr}")
            raise e

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
            node_str = f"{indent_str}{logstr.file(node.data)}:"
            logger.note(node_str, verbose=self.verbose)
        elif node_type == "token":
            node_str = (
                f'{indent_str}{logstr.file(node.type)}: "{logstr.line(node.value)}"'
            )
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

    def str(self, tree: Tree):
        self.tree_str = ""
        self.traverse_node(tree, level=0, node_func=self.log_node)
        return self.tree_str
