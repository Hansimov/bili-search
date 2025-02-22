import re

from copy import deepcopy
from lark import Token, Tree
from tclogger import logger, logstr, dict_to_str
from typing import Union, Literal


class DslNode:
    def __init__(
        self,
        key: str = "",
        value: str = "",
        parent: "DslNode" = None,
        # DO NOT use [] for default value!
        # As it will be shared among all instances!
        children: list["DslNode"] = None,
    ):
        self.key = key
        self.value = value
        self.parent = parent
        self.children = children or []

    def log_node(self, node: "DslNode", level: int = 0):
        indent_str = " " * 2 * level
        if level > 0:
            indent_str += "- "
        node_str = f"{indent_str}{logstr.file(node.key)}: {logstr.line(node.value)}"
        # logger.mesg(node_str)
        self.node_str += f"{node_str}\n"

    def recursive_log_node(self, node: "DslNode", level: int = 0):
        self.log_node(node, level=level)
        for child in node.children:
            self.recursive_log_node(child, level=level + 1)

    def __str__(self):
        self.node_str = ""
        self.recursive_log_node(self, level=0)
        return self.node_str

    def __repr__(self):
        return self.__str__()


class DslTreeProcessor:
    @staticmethod
    def get_node_type(node: Union[Tree, Token]) -> Literal["tree", "token", "unknown"]:
        if isinstance(node, Tree):
            return "tree"
        elif isinstance(node, Token):
            return "token"
        else:
            return "unknown"

    @staticmethod
    def get_siblings(node: DslNode) -> list[DslNode]:
        if node.parent:
            siblings = deepcopy(node.parent.children)
            siblings.remove(node)
            return siblings
        else:
            return []

    @staticmethod
    def connect_node_to_parent(node: DslNode, parent: DslNode):
        """Add node to parent's children"""
        node.parent = parent
        parent.children.append(node)

    @staticmethod
    def disconnect_from_parent(node: DslNode):
        """Remove node from its parent's children"""
        if node.parent:
            node.parent.children.remove(node)
            node.parent = None

    @staticmethod
    def graft_node_to_parent(node: DslNode, parent: DslNode):
        """Remove current node from its parent's children, and append it to the new parent"""
        DslTreeProcessor.disconnect_from_parent(node)
        DslTreeProcessor.connect_node_to_parent(node, parent)
        return parent

    @staticmethod
    def graft_children_to_parent(node: DslNode) -> DslNode:
        """Remove current node from its parent's children, and append its children to its parent"""
        parent = node.parent
        for child in node.children:
            DslTreeProcessor.graft_node_to_parent(child, parent)
        DslTreeProcessor.disconnect_from_parent(node)
        return parent


class DslTreeBuilder(DslTreeProcessor):
    def recursive_build_tree(self, node: Union[Tree, Token], parent: DslNode):
        node_type = self.get_node_type(node)
        if node_type == "tree":
            dsl_node = DslNode(key=str(node.data), parent=parent)
            parent.children.append(dsl_node)
            for child in node.children:
                self.recursive_build_tree(child, dsl_node)
        elif node_type == "token":
            dsl_node = DslNode(key=node.type, value=node.value, parent=parent)
            parent.children.append(dsl_node)
        else:
            logger.warn(f"× Unknown node type: {node_type}")

    def build_dsl_tree_from_lark_tree(self, node: Tree) -> DslNode:
        root = DslNode("root")
        self.recursive_build_tree(node, root)
        return root.children[0]


class DslExprNode(DslNode):
    pass


START_EXPR = "start"
MAIN_EXPRS = ["expr", "expr_error"]
ATOM_EXPRS = ["atom_expr"]
PA_EXPRS = ["pa_expr"]
BOOL_EXPRS = ["and_expr", "or_expr", "co_expr"]
BOOL_OPS = ["or", "and"]
ATOMS = [
    *["date_expr", "user_expr", "uid_expr", "stat_expr"],
    *["region_expr", "word_expr", "text_expr"],
]


class DslTreeExprGrouper(DslTreeProcessor):
    """Group nodes in DslTree to DslExprNode.

    Priorities of exprs:
        atom > pa > and > co > or
    """

    @staticmethod
    def is_start(node: DslNode):
        return node.key == START_EXPR

    @staticmethod
    def is_atom_expr(node: DslNode):
        return node.key in ATOM_EXPRS

    @staticmethod
    def is_atom(node: DslNode):
        return node.key in ATOMS

    @staticmethod
    def is_pa_expr(node: DslNode):
        return node.key in PA_EXPRS

    @staticmethod
    def is_lp(node: DslNode):
        return node.key == "lp"

    @staticmethod
    def is_rp(node: DslNode):
        return node.key == "rp"

    @staticmethod
    def is_bool_op(node: DslNode):
        return node.key in BOOL_OPS

    @staticmethod
    def is_bool_expr(node: DslNode):
        return node.key in BOOL_EXPRS

    @staticmethod
    def get_key(node: DslNode):
        return node.key

    @staticmethod
    def construct_expr_node_from_dsl_node(node: DslNode):
        return DslExprNode(node.key, node.value, node.parent, node.children)

    def group(self, node: DslNode) -> DslExprNode:
        children = node.children

        if self.is_start(node):
            expr_node = DslExprNode("start")
            for child in children:
                self.connect_node_to_parent(self.group(child), expr_node)
            return expr_node
        elif self.is_pa_expr(node):
            expr_node = DslExprNode("pa")
            for child in children:
                if self.is_lp(child) or self.is_rp(child):
                    pass
                elif self.is_bool_expr(child) or self.is_bool_expr(child):
                    self.connect_node_to_parent(self.group(child), expr_node)
                else:
                    raise ValueError(f"Invalid pa_expr: {child.key}")
            return expr_node
        elif self.is_bool_expr(node):
            expr_key = re.sub("_expr$", "", self.get_key(node))
            expr_node = DslExprNode(expr_key)
            non_op_children = [
                child for child in children if not self.is_bool_op(child)
            ]
            for child in non_op_children:
                self.connect_node_to_parent(self.group(child), expr_node)
            return expr_node
        elif self.is_atom_expr(node):
            expr_node = DslExprNode("atom")
            for child in children:
                if self.is_atom(child):
                    self.connect_node_to_parent(self.group(child), expr_node)
                else:
                    raise ValueError(f"Invalid atom_expr: {child.key}")
            return expr_node
        else:
            print(node.key)
        return self.construct_expr_node_from_dsl_node(node)

    def group_dsl_tree_to_expr_tree(self, node: DslNode) -> DslExprNode:
        root = DslExprNode("root")
        expr_tree = self.group(node)
        self.connect_node_to_parent(expr_tree, root)
        return root
