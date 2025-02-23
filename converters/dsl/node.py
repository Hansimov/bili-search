import re

from copy import deepcopy
from lark import Token, Tree
from tclogger import logger, logstr
from typing import Union, Literal, Any

from converters.dsl.constants import START_EXPR, ATOM_EXPRS, ATOMS
from converters.dsl.constants import PA_EXPRS, BOOL_OPS, BOOL_EXPRS


class DslNode:
    KEY_LOGSTRS = {
        0: logstr.hint,
        1: logstr.note,
        2: logstr.mesg,
        3: logstr.file,
    }

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

    def get_key_logstr(self, level: int = 0):
        return self.KEY_LOGSTRS.get(level % len(self.KEY_LOGSTRS), logstr.note)

    def log_node(self, node: "DslNode", level: int = 0):
        indent_str = " " * 2 * level
        if level > 0:
            indent_str += "- "

        node_key_str = self.get_key_logstr(level)(node.key)
        node_val_str = logstr.line(node.value)

        node_str = f"{indent_str}{node_key_str}: {node_val_str}"
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

    def find_child_with_key(
        self, key: Union[str, list[str]], raise_error: bool = True
    ) -> Union["DslExprNode", None]:
        queue = [self]
        while queue:
            current = queue.pop(0)
            if isinstance(key, str):
                if current.key == key:
                    return current
            else:
                if current.key in key:
                    return current
            queue.extend(current.children)
        if raise_error:
            err_mesg = logstr.warn(f"× Not found: <{logstr.file(key)}>")
            raise ValueError(err_mesg)
        else:
            return None

    def get_value_by_key(self, key: str, raise_error: bool = True) -> Union[Any, None]:
        child = self.find_child_with_key(key, raise_error=raise_error)
        if child:
            return child.value
        else:
            return None

    def get_value_dict_by_keys(
        self, keys: list[str], raise_error: bool = False
    ) -> dict[str, Any]:
        return {
            key: self.get_value_by_key(key, raise_error=raise_error) for key in keys
        }

    def get_deepest_node_key(self) -> str:
        if not self.children:
            return self.key
        else:
            return self.children[0].get_deepest_node_key()

    def is_start(self):
        return self.key == START_EXPR

    def is_atom_expr(self):
        return self.key in ATOM_EXPRS

    def is_atom(self):
        return self.key in ATOMS

    def is_pa_expr(self):
        return self.key in PA_EXPRS

    def is_lp(self):
        return self.key == "lp"

    def is_rp(self):
        return self.key == "rp"

    def is_bool_op(self):
        return self.key in BOOL_OPS

    def is_bool_expr(self):
        return self.key in BOOL_EXPRS

    def get_key(self):
        return self.key


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
    def has_only_one_child(node: DslNode) -> bool:
        return len(node.children) == 1

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
            parent.value = node.value
        else:
            logger.warn(f"× Unknown node type: {node_type}")

    def build_dsl_tree_from_lark_tree(self, node: Tree) -> DslNode:
        root = DslNode("root")
        self.recursive_build_tree(node, root)
        return root.children[0]


class DslExprNode(DslNode):
    pass


class DslTreeExprGrouper(DslTreeProcessor):
    """Group nodes in DslTree to DslExprNode.

    Priorities of exprs:
        atom > pa > and > co > or
    """

    @staticmethod
    def construct_expr_node_from_dsl_node(node: DslNode):
        return DslExprNode(node.key, node.value, node.parent, node.children)

    def group(self, node: DslNode) -> DslExprNode:
        children = node.children

        if node.is_start():
            expr_node = DslExprNode("start")
            for child in children:
                self.connect_node_to_parent(self.group(child), expr_node)
            return expr_node
        elif node.is_pa_expr():
            expr_node = DslExprNode("pa")
            for child in children:
                if child.is_lp() or child.is_rp():
                    pass
                elif child.is_bool_expr() or child.is_bool_expr():
                    self.connect_node_to_parent(self.group(child), expr_node)
                else:
                    raise ValueError(f"Invalid pa_expr: {child.key}")
            return expr_node
        elif node.is_bool_expr():
            expr_key = re.sub("_expr$", "", node.key)
            expr_node = DslExprNode(expr_key)
            non_op_children = [child for child in children if not child.is_bool_op()]
            for child in non_op_children:
                self.connect_node_to_parent(self.group(child), expr_node)
            return expr_node
        elif node.is_atom_expr():
            expr_node = DslExprNode("atom")
            for child in children:
                if child.is_atom():
                    self.connect_node_to_parent(self.group(child), expr_node)
                else:
                    raise ValueError(f"Invalid atom_expr: {child.key}")
            return expr_node
        else:
            return self.construct_expr_node_from_dsl_node(node)

    def group_dsl_tree_to_expr_tree(self, node: DslNode) -> DslExprNode:
        return self.group(node)
