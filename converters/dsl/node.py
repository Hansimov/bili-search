import re

from copy import deepcopy
from collections import defaultdict
from lark import Token, Tree
from tclogger import logger, logstr
from typing import Union, Literal, Any

from converters.dsl.constants import START_EXPR, MAIN_EXPRS, ATOM_EXPRS, ITEM_EXPRS
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
        self.extras = {}

    def copy(self) -> "DslNode":
        return deepcopy(self)

    def get_key_logstr(self, level: int = 0):
        return self.KEY_LOGSTRS.get(level % len(self.KEY_LOGSTRS), logstr.note)

    def log_node(self, node: "DslNode", level: int = 0):
        indent_str = " " * 2 * level
        if level > 0:
            indent_str += "- "
        node_key_str = self.get_key_logstr(level)(node.key)
        node_val_str = logstr.okay(node.value)
        node_str = f"{indent_str}{node_key_str}: {node_val_str}"
        # logger.mesg(node_str)
        self.node_str += f"{node_str}\n"

    def recursive_log_node(self, node: "DslNode", level: int = 0):
        self.log_node(node, level=level)
        for child in node.children:
            self.recursive_log_node(child, level=level + 1)

    def yaml(self):
        self.node_str = ""
        self.recursive_log_node(self, level=0)
        return self.node_str

    def __repr__(self):
        atom_nodes = self.find_all_childs_with_key("atom")
        word_expr_num = sum(
            1 for atom_node in atom_nodes if atom_node.first_child_key == "word_expr"
        )
        other_expr_num = len(atom_nodes) - word_expr_num
        return f"<DslNode> ({word_expr_num} word_exprs, {other_expr_num} filters)"

    @property
    def first_child(self) -> Union["DslNode", None]:
        if isinstance(self.children, list) and len(self.children) > 0:
            return self.children[0]
        else:
            return None

    @property
    def first_child_key(self) -> str:
        if self.children:
            return self.first_child.key
        else:
            return ""

    def has_only_one_child(self) -> bool:
        return len(self.children) == 1

    def find_child_with_key(
        self,
        key: Union[str, list[str]],
        raise_error: bool = False,
        use_re: bool = False,
    ) -> Union["DslNode", None]:
        queue = [self]
        while queue:
            current = queue.pop(0)
            if current.is_key(key, use_re=use_re):
                return current
            queue.extend(current.children)
        if raise_error:
            err_mesg = logstr.warn(f"× Not found: <{logstr.file(key)}>")
            raise ValueError(err_mesg)
        else:
            return None

    def find_all_childs_with_key(
        self,
        key: Union[str, list[str]],
        raise_error: bool = False,
        use_re: bool = False,
        max_level: int = None,
        exclude_self: bool = False,
    ) -> list["DslNode"]:
        """max_level:
        - None: no limit
        - 0: only check current node
        - 1: only check children of current node
        - etc."""
        res = []
        level_counts = defaultdict(int)

        if exclude_self:
            queue = self.children
            current_level = 1
            level_counts[1] = len(queue)
        else:
            queue = [self]
            current_level = 0
            level_counts[0] = 1

        while queue:
            current = queue.pop(0)
            if current.is_key(key, use_re=use_re):
                res.append(current)
            queue.extend(current.children)
            if max_level is not None:
                level_counts[current_level + 1] += len(current.children)
                level_counts[current_level] -= 1
                if level_counts[current_level] <= 0:
                    current_level += 1
                if current_level > max_level:
                    break

        if not res and raise_error:
            err_mesg = logstr.warn(f"× Not found: <{logstr.file(key)}>")
            raise ValueError(err_mesg)
        return res

    def find_parent_with_key(
        self,
        key: Union[str, list[str]],
        raise_error: bool = False,
        use_re: bool = False,
    ) -> Union["DslNode", None]:
        queue = [self]
        while queue:
            current = queue.pop(0)
            if current is None or current.is_key(key, use_re=use_re):
                return current
            queue.append(current.parent)
        if raise_error:
            err_mesg = logstr.warn(f"× Not found: <{logstr.file(key)}>")
            raise ValueError(err_mesg)
        else:
            return None

    def find_sibling_with_key(
        self,
        key: Union[str, list[str]],
        raise_error: bool = False,
        use_re: bool = False,
    ) -> Union["DslNode", None]:
        if self.parent:
            siblings = self.parent.children
            for sibling in siblings:
                if sibling is not self and sibling.is_key(key, use_re=use_re):
                    return sibling
        if raise_error:
            err_mesg = logstr.warn(f"× Not found: <{logstr.file(key)}>")
            raise ValueError(err_mesg)
        else:
            return None

    def get_value_by_key(self, key: str, raise_error: bool = False) -> Union[Any, None]:
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

    def get_deepest_node(self) -> "DslNode":
        if not self.children:
            return self
        else:
            return self.first_child.get_deepest_node()

    def get_deepest_node_key(self) -> str:
        if not self.children:
            return self.key
        else:
            return self.first_child.get_deepest_node_key()

    def set_deepest_node_key(self, key: str):
        deepest_node = self.get_deepest_node()
        deepest_node.key = key

    def get_deepest_node_value(self) -> str:
        if not self.children:
            return self.value
        else:
            return self.first_child.get_deepest_node_value()

    def set_deepest_node_value(self, value: str):
        deepest_node = self.get_deepest_node()
        deepest_node.value = value

    def get_next_sibling(self) -> "DslNode":
        if self.parent:
            siblings = self.parent.children
            if siblings and self in siblings:
                self_idx = siblings.index(self)
                if self_idx < len(siblings) - 1:
                    return siblings[self_idx + 1]
        return None

    def get_prev_sibling(self) -> "DslNode":
        if self.parent:
            siblings = self.parent.children
            if siblings and self in siblings:
                self_idx = siblings.index(self)
                if self_idx > 0:
                    return siblings[self_idx - 1]
        return None

    def get_next_sibling_key(self) -> str:
        next_sibling = self.get_next_sibling()
        if next_sibling:
            return next_sibling.key
        else:
            return None

    def get_prev_sibling_key(self) -> str:
        prev_sibling = self.get_prev_sibling()
        if prev_sibling:
            return prev_sibling.key
        else:
            return None

    def is_start(self):
        return self.key == START_EXPR

    def is_main(self):
        return self.key in MAIN_EXPRS

    def is_atom_expr(self):
        return self.key in ATOM_EXPRS

    def is_item_expr(self):
        return self.key in ITEM_EXPRS

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

    def is_key(self, key: Union[str, list[str]], use_re: bool = False) -> bool:
        if isinstance(key, list):
            if use_re:
                for k in key:
                    if re.match(k, self.key):
                        return True
                return False
            else:
                return self.key in key
        else:
            if use_re:
                return bool(re.match(key, self.key))
            else:
                return self.key == key

    def disconnect_from_parent(self):
        if self.parent:
            self.parent.children.remove(self)
            self.parent = None

    def connect_to_parent(self, parent: "DslNode", insert_index: int = None):
        self.parent = parent
        if insert_index is not None:
            parent.children.insert(insert_index, self)
        else:
            parent.children.append(self)

    def graft_to_new_parent(self, parent: "DslNode", insert_index: int = None):
        self.disconnect_from_parent()
        self.connect_to_parent(parent, insert_index=insert_index)

    def graft_children_to_parent(self, keep_order: bool = True) -> "DslNode":
        """Remove current node from its parent's children, and append its children to its parent.
        Return the parent of current node.
        """
        parent = self.parent
        children_copy = list(self.children)
        if keep_order:
            insert_index = parent.children.index(self)
            for child in children_copy:
                child.graft_to_new_parent(parent, insert_index)
                insert_index += 1
        else:
            for child in children_copy:
                child.graft_to_new_parent(parent)
        self.disconnect_from_parent()
        return parent

    def insert_subparent(self, subparent: "DslNode") -> "DslNode":
        """Insert mid_subparent between itself and its childs,
        which means subparent will be new parent of self's children, while self will be parent of subparent.
        """
        child_copy = list(self.children)
        for child in child_copy:
            child.graft_to_new_parent(subparent)
        subparent.connect_to_parent(self)
        return subparent


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
    def get_atom_child_expr_key(self) -> str:
        atom_node = self.find_child_with_key("atom")
        if atom_node:
            return atom_node.first_child_key
        else:
            return ""

    def has_level_1_atom_child(self) -> bool:
        for child in self.children:
            if child.is_key("atom"):
                return True
        return False

    def get_all_atom_childs_expr_keys(self) -> list[str]:
        atom_nodes = self.find_all_childs_with_key("atom")
        return [atom_node.first_child_key for atom_node in atom_nodes]

    def all_atom_childs_are_word_expr(self) -> bool:
        atom_nodes = self.find_all_childs_with_key("atom")
        if not atom_nodes:
            return False
        for atom_node in atom_nodes:
            if not atom_node.first_child.is_key("word_expr"):
                return False
        return True

    def get_all_bool_childs_keys(self) -> list[str]:
        bool_nodes = self.find_all_childs_with_key(BOOL_OPS)
        return [bool_node.key for bool_node in bool_nodes]

    def all_bool_childs_are_co_and(self) -> bool:
        bool_nodes = self.find_all_childs_with_key(BOOL_OPS)
        for bool_node in bool_nodes:
            if not bool_node.is_key(["co", "and"]):
                return False
        return True

    def filter_atoms_by_keys(
        self, include_keys: list[str] = [], exclude_keys: list[str] = []
    ) -> "DslExprNode":
        new_node = self.copy()
        atom_nodes = new_node.find_all_childs_with_key("atom")
        for atom_node in atom_nodes:
            atom_key = atom_node.first_child_key
            if include_keys and atom_key not in include_keys:
                atom_node.disconnect_from_parent()
            elif exclude_keys and atom_key in exclude_keys:
                atom_node.disconnect_from_parent()
        return new_node


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
                self.group(child).connect_to_parent(expr_node)
            return expr_node
        elif node.is_main():
            expr_node = DslExprNode("expr")
            for child in children:
                self.group(child).connect_to_parent(expr_node)
            return expr_node
        elif node.is_pa_expr():
            expr_node = DslExprNode("pa")
            for child in children:
                if child.is_lp() or child.is_rp():
                    pass
                elif child.is_bool_expr() or child.is_atom_expr() or child.is_pa_expr():
                    self.group(child).connect_to_parent(expr_node)
                else:
                    raise ValueError(f"Invalid pa_expr: {child.key}")
            return expr_node
        elif node.is_bool_expr():
            expr_key = re.sub("_expr$", "", node.key)
            expr_node = DslExprNode(expr_key)
            non_op_children = [child for child in children if not child.is_bool_op()]
            for child in non_op_children:
                self.group(child).connect_to_parent(expr_node)
            return expr_node
        elif node.is_atom_expr():
            expr_node = DslExprNode("atom")
            for child in children:
                if child.is_item_expr():
                    self.group(child).connect_to_parent(expr_node)
                else:
                    raise ValueError(f"Invalid atom_expr: {child.key}")
            return expr_node
        else:
            return self.construct_expr_node_from_dsl_node(node)

    def group_dsl_tree_to_expr_tree(self, node: DslNode) -> DslExprNode:
        return self.group(node)


class DslExprTreeFlatter(DslTreeProcessor):
    def all_childs_are_atom(self, node: DslExprNode) -> bool:
        for child in node.children:
            if not child.is_atom_expr():
                return False
        return True

    def all_childs_are_atom_co(self, co_node: DslExprNode) -> bool:
        """node key is `co` or `and`.
        Only when all bool expr children are `co` or `and`, and all atom expr children are `word_expr`, return True, otherwise return False.
        """
        bool_childs = co_node.find_all_childs_with_key(BOOL_OPS)
        for bool_child in bool_childs:
            if not bool_child.is_key(["co", "and"]):
                return False
        item_childs = co_node.find_all_childs_with_key(ITEM_EXPRS)
        for item_child in item_childs:
            if not item_child.is_key("word_expr"):
                return False
        return True

    def flatten_word_nodes_under_co_node(self, co_node: DslExprNode) -> DslExprNode:
        """node key is `co` or `and`.
        Flatten its word_expr_nodes to same level, and connect to a new single node.
        Then replace original node with this new node.
        """
        new_co_node = DslExprNode(co_node.key)
        atom_nodes = co_node.find_all_childs_with_key("atom")
        for atom_node in atom_nodes:
            atom_node.graft_to_new_parent(new_co_node)
        new_co_node.extras["atoms_type"] = "word_expr"
        parent = co_node.parent
        co_node.disconnect_from_parent()
        new_co_node.connect_to_parent(parent)
        return new_co_node

    def flatten_pa_node(self, pa_node: DslExprNode, queue: list[DslExprNode]):
        """node key is `pa`.
        If it has only one child, replace it with its child.
        """
        if pa_node.has_only_one_child():
            first_child = pa_node.first_child
            pa_node.graft_children_to_parent()
            queue.append(first_child)
        else:
            queue.extend(pa_node.children)

    def flatten_or_node(self, or_node: DslExprNode, queue: list[DslExprNode]):
        """node key is `or`.
        Flatten its atom_nodes and or_nodes to same level, and connect to a new single `or` node.
        """
        if or_node.parent.is_key("or"):
            childs = list(or_node.children)
            or_node.graft_children_to_parent()
            queue.extend(childs)
        else:
            queue.extend(or_node.children)

    def flatten_co_node(self, co_node: DslExprNode, queue: list[DslExprNode]):
        """node key is `co` or `and`."""
        if self.all_childs_are_atom(co_node):
            return
        if self.all_childs_are_atom_co(co_node):
            self.flatten_word_nodes_under_co_node(co_node)
        else:
            queue.extend(co_node.children)

    def flatten(self, node: DslExprNode) -> DslExprNode:
        """node key is `start`"""
        if node.first_child.is_key("expr"):
            expr_node: DslExprNode = node.first_child
            if expr_node.has_level_1_atom_child():
                # this condition often appears when there is `expr_error` in the first-level child of root `expr` node
                # need to put all childs under the `co` expr node
                co_node = DslExprNode("co")
                expr_node.insert_subparent(co_node)

        queue = node.find_all_childs_with_key(BOOL_OPS, max_level=1)
        while queue:
            current = queue.pop(0)
            if current.is_key("atom"):
                continue
            elif current.is_key("pa"):
                self.flatten_pa_node(current, queue)
            elif current.is_key(["co", "and"]):
                self.flatten_co_node(current, queue)
            elif current.is_key("or"):
                self.flatten_or_node(current, queue)
            else:
                logger.warn(f"× Unknown current.key: <{current.key}>")
        return node
