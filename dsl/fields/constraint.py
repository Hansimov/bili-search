"""Token constraint expression handler for DSL.

Converts word_expr nodes with +/- prefixes into es_tok_constraints.

Syntax:
  +token       → must have_token (AND by default with other constraints)
  -token       → must NOT have_token
  !token       → same as -token

Complex boolean:
  +A +B        → AND(have_token(A), have_token(B))
  +A -B        → AND(have_token(A), NOT(have_token(B)))
  (+A & +B)    → AND(have_token(A), have_token(B))
  (+A | +B)    → OR(have_token(A), have_token(B))
  (+A & +B) | (+C & -D)
               → OR(AND(have_token(A), have_token(B)),
                     AND(have_token(C), NOT(have_token(D))))

Output format: es_tok_constraints dict, e.g.:
  {"es_tok_constraints": {"constraints": [...], "fields": [...]}}
"""

from typing import Union
from dsl.node import DslExprNode
from dsl.constants import BOOL_OPS, TEXT_TYPES

QUOTES_TO_REMOVE = '""《》（）【】"'


def is_constraint_word_expr(node: DslExprNode) -> bool:
    """Check if a word_expr node is a constraint (has +/-/! prefix).
    node key should be 'word_expr'."""
    word_pp_node = node.find_child_with_key("word_pp")
    return word_pp_node is not None


def get_constraint_text(node: DslExprNode) -> str:
    """Extract the text from a constraint word_expr node.
    node key should be 'word_expr'."""
    text_node = node.find_child_with_key(TEXT_TYPES)
    if not text_node:
        return ""
    text = text_node.get_deepest_node_value()
    if not text:
        return ""
    text = text.strip(" ")
    for q in QUOTES_TO_REMOVE:
        text = text.replace(q, "")
    return text


def get_constraint_pp_key(node: DslExprNode) -> str:
    """Get the prefix type key: 'pl', 'mi', or 'nq'.
    node key should be 'word_expr'."""
    word_pp_node = node.find_child_with_key("word_pp")
    if word_pp_node:
        return word_pp_node.get_deepest_node_key()
    return ""


def word_expr_to_constraint(node: DslExprNode) -> dict:
    """Convert a +/-/! prefixed word_expr to a constraint dict.
    node key should be 'word_expr'.

    Returns:
        {"have_token": ["text"]} for + prefix
        {"NOT": {"have_token": ["text"]}} for -/! prefix
    """
    pp_key = get_constraint_pp_key(node)
    text = get_constraint_text(node)
    if not text:
        return {}

    constraint = {"have_token": [text]}
    if pp_key in ["mi", "nq"]:
        constraint = {"NOT": constraint}
    return constraint


class ConstraintTreeConverter:
    """Convert a boolean tree of constraint word_exprs to es_tok_constraints."""

    def is_pure_constraint_subtree(self, node: DslExprNode) -> bool:
        """Check if a subtree contains ONLY constraint word_exprs (no regular words/filters).
        Returns True if all atom children with word_expr have +/-/! prefix."""
        word_expr_nodes = node.find_all_childs_with_key("word_expr")
        if not word_expr_nodes:
            return False
        for word_node in word_expr_nodes:
            if not is_constraint_word_expr(word_node):
                return False
        # Also check there are no non-word filter exprs
        from dsl.constants import FILTER_EXPRS

        filter_nodes = node.find_all_childs_with_key(FILTER_EXPRS)
        # qmod_expr is OK in constraint subtrees (it's a mode flag, not a filter)
        non_qmod_filters = [n for n in filter_nodes if not n.is_key("qmod_expr")]
        if non_qmod_filters:
            return False
        return True

    def node_to_constraint(self, node: DslExprNode) -> Union[dict, None]:
        """Convert a DslExprNode to a constraint dict recursively.

        Args:
            node: A node from the expr tree. Can be:
              - 'atom' with constraint word_expr child
              - 'co' or 'and' with constraint children (AND combination)
              - 'or' with constraint children (OR combination)
              - 'pa' wrapping a constraint expression
              - 'expr' or 'start' wrapping a constraint expression

        Returns:
            Constraint dict or None if not a constraint node.
        """
        if node.is_key(["start", "expr"]):
            if node.first_child:
                return self.node_to_constraint(node.first_child)
            return None

        if node.is_key("atom"):
            word_expr = node.find_child_with_key("word_expr")
            if word_expr and is_constraint_word_expr(word_expr):
                return word_expr_to_constraint(word_expr)
            return None

        if node.is_key("pa"):
            if node.first_child:
                return self.node_to_constraint(node.first_child)
            return None

        if node.is_key(["co", "and"]):
            constraints = []
            for child in node.children:
                c = self.node_to_constraint(child)
                if c:
                    constraints.append(c)
            if len(constraints) == 0:
                return None
            elif len(constraints) == 1:
                return constraints[0]
            else:
                return {"AND": constraints}

        if node.is_key("or"):
            constraints = []
            for child in node.children:
                c = self.node_to_constraint(child)
                if c:
                    constraints.append(c)
            if len(constraints) == 0:
                return None
            elif len(constraints) == 1:
                return constraints[0]
            else:
                return {"OR": constraints}

        return None

    def extract_constraints_from_tree(
        self, expr_tree: DslExprNode, fields: list[str] = None
    ) -> tuple[list[dict], DslExprNode]:
        """Extract constraint word_exprs from the expr tree.

        Finds constraint word_expr atoms, converts them to constraint dicts,
        and removes them from the tree. Handles boolean structure.

        Args:
            expr_tree: The full expression tree.
            fields: Default fields for constraints (e.g., ["title.words", "tags.words"]).

        Returns:
            Tuple of:
            - constraints: list of constraint dicts for es_tok_constraints
            - cleaned_tree: expr tree with constraint atoms removed
        """
        # Find all word_expr nodes and check which are constraints
        word_expr_nodes = expr_tree.find_all_childs_with_key("word_expr")
        constraint_word_nodes = [
            n for n in word_expr_nodes if is_constraint_word_expr(n)
        ]

        if not constraint_word_nodes:
            return [], expr_tree

        # Check if the entire subtree (excluding non-constraint atoms) forms
        # a pure constraint expression. If so, convert the whole boolean structure.
        # For mixed trees, extract constraints individually.

        # Strategy: For each constraint word_expr, find its closest boolean parent
        # that is a pure constraint subtree. Convert that subtree to a constraint.
        # Then remove those atoms from the tree.

        # Simpler strategy: collect all constraint atoms individually,
        # combine them with AND at the top level.
        constraints = []
        for word_node in constraint_word_nodes:
            c = word_expr_to_constraint(word_node)
            if c:
                constraints.append(c)
            # Remove the atom containing this constraint from the tree
            atom_node = word_node.find_parent_with_key("atom")
            if atom_node and atom_node.parent:
                atom_node.disconnect_from_parent()

        return constraints, expr_tree

    def extract_constraints_with_bool_structure(
        self, expr_tree: DslExprNode, fields: list[str] = None
    ) -> tuple[list[dict], DslExprNode]:
        """Extract constraints while preserving boolean structure.

        For expressions like `(+A & +B) | (+C & -D)`, this preserves the
        OR/AND structure in the constraint output.

        For mixed expressions like `世界 +A -B`, only the constraints are
        extracted with AND combination.

        Args:
            expr_tree: The full expression tree.
            fields: Default fields for constraints.

        Returns:
            Tuple of (constraints list, modified tree).
        """
        word_expr_nodes = expr_tree.find_all_childs_with_key("word_expr")
        constraint_word_nodes = [
            n for n in word_expr_nodes if is_constraint_word_expr(n)
        ]

        if not constraint_word_nodes:
            return [], expr_tree

        # Find the closest common boolean ancestor of all constraint nodes
        # If that ancestor is a pure constraint subtree, convert with structure
        # Otherwise, extract individually with AND

        # Check each boolean parent for pure constraint status
        bool_parents = set()
        for word_node in constraint_word_nodes:
            parent = word_node.find_parent_with_key(BOOL_OPS)
            if parent:
                bool_parents.add(id(parent))

        # Try to find the highest pure-constraint boolean subtree
        # Walk up from constraint nodes to find pure constraint subtrees
        pure_constraint_roots = []
        processed_atoms = set()

        for word_node in constraint_word_nodes:
            atom_node = word_node.find_parent_with_key("atom")
            if atom_node and id(atom_node) in processed_atoms:
                continue
            if atom_node:
                processed_atoms.add(id(atom_node))

            # Walk up through boolean parents to find the highest pure constraint root
            current = atom_node
            best_root = None
            while current and current.parent:
                parent = current.parent
                if parent.is_key(BOOL_OPS) and self.is_pure_constraint_subtree(parent):
                    best_root = parent
                    current = parent
                elif parent.is_key(["start", "expr"]):
                    break
                else:
                    break

            if best_root and id(best_root) not in [
                id(r) for r in pure_constraint_roots
            ]:
                pure_constraint_roots.append(best_root)

        constraints = []
        if pure_constraint_roots:
            for root in pure_constraint_roots:
                constraint = self.node_to_constraint(root)
                if constraint:
                    constraints.append(constraint)
                # Remove the constraint subtree from the main tree
                root.disconnect_from_parent()
        else:
            # No pure constraint subtrees found, extract individually
            for word_node in constraint_word_nodes:
                c = word_expr_to_constraint(word_node)
                if c:
                    constraints.append(c)
                atom_node = word_node.find_parent_with_key("atom")
                if atom_node and atom_node.parent:
                    atom_node.disconnect_from_parent()

        return constraints, expr_tree

    def normalize_constraints_in_tree(
        self, expr_tree: DslExprNode
    ) -> tuple[list[str], list[str], DslExprNode]:
        """Normalize constraint word_exprs in the expression tree.

        For `+token` (must-include): extract the text and remove the atom from
        the tree. The caller should add these as individual bool.filter clauses,
        ensuring each +token is independently required (AND semantics).

        For `-token`/`!token` (must-exclude): extract the text and remove the
        atom from the tree. The caller should add these as bool.must_not
        clauses using the same query mechanism as regular word search.

        Regular (unprefixed) words remain in the tree and go through the normal
        query pipeline (merged into a single es_tok_query_string query).

        Args:
            expr_tree: The full expression tree (will be modified in-place).

        Returns:
            Tuple of:
            - must_have_texts: list of text strings that must be present (from + tokens)
            - must_not_texts: list of text strings to exclude (from - tokens)
            - modified expr_tree: constraint tokens removed, regular words kept
        """
        word_expr_nodes = expr_tree.find_all_childs_with_key("word_expr")
        constraint_word_nodes = [
            n for n in word_expr_nodes if is_constraint_word_expr(n)
        ]

        if not constraint_word_nodes:
            return [], [], expr_tree

        must_have_texts = []
        must_not_texts = []

        for word_node in constraint_word_nodes:
            pp_key = get_constraint_pp_key(word_node)
            text = get_constraint_text(word_node)
            if not text:
                continue

            if pp_key == "pl":
                # +token: extract text and remove atom from tree
                must_have_texts.append(text)
                atom_node = word_node.find_parent_with_key("atom")
                if atom_node and atom_node.parent:
                    atom_node.disconnect_from_parent()
            elif pp_key in ["mi", "nq"]:
                # -token/!token: extract text and remove atom from tree
                must_not_texts.append(text)
                atom_node = word_node.find_parent_with_key("atom")
                if atom_node and atom_node.parent:
                    atom_node.disconnect_from_parent()

        return must_have_texts, must_not_texts, expr_tree


def constraints_to_filter(constraints: list[dict], fields: list[str] = None) -> dict:
    """Convert a list of constraint dicts to an es_tok_constraints filter.

    Args:
        constraints: List of constraint dicts.
        fields: Default fields for constraints.

    Returns:
        es_tok_constraints filter dict.
    """
    if not constraints:
        return {}
    body = {"constraints": constraints}
    if fields:
        body["fields"] = fields
    return {"es_tok_constraints": body}
