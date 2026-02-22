from tclogger import tcdatetime, get_now

from dsl.node import DslExprNode
from dsl.elastic import DslExprToElasticConverter, DslTreeToExprConstructor
from dsl.fields.word import WordNodeToExprConstructor
from converters.times import DateFormatChecker


class WordNodeExpander:
    def __init__(self):
        self.elastic_converter = DslExprToElasticConverter()
        self.date_checker = DateFormatChecker()
        self.next_year_start_dt = tcdatetime(year=get_now().year + 1, month=1, day=1)

    def replace_word_node(self, word_node: DslExprNode) -> DslExprNode:
        """word_node key is `word_expr`. Add a `date_expr` node as its sibling, create a new `or` node as its parent, and connect `word_expr` and `date_expr` nodes to `or` node, and graft `or` node to the original parent of `word_expr` node."""
        word_atom = word_node.find_parent_with_key("atom")
        word_atom_parent = word_atom.parent
        word_atom.disconnect_from_parent()

        text = word_node.get_deepest_node_value()
        date_expr = f"d={text}"
        date_atom_root = self.elastic_converter.construct_expr_tree(date_expr)
        date_atom = date_atom_root.find_child_with_key("atom")
        date_atom.disconnect_from_parent()

        or_node = DslExprNode("or")
        word_atom.connect_to_parent(or_node)
        date_atom.connect_to_parent(or_node)
        or_node.connect_to_parent(word_atom_parent)

        return or_node

    def expand_date_formatted_word_node(self, node: DslExprNode) -> DslExprNode:
        """node key is `word_expr`. Expand a data-formated `word_expr` node to a `or` node with atom nodes with `word_expr` and `date_expr`."""
        word_val_nodes = node.find_all_childs_with_key("word_val_single")
        # Do not expand list of word_val_single nodes or non-word nodes
        if len(word_val_nodes) != 1:
            return node
        word_val_node = word_val_nodes[0]
        text = word_val_node.get_deepest_node_value()
        if text and self.date_checker.is_in_date_range(
            text, start="2009-09-09", end=self.next_year_start_dt
        ):
            word_val_node.extras["is_date_format"] = True
            return self.replace_word_node(node)
        else:
            return node

    def expand_expr_tree(self, node: DslExprNode) -> DslExprNode:
        """Expand all `word_expr` nodes that have date format in the expr tree."""
        word_expr_nodes = node.find_all_childs_with_key("word_expr")
        for word_expr_node in word_expr_nodes:
            self.expand_date_formatted_word_node(word_expr_node)
        return node


class DslExprRewriter:
    def __init__(self):
        self.elastic_converter = DslExprToElasticConverter()
        self.expr_constructor = DslTreeToExprConstructor()
        self.word_constructor = WordNodeToExprConstructor()
        self.word_expander = WordNodeExpander()

    def get_words_from_expr_tree(
        self, expr_tree: DslExprNode
    ) -> tuple[list[str], list[str], list[str]]:
        """Extract scoring keywords, date keywords, and constraint texts from tree.

        Constraint words (+token/-token) are separated from scoring keywords
        because they don't produce BM25 scoring — they become es_tok_constraints
        filters. Only regular word_expr nodes produce BM25 scoring clauses.

        Returns:
            (words_body, words_date, constraint_texts):
            - words_body: Regular keywords that generate BM25 scoring.
            - words_date: Date-formatted keywords for date field matching.
            - constraint_texts: +token texts (for display/embedding, NOT scoring).
              -token texts are excluded entirely.
        """
        from dsl.fields.constraint import (
            is_constraint_word_expr,
            get_constraint_pp_key,
            get_constraint_text,
        )

        word_expr_nodes = expr_tree.find_all_childs_with_key("word_expr")
        words_body = []
        words_date = []
        constraint_texts = []
        for word_expr_node in word_expr_nodes:
            # Handle constraint words (+token/-token) specially
            if is_constraint_word_expr(word_expr_node):
                pp_key = get_constraint_pp_key(word_expr_node)
                if pp_key in ["mi", "nq"]:
                    # -token/!token: exclude from keywords entirely
                    continue
                # +token: collect as constraint text (NOT a scoring keyword)
                text = get_constraint_text(word_expr_node)
                if text:
                    constraint_texts.append(text)
                continue

            word_val_node = word_expr_node.find_child_with_key("word_val_single")
            word_expr = self.word_constructor.construct(word_expr_node)
            if word_val_node.extras.get("is_date_format", False):
                words_date.append(word_expr)
            else:
                words_body.append(word_expr)
        return words_body, words_date, constraint_texts

    def get_constraint_filter_from_expr_tree(self, expr_tree: DslExprNode) -> dict:
        """Build es_tok_constraints filter from +/- constraint words in the tree.

        This extracts constraint information from the expression tree and returns
        an es_tok_constraints dict suitable for use as a KNN pre-filter or as a
        standalone constraint filter.

        Args:
            expr_tree: Expression tree containing word_expr nodes.

        Returns:
            es_tok_constraints dict, or empty dict if no constraints found.
            Example: {"es_tok_constraints": {"constraints": [...], "fields": [...]}}
        """
        from dsl.fields.constraint import (
            is_constraint_word_expr,
            get_constraint_pp_key,
            get_constraint_text,
        )
        from elastics.videos.constants import SEARCH_MATCH_FIELDS

        word_expr_nodes = expr_tree.find_all_childs_with_key("word_expr")
        constraints = []
        for word_node in word_expr_nodes:
            if not is_constraint_word_expr(word_node):
                continue
            pp_key = get_constraint_pp_key(word_node)
            text = get_constraint_text(word_node)
            if not text:
                continue
            if pp_key == "pl":
                t = text.lower()
                constraints.append({"have_token": [t]})
            elif pp_key in ["mi", "nq"]:
                t = text.lower()
                constraints.append({"NOT": {"have_token": [t]}})

        if not constraints:
            return {}
        return {
            "es_tok_constraints": {
                "constraints": constraints,
                "fields": SEARCH_MATCH_FIELDS,
            }
        }

    def replace_words_in_expr_tree(
        self, expr_tree: DslExprNode, qword_hword_dict: dict[str, str]
    ) -> DslExprNode:
        """Rewrite the expr tree with the given qword_hword_dict."""
        if not qword_hword_dict:
            return expr_tree
        new_expr_tree = expr_tree.copy()
        word_nodes = new_expr_tree.find_all_childs_with_key("word_val_single")
        for word_node in word_nodes:
            word = word_node.get_deepest_node_value().lower()
            if word in qword_hword_dict:
                new_word = qword_hword_dict[word]
                word_node.set_deepest_node_value(new_word)
        return new_expr_tree

    def expr_to_tree(self, expr: str) -> DslExprNode:
        expr_tree = self.elastic_converter.construct_expr_tree(expr)
        expr_tree = self.word_expander.expand_expr_tree(expr_tree)
        return expr_tree

    def get_query_info(self, expr: str) -> dict:
        """Example output:
        ```python
        {
            "query":             str,
            "words_expr":        str
            "keywords_body":     list[str]  # Scoring keywords only (no constraints)
            "keywords_date":     list[str]
            "constraint_texts":  list[str]  # +token texts (for display/embedding)
            "query_expr_tree":   DslExprNode(...)
            "constraint_filter": dict  # es_tok_constraints dict, or {}
        }
        ```
        #ANCHOR[id=query_info]
        """
        expr_tree = self.expr_to_tree(expr)
        words_expr_tree = expr_tree.filter_atoms_by_keys(include_keys=["word_expr"])
        words_body, words_date, constraint_texts = self.get_words_from_expr_tree(
            words_expr_tree
        )
        words_expr = self.expr_constructor.construct(words_expr_tree)
        constraint_filter = self.get_constraint_filter_from_expr_tree(words_expr_tree)
        return {
            "query": expr,
            "words_expr": words_expr,
            "keywords_body": words_body,
            "keywords_date": words_date,
            "constraint_texts": constraint_texts,
            "query_expr_tree": expr_tree,
            "constraint_filter": constraint_filter,
        }

    def rewrite_query_info_by_suggest_info(
        self, query_info: dict = {}, suggest_info: dict = {}, threshold: int = 2
    ) -> dict:
        """Get rewrite_info with query_info and suggest_info.

        Input:
            - Format of query_info:   # LINK dsl/rewrite.py#query_info
            - Format of suggest_info: # LINK elastics/videos/hits.py#suggest_info

        Example output:
        ```python
        {
            "rewrited":                bool,
            "is_original_in_rewrites": bool,
            "rewrited_expr_trees":     list[DslExprNode],
            "rewrited_word_exprs":     list[str],
        }
        ```

        #ANCHOR[id=rewrite_info]
        """
        rewrite_info = {"rewrited": False, "is_original_in_rewrites": False}
        if not query_info or not suggest_info:
            return rewrite_info
        # expr_tree = query_info["query_expr_tree"]
        query = query_info["query"]
        expr_tree = self.expr_to_tree(query)
        group_replaces_count = suggest_info.get("group_replaces_count", {})
        rewrited_expr_trees: list[DslExprNode] = []
        rewrited_word_exprs: list[str] = []
        for group_replaces, group_count in group_replaces_count:
            # this line is used in frontend to check if original query is in rewrites,
            # to determine whether the original query should be displayed or not in rewrite options
            if len(group_replaces) == 0:
                rewrite_info["is_original_in_rewrites"] = True
            # replace qword with hword in expr_tree
            qword_hword_dict = {}
            for i in range(0, len(group_replaces), 2):
                # loop over group_replaces by pairs
                qword = group_replaces[i]
                hword = group_replaces[i + 1]
                qword_hword_dict[qword] = hword
            rewrited_expr_tree = self.replace_words_in_expr_tree(
                expr_tree, qword_hword_dict=qword_hword_dict
            )
            rewrited_expr_trees.append(rewrited_expr_tree)
            # only keep word_expr atoms for expr display
            rewrited_word_expr_tree = rewrited_expr_tree.filter_atoms_by_keys(
                include_keys=["word_expr"]
            )
            rewrited_word_expr = self.expr_constructor.construct(
                rewrited_word_expr_tree
            )
            rewrited_word_exprs.append(rewrited_word_expr)
        rewrite_info["rewrited_expr_trees"] = rewrited_expr_trees
        rewrite_info["rewrited_word_exprs"] = rewrited_word_exprs
        if rewrited_word_exprs:
            rewrite_info["rewrited"] = True
        return rewrite_info
