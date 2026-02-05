"""Query mode (qmod) expression parser for DSL.

Parses expressions like:
- q=w     (word-based retrieval only)
- q=v     (vector-based KNN retrieval only)
- qm=wv   (word + vector, hybrid)
- qmod=vw (vector + word, same as wv)
- q=wr    (word + rerank with embeddings)
- q=vr    (vector + rerank)
- q=wvr   (word + vector + rerank)

Each character represents an independent mode that can be combined:
- w: word-based retrieval
- v: vector-based KNN retrieval
- r: rerank results using float embeddings for precise similarity

Valid combinations require at least 'w' or 'v':
- w, v, wv: retrieval modes without reranking
- wr, vr, wvr: retrieval modes with reranking

When multiple retrieval modes are specified (e.g., "wv"), results from all modes
are fused together using RRF or weighted combination. The 'r' mode applies
reranking to the fused results for improved precision.
"""

from typing import Literal, Union

from converters.dsl.node import DslExprNode
from converters.dsl.constants import QMOD_CHARS


# Single query mode types (retrieval modes)
QMOD_RETRIEVAL_TYPE = Literal["word", "vector"]

# All mode types including rerank
QMOD_SINGLE_TYPE = Literal["word", "vector", "rerank"]

# Default query mode - hybrid search for best results (without rerank for speed)
QMOD = ["word", "vector"]

# Character to mode name mapping
QMOD_CHAR_MAP = {
    "w": "word",
    "v": "vector",
    "r": "rerank",
}

# Mode name to character mapping (reverse)
QMOD_NAME_MAP = {v: k for k, v in QMOD_CHAR_MAP.items()}


def parse_qmod_str(mode_str: str) -> list[str]:
    """Parse qmod string into list of mode names.

    Args:
        mode_str: Raw mode string like "w", "v", "wv", "vw", "wr", "vr", "wvr".

    Returns:
        List of mode names like ["word"], ["vector"], ["word", "vector"], ["word", "rerank"].
        Always in canonical order: word, vector, rerank.
        Returns default (["word", "vector"]) if no valid retrieval mode is specified.
    """
    mode_str = mode_str.lower().strip()
    modes = set()

    for char in mode_str:
        if char in QMOD_CHAR_MAP:
            mode_name = QMOD_CHAR_MAP[char]
            modes.add(mode_name)

    # Validate: must have at least 'w' or 'v' for retrieval
    has_retrieval = "word" in modes or "vector" in modes
    if not has_retrieval:
        return QMOD.copy()

    # Return in canonical order: word, vector, rerank
    canonical_order = ["word", "vector", "rerank"]
    return [m for m in canonical_order if m in modes]


def normalize_qmod(modes: Union[str, list[str]]) -> list[str]:
    """Normalize qmod to list of mode names.

    Args:
        modes: String like "wv" or "wvr" or list like ["word", "vector"].

    Returns:
        List of mode names. Returns default if no valid retrieval mode.
    """
    if isinstance(modes, str):
        return parse_qmod_str(modes)
    elif isinstance(modes, list):
        # Already a list, ensure valid mode names
        valid_modes = []
        for mode in modes:
            if mode in QMOD_NAME_MAP:
                # It's a mode name like "word"
                if mode not in valid_modes:
                    valid_modes.append(mode)
            elif mode in QMOD_CHAR_MAP:
                # It's a char like "w"
                mode_name = QMOD_CHAR_MAP[mode]
                if mode_name not in valid_modes:
                    valid_modes.append(mode_name)
        # Validate: must have at least 'w' or 'v' for retrieval
        has_retrieval = "word" in valid_modes or "vector" in valid_modes
        if not has_retrieval:
            return QMOD.copy()
        # Return in canonical order
        canonical_order = ["word", "vector", "rerank"]
        return [m for m in canonical_order if m in valid_modes]
    else:
        return QMOD.copy()


def qmod_to_str(modes: list[str]) -> str:
    """Convert list of mode names to string representation.

    Args:
        modes: List like ["word", "vector"].

    Returns:
        String like "wv".
    """
    chars = []
    for mode in modes:
        if mode in QMOD_NAME_MAP:
            chars.append(QMOD_NAME_MAP[mode])
    return "".join(chars)


def is_hybrid_qmod(modes: list[str]) -> bool:
    """Check if qmod represents hybrid search (multiple retrieval modes).

    Args:
        modes: List of mode names.

    Returns:
        True if both word and vector retrieval modes are specified.
    """
    return "word" in modes and "vector" in modes


def has_rerank_qmod(modes: list[str]) -> bool:
    """Check if qmod includes rerank mode.

    Args:
        modes: List of mode names.

    Returns:
        True if rerank mode is specified.
    """
    return "rerank" in modes


def get_retrieval_modes(modes: list[str]) -> list[str]:
    """Get only the retrieval modes (excluding rerank).

    Args:
        modes: List of all mode names.

    Returns:
        List of retrieval mode names (word and/or vector only).
    """
    return [m for m in modes if m in ["word", "vector"]]


class QmodExprParser:
    """Parser for qmod_expr DSL nodes."""

    def parse(self, node: DslExprNode) -> list[str]:
        """Parse qmod_expr node and return list of mode names.

        Args:
            node: DslExprNode with key "qmod_expr" or containing it.

        Returns:
            List of mode names like ["word"] or ["word", "vector"].
        """
        qmod_node = node.find_child_with_key("qmod_expr")
        if not qmod_node:
            return QMOD.copy()

        val_node = qmod_node.find_child_with_key("qmod_val")
        if not val_node:
            return QMOD.copy()

        mode_str = val_node.get_deepest_node_value().lower()
        return parse_qmod_str(mode_str)


class QmodExprElasticConverter:
    """Converter for qmod_expr to Elasticsearch query.

    Note: qmod_expr doesn't generate Elasticsearch filter clauses.
    It's used to control the search mode at a higher level.
    """

    def __init__(self):
        self.parser = QmodExprParser()

    def convert(self, node: DslExprNode) -> dict:
        """Convert qmod_expr node.

        Since qmod doesn't produce ES filters, returns empty dict.
        The mode is extracted and used by the searcher/explorer to decide
        which search method to use.

        Args:
            node: DslExprNode with qmod_expr.

        Returns:
            Empty dict (qmod doesn't produce ES filters).
        """
        # qmod_expr doesn't produce ES filter clauses
        # It's a control expression for search mode selection
        return {}

    def get_modes(self, node: DslExprNode) -> list[str]:
        """Get the query modes from the node.

        Args:
            node: DslExprNode with qmod_expr.

        Returns:
            List of mode names.
        """
        return self.parser.parse(node)


def extract_qmod_from_expr_tree(expr_tree: DslExprNode) -> list[str]:
    """Extract query modes from an expression tree.

    Searches the tree for qmod_expr and returns the modes.
    If no qmod_expr is found, returns default mode ["word"].

    Args:
        expr_tree: Root of the DSL expression tree.

    Returns:
        List of mode names like ["word"] or ["word", "vector"].
    """
    parser = QmodExprParser()

    qmod_nodes = expr_tree.find_all_childs_with_key("qmod_expr")
    if not qmod_nodes:
        return QMOD.copy()

    # Use the last qmod_expr found (if multiple are specified)
    for node in reversed(qmod_nodes):
        modes = parser.parse(node)
        if modes:
            return modes

    return QMOD.copy()


def test_qmod_parser():
    """Test qmod parsing."""
    from converters.dsl.elastic import DslExprToElasticConverter
    from tclogger import logger

    converter = DslExprToElasticConverter()

    test_cases = [
        ("q=w", ["word"]),
        ("q=v", ["vector"]),
        ("q=wv", ["word", "vector"]),
        ("q=vw", ["word", "vector"]),  # canonical order
        ("qm=w", ["word"]),
        ("qmod=v", ["vector"]),
        ("黑神话 q=v", ["vector"]),
        ("黑神话 q=wv v>1w", ["word", "vector"]),
        ("黑神话 悟空", ["word", "vector"]),  # default is hybrid
        # New rerank mode tests
        ("q=wr", ["word", "rerank"]),
        ("q=vr", ["vector", "rerank"]),
        ("q=wvr", ["word", "vector", "rerank"]),
        ("q=rvw", ["word", "vector", "rerank"]),  # canonical order
        ("q=r", ["word", "vector"]),  # r alone invalid, returns default
    ]

    for test_str, expected in test_cases:
        expr_tree = converter.construct_expr_tree(test_str)
        modes = extract_qmod_from_expr_tree(expr_tree)
        status = "✓" if modes == expected else "×"
        logger.note(f"{status} Query: [{test_str}]")
        logger.mesg(f"  Modes: {modes} (expected: {expected})")


if __name__ == "__main__":
    test_qmod_parser()

    # python -m converters.dsl.fields.qmod
