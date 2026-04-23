"""Search scope expression parser for DSL.

Parses expressions like:
- scope=t    (title only)
- s=tg       (title + tags)
- scope=n    (owner.name only)
- scope=d    (desc only)
- scope!=d   (search every default field except desc)

Aliases:
- u -> owner.name
- v -> title + tags

Scope expressions do not generate Elasticsearch filter clauses directly.
They are extracted upstream to limit the fields used for text matching and
constraint-field construction.
"""

from converters.query.field import deboost_field
from dsl.node import DslExprNode


SCOPE_DEFAULT_FIELDS = ["title", "tags", "owner.name", "desc"]
SCOPE_FIELD_CHAR_MAP = {
    "t": ["title"],
    "g": ["tags"],
    "n": ["owner.name"],
    "u": ["owner.name"],
    "d": ["desc"],
    "v": ["title", "tags"],
}
SCOPE_CONSTRAINT_FIELD_MAP = {
    "title": "title.words",
    "tags": "tags.words",
    "owner.name": "owner.name.words",
    "desc": "desc.words",
}


def parse_scope_val(scope_val: str) -> list[str]:
    fields: list[str] = []
    for char in str(scope_val or "").lower().strip():
        for field in SCOPE_FIELD_CHAR_MAP.get(char, []):
            if field not in fields:
                fields.append(field)
    return fields


def normalize_scope_fields(
    fields: list[str],
    operator: str = "eq",
    default_fields: list[str] | None = None,
) -> list[str]:
    default_fields = list(default_fields or SCOPE_DEFAULT_FIELDS)
    if operator == "neq":
        fields = [field for field in default_fields if field not in set(fields)]
    return fields or default_fields


class ScopeExprParser:
    """Parser for scope_expr DSL nodes."""

    def parse(self, node: DslExprNode) -> dict:
        scope_node = node.find_child_with_key("scope_expr")
        if not scope_node:
            return {}

        op_node = scope_node.find_child_with_key("scope_op")
        val_node = scope_node.find_child_with_key("scope_val")
        if not op_node or not val_node:
            return {}

        operator = op_node.get_deepest_node_key()
        raw_value = str(val_node.get_deepest_node_value() or "").lower()
        fields = normalize_scope_fields(parse_scope_val(raw_value), operator=operator)
        return {
            "op": operator,
            "mode": "exclude" if operator == "neq" else "include",
            "raw": raw_value,
            "fields": fields,
        }


class ScopeExprElasticConverter:
    """Converter for scope_expr to Elasticsearch query.

    Scope expressions are handled at the searcher layer, so this converter
    intentionally returns an empty dict.
    """

    def __init__(self):
        self.parser = ScopeExprParser()

    def convert(self, node: DslExprNode) -> dict:
        return {}

    def get_scope_info(self, node: DslExprNode) -> dict:
        return self.parser.parse(node)


def extract_scope_from_expr_tree(expr_tree: DslExprNode) -> dict:
    scope_nodes = expr_tree.find_all_childs_with_key("scope_expr")
    if not scope_nodes:
        return {}

    parser = ScopeExprParser()
    for node in reversed(scope_nodes):
        scope_info = parser.parse(node)
        if scope_info.get("fields"):
            return scope_info
    return {}


def filter_fields_by_scope(fields: list[str], scope_info: dict | None) -> list[str]:
    scope_fields = list((scope_info or {}).get("fields") or [])
    if not scope_fields:
        return list(fields)

    filtered = []
    for field in fields:
        deboosted_field = deboost_field(field)
        if any(
            deboosted_field == scope_field
            or deboosted_field.startswith(f"{scope_field}.")
            for scope_field in scope_fields
        ):
            filtered.append(field)
    return filtered


def get_scope_constraint_fields(
    scope_info: dict | None,
    default_fields: list[str] | None = None,
) -> list[str]:
    default_fields = list(default_fields or [])
    scope_fields = list((scope_info or {}).get("fields") or [])
    if not scope_fields:
        return default_fields

    constraint_fields: list[str] = []
    for field in scope_fields:
        constraint_field = SCOPE_CONSTRAINT_FIELD_MAP.get(field)
        if constraint_field and constraint_field not in constraint_fields:
            constraint_fields.append(constraint_field)
    return constraint_fields
