from typing import Literal

OP_REL_TYPE = Literal["eqs", "neqs", "lt", "gt", "leqs", "geqs"]

USER_OPS = ["eq", "neq", "at_eq", "at_neq"]
WORD_OPS = ["eq", "neq", "qs"]

START_EXPR = "start"
MAIN_EXPRS = ["expr", "expr_error"]
ATOM_EXPRS = ["atom_expr"]
PA_EXPRS = ["pa_expr"]
BOOL_EXPRS = ["and_expr", "or_expr", "co_expr"]
BOOL_OPS = ["or", "and", "co", "pa"]
ATOMS = [
    *["date_expr", "user_expr", "uid_expr", "stat_expr"],
    *["region_expr", "word_expr"],
]
TEXT_TYPES = ["text_quoted", "text_strict", "text_plain"]

ES_BOOL_OPS = ["must", "filter", "must_not", "should"]
ES_BOOL_OP_TYPE = Literal["must", "should", "must_not", "filter"]
