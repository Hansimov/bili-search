from typing import Literal

OP_REL_TYPE = Literal["eqs", "neqs", "lt", "gt", "leqs", "geqs"]

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
