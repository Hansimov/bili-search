from typing import Literal

OP_REL_TYPE = Literal["eqs", "neqs", "lt", "gt", "leqs", "geqs"]

USER_OPS = ["eq", "neq", "at_eq", "at_neq"]
WORD_OPS = ["eq", "neq", "pl", "mi", "nq", "qs", "fz"]

START_EXPR = "start"
MAIN_EXPRS = ["expr", "expr_with_error"]
ATOM_EXPRS = ["atom_expr"]
PA_EXPRS = ["pa_expr"]
BOOL_EXPRS = ["and_expr", "or_expr", "co_expr"]
BOOL_OPS = ["or", "and", "co", "pa"]

WORD_EXPRS = ["word_expr"]
FILTER_EXPRS = [
    *["bvid_expr", "date_expr", "user_expr", "uid_expr"],
    *["stat_expr", "dura_expr", "region_expr"],
]
ITEM_EXPRS = [*FILTER_EXPRS, *WORD_EXPRS]
TEXT_TYPES = ["text_quoted", "text_strict", "text_plain"]

ES_BOOL_OPS = ["must", "filter", "must_not", "should"]
ES_BOOL_OP_TYPE = Literal["must", "filter", "must_not", "should"]

MSM = "minimum_should_match"
BM = "bool.must"
BMM = f"bool.must.multi_match"
BMS = f"bool.must.query_string"
BMKS = f"bool.must.es_tok_query_string"
BMMQ = f"bool.must.multi_match.query"
BMSQ = f"bool.must.multi_match.query"

BM_MAP = {
    "multi_match": {
        "BM": "bool.must.multi_match",
        "BMQ": "bool.must.multi_match.query",
    },
    "query_string": {
        "BM": "bool.must.query_string",
        "BMQ": "bool.must.query_string.query",
    },
    "es_tok_query_string": {
        "BM": "bool.must.es_tok_query_string",
        "BMQ": "bool.must.es_tok_query_string.query",
    },
}


STAT_KEYS = ["view", "like", "coin", "favorite", "reply", "danmaku", "share"]
