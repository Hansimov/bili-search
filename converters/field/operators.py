OP_EN_MAP = {">": "gt", "<": "lt", ">=": "gte", "<=": "lte"}
OP_ZH_MAP = {"》": "gt", "《": "lt", "》=": "gte", "《=": "lte"}
OP_MAP = {**OP_EN_MAP, **OP_ZH_MAP}

BRACKET_EN_MAP = {"(": "gt", ")": "lt", "[": "gte", "]": "lte"}
BRACKET_ZH_MAP = {"（": "gt", "）": "lt", "【": "gte", "】": "lte"}
BRACKET_MAP = {**BRACKET_EN_MAP, **BRACKET_ZH_MAP}
