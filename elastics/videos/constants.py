from typing import Literal

# source fields
SOURCE_FIELDS = [
    "title",
    "bvid",
    "owner",
    "pic",
    "duration",
    "desc",
    "stat",
    "tname",
    "tags",
    "pubdate_str",
    "insert_at_str",
]
DOC_EXCLUDED_SOURCE_FIELDS = []

# search match fields
# SEARCH_MATCH_FIELDS_DEFAULT = ["title", "tags", "owner.name"]
SEARCH_MATCH_FIELDS_DEFAULT = ["title", "tags", "owner.name", "desc"]
SEARCH_MATCH_FIELDS_WORDS = [f"{field}.words" for field in SEARCH_MATCH_FIELDS_DEFAULT]
SEARCH_MATCH_FIELDS_PINYIN = [
    f"{field}.pinyin" for field in SEARCH_MATCH_FIELDS_DEFAULT
]
SEARCH_MATCH_FIELDS = [
    # *SEARCH_MATCH_FIELDS_DEFAULT,
    *SEARCH_MATCH_FIELDS_WORDS,
    # *SEARCH_MATCH_FIELDS_PINYIN,
]

# suggest match fields
SUGGEST_MATCH_FIELDS_DFAULT = ["title", "tags", "owner.name"]
SUGGEST_MATCH_FIELDS_WORDS = [f"{field}.words" for field in SUGGEST_MATCH_FIELDS_DFAULT]
SUGGEST_MATCH_FIELDS_PINYIN = [
    f"{field}.pinyin" for field in SUGGEST_MATCH_FIELDS_DFAULT
]
SUGGEST_MATCH_FIELDS = [
    # *SUGGEST_MATCH_FIELDS_DFAULT,
    *SUGGEST_MATCH_FIELDS_WORDS,
    *SUGGEST_MATCH_FIELDS_PINYIN,
]

# date match fields
DATE_MATCH_FIELDS_DEFAULT = ["title"]
DATE_MATCH_FIELDS_WORDS = [f"{field}.words" for field in DATE_MATCH_FIELDS_DEFAULT]
DATE_MATCH_FIELDS = [
    # *DATE_MATCH_FIELDS_DEFAULT,
    *DATE_MATCH_FIELDS_WORDS,
]

# boosted fields
SEARCH_BOOSTED_FIELDS = {
    "title": 2.5,
    "title.words": 2.5,
    "title.pinyin": 0.25,
    "tags": 3.5,
    "tags.words": 3.5,
    "tags.pinyin": 0.35,
    "owner.name": 2,
    "owner.name.words": 2,
    "owner.name.pinyin": 0.2,
    "desc": 0.1,
    "desc.words": 0.1,
    "desc.pinyin": 0.01,
}
SUGGEST_BOOSTED_FIELDS = {
    "title": 2.5,
    "title.words": 2.5,
    "title.pinyin": 0.5,
    "tags": 3.5,
    "tags.words": 3.5,
    "tags.pinyin": 0.7,
    "owner.name": 2,
    "owner.name.words": 2,
    "owner.name.pinyin": 0.4,
}
DATE_BOOSTED_FIELDS = {
    "title": 0.01,
    "title.words": 0.01,
    "owner.name": 0.01,
    "owner.name.words": 0.01,
    "desc": 0.003,
    "desc.words": 0.003,
    "tags": 0.008,
    "tags.words": 0.008,
}

# combined fields
SEARCH_COMBINED_FIELDS_LIST = [["title", "tags"]]
SUGGEST_COMBINED_FIELDS_LIST = [["title", "tags"]]

# match type, bool and operator
MATCH_TYPE = Literal[
    "best_fields",
    "most_fields",
    "cross_fields",
    "phrase",
    "phrase_prefix",
    "bool_prefix",
]
MATCH_BOOL = Literal["must", "should", "must_not", "filter"]
MATCH_OPERATOR = Literal["or", "and"]

# match type, bool and operator
SEARCH_MATCH_TYPE = "phrase_prefix"
SEARCH_MATCH_BOOL = "must"
SEARCH_MATCH_OPERATOR = "or"
SUGGEST_MATCH_TYPE = "phrase_prefix"
SUGGEST_MATCH_BOOL = SEARCH_MATCH_BOOL
SUGGEST_MATCH_OPERATOR = SEARCH_MATCH_OPERATOR

# search detail levels
SEARCH_DETAIL_LEVELS = {
    1: {
        "match_type": SEARCH_MATCH_TYPE,
        "bool": SEARCH_MATCH_BOOL,
        "filters": [
            {"range": {"stat.view": {"gte": 100}}},
            {"range": {"stat.coin": {"gte": 1}}},
        ],
        "timeout": 2,
    },
    2: {
        "match_type": SEARCH_MATCH_TYPE,
        "bool": SEARCH_MATCH_BOOL,
        "filters": [
            {"range": {"stat.view": {"gte": 1000}}},
            {"range": {"stat.coin": {"gte": 10}}},
        ],
        "timeout": 3,
    },
    3: {
        "match_type": "phrase",
        "bool": SEARCH_MATCH_BOOL,
        "filters": [
            {"range": {"stat.view": {"gte": 10000}}},
            {"range": {"stat.coin": {"gte": 15}}},
        ],
        "timeout": 4,
    },
}
MAX_SEARCH_DETAIL_LEVEL = 3

# suggest detail levels
SUGGEST_DETAIL_LEVELS = {
    1: {
        "match_type": SUGGEST_MATCH_TYPE,
        "bool": SUGGEST_MATCH_BOOL,
        "filters": [
            {"range": {"stat.view": {"gte": 200}}},
            {"range": {"stat.coin": {"gte": 5}}},
        ],
        "timeout": 1.5,
    },
    2: {
        "match_type": SUGGEST_MATCH_TYPE,
        "bool": SUGGEST_MATCH_BOOL,
        "filters": [
            {"range": {"stat.view": {"gte": 1000}}},
            {"range": {"stat.coin": {"gte": 10}}},
        ],
        "timeout": 2,
    },
    3: {
        "match_type": SUGGEST_MATCH_TYPE,
        "bool": SUGGEST_MATCH_BOOL,
        "filters": [
            {"range": {"stat.view": {"gt": 10000}}},
            {"range": {"stat.coin": {"gt": 20}}},
        ],
        "timeout": 3,
    },
}
MAX_SUGGEST_DETAIL_LEVEL = 3

# limits
SEARCH_LIMIT = 50
SUGGEST_LIMIT = 10

# timeout
SEARCH_TIMEOUT = 2
SUGGEST_TIMEOUT = 1.5

# This constant is to contain more hits for redundance,
# as drop_no_highlights would drop some hits
NO_HIGHLIGHT_REDUNDANCE_RATIO = 2
