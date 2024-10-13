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
SEARCH_MATCH_FIELDS_DEFAULT = ["title", "tags", "owner.name", "desc"]
SEARCH_MATCH_FIELDS_WORDS = [f"{field}.words" for field in SEARCH_MATCH_FIELDS_DEFAULT]
SEARCH_MATCH_FIELDS_PINYIN = [
    f"{field}.pinyin" for field in SEARCH_MATCH_FIELDS_DEFAULT
]
SEARCH_MATCH_FIELDS = [
    # *SEARCH_MATCH_FIELDS_DEFAULT,
    *SEARCH_MATCH_FIELDS_WORDS,
    *SEARCH_MATCH_FIELDS_PINYIN,
    "pubdate_str",
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
    "pubdate_str",
]

# date match fields
DATE_MATCH_FIELDS_DEFAULT = ["title", "desc", "tags"]
DATE_MATCH_FIELDS_WORDS = [f"{field}.words" for field in DATE_MATCH_FIELDS_DEFAULT]
DATE_MATCH_FIELDS = [
    # *DATE_MATCH_FIELDS_DEFAULT,
    *DATE_MATCH_FIELDS_WORDS,
    "pubdate_str",
]

# boosted fields
SEARCH_BOOSTED_FIELDS = {
    "title": 2.5,
    "title.words": 2.5,
    "title.pinyin": 0.25,
    "tags": 2,
    "tags.words": 2,
    "tags.pinyin": 0.2,
    "owner.name": 2,
    "owner.name.words": 2,
    "owner.name.pinyin": 0.2,
    "desc": 0.1,
    "desc.words": 0.1,
    "desc.pinyin": 0.01,
    "pubdate_str": 2.5,
}
SUGGEST_BOOSTED_FIELDS = {
    "title": 2.5,
    "title.words": 2.5,
    "title.pinyin": 0.5,
    "tags": 2,
    "tags.words": 2,
    "tags.pinyin": 0.4,
    "owner.name": 2,
    "owner.name.words": 2,
    "owner.name.pinyin": 0.4,
    "pubdate_str": 2.5,
}
DATE_BOOSTED_FIELDS = {
    "title": 0.1,
    "title.words": 0.1,
    "owner.name": 0.1,
    "owner.name.words": 0.1,
    "desc": 0.03,
    "desc.words": 0.03,
    "tags": 0.08,
    "tags.words": 0.08,
    "pubdate_str": 2.5,
}

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

# search match type, bool and operator
SEARCH_MATCH_TYPE = "phrase_prefix"
SUGGEST_MATCH_TYPE = "phrase_prefix"
SEARCH_MATCH_BOOL = "must"
SEARCH_MATCH_OPERATOR = "or"

# search detail levels
SEARCH_DETAIL_LEVELS = {
    1: {"match_type": "phrase", "bool": "must"},
    2: {"match_type": "cross_fields", "bool": "must", "operator": "and"},
    3: {"match_type": "most_fields", "bool": "should"},
}
MAX_SEARCH_DETAIL_LEVEL = 3

# limits
SUGGEST_LIMIT = 10
SEARCH_LIMIT = 50

# This constant is to contain more hits for redundance,
# as drop_no_highlights would drop some hits
NO_HIGHLIGHT_REDUNDANCE_RATIO = 2
