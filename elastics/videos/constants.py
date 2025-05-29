from typing import Literal

VIDEOS_INDEX_DEFAULT = "bili_videos_dev4"

SEARCH_REQUEST_TYPE = Literal["search", "suggest"]
SEARCH_REQUEST_TYPE_DEFAULT = "search"
API_REQUEST_TYPE = Literal["search", "suggest", "random", "latest", "doc"]

# source fields
SOURCE_FIELDS = [
    *["bvid", "title", "desc"],
    *["tid", "ptid", "tname", "rtags", "tags"],
    *["owner", "pic", "duration", "stat"],
    *["pubdate", "insert_at"],
]
DOC_EXCLUDED_SOURCE_FIELDS = []

# search match fields
# SEARCH_MATCH_FIELDS_DEFAULT = ["title"]
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
BIAS_BOOSTED_FIELDS = {
    "title": 3,
    "title.words": 3,
    "tags": 2.5,
    "tags.words": 2.5,
    "owner.name": 2,
    "owner.name.words": 2,
}
NORM_BOOSTED_FIELDS = {
    "title": 1,
    "title.words": 1,
    "tags": 1,
    "tags.words": 1,
    "owner.name": 1,
    "owner.name.words": 1,
}
SEARCH_BOOSTED_FIELDS = {
    **BIAS_BOOSTED_FIELDS,
    # **NORM_BOOSTED_FIELDS,
    "title.pinyin": 0.25,
    "tags.pinyin": 0.2,
    "owner.name.pinyin": 0.2,
    "desc": 0.1,
    "desc.words": 0.1,
    "desc.pinyin": 0.01,
}
EXPLORE_BOOSTED_FIELDS = {
    **BIAS_BOOSTED_FIELDS,
    # **NORM_BOOSTED_FIELDS,
    "title.pinyin": 0.25,
    "tags.pinyin": 0.2,
    "owner.name.pinyin": 0.2,
    "desc": 0.1,
    "desc.words": 0.1,
    "desc.pinyin": 0.01,
}
SUGGEST_BOOSTED_FIELDS = {
    # **BIAS_BOOSTED_FIELDS,
    **NORM_BOOSTED_FIELDS,
    "title.pinyin": 0.5,
    "tags.pinyin": 0.4,
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

# query type
QUERY_TYPE = Literal["multi_match", "combined_fields", "query_string"]
# QUERY_TYPE_DEFAULT = "multi_match"
QUERY_TYPE_DEFAULT = "query_string"

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

# SEARCH_MATCH_TYPE = "phrase"
# SEARCH_MATCH_TYPE = "phrase_prefix"
SEARCH_MATCH_TYPE = "cross_fields"

# SUGGEST_MATCH_TYPE = "phrase_prefix"
SUGGEST_MATCH_TYPE = "cross_fields"

SEARCH_MATCH_BOOL = "must"
SUGGEST_MATCH_BOOL = SEARCH_MATCH_BOOL

SEARCH_MATCH_OPERATOR = "or"
SUGGEST_MATCH_OPERATOR = SEARCH_MATCH_OPERATOR

USE_SCRIPT_SCORE_DEFAULT = False
TRACK_TOTAL_HITS = True
IS_HIGHLIGHT = True

# search detail levels
SEARCH_DETAIL_LEVELS = {
    1: {
        "match_type": SEARCH_MATCH_TYPE,
        "bool": SEARCH_MATCH_BOOL,
        "filters": [
            {"range": {"stat.view": {"gte": 100}}},
        ],
        "timeout": 3,
    },
    2: {
        "match_type": SEARCH_MATCH_TYPE,
        "bool": SEARCH_MATCH_BOOL,
        "filters": [
            {"range": {"stat.view": {"gte": 100}}},
        ],
        "timeout": 6,
    },
}
MAX_SEARCH_DETAIL_LEVEL = 2

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
EXPLORE_TIMEOUT = 5
SUGGEST_TIMEOUT = 1.5
AGG_TIMEOUT = 3
TERMINATE_AFTER = 1000000

# aggregation
AGG_PERCENTS = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9, 99.99, 100]
AGG_SORT_FIELD = "_score"
AGG_SORT_ORDER = "desc"

# This constant is to contain more hits for redundance,
# as drop_no_highlights would drop some hits
NO_HIGHLIGHT_REDUNDANCE_RATIO = 2

# search analyzer
# defined in: [bili-scraper/converters/elastic/video_index_settings_v3.py]
SEARCH_ANALYZER_NAME = "chinese_analyzer"
