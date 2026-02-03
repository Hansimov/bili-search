from typing import Literal

ELASTIC_VIDEOS_PRO_INDEX = "bili_videos_pro1"
ELASTIC_VIDEOS_DEV_INDEX = "bili_videos_dev6"
ELASTIC_DEV = "elastic_dev"
ELASTIC_PRO = "elastic_pro"

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
QUERY_TYPE = Literal[
    "multi_match", "combined_fields", "query_string", "es_tok_query_string"
]
# QUERY_TYPE_DEFAULT = "multi_match"
# QUERY_TYPE_DEFAULT = "query_string"
QUERY_TYPE_DEFAULT = "es_tok_query_string"

ES_TOK_QUERY_STRING_MAX_FREQ = 1000000
ES_TOK_QUERY_STRING_MIN_KEPT_TOKENS_COUNT = 2
ES_TOK_QUERY_STRING_MIN_KEPT_TOKENS_RATIO = -1.0

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
RANK_METHOD_TYPE = Literal["heads", "rrf", "stats", "relevance", "tiered"]
RANK_METHOD_DEFAULT = "stats"

# =============================================================================
# Explore Settings
# =============================================================================

# Explore result limits
EXPLORE_RANK_TOP_K = 400  # max results to return after ranking
EXPLORE_GROUP_OWNER_LIMIT = 25  # max author groups to return
EXPLORE_MOST_RELEVANT_LIMIT = 10000  # max docs to scan for relevance

# =============================================================================
# Ranking Configuration
# =============================================================================

# Relevance-only ranking settings (for vector search)
# When rank_method="relevance", only vector similarity score matters
# No stats/pubdate weighting - pure relevance ranking
RELEVANCE_MIN_SCORE = 0.4  # minimum normalized score to be considered relevant
RELEVANCE_SCORE_POWER = 2.0  # power transform to amplify score differences

# RRF (Reciprocal Rank Fusion) settings for multi-metric ranking
RRF_K = 60  # RRF constant k
RRF_HEAP_SIZE = 2000  # max items to consider per metric
RRF_HEAP_RATIO = 5  # heap_size = max(input, top_k * ratio)

# RRF weights for different metrics
RRF_WEIGHTS = {
    "pubdate": 2.0,  # publish date timestamp
    "stat.view": 1.0,
    "stat.favorite": 1.0,
    "stat.coin": 1.0,
    "score": 5.0,  # relevance score (highest weight)
}

# Stats-based ranking: relevance gating
# Results with score < RELATE_GATE_RATIO * max_score are filtered
RELATE_GATE_RATIO = 0.5  # higher = more selective
RELATE_GATE_COUNT = 2000  # max results to keep
RELATE_SCORE_POWER = 4  # power transform for relevance score

# =============================================================================
# Tiered Ranking (Hybrid Search)
# =============================================================================
# Tiered ranking for hybrid search:
# - High relevance zone: sort by stats/recency (热度+最新)
# - Low relevance zone: sort strictly by relevance
#
# ONLY results in the high relevance zone get popularity/recency boost.
# This prevents low-relevance but high-popularity content from ranking too high.

# High relevance threshold: only items with score >= threshold * max_score
# get the stats/recency boost. Items below this are sorted by relevance only.
# 0.7 means only top 30% by relevance score qualify for popularity boost
TIERED_HIGH_RELEVANCE_THRESHOLD = 0.7

# Within high relevance zone, further group by similarity for tie-breaking
# Items within this relative diff are considered "equally relevant"
TIERED_SIMILARITY_THRESHOLD = 0.05  # 5% relative difference

# Weights for secondary sort within high relevance zone
TIERED_STATS_WEIGHT = 0.7  # weight for popularity (view, coin, etc.)
TIERED_RECENCY_WEIGHT = 0.3  # weight for recency (pubdate)

# =============================================================================
# Query Mode Settings
# =============================================================================

# Query mode (qmod): controls word/vector/rerank search
# Each character represents a mode: w=word, v=vector, r=rerank
# Multiple chars enable hybrid search (e.g., "wv" = word+vector)
# Rerank mode (r) enables float embedding reranking for precise similarity
QMOD_SINGLE_TYPE = Literal["word", "vector", "rerank"]
QMOD_DEFAULT = ["word", "vector"]  # default to hybrid search (no rerank for speed)

# Hybrid search settings
# Word and vector weights are balanced 0.5:0.5 for fair fusion
# RRF fusion is rank-based, so these weights are used for weighted fusion mode
HYBRID_WORD_WEIGHT = 0.5  # weight for word-based score in hybrid mode
HYBRID_VECTOR_WEIGHT = 0.5  # weight for vector-based score in hybrid mode
HYBRID_RRF_K = 60  # k parameter for RRF fusion

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
RANK_TOP_K = 50
AGG_TOP_K = 1000

# timeout
SEARCH_TIMEOUT = 2
EXPLORE_TIMEOUT = 5
SUGGEST_TIMEOUT = 1.5
AGG_TIMEOUT = 3
TERMINATE_AFTER = 2000000

# KNN search settings
# text_emb is a 2048-bit vector stored as dense_vector with element_type="bit"
KNN_TEXT_EMB_FIELD = "text_emb"
# KNN_K: Number of nearest neighbors to return from each shard
# Higher values improve recall but increase latency
# For bit vectors with Hamming distance, larger K is needed for good recall
# Note: Should be >= KNN_RERANK_MAX_HITS for optimal reranking
KNN_K = 1000  # Must match KNN_RERANK_MAX_HITS for full reranking
# KNN_NUM_CANDIDATES: Candidates to consider per shard before selecting top K
# Should be significantly larger than K for good recall
# For 2048-bit vectors, more candidates help find relevant items
# Higher values improve recall but increase latency - ES does internal oversampling
KNN_NUM_CANDIDATES = 4000  # 4x of K for good bit vector recall
KNN_TIMEOUT = 8  # timeout for KNN search
KNN_SIMILARITY_TYPE = Literal["hamming", "l2_norm", "cosine"]
KNN_SIMILARITY_DEFAULT = "hamming"  # for bit vectors, hamming is most efficient
KNN_LSH_BITN = 2048  # LSH bit count, must match text_emb dims

# =============================================================================
# KNN Reranking Settings
# =============================================================================
# Reranking uses float embeddings for precise similarity calculation
# This compensates for precision loss in LSH bit vector quantization

# Whether to enable reranking by default for KNN search
KNN_RERANK_ENABLED = True

# Maximum number of hits to rerank (trade-off between quality and latency)
# Set to 0 to disable reranking
# Higher values improve recall at cost of embedding API calls
# Note: Values over 200 significantly increase latency
KNN_RERANK_MAX_HITS = 1000

# Boost factors for keyword matching during rerank
# These help surface results that contain exact query terms
KNN_RERANK_KEYWORD_BOOST = 1.5  # Boost when keyword found in tags/desc
KNN_RERANK_TITLE_KEYWORD_BOOST = 2.5  # Higher boost for title matches

# Text fields to use for document embedding during rerank
KNN_RERANK_TEXT_FIELDS = ["title", "tags", "desc", "owner.name"]

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
