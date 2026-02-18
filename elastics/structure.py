from copy import deepcopy
from tclogger import logger
from typing import Union

from converters.query.field import is_pinyin_field, deboost_field, boost_fields
from converters.query.field import remove_suffixes_from_fields
from elastics.videos.constants import SEARCH_MATCH_FIELDS, SEARCH_BOOSTED_FIELDS
from elastics.videos.constants import DATE_MATCH_FIELDS, DATE_BOOSTED_FIELDS


def get_es_source_val(d: dict, key: str):
    keys = key.split(".")
    dd = deepcopy(d)
    for key in keys:
        if isinstance(dd, dict) and key in dd:
            dd = dd[key]
        else:
            joined_key = ".".join(keys)
            if joined_key in dd:
                return dd[joined_key]
            else:
                return None

    return dd


def get_highlight_settings(
    match_fields: list[str],
    removable_suffixes: list[str] = [".words"],
    tag: str = "hit",
):
    highlight_fields = [
        deboost_field(field) for field in match_fields if not is_pinyin_field(field)
    ]
    if removable_suffixes:
        highlight_fields.extend(
            remove_suffixes_from_fields(highlight_fields, suffixes=removable_suffixes)
        )

    highlight_fields = sorted(list(set(highlight_fields)))
    highlight_fields_dict = {field: {} for field in highlight_fields}

    highlight_settings = {
        "pre_tags": [f"<{tag}>"],
        "post_tags": [f"</{tag}>"],
        "fields": highlight_fields_dict,
    }
    return highlight_settings


def construct_boosted_fields(
    match_fields: list[str] = SEARCH_MATCH_FIELDS,
    boost: bool = True,
    boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
    use_pinyin: bool = False,
) -> tuple[list[str], list[str]]:
    if not use_pinyin:
        match_fields = [
            field for field in match_fields if not field.endswith(".pinyin")
        ]
    date_fields = [
        field
        for field in match_fields
        if not field.endswith(".pinyin")
        and any(field.startswith(date_field) for date_field in DATE_MATCH_FIELDS)
    ]
    if boost:
        boosted_match_fields = boost_fields(match_fields, boosted_fields)
        boosted_date_fields = boost_fields(date_fields, DATE_BOOSTED_FIELDS)
    else:
        boosted_match_fields = match_fields
        boosted_date_fields = date_fields
    return boosted_match_fields, boosted_date_fields


def set_timeout(body: dict, timeout: Union[int, float, str] = None):
    if timeout is not None:
        if isinstance(timeout, str):
            body["timeout"] = timeout
        elif isinstance(timeout, (int, float)):
            timeout_str = round(timeout * 1000)
            body["timeout"] = f"{timeout_str}ms"
        else:
            logger.warn(f"× Invalid type of `timeout`: {type(timeout)}")
    return body


def set_min_score(body: dict, min_score: float = None):
    if min_score is not None:
        body["min_score"] = min_score
    return body


def set_terminate_after(body: dict, terminate_after: int = None):
    if terminate_after is not None:
        body["terminate_after"] = terminate_after
    return body


def set_profile(body: dict, profile: bool = None):
    if profile is not None:
        body["profile"] = profile
    return body


def construct_constraint_filter(
    constraints: list[dict],
    fields: list[str] = None,
) -> dict:
    """Construct an es_tok_constraints query dict for use as a KNN pre-filter.

    This builds a query that leverages the es-tok plugin's constraint system
    to filter documents based on their indexed tokens. When used as a KNN
    filter, it acts as a pre-filter — ES only considers documents matching
    the constraints during HNSW graph exploration.

    Each constraint can optionally include a "fields" key to specify which
    index fields that particular constraint checks. When a constraint does
    not include "fields", the top-level `fields` parameter is used (or all
    fields if neither is specified).

    Args:
        constraints: List of constraint dicts. Each dict represents a
            constraint condition. Supported formats:
            - {"have_token": ["token1", "token2"]}
                Document must have at least one of these tokens.
            - {"have_token": ["token1"], "fields": ["title"]}
                Same as above, checking only the "title" field.
            - {"with_prefixes": ["prefix1"]}
                Document must have a token starting with one of these.
            - {"with_suffixes": ["suffix1"]}
                Document must have a token ending with one of these.
            - {"with_contains": ["substr1"]}
                Document must have a token containing one of these.
            - {"with_patterns": ["regex1"]}
                Document must have a token matching one of these regexes.
            - {"AND": {"have_token": [...]}}
                Explicit AND wrapping (same as bare condition).
            - {"OR": [{"have_token": [...]}, {"with_prefixes": [...]}]}
                At least one sub-constraint must match.
            - {"NOT": {"have_token": [...]}}
                The sub-constraint must NOT match.
            - {"NOT": {"have_token": [...]}, "fields": ["title"]}
                NOT constraint checking only "title" field.
        fields: Optional default fields for constraints without per-
            constraint fields. If None, the plugin uses all fields ("*").

    Returns:
        Query dict like:
        {"es_tok_constraints": {"constraints": [...], "fields": [...]}}

    Example:
        >>> # Filter: must have "影视飓风" in title/tags, must NOT have "广告"
        >>> construct_constraint_filter(
        ...     constraints=[
        ...         {"have_token": ["影视飓风"], "fields": ["title", "tags"]},
        ...         {"NOT": {"have_token": ["广告"]}},
        ...     ],
        ...     fields=["title", "tags"],
        ... )
        {'es_tok_constraints': {'constraints': [...], 'fields': [...]}}
    """
    body = {"constraints": constraints}
    if fields:
        body["fields"] = fields
    return {"es_tok_constraints": body}


def construct_knn_query(
    query_vector: list[int],
    field: str = "text_emb",
    k: int = 100,
    num_candidates: int = 1000,
    filter_clauses: list[dict] = None,
    constraint_filter: dict = None,
    similarity: float = None,
) -> dict:
    """Construct a KNN query for Elasticsearch.

    Args:
        query_vector: The query vector as a byte array (signed int8 list).
        field: The dense_vector field name to search.
        k: Number of nearest neighbors to return.
        num_candidates: Number of candidates to consider per shard.
        filter_clauses: Optional list of filter clauses to apply.
        constraint_filter: Optional es_tok_constraints query dict for
            token-level filtering. Applied as a pre-filter alongside
            other filter_clauses.
        similarity: Optional minimum similarity threshold.

    Returns:
        KNN query dict for use in ES search body.

    Note:
        For bit vectors (element_type="bit"), Elasticsearch uses Hamming distance.
        The score is calculated as: (num_bits - hamming_distance) / num_bits
        So higher scores mean more similar vectors.
    """
    knn_query = {
        "field": field,
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates,
    }

    # Merge filter_clauses and constraint_filter
    all_filters = list(filter_clauses) if filter_clauses else []
    if constraint_filter:
        all_filters.append(constraint_filter)

    if all_filters:
        # Combine filters into a bool query
        if len(all_filters) == 1:
            knn_query["filter"] = all_filters[0]
        else:
            knn_query["filter"] = {"bool": {"filter": all_filters}}

    if similarity is not None:
        knn_query["similarity"] = similarity

    return knn_query


def construct_knn_search_body(
    knn_query: dict,
    source_fields: list[str] = None,
    size: int = 50,
    timeout: Union[int, float, str] = None,
    track_total_hits: bool = True,
    is_explain: bool = False,
) -> dict:
    """Construct a complete search body with KNN query.

    Args:
        knn_query: The KNN query dict from construct_knn_query().
        source_fields: Fields to include in _source.
        size: Number of results to return (overrides k if larger).
        timeout: Search timeout.
        track_total_hits: Whether to track total hits.
        is_explain: Whether to include explanation.

    Returns:
        Complete search body dict for ES client.
    """
    body = {
        "knn": knn_query,
        "track_total_hits": track_total_hits,
        "explain": is_explain,
    }

    if source_fields:
        body["_source"] = source_fields

    if size:
        body["size"] = size

    if timeout is not None:
        body = set_timeout(body, timeout=timeout)

    return body


def analyze_tokens(
    es_client,
    index_name: str,
    text: str,
    analyzer: str = "chinese_analyzer",
) -> list[dict]:
    """Call ES _analyze API and return token list from the es-tok tokenizer.

    Each token dict contains:
        - token: the term string
        - start_offset: start position in original text
        - end_offset: end position in original text
        - type: token type (e.g., "vocab", "categ")
        - position: token position in stream

    Args:
        es_client: Elasticsearch client instance (e.g., searcher.es.client).
        index_name: Index name whose analyzer config to use.
        text: The text to analyze.
        analyzer: Analyzer name configured in the index mapping.

    Returns:
        List of token dicts from the analyzer response.
    """
    res = es_client.indices.analyze(
        index=index_name,
        body={"analyzer": analyzer, "text": text},
    )
    return res.body.get("tokens", [])


def select_covering_tokens(
    tokens: list[dict],
    text: str,
    min_token_len: int = 2,
) -> list[str]:
    """Select a minimal set of non-overlapping tokens that cover the query.

    Uses a greedy algorithm: at each uncovered position, pick the longest
    token that starts at that position. When a single token covers the
    entire query, prefers a 2-token split (AND of shorter components is
    more flexible and catches more relevant documents than requiring the
    exact compound token).

    For example:
        "小红书推荐系统" → ["小红书", "推荐系统"]
        "通义实验室"     → ["通义实验室"]
        "吴恩达大模型"   → ["吴恩达", "大模型"] (split preferred over compound)

    Args:
        tokens: Token list from analyze_tokens().
        text: The original query text.
        min_token_len: Minimum character length for a useful token.

    Returns:
        List of token strings forming the covering set.
    """
    text_len = len(text)
    if text_len == 0:
        return []

    # Build lookup: start_offset → list of (end_offset, token_str)
    # sorted by length descending (longest first)
    by_start: dict[int, list[tuple[int, str]]] = {}
    for t in tokens:
        tok_str = t["token"]
        if len(tok_str) < min_token_len:
            continue
        start = t["start_offset"]
        end = t["end_offset"]
        if start not in by_start:
            by_start[start] = []
        by_start[start].append((end, tok_str))

    for start in by_start:
        by_start[start].sort(key=lambda x: x[0], reverse=True)  # longest first

    def _greedy_cover(exclude_full_span: bool = False) -> list[str]:
        """Greedy covering: pick longest token at each position.

        Args:
            exclude_full_span: If True, skip tokens that cover the
                entire text (start=0, end=text_len).
        """
        result = []
        pos = 0
        while pos < text_len:
            if pos in by_start:
                best = None
                for end, tok_str in by_start[pos]:
                    if exclude_full_span and pos == 0 and end == text_len:
                        continue
                    best = (end, tok_str)
                    break  # already sorted longest-first
                if best:
                    result.append(best[1])
                    pos = best[0]
                else:
                    pos += 1
            else:
                pos += 1
        return result

    # First try the standard greedy covering
    covering = _greedy_cover(exclude_full_span=False)

    # If a single token covers the entire query, try splitting into
    # shorter components. AND of 2 tokens is more flexible than requiring
    # the exact compound token (e.g., "吴恩达" AND "大模型" matches more
    # relevant docs than requiring "吴恩达大模型" as a single token).
    if len(covering) == 1 and len(covering[0]) == text_len:
        split_covering = _greedy_cover(exclude_full_span=True)
        # Use the split if it produces 2+ tokens covering the full text
        if len(split_covering) >= 2:
            # Verify the split covers the full text
            total_covered = sum(len(t) for t in split_covering)
            if total_covered >= text_len:
                covering = split_covering

    return covering


def build_auto_constraint_filter(
    es_client,
    index_name: str,
    query: str,
    analyzer: str = "chinese_analyzer",
    fields: list[str] = None,
    min_token_len: int = 2,
) -> dict:
    """Automatically build a constraint filter from query tokenization.

    Analyzes the query using the index's tokenizer, selects covering
    tokens (the minimal set of longest non-overlapping tokens that
    cover the query text), and builds an AND constraint filter.

    This ensures that KNN results contain the key terms from the query
    in their indexed text fields, dramatically improving precision for
    compound queries like "小红书推荐系统" or "通义实验室".

    Args:
        es_client: Elasticsearch client instance.
        index_name: Index name for tokenizer config.
        query: The search query text.
        analyzer: Analyzer name (default: "chinese_analyzer").
        fields: Target fields for constraints (e.g., ["title.words",
            "tags.words"]). If None, all fields are searched.
        min_token_len: Minimum token length to consider.

    Returns:
        Constraint filter dict, or None if no useful tokens found.

    Example:
        >>> cf = build_auto_constraint_filter(
        ...     es_client, "bili_videos_dev6", "小红书推荐系统",
        ...     fields=["title.words", "tags.words"])
        >>> # Returns: {"es_tok_constraints": {"constraints": [
        >>> #   {"have_token": ["小红书"]}, {"have_token": ["推荐系统"]}
        >>> # ], "fields": ["title.words", "tags.words"]}}
    """
    tokens = analyze_tokens(es_client, index_name, query, analyzer)
    if not tokens:
        return None

    covering = select_covering_tokens(tokens, query, min_token_len=min_token_len)
    if not covering:
        return None

    constraints = [{"have_token": [tok]} for tok in covering]
    return construct_constraint_filter(constraints, fields=fields)


if __name__ == "__main__":
    d = {
        "owner": {"name": {"space": "value1"}},
        "stat": {"rights": {"view": "value2"}},
        "pubdate.time": "value3",
    }

    k1 = "owner.name.space"
    k2 = "stat.rights.view"
    k3 = "pubdate.time"
    for k in [k1, k2, k3]:
        print(f"{k}: {get_es_source_val(d,k)}")

    # python -m elastics.structure
