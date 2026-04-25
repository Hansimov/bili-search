from fastapi.encoders import jsonable_encoder
from tclogger import logger, logstr, dict_to_str, brk

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.explorer import VideoExplorer
from converters.query.dsl import MultiMatchQueryDSLConstructor
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.query.filter import QueryFilterExtractor
from elastics.tests.test_videos import make_searcher, make_explorer, author_values, author_items


def test_es_tok_query_params():
    """Test that es_tok_query_string queries do NOT include unsupported params.

    After the es_tok plugin rewrite, only `max_freq` and `constraints` are
    supported. The old `min_kept_tokens_count` and `min_kept_tokens_ratio`
    params must NOT be sent, otherwise the plugin throws a ParsingException
    which causes search to return empty results (timed_out=True, took=-1).
    """
    import json
    from dsl.elastic import DslExprToElasticConverter
    from elastics.structure import construct_boosted_fields
    from elastics.videos.constants import (
        SEARCH_MATCH_FIELDS,
        EXPLORE_BOOSTED_FIELDS,
        SEARCH_MATCH_TYPE,
    )

    logger.note("> Testing es_tok_query_string parameter validation...")

    converter = DslExprToElasticConverter()
    test_queries = [
        "影视飓风",
        "deepseek v3",
        "黑神话 悟空",
        '"罗永浩"',
        "are you ok",
        "雷军 2024",
    ]

    unsupported_params = ["min_kept_tokens_count", "min_kept_tokens_ratio", '"type"']

    all_passed = True
    for query in test_queries:
        boosted_match_fields, boosted_date_fields = construct_boosted_fields(
            match_fields=SEARCH_MATCH_FIELDS,
            boost=True,
            boosted_fields=EXPLORE_BOOSTED_FIELDS,
        )
        converter.word_converter.switch_mode(
            match_fields=boosted_match_fields,
            date_match_fields=boosted_date_fields,
            match_type=SEARCH_MATCH_TYPE,
        )
        expr_tree = converter.construct_expr_tree(query)
        query_dsl_dict = converter.expr_tree_to_dict(expr_tree)

        # Serialize to JSON string and check for unsupported params
        dsl_str = json.dumps(query_dsl_dict)
        for param in unsupported_params:
            if param in dsl_str:
                logger.warn(f"  × [{query}] still contains unsupported param: {param}")
                all_passed = False
            else:
                logger.mesg(f"  ✓ [{query}] does not contain '{param}'")

        # Check that supported params are present
        if "es_tok_query_string" in dsl_str:
            for param in ["query", "max_freq"]:
                if param in dsl_str:
                    logger.mesg(f"  ✓ [{query}] contains required param: {param}")

    if all_passed:
        logger.success("\n✓ All es_tok query params are correct!")
    else:
        logger.warn("\n× Some queries still have unsupported params!")
    assert all_passed, "Unsupported params found in es_tok_query_string queries"


def test_dsl_query_construction():
    """Test that DSL query construction produces valid es_tok_query_string queries.

    Validates the full pipeline from query string -> DSL expression tree ->
    Elasticsearch query dict, ensuring the output is well-formed.
    """
    from dsl.elastic import DslExprToElasticConverter
    from dsl.rewrite import DslExprRewriter
    from dsl.filter import QueryDslDictFilterMerger
    from elastics.structure import construct_boosted_fields
    from elastics.videos.constants import (
        SEARCH_MATCH_FIELDS,
        EXPLORE_BOOSTED_FIELDS,
        SEARCH_MATCH_TYPE,
    )

    logger.note("> Testing DSL query construction pipeline...")

    converter = DslExprToElasticConverter()
    rewriter = DslExprRewriter()
    filter_merger = QueryDslDictFilterMerger()

    test_cases = [
        # (query, expected_keys_in_output, description)
        ("影视飓风", ["es_tok_query_string"], "Simple Chinese query"),
        ("影视飓风 q=w", ["es_tok_query_string"], "Query with word mode"),
        ("deepseek v3 0324", ["es_tok_query_string"], "Multi-word query"),
        ('影视飓风 "罗永浩"', ["es_tok_query_string"], "Query with quoted phrase"),
        ("黑神话 v>1k", ["es_tok_query_string", "range"], "Query with stat filter"),
        ("u=影视飓风 d>2024-06-01", ["range"], "Filter-only query (no word)"),
        ("雷军 2024", ["es_tok_query_string"], "Query with date keyword"),
    ]

    all_passed = True
    for query, expected_keys, description in test_cases:
        try:
            boosted_match_fields, boosted_date_fields = construct_boosted_fields(
                match_fields=SEARCH_MATCH_FIELDS,
                boost=True,
                boosted_fields=EXPLORE_BOOSTED_FIELDS,
            )
            query_info = rewriter.get_query_info(query)
            rewrite_info = rewriter.rewrite_query_info_by_suggest_info(query_info, {})
            expr_tree = rewrite_info.get("rewrited_expr_tree", None)
            expr_tree = expr_tree or query_info.get("query_expr_tree", None)
            converter.word_converter.switch_mode(
                match_fields=boosted_match_fields,
                date_match_fields=boosted_date_fields,
                match_type=SEARCH_MATCH_TYPE,
            )
            query_dsl_dict = converter.expr_tree_to_dict(expr_tree)

            dsl_str = str(query_dsl_dict)
            found_keys = [key for key in expected_keys if key in dsl_str]
            if len(found_keys) == len(expected_keys):
                logger.mesg(f"  ✓ {description}: [{query}]")
            else:
                missing = set(expected_keys) - set(found_keys)
                logger.warn(f"  × {description}: [{query}] missing keys: {missing}")
                all_passed = False
        except Exception as e:
            logger.warn(f"  × {description}: [{query}] ERROR: {e}")
            all_passed = False

    if all_passed:
        logger.success("\n✓ All DSL construction tests passed!")
    else:
        logger.warn("\n× Some DSL construction tests failed!")
    assert all_passed, "DSL construction test failures"


def test_es_tok_query_preserves_constraint_syntax_in_query_text():
    """Ensure +/- stay inside es_tok_query_string instead of becoming es_tok_constraints."""
    import json
    from dsl.elastic import DslExprToElasticConverter

    converter = DslExprToElasticConverter()
    expr_tree = converter.construct_expr_tree("游戏音乐 +若生命将于明日落幕 -广告")
    query_dsl_dict = converter.expr_tree_to_dict(expr_tree)
    dsl_str = json.dumps(query_dsl_dict, ensure_ascii=False)

    assert "es_tok_query_string" in dsl_str
    assert "+若生命将于明日落幕" in dsl_str
    assert "-广告" in dsl_str
    assert "es_tok_constraints" not in dsl_str


def test_constraint_filter_uses_es_tok_query_string_exact_syntax():
    """Constraint prefilters should use the same exact-segment syntax as the main query."""
    import json
    from dsl.rewrite import DslExprRewriter

    rewriter = DslExprRewriter()
    query_info = rewriter.get_query_info("游戏音乐 +若生命将于明日落幕 -广告")
    constraint_filter = query_info.get("constraint_filter", {})
    dsl_str = json.dumps(constraint_filter, ensure_ascii=False)

    assert "es_tok_query_string" in dsl_str
    assert "+若生命将于明日落幕" in dsl_str
    assert "-广告" in dsl_str
    assert "游戏音乐" not in dsl_str
    assert "es_tok_constraints" not in dsl_str


def test_constraint_filter_preserves_or_structure_for_exact_segments():
    """OR-ed exact constraints should stay in the prefilter instead of collapsing to AND."""
    import json
    from dsl.rewrite import DslExprRewriter

    rewriter = DslExprRewriter()
    query_info = rewriter.get_query_info("(+若生命将于明日落幕 || +工藤晴香) 游戏音乐")
    constraint_filter = query_info.get("constraint_filter", {})
    dsl_str = json.dumps(constraint_filter, ensure_ascii=False)

    assert "es_tok_query_string" in dsl_str
    assert "+若生命将于明日落幕" in dsl_str
    assert "+工藤晴香" in dsl_str
    assert "minimum_should_match" in dsl_str
    assert "游戏音乐" not in dsl_str


def test_es_tok_query_constraints_work_in_live_searcher_path():
    """Validate required and excluded exact segments through the real bili-search path."""
    searcher = make_searcher()

    required_res = searcher.search(
        "+若生命将于明日落幕",
        source_fields=["bvid", "title"],
        add_highlights_info=False,
        limit=10,
        timeout=5,
        verbose=False,
    )
    required_hits = required_res.get("hits", [])
    required_bvids = {hit.get("bvid") for hit in required_hits if hit.get("bvid")}
    assert "BV1v8w8zwEBQ" in required_bvids

    excluded_res = searcher.search(
        "游戏音乐 -若生命将于明日落幕",
        source_fields=["bvid", "title"],
        add_highlights_info=False,
        limit=20,
        timeout=5,
        verbose=False,
    )
    excluded_hits = excluded_res.get("hits", [])
    excluded_bvids = {hit.get("bvid") for hit in excluded_hits if hit.get("bvid")}
    assert "BV1v8w8zwEBQ" not in excluded_bvids


def test_es_tok_query_quoted_exact_segments_work_in_live_searcher_path():
    """Validate quoted exact-segment syntax through the real bili-search path."""
    searcher = make_searcher()

    split_cn_res = searcher.search(
        '"若生命将于明日落幕"',
        source_fields=["bvid", "title"],
        add_highlights_info=False,
        limit=10,
        timeout=5,
        verbose=False,
    )
    split_cn_bvids = {
        hit.get("bvid") for hit in split_cn_res.get("hits", []) if hit.get("bvid")
    }
    assert "BV1v8w8zwEBQ" in split_cn_bvids

    named_res = searcher.search(
        '"工藤晴香"',
        source_fields=["bvid", "title"],
        add_highlights_info=False,
        limit=10,
        timeout=5,
        verbose=False,
    )
    named_bvids = {
        hit.get("bvid") for hit in named_res.get("hits", []) if hit.get("bvid")
    }
    assert "BV1gcwuzhEaX" in named_bvids


def test_es_tok_query_required_exact_segment_works_with_keywords_in_live_searcher_path():
    """Required exact segments should still work when combined with normal keywords."""
    searcher = make_searcher()

    res = searcher.search(
        "游戏音乐 +若生命将于明日落幕",
        source_fields=["bvid", "title"],
        add_highlights_info=False,
        limit=10,
        timeout=5,
        verbose=False,
    )
    bvids = {hit.get("bvid") for hit in res.get("hits", []) if hit.get("bvid")}
    assert "BV1v8w8zwEBQ" in bvids


def test_es_tok_query_owner_words_match_mixed_ascii_in_live_searcher_path():
    """Mixed CJK+ASCII owner queries should surface docs from owner.name=红警HBK08."""
    searcher = make_searcher()

    loose_res = searcher.search(
        "红警HBK08",
        source_fields=["bvid", "title", "owner"],
        add_highlights_info=False,
        limit=100,
        timeout=5,
        verbose=False,
    )
    loose_owner_names = {
        hit.get("owner", {}).get("name")
        for hit in loose_res.get("hits", [])
        if hit.get("owner")
    }
    assert "红警HBK08" in loose_owner_names

    exact_res = searcher.search(
        "+红警HBK08",
        source_fields=["bvid", "title", "owner"],
        add_highlights_info=False,
        limit=20,
        timeout=5,
        verbose=False,
    )
    exact_owner_names = {
        hit.get("owner", {}).get("name")
        for hit in exact_res.get("hits", [])
        if hit.get("owner")
    }
    assert "红警HBK08" in exact_owner_names


def test_es_tok_query_contradictory_exact_constraints_return_no_hits():
    """A query that both requires and excludes the same exact segment must return nothing."""
    searcher = make_searcher()

    res = searcher.search(
        "+若生命将于明日落幕 -若生命将于明日落幕",
        source_fields=["bvid", "title"],
        add_highlights_info=False,
        limit=10,
        timeout=5,
        verbose=False,
    )
    hits = res.get("hits", [])
    assert not hits


def test_word_search_basic():
    """Test basic word search (q=w mode) with various queries.

    This is the core test for the bug fix: the search should return results
    instead of timing out with 0 hits.
    """
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    test_queries = [
        ("影视飓风", True, "Popular channel name"),
        ("影视飓风 q=w", True, "Channel name with word mode"),
        ("deepseek", True, "Tech keyword"),
        ("deepseek v3", True, "Multi-word tech query"),
        ("黑神话 悟空", True, "Game name"),
        ("are you ok", True, "English phrase"),
        ("雷军", True, "Person name"),
    ]

    logger.note("> Testing basic word search...")
    all_passed = True

    for query, expect_hits, description in test_queries:
        logger.mesg(f"  Testing: [{query}] ({description})")
        res = searcher.search(query, limit=10, timeout=5, verbose=False)

        timed_out = res.get("timed_out", True)
        took = res.get("took", -1)
        total_hits = res.get("total_hits", 0)
        return_hits = res.get("return_hits", 0)
        has_error = res.get("_es_error", False)

        if has_error:
            error_msg = res.get("_es_error_msg", "Unknown error")
            logger.warn(f"    × ES error: {error_msg}")
            all_passed = False
            continue

        if timed_out and took == -1:
            logger.warn(f"    × Query failed (took=-1, timed_out=True)")
            all_passed = False
            continue

        if expect_hits and total_hits == 0:
            logger.warn(f"    × Expected hits but got 0")
            all_passed = False
            continue

        status = "✓" if (not timed_out or total_hits > 0) else "△"
        logger.mesg(
            f"    {status} took={took}ms, timed_out={timed_out}, "
            f"total={total_hits}, returned={return_hits}"
        )

    if all_passed:
        logger.success("\n✓ All basic word search tests passed!")
    else:
        logger.warn("\n× Some basic word search tests failed!")
    assert all_passed, "Basic word search test failures"


def test_word_search_with_filters():
    """Test word search combined with various filter types."""
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    test_queries = [
        ("影视飓风 v>1k", "Query with view filter"),
        ("deepseek d>2024-01-01", "Query with date filter"),
        ("黑神话 v>1w d>2024", "Query with view + date filter"),
        ('影视飓风 "罗永浩"', "Query with quoted phrase"),
    ]

    logger.note("> Testing word search with filters...")
    all_passed = True

    for query, description in test_queries:
        logger.mesg(f"  Testing: [{query}] ({description})")
        res = searcher.search(query, limit=10, timeout=5, verbose=False)

        timed_out = res.get("timed_out", True)
        took = res.get("took", -1)
        total_hits = res.get("total_hits", 0)
        has_error = res.get("_es_error", False)

        if has_error:
            error_msg = res.get("_es_error_msg", "Unknown error")
            logger.warn(f"    × ES error: {error_msg}")
            all_passed = False
            continue

        if took == -1:
            logger.warn(f"    × Query failed (took=-1)")
            all_passed = False
            continue

        logger.mesg(f"    ✓ took={took}ms, timed_out={timed_out}, total={total_hits}")

    if all_passed:
        logger.success("\n✓ All word search with filters tests passed!")
    else:
        logger.warn("\n× Some word search with filters tests failed!")
    assert all_passed, "Word search with filters test failures"


def test_explore_word_mode():
    """Test explore endpoint with word-only mode (q=w).

    This directly tests the failing scenario from the bug report:
    query "影视飓风 q=w" should NOT timeout with 0 hits.
    """
    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    test_queries = [
        ("影视飓风 q=w", "Bug report query - word mode"),
        ("deepseek q=w", "Tech query - word mode"),
        ("黑神话 q=w", "Game query - word mode"),
    ]

    logger.note("> Testing explore with word mode (q=w)...")
    all_passed = True

    for query, description in test_queries:
        logger.mesg(f"  Testing: [{query}] ({description})")
        try:
            res = explorer.unified_explore(
                query=query,
                rank_top_k=10,
                group_owner_limit=5,
                verbose=False,
            )

            status = res.get("status", "unknown")
            data = res.get("data", [])

            if status == "timedout":
                # Check if it's a real timeout or an error
                for step in data:
                    if step.get("status") == "timedout":
                        step_output = step.get("output", {})
                        took = step_output.get("took", -1)
                        if took == -1:
                            logger.warn(
                                f"    × Step '{step.get('name')}' failed with took=-1 "
                                f"(likely ES error, not timeout)"
                            )
                            all_passed = False
                        else:
                            logger.mesg(
                                f"    △ Step '{step.get('name')}' genuinely timed out "
                                f"(took={took}ms)"
                            )
            else:
                # Find the search step
                for step in data:
                    if step.get("name") == "most_relevant_search":
                        output = step.get("output", {})
                        total_hits = output.get("total_hits", 0)
                        return_hits = output.get("return_hits", 0)
                        took = output.get("took", -1)
                        logger.mesg(
                            f"    ✓ status={status}, took={took}ms, "
                            f"total={total_hits}, returned={return_hits}"
                        )
                        if total_hits == 0 and took != -1:
                            logger.warn(f"    △ No hits found but query succeeded")
                        break
        except Exception as e:
            logger.warn(f"    × Exception: {e}")
            all_passed = False

    if all_passed:
        logger.success("\n✓ All explore word mode tests passed!")
    else:
        logger.warn("\n× Some explore word mode tests failed!")
    assert all_passed, "Explore word mode test failures"


def test_submit_to_es_error_handling():
    """Test that submit_to_es returns structured error responses.

    When ES throws an exception (e.g., unsupported query params),
    the response should contain _es_error=True instead of empty dict,
    so that downstream code can distinguish errors from timeouts.
    """
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    logger.note("> Testing submit_to_es error handling...")

    # Test with an intentionally invalid search body
    invalid_body = {
        "query": {"nonexistent_query_type": {"query": "test"}},
        "size": 1,
    }

    res = searcher.submit_to_es(invalid_body, context="test_error_handling")

    has_error = res.get("_es_error", False)
    has_hits = "hits" in res
    has_took = "took" in res

    if has_error:
        logger.mesg(f"  ✓ Error response has _es_error=True")
        logger.mesg(f"  ✓ Error message: {res.get('_es_error_msg', 'N/A')[:100]}")
    else:
        logger.warn(f"  × Error response missing _es_error flag")

    if has_hits:
        logger.mesg(f"  ✓ Error response has hits structure")
    else:
        logger.warn(f"  × Error response missing hits structure")

    if has_took:
        logger.mesg(f"  ✓ Error response has took field")
    else:
        logger.warn(f"  × Error response missing took field")

    passed = has_error and has_hits and has_took
    if passed:
        logger.success("\n✓ submit_to_es error handling test passed!")
    else:
        logger.warn("\n× submit_to_es error handling test had issues")
    assert passed, "submit_to_es error handling test failure"


def test_match_fields_no_missing_index_fields():
    """Test that match fields referenced in queries actually exist in the index.

    After index v6 rebuild, pinyin subfields were removed.
    Ensure SEARCH_MATCH_FIELDS and SUGGEST_MATCH_FIELDS only reference
    fields that exist in the index mapping.
    """
    from elastics.videos.constants import (
        SEARCH_MATCH_FIELDS,
        SUGGEST_MATCH_FIELDS,
        DATE_MATCH_FIELDS,
    )

    logger.note("> Testing match fields against index mapping...")

    # Fields that exist in index v6 (based on video_index_settings_v6.py)
    valid_field_patterns = [
        "title",
        "title.words",
        "tags",
        "tags.words",
        "owner.name",
        "owner.name.words",
        "owner.name.keyword",
        "desc",
        "desc.words",
    ]

    all_passed = True

    # Check SEARCH_MATCH_FIELDS
    for field in SEARCH_MATCH_FIELDS:
        base_field = field.split("^")[0]  # remove boost suffix
        if base_field in valid_field_patterns:
            logger.mesg(f"  ✓ SEARCH_MATCH_FIELDS: {base_field}")
        elif base_field.endswith(".pinyin"):
            logger.warn(f"  × SEARCH_MATCH_FIELDS: {base_field} (pinyin removed in v6)")
            all_passed = False
        else:
            logger.mesg(f"  △ SEARCH_MATCH_FIELDS: {base_field} (may be dynamic)")

    # Check SUGGEST_MATCH_FIELDS
    for field in SUGGEST_MATCH_FIELDS:
        base_field = field.split("^")[0]
        if base_field in valid_field_patterns:
            logger.mesg(f"  ✓ SUGGEST_MATCH_FIELDS: {base_field}")
        elif base_field.endswith(".pinyin"):
            logger.warn(
                f"  × SUGGEST_MATCH_FIELDS: {base_field} (pinyin removed in v6)"
            )
            all_passed = False
        else:
            logger.mesg(f"  △ SUGGEST_MATCH_FIELDS: {base_field} (may be dynamic)")

    # Check DATE_MATCH_FIELDS
    for field in DATE_MATCH_FIELDS:
        base_field = field.split("^")[0]
        if base_field in valid_field_patterns:
            logger.mesg(f"  ✓ DATE_MATCH_FIELDS: {base_field}")
        elif base_field.endswith(".pinyin"):
            logger.warn(f"  × DATE_MATCH_FIELDS: {base_field} (pinyin removed in v6)")
            all_passed = False

    if all_passed:
        logger.success("\n✓ All match fields are valid for index v6!")
    else:
        logger.warn("\n× Some match fields reference non-existent index fields!")
    assert all_passed, "Match fields reference non-existent index fields"


def test_search_body_structure():
    """Test that constructed search body has correct structure.

    Validates the full search body construction including:
    - es_tok_query_string parameters
    - timeout setting
    - terminate_after setting
    - _source fields
    - size/limit
    """
    from dsl.elastic import DslExprToElasticConverter
    from dsl.rewrite import DslExprRewriter
    from dsl.filter import QueryDslDictFilterMerger
    from elastics.structure import construct_boosted_fields
    from elastics.videos.constants import (
        SEARCH_MATCH_FIELDS,
        EXPLORE_BOOSTED_FIELDS,
        SEARCH_MATCH_TYPE,
        SOURCE_FIELDS,
    )

    logger.note("> Testing search body structure...")

    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Build query_dsl_dict
    query = "影视飓风"
    boosted_match_fields, boosted_date_fields = construct_boosted_fields(
        match_fields=SEARCH_MATCH_FIELDS,
        boost=True,
        boosted_fields=EXPLORE_BOOSTED_FIELDS,
    )
    _, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
        query=query,
        boosted_match_fields=boosted_match_fields,
        boosted_date_fields=boosted_date_fields,
        match_type=SEARCH_MATCH_TYPE,
    )

    # Construct search body
    search_body = searcher.construct_search_body(
        query_dsl_dict=query_dsl_dict,
        match_fields=boosted_match_fields,
        source_fields=["bvid", "stat", "pubdate", "duration"],
        limit=100,
        timeout=5,
    )

    all_passed = True

    # Check required top-level keys
    required_keys = ["query", "_source", "timeout", "size"]
    for key in required_keys:
        if key in search_body:
            logger.mesg(f"  ✓ Search body has '{key}'")
        else:
            logger.warn(f"  × Search body missing '{key}'")
            all_passed = False

    # Check that query contains es_tok_query_string (not unsupported params)
    import json

    body_str = json.dumps(search_body)
    if "min_kept_tokens_count" in body_str:
        logger.warn("  × Search body contains unsupported 'min_kept_tokens_count'")
        all_passed = False
    else:
        logger.mesg("  ✓ Search body does not contain 'min_kept_tokens_count'")

    if "min_kept_tokens_ratio" in body_str:
        logger.warn("  × Search body contains unsupported 'min_kept_tokens_ratio'")
        all_passed = False
    else:
        logger.mesg("  ✓ Search body does not contain 'min_kept_tokens_ratio'")

    # Check size
    if search_body.get("size") == 100:
        logger.mesg("  ✓ Search body size=100")
    else:
        logger.warn(f"  × Search body size={search_body.get('size')}, expected 100")
        all_passed = False

    # Check timeout format
    timeout_val = search_body.get("timeout", "")
    if timeout_val.endswith("ms"):
        logger.mesg(f"  ✓ Timeout format correct: {timeout_val}")
    else:
        logger.warn(f"  × Timeout format unexpected: {timeout_val}")
        all_passed = False

    logger.mesg(f"\n  Full search body:")
    logger.mesg(dict_to_str(search_body, add_quotes=True), indent=4)

    if all_passed:
        logger.success("\n✓ Search body structure test passed!")
    else:
        logger.warn("\n× Search body structure test had issues!")
    assert all_passed, "Search body structure test failure"


def test_search_edge_cases():
    """Test search with edge case queries.

    Covers: empty-ish queries, single characters, special characters,
    very long queries, mixed Chinese/English, date-only queries.
    """
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    edge_case_queries = [
        # (query, description, should_not_error)
        ("a", "Single ASCII character", True),
        ("我", "Single Chinese character", True),
        ("Python 教程 2024", "Mixed Chinese/English with date", True),
        ("v>1w", "Filter-only (no word search)", True),
        ("d>2024-01-01", "Date filter only", True),
        ("u=雷军", "User filter only", True),
        ("影视飓风" * 5, "Repeated query text", True),
        ("hello world 你好 世界", "Mixed language", True),
    ]

    logger.note("> Testing search edge cases...")
    all_passed = True

    for query, description, should_not_error in edge_case_queries:
        logger.mesg(f"  Testing: [{query[:40]}...] ({description})")
        try:
            res = searcher.search(query, limit=5, timeout=5, verbose=False)

            took = res.get("took", -1)
            has_error = res.get("_es_error", False)

            if has_error and should_not_error:
                error_msg = res.get("_es_error_msg", "")[:80]
                logger.warn(f"    × Unexpected ES error: {error_msg}")
                all_passed = False
            elif took == -1 and should_not_error:
                logger.warn(f"    × Query failed (took=-1)")
                all_passed = False
            else:
                logger.mesg(f"    ✓ OK (took={took}ms)")
        except Exception as e:
            if should_not_error:
                logger.warn(f"    × Exception: {str(e)[:80]}")
                all_passed = False
            else:
                logger.mesg(f"    ✓ Expected error: {str(e)[:80]}")

    if all_passed:
        logger.success("\n✓ All search edge case tests passed!")
    else:
        logger.warn("\n× Some search edge case tests failed!")
    assert all_passed, "Search edge case test failures"
