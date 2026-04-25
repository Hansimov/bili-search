"""Constraint routing tests and manual runner for DSL coverage."""

import json
import sys

from tclogger import logger, logstr


def run_test_suite(name, test_fn):
    """Wrapper to run a test suite with result logging."""
    logger.note("=" * 60)
    result = test_fn()
    print()
    return (name, result)


def test_constraint_only_routing():
    """Test Bug 1: constraint-only queries must route to filter_only path.

    For queries like "+seedance +2.0", "+红警 +08", "+达芬奇":
    - keywords_body must be EMPTY (no scoring keywords)
    - constraint_texts must contain the +token texts
    - constraint_filter must be present and keep exact-segment syntax only
    - The standalone DSL dict must preserve +/- inside es_tok_query_string.

    This is the root cause of Bug 1: when constraint texts were in keywords_body,
    has_search_keywords() returned True, sending constraint-only queries through
    the BM25 recall + diversified ranking pipeline where all docs get identical
    _score, producing uniform rank_score ≈ 0.7.
    """
    logger.note("TEST: Constraint-Only Routing (Bug 1)")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter
    from dsl.rewrite import DslExprRewriter

    rewriter = DslExprRewriter()
    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    # Test case from the original bug report
    constraint_only_cases = [
        # (query, expected_constraint_texts, expected_keywords_body, desc)
        ("+seedance +2.0", ["seedance", "2.0"], [], "Bug 1 original case"),
        ("+红警 +08", ["红警", "08"], [], "CJK + numeric constraints"),
        ("+达芬奇", ["达芬奇"], [], "Single constraint"),
        (
            "+影视飓风 +评测",
            ["影视飓风", "评测"],
            [],
            "Two CJK constraints",
        ),
    ]

    for query, exp_ct, exp_kb, desc in constraint_only_cases:
        info = rewriter.get_query_info(query)
        kb = info.get("keywords_body", [])
        ct = info.get("constraint_texts", [])
        cf = info.get("constraint_filter", {})

        # 1. keywords_body must be empty (no scoring keywords)
        if kb == exp_kb:
            logger.mesg(f"{ok} {desc}: keywords_body={kb} (empty, correct)")
            passed += 1
        else:
            logger.fail(f"{fail} {desc}: keywords_body={kb}, expected {exp_kb}")
            failed += 1

        # 2. constraint_texts must contain +token texts
        if ct == exp_ct:
            logger.mesg(f"{ok} {desc}: constraint_texts={ct}")
            passed += 1
        else:
            logger.fail(f"{fail} {desc}: constraint_texts={ct}, expected {exp_ct}")
            failed += 1

        cf_str = json.dumps(cf, ensure_ascii=False)
        if (
            cf
            and "es_tok_query_string" in cf_str
            and all(token in cf_str for token in exp_ct)
        ):
            logger.mesg(f"{ok} {desc}: has exact-segment constraint_filter")
            passed += 1
        else:
            logger.fail(f"{fail} {desc}: bad constraint_filter={cf_str}")
            failed += 1

        dsl_str = json.dumps(converter.expr_to_dict(query), ensure_ascii=False)
        if (
            "es_tok_query_string" in dsl_str
            and all(token in dsl_str for token in exp_ct)
            and "es_tok_constraints" not in dsl_str
        ):
            logger.mesg(f"{ok} {desc}: standalone DSL keeps exact-segment syntax")
            passed += 1
        else:
            logger.fail(f"{fail} {desc}: unexpected standalone DSL={dsl_str}")
            failed += 1

    logger.note(f"Constraint-only routing: {passed} passed, {failed} failed")
    assert failed == 0


def test_constraint_plus_keyword_routing():
    """Test Bug 2: constraint+keyword queries must have correct routing and tagging.

    For queries like "+seedance +2.0 科幻":
    - keywords_body must contain only scoring keywords ["科幻"]
    - constraint_texts must contain +token texts ["seedance", "2.0"]
    - constraint_filter must keep only constraint terms in exact-segment syntax
    - standalone DSL must keep both scoring words and +/- terms inline
    - Title match tagging must handle +/- prefixes correctly
    """
    logger.note("TEST: Constraint + Keyword Routing (Bug 2)")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter
    from dsl.rewrite import DslExprRewriter

    rewriter = DslExprRewriter()
    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    mixed_cases = [
        # (query, expected_kb, expected_ct, desc)
        ("+seedance +2.0 科幻", ["科幻"], ["seedance", "2.0"], "Bug 2 original case"),
        ("+红警 +08 一块地", ["一块地"], ["红警", "08"], "CJK constraint + keyword"),
        (
            "+达芬奇 -广告 文艺复兴",
            ["文艺复兴"],
            ["达芬奇"],
            "Plus + minus + keyword",
        ),
        (
            "好看的 +4K +HDR",
            ["好看的"],
            ["4K", "HDR"],
            "Keyword first, then constraints",
        ),
    ]

    for query, exp_kb, exp_ct, desc in mixed_cases:
        info = rewriter.get_query_info(query)
        kb = info.get("keywords_body", [])
        ct = info.get("constraint_texts", [])
        cf = info.get("constraint_filter", {})

        # 1. keywords_body must contain only scoring keywords
        if kb == exp_kb:
            logger.mesg(f"{ok} {desc}: keywords_body={kb}")
            passed += 1
        else:
            logger.fail(f"{fail} {desc}: keywords_body={kb}, expected {exp_kb}")
            failed += 1

        # 2. constraint_texts must contain +token texts
        if ct == exp_ct:
            logger.mesg(f"{ok} {desc}: constraint_texts={ct}")
            passed += 1
        else:
            logger.fail(f"{fail} {desc}: constraint_texts={ct}, expected {exp_ct}")
            failed += 1

        cf_str = json.dumps(cf, ensure_ascii=False)
        if (
            cf
            and "es_tok_query_string" in cf_str
            and all(token in cf_str for token in exp_ct)
            and all(token not in cf_str for token in exp_kb)
        ):
            logger.mesg(f"{ok} {desc}: constraint_filter keeps only constraint terms")
            passed += 1
        else:
            logger.fail(f"{fail} {desc}: bad constraint_filter={cf_str}")
            failed += 1

        dsl_str = json.dumps(converter.expr_to_dict(query), ensure_ascii=False)
        if (
            "es_tok_query_string" in dsl_str
            and all(token in dsl_str for token in [*exp_kb, *exp_ct])
            and "es_tok_constraints" not in dsl_str
        ):
            logger.mesg(
                f"{ok} {desc}: standalone DSL keeps scoring words + exact tokens"
            )
            passed += 1
        else:
            logger.fail(f"{fail} {desc}: unexpected standalone DSL={dsl_str}")
            failed += 1

    # --- Test title match tagging strips constraint prefixes ---
    from recalls.word import MultiLaneWordRecall

    tag_cases = [
        # (query, title, tags, expected_matched)
        (
            "+seedance +2.0 科幻",
            "Seedance 2.0 科幻风格演示",
            "AI 视频生成",
            True,
        ),
        (
            "+红警 +08 解说",
            "红警08解说视频合集",
            "红警 游戏",
            True,
        ),
        (
            "+达芬奇 文艺复兴",
            "达芬奇与文艺复兴",
            "",
            True,
        ),
        (
            "+seedance +2.0 科幻",
            "完全无关的标题",
            "无关标签",
            False,
        ),
    ]

    for query, title, tags, expected in tag_cases:
        hits = [{"title": title, "tags": tags}]
        MultiLaneWordRecall._tag_title_matches(hits, query)
        matched = hits[0].get("_title_matched", False)
        if matched == expected:
            logger.mesg(f"{ok} title_match('{query}', '{title}'): {matched}")
            passed += 1
        else:
            logger.fail(
                f"{fail} title_match('{query}', '{title}'): "
                f"got {matched}, expected {expected}"
            )
            failed += 1

    logger.note(f"Constraint + keyword routing: {passed} passed, {failed} failed")
    assert failed == 0


def test_constraint_filter_in_filter_only():
    """Test that constraint filters are included in filter-only search path.

    When get_filters_from_query() is called for a constraint-only query,
    the returned filter_clauses must include the exact-segment constraint_filter.
    Without this, constraint-only queries would return ALL docs (no filtering).
    """
    logger.note("TEST: Constraint Filter in Filter-Only Path")
    logger.note("=" * 60)

    from dsl.rewrite import DslExprRewriter

    rewriter = DslExprRewriter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    # Verify constraint_filter is present in query_info for constraint queries
    cases = [
        ("+seedance +2.0", ["seedance", "2.0"], [], "Two plus constraints"),
        ("+红警 +08", ["红警", "08"], [], "CJK + numeric"),
        ("+达芬奇", ["达芬奇"], [], "Single constraint"),
        ("+达芬奇 -广告", ["达芬奇", "广告"], [], "Plus + minus"),
        ("+seedance +2.0 科幻", ["seedance", "2.0"], ["科幻"], "Constraints + keyword"),
    ]

    for query, expected_tokens, forbidden_tokens, desc in cases:
        info = rewriter.get_query_info(query)
        cf = info.get("constraint_filter", {})
        cf_str = json.dumps(cf, ensure_ascii=False)

        if (
            cf
            and "es_tok_query_string" in cf_str
            and all(token in cf_str for token in expected_tokens)
            and all(token not in cf_str for token in forbidden_tokens)
        ):
            logger.mesg(
                f"{ok} {desc}: constraint_filter keeps exact-segment terms only"
            )
            passed += 1
        else:
            logger.fail(f"{fail} {desc}: unexpected constraint_filter={cf_str}")
            failed += 1

    # Verify constraint-only queries have empty keywords_body
    constraint_only = ["+seedance +2.0", "+红警 +08", "+达芬奇"]
    for query in constraint_only:
        info = rewriter.get_query_info(query)
        kb = info.get("keywords_body", [])
        if not kb:
            logger.mesg(f"{ok} '{query}': empty keywords_body (filter-only)")
            passed += 1
        else:
            logger.fail(
                f"{fail} '{query}': non-empty keywords_body={kb} "
                f"(should be empty for filter-only routing)"
            )
            failed += 1

    # Verify mixed queries have non-empty keywords_body
    mixed = [
        ("+seedance +2.0 科幻", ["科幻"]),
        ("+红警 +08 一块地", ["一块地"]),
    ]
    for query, expected_kb in mixed:
        info = rewriter.get_query_info(query)
        kb = info.get("keywords_body", [])
        if kb == expected_kb:
            logger.mesg(f"{ok} '{query}': keywords_body={kb} (has scoring keywords)")
            passed += 1
        else:
            logger.fail(f"{fail} '{query}': keywords_body={kb}, expected {expected_kb}")
            failed += 1

    logger.note(f"Constraint filter in filter-only: {passed} passed, {failed} failed")
    assert failed == 0


def run_all_tests():
    """Run all test suites and report results."""
    from test_dsl import (
        test_atom_parsing,
        test_bool_parsing,
        test_constraint_boolean,
        test_constraint_fields,
        test_constraint_mixed,
        test_constraint_regression,
        test_constraint_simple,
        test_constraint_tree_converter,
        test_constraint_unit,
        test_edge_cases,
        test_elastic_conversion,
        test_filter_merger,
        test_knn_filter_extraction,
        test_qmod,
        test_user_exclusion,
    )

    logger.note("=" * 60)
    logger.note(" DSL Module — Comprehensive Test Suite")
    logger.note("=" * 60)
    print()

    results = []
    results.append(run_test_suite("Atom Parsing", test_atom_parsing))
    results.append(run_test_suite("Bool Parsing", test_bool_parsing))
    results.append(run_test_suite("Elastic Conversion", test_elastic_conversion))
    results.append(run_test_suite("User Exclusion", test_user_exclusion))
    results.append(run_test_suite("KNN Filter Extraction", test_knn_filter_extraction))
    results.append(run_test_suite("Filter Merger", test_filter_merger))
    results.append(run_test_suite("Constraint Unit", test_constraint_unit))
    results.append(run_test_suite("Constraint Simple", test_constraint_simple))
    results.append(run_test_suite("Constraint Mixed", test_constraint_mixed))
    results.append(run_test_suite("Constraint Boolean", test_constraint_boolean))
    results.append(run_test_suite("Constraint Fields", test_constraint_fields))
    results.append(
        run_test_suite("Constraint Tree Converter", test_constraint_tree_converter)
    )
    results.append(run_test_suite("Constraint Regression", test_constraint_regression))
    results.append(run_test_suite("Query Mode", test_qmod))
    results.append(run_test_suite("Edge Cases", test_edge_cases))
    results.append(
        run_test_suite("Constraint-Only Routing", test_constraint_only_routing)
    )
    results.append(
        run_test_suite(
            "Constraint+Keyword Routing", test_constraint_plus_keyword_routing
        )
    )
    results.append(
        run_test_suite(
            "Constraint Filter-Only Path", test_constraint_filter_in_filter_only
        )
    )

    logger.note("=" * 60)
    logger.note("SUMMARY")
    logger.note("=" * 60)

    total_pass = 0
    total_fail = 0
    for name, test_passed in results:
        status = logstr.okay("PASS") if test_passed else logstr.fail("FAIL")
        logger.mesg(f"  {status} {name}")
        if test_passed:
            total_pass += 1
        else:
            total_fail += 1

    print()
    logger.note(f"Total: {total_pass} suites passed, {total_fail} suites failed")
    if total_fail == 0:
        logger.success("All test suites passed!")
    else:
        logger.fail(f"{total_fail} test suite(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()

    # python tests/dsl/test_dsl_constraint_routing.py
