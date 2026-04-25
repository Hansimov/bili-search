"""Constraint and qmod coverage for the DSL module."""

import json

from tclogger import logger, logstr


def test_constraint_unit():
    """Unit tests for constraint.py functions."""
    logger.note("TEST: Constraint Unit Functions")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter
    from dsl.fields.constraint import (
        is_constraint_word_expr,
        get_constraint_text,
        get_constraint_pp_key,
        word_expr_to_constraint,
        ConstraintTreeConverter,
        constraints_to_filter,
    )

    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    # --- is_constraint_word_expr ---
    plus_tree = converter.construct_expr_tree("+影视飓风")
    plus_word_nodes = plus_tree.find_all_childs_with_key("word_expr")
    if plus_word_nodes and is_constraint_word_expr(plus_word_nodes[0]):
        logger.mesg(f"{ok} is_constraint_word_expr(+影视飓风) = True")
        passed += 1
    else:
        logger.fail(f"{fail} is_constraint_word_expr(+影视飓风) should be True")
        failed += 1

    plain_tree = converter.construct_expr_tree("影视飓风")
    plain_word_nodes = plain_tree.find_all_childs_with_key("word_expr")
    if plain_word_nodes and not is_constraint_word_expr(plain_word_nodes[0]):
        logger.mesg(f"{ok} is_constraint_word_expr(影视飓风) = False")
        passed += 1
    else:
        logger.fail(f"{fail} is_constraint_word_expr(影视飓风) should be False")
        failed += 1

    # --- get_constraint_pp_key ---
    pp = get_constraint_pp_key(plus_word_nodes[0])
    if pp == "pl":
        logger.mesg(f'{ok} get_constraint_pp_key(+影视飓风) = "pl"')
        passed += 1
    else:
        logger.fail(
            f'{fail} get_constraint_pp_key(+影视飓风) expected "pl", got "{pp}"'
        )
        failed += 1

    minus_tree = converter.construct_expr_tree("-广告")
    minus_word_nodes = minus_tree.find_all_childs_with_key("word_expr")
    pp_mi = get_constraint_pp_key(minus_word_nodes[0])
    if pp_mi == "mi":
        logger.mesg(f'{ok} get_constraint_pp_key(-广告) = "mi"')
        passed += 1
    else:
        logger.fail(f'{fail} get_constraint_pp_key(-广告) expected "mi", got "{pp_mi}"')
        failed += 1

    bang_tree = converter.construct_expr_tree("!广告")
    bang_word_nodes = bang_tree.find_all_childs_with_key("word_expr")
    pp_nq = get_constraint_pp_key(bang_word_nodes[0])
    if pp_nq == "nq":
        logger.mesg(f'{ok} get_constraint_pp_key(!广告) = "nq"')
        passed += 1
    else:
        logger.fail(f'{fail} get_constraint_pp_key(!广告) expected "nq", got "{pp_nq}"')
        failed += 1

    # --- get_constraint_text ---
    text = get_constraint_text(plus_word_nodes[0])
    if text == "影视飓风":
        logger.mesg(f'{ok} get_constraint_text(+影视飓风) = "影视飓风"')
        passed += 1
    else:
        logger.fail(
            f'{fail} get_constraint_text(+影视飓风) expected "影视飓风", got "{text}"'
        )
        failed += 1

    quoted_tree = converter.construct_expr_tree('+"影视飓风"')
    quoted_word_nodes = quoted_tree.find_all_childs_with_key("word_expr")
    quoted_text = get_constraint_text(quoted_word_nodes[0])
    if quoted_text == "影视飓风":
        logger.mesg(f'{ok} get_constraint_text(+"影视飓风") strips quotes')
        passed += 1
    else:
        logger.fail(
            f'{fail} get_constraint_text(+"影视飓风") expected "影视飓风", got "{quoted_text}"'
        )
        failed += 1

    # --- word_expr_to_constraint ---
    c_plus = word_expr_to_constraint(plus_word_nodes[0])
    if c_plus == {"have_token": ["影视飓风"]}:
        logger.mesg(f"{ok} word_expr_to_constraint(+影视飓风) = have_token")
        passed += 1
    else:
        logger.fail(f"{fail} word_expr_to_constraint(+影视飓风): {c_plus}")
        failed += 1

    c_minus = word_expr_to_constraint(minus_word_nodes[0])
    if c_minus == {"NOT": {"have_token": ["广告"]}}:
        logger.mesg(f"{ok} word_expr_to_constraint(-广告) = NOT have_token")
        passed += 1
    else:
        logger.fail(f"{fail} word_expr_to_constraint(-广告): {c_minus}")
        failed += 1

    c_bang = word_expr_to_constraint(bang_word_nodes[0])
    if c_bang == {"NOT": {"have_token": ["广告"]}}:
        logger.mesg(f"{ok} word_expr_to_constraint(!广告) = NOT have_token")
        passed += 1
    else:
        logger.fail(f"{fail} word_expr_to_constraint(!广告): {c_bang}")
        failed += 1

    # --- constraints_to_filter ---
    constraints = [{"have_token": ["A"]}, {"NOT": {"have_token": ["B"]}}]
    fields = ["title.words", "tags.words"]
    cf = constraints_to_filter(constraints, fields=fields)
    if (
        "es_tok_constraints" in cf
        and cf["es_tok_constraints"]["constraints"] == constraints
        and cf["es_tok_constraints"]["fields"] == fields
    ):
        logger.mesg(f"{ok} constraints_to_filter: correct structure")
        passed += 1
    else:
        logger.fail(f"{fail} constraints_to_filter: {cf}")
        failed += 1

    # --- constraints_to_filter empty ---
    empty = constraints_to_filter([])
    if empty == {}:
        logger.mesg(f"{ok} constraints_to_filter([]) = {{}}")
        passed += 1
    else:
        logger.fail(f"{fail} constraints_to_filter([]) = {empty}")
        failed += 1

    logger.note(f"Constraint unit: {passed} passed, {failed} failed")
    assert failed == 0


def test_constraint_simple():
    """Test simple +/- tokens stay inline in es_tok_query_string."""
    logger.note("TEST: Token Constraints (Simple)")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter

    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    def get_query_text(d: dict) -> str:
        return (
            d.get("bool", {})
            .get("must", {})
            .get("es_tok_query_string", {})
            .get("query", "")
        )

    test_cases = [
        (
            "+影视飓风",
            "+constraint stays inside es_tok_query_string",
            lambda query_text, dsl_str: query_text == "+影视飓风"
            and "es_tok_constraints" not in dsl_str,
        ),
        (
            "-广告",
            "-constraint stays inside es_tok_query_string",
            lambda query_text, dsl_str: query_text == "-广告"
            and "es_tok_constraints" not in dsl_str,
        ),
        (
            "!广告",
            "! preserves exclusion syntax inline",
            lambda query_text, dsl_str: query_text == "!广告"
            and "es_tok_constraints" not in dsl_str,
        ),
        (
            "+影视飓风 -小米",
            "+/- mix stays in one exact-segment query string",
            lambda query_text, dsl_str: query_text == "+影视飓风 -小米"
            and "es_tok_constraints" not in dsl_str,
        ),
        (
            '+"影视飓风"',
            "quoted +constraint keeps exact quoted syntax",
            lambda query_text, dsl_str: query_text == '+"影视飓风"'
            and "es_tok_constraints" not in dsl_str,
        ),
        (
            '+"影视飓风" !"李四维"',
            "two exact-segment constraints stay inline",
            lambda query_text, dsl_str: '+"影视飓风"' in query_text.replace(" ", "")
            and '!"李四维"' in query_text.replace(" ", "")
            and "es_tok_constraints" not in dsl_str,
        ),
    ]

    for query, desc, check_fn in test_cases:
        elastic_dict = converter.expr_to_dict(query)
        dsl_str = json.dumps(elastic_dict, ensure_ascii=False)
        query_text = get_query_text(elastic_dict)
        if check_fn(query_text, dsl_str):
            logger.mesg(f"{ok} {query}: {desc}")
            passed += 1
        else:
            logger.fail(f"{fail} {query}: {desc}")
            logger.fail(f"  query={query_text}")
            logger.fail(f"  dsl={dsl_str}")
            failed += 1

    logger.note(f"Simple constraints: {passed} passed, {failed} failed")
    assert failed == 0


def test_constraint_mixed():
    """Test +/- tokens mixed with regular words and filters."""
    logger.note("TEST: Token Constraints (Mixed)")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter

    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    def get_query_text(d: dict) -> str:
        return (
            d.get("bool", {})
            .get("must", {})
            .get("es_tok_query_string", {})
            .get("query", "")
        )

    test_cases = [
        (
            '世界 +"影视飓风" -小米',
            "word and exact-segment tokens are merged into es_tok_query_string",
            lambda query_text, dsl_str: query_text == '世界 +"影视飓风" -小米'
            and "es_tok_constraints" not in dsl_str,
        ),
        (
            "+影视飓风 -小米 v>10k",
            "constraints stay inline while stat filters stay in bool.filter",
            lambda query_text, dsl_str: query_text == "+影视飓风 -小米"
            and "range" in dsl_str
            and "es_tok_constraints" not in dsl_str,
        ),
        (
            "影视飓风 d=2024",
            "no +/- still uses normal query + date filter path",
            lambda query_text, dsl_str: query_text == "影视飓风"
            and "es_tok_constraints" not in dsl_str,
        ),
        (
            "+影视飓风 u=何同学",
            "+constraint + user filter keeps exact token inline and user filter separate",
            lambda query_text, dsl_str: query_text == "+影视飓风"
            and ("term" in dsl_str or "terms" in dsl_str)
            and "何同学" in dsl_str
            and "es_tok_constraints" not in dsl_str,
        ),
        (
            "世界 +影视飓风 u!=[何同学] v>10k",
            "word + constraint + filters keeps roles separated",
            lambda query_text, dsl_str: query_text == "世界 +影视飓风"
            and "何同学" in dsl_str
            and "range" in dsl_str
            and "es_tok_constraints" not in dsl_str,
        ),
    ]

    for query, desc, check_fn in test_cases:
        elastic_dict = converter.expr_to_dict(query)
        dsl_str = json.dumps(elastic_dict, ensure_ascii=False)
        query_text = get_query_text(elastic_dict)
        if check_fn(query_text, dsl_str):
            logger.mesg(f"{ok} {query}: {desc}")
            passed += 1
        else:
            logger.fail(f"{fail} {query}: {desc}")
            logger.fail(f"  query={query_text}")
            logger.fail(f"  dsl={dsl_str}")
            failed += 1

    logger.note(f"Mixed constraints: {passed} passed, {failed} failed")
    assert failed == 0


def test_constraint_boolean():
    """Test boolean exact-segment expressions keep their structure."""
    logger.note("TEST: Token Constraints (Boolean)")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter

    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    test_cases = [
        (
            "(+影视飓风 & +小米) | (+小米 & -苹果)",
            "OR of AND groups keeps exact-segment tokens inline",
            lambda s: "es_tok_query_string" in s
            and "影视飓风" in s
            and "小米" in s
            and "苹果" in s
            and "minimum_should_match" in s
            and "es_tok_constraints" not in s,
        ),
        (
            "+A & +B",
            "AND of two + tokens stays inline",
            lambda s: "es_tok_query_string" in s
            and "+A" in s
            and "+B" in s
            and "es_tok_constraints" not in s,
        ),
        (
            "+A | +B",
            "OR of + constraints keeps should structure",
            lambda s: "es_tok_query_string" in s
            and "+A" in s
            and "+B" in s
            and "minimum_should_match" in s
            and "es_tok_constraints" not in s,
        ),
        (
            '(+"雷军" || +"小米") (+"影视飓风" || +"tim")',
            "two OR groups of + preserve exact-segment syntax",
            lambda s: "es_tok_query_string" in s
            and "雷军" in s
            and "小米" in s
            and "影视飓风" in s
            and "tim" in s
            and "minimum_should_match" in s
            and "es_tok_constraints" not in s,
        ),
    ]

    for query, desc, check_fn in test_cases:
        dsl_str = json.dumps(converter.expr_to_dict(query), ensure_ascii=False)
        if check_fn(dsl_str):
            logger.mesg(f"{ok} {query}: {desc}")
            passed += 1
        else:
            logger.fail(f"{fail} {query}: {desc}")
            logger.fail(f"  got: {dsl_str}")
            failed += 1

    logger.note(f"Boolean constraints: {passed} passed, {failed} failed")
    assert failed == 0


def test_constraint_fields():
    """Test that exact-segment queries still target the normal search fields."""
    logger.note("TEST: Token Constraints (Fields)")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter
    from elastics.videos.constants import SEARCH_MATCH_FIELDS

    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    def get_query_dict(d: dict) -> dict:
        must_clause = d.get("bool", {}).get("must", {})
        if isinstance(must_clause, dict):
            return must_clause.get("es_tok_query_string", {})
        return {}

    d_plus = converter.expr_to_dict("+影视飓风")
    plus_query = get_query_dict(d_plus)
    plus_fields = plus_query.get("fields", [])
    if plus_query.get("query") == "+影视飓风" and set(plus_fields) == set(
        SEARCH_MATCH_FIELDS
    ):
        logger.mesg(f"{ok} +constraint uses normal es_tok_query_string fields")
        passed += 1
    else:
        logger.fail(f"{fail} +constraint query dict: {plus_query}")
        failed += 1

    if "es_tok_constraints" not in json.dumps(d_plus, ensure_ascii=False):
        logger.mesg(f"{ok} +constraint does not emit es_tok_constraints")
        passed += 1
    else:
        logger.fail(f"{fail} Unexpected es_tok_constraints in +constraint output")
        failed += 1

    d_minus = converter.expr_to_dict("-广告")
    minus_query = get_query_dict(d_minus)
    if minus_query.get("query") == "-广告" and set(
        minus_query.get("fields", [])
    ) == set(SEARCH_MATCH_FIELDS):
        logger.mesg(f"{ok} -constraint keeps exclusion syntax on normal fields")
        passed += 1
    else:
        logger.fail(f"{fail} -constraint query dict: {minus_query}")
        failed += 1

    d_both = converter.expr_to_dict("+A -B")
    both_query = get_query_dict(d_both)
    if both_query.get("query") == "+A -B":
        logger.mesg(f"{ok} +A -B stays in a single exact-segment query string")
        passed += 1
    else:
        logger.fail(f"{fail} +A -B query dict: {both_query}")
        failed += 1

    if "es_tok_constraints" not in json.dumps(d_both, ensure_ascii=False):
        logger.mesg(f"{ok} +A -B still avoids es_tok_constraints")
        passed += 1
    else:
        logger.fail(f"{fail} Unexpected es_tok_constraints: {d_both}")
        failed += 1

    logger.note(f"Constraint fields: {passed} passed, {failed} failed")
    assert failed == 0


def test_qmod():
    """Test query mode parsing."""
    logger.note("TEST: Query Mode (qmod)")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter
    from dsl.fields.qmod import extract_qmod_from_expr_tree

    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    test_cases = [
        ("q=w", ["word"]),
        ("q=v", ["vector"]),
        ("q=wv", ["word", "vector"]),
        ("q=vw", ["word", "vector"]),
        ("q=wr", ["word", "rerank"]),
        ("q=wvr", ["word", "vector", "rerank"]),
        ("黑神话 q=v", ["vector"]),
        ("黑神话 悟空", ["word", "vector"]),  # default is hybrid
    ]

    for query, expected in test_cases:
        expr_tree = converter.construct_expr_tree(query)
        modes = extract_qmod_from_expr_tree(expr_tree)
        if modes == expected:
            logger.mesg(f"{ok} {query}: {modes}")
            passed += 1
        else:
            logger.fail(f"{fail} {query}: expected {expected}, got {modes}")
            failed += 1

    logger.note(f"Qmod: {passed} passed, {failed} failed")
    assert failed == 0


def test_constraint_tree_converter():
    """Test ConstraintTreeConverter methods directly."""
    logger.note("TEST: ConstraintTreeConverter")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter
    from dsl.fields.constraint import ConstraintTreeConverter

    converter = DslExprToElasticConverter()
    ctc = ConstraintTreeConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    # --- Pure constraint subtree detection ---
    pure_tree = converter.construct_expr_tree("+A +B")
    flat_tree = converter.flatter.flatten(pure_tree)
    if ctc.is_pure_constraint_subtree(flat_tree):
        logger.mesg(f"{ok} is_pure_constraint_subtree(+A +B) = True")
        passed += 1
    else:
        logger.fail(f"{fail} is_pure_constraint_subtree(+A +B) should be True")
        failed += 1

    mixed_tree = converter.construct_expr_tree("A +B")
    flat_mixed = converter.flatter.flatten(mixed_tree)
    if not ctc.is_pure_constraint_subtree(flat_mixed):
        logger.mesg(f"{ok} is_pure_constraint_subtree(A +B) = False (mixed)")
        passed += 1
    else:
        logger.fail(f"{fail} is_pure_constraint_subtree(A +B) should be False")
        failed += 1

    filter_tree = converter.construct_expr_tree("+A v>10k")
    flat_filter = converter.flatter.flatten(filter_tree)
    if not ctc.is_pure_constraint_subtree(flat_filter):
        logger.mesg(f"{ok} is_pure_constraint_subtree(+A v>10k) = False (has filter)")
        passed += 1
    else:
        logger.fail(f"{fail} is_pure_constraint_subtree(+A v>10k) should be False")
        failed += 1

    # --- node_to_constraint for simple atoms ---
    atom_tree = converter.construct_expr_tree("+hello")
    flat_atom = converter.flatter.flatten(atom_tree)
    constraint = ctc.node_to_constraint(flat_atom)
    if constraint and "have_token" in constraint:
        logger.mesg(f"{ok} node_to_constraint(+hello): {constraint}")
        passed += 1
    else:
        logger.fail(f"{fail} node_to_constraint(+hello): {constraint}")
        failed += 1

    # --- node_to_constraint for AND ---
    and_tree = converter.construct_expr_tree("+A & +B")
    flat_and = converter.flatter.flatten(and_tree)
    and_constraint = ctc.node_to_constraint(flat_and)
    if and_constraint and "AND" in and_constraint and len(and_constraint["AND"]) == 2:
        logger.mesg(f"{ok} node_to_constraint(+A & +B): AND of 2")
        passed += 1
    else:
        logger.fail(f"{fail} node_to_constraint(+A & +B): {and_constraint}")
        failed += 1

    # --- node_to_constraint for OR ---
    or_tree = converter.construct_expr_tree("+A | +B")
    flat_or = converter.flatter.flatten(or_tree)
    or_constraint = ctc.node_to_constraint(flat_or)
    if or_constraint and "OR" in or_constraint and len(or_constraint["OR"]) == 2:
        logger.mesg(f"{ok} node_to_constraint(+A | +B): OR of 2")
        passed += 1
    else:
        logger.fail(f"{fail} node_to_constraint(+A | +B): {or_constraint}")
        failed += 1

    # --- extract via normalize_constraints_in_tree: simple ---
    simple_tree = converter.construct_expr_tree("+X -Y")
    flat_simple = converter.flatter.flatten(simple_tree)
    must_have_texts, must_not_texts, remaining = ctc.normalize_constraints_in_tree(
        flat_simple
    )
    if len(must_have_texts) == 1 and must_have_texts[0] == "X":
        logger.mesg(f"{ok} normalize_constraints(+X -Y): must_have=['X']")
        passed += 1
    else:
        logger.fail(f"{fail} normalize_constraints(+X -Y): must_have={must_have_texts}")
        failed += 1

    if len(must_not_texts) == 1 and must_not_texts[0] == "Y":
        logger.mesg(f"{ok} normalize_constraints(+X -Y): must_not=['Y']")
        passed += 1
    else:
        logger.fail(f"{fail} normalize_constraints(+X -Y): must_not={must_not_texts}")
        failed += 1

    # --- normalize_constraints_in_tree: no constraints ---
    no_constraint_tree = converter.construct_expr_tree("hello world")
    flat_nc = converter.flatter.flatten(no_constraint_tree)
    nc_must_have, nc_must_not, nc_remaining = ctc.normalize_constraints_in_tree(flat_nc)
    if len(nc_must_have) == 0 and len(nc_must_not) == 0:
        logger.mesg(f"{ok} normalize_constraints(hello world): 0 must_have, 0 must_not")
        passed += 1
    else:
        logger.fail(
            f"{fail} normalize_constraints(hello world): must_have={nc_must_have}, must_not={nc_must_not}"
        )
        failed += 1

    logger.note(f"ConstraintTreeConverter: {passed} passed, {failed} failed")
    assert failed == 0


def test_constraint_regression():
    """Test current regression-sensitive behavior for exact-segment constraints."""
    logger.note("TEST: Constraint Regression (Bug Fixes)")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter
    from dsl.rewrite import DslExprRewriter

    converter = DslExprToElasticConverter()
    rewriter = DslExprRewriter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    d_plain = converter.expr_to_dict("08 V神")
    d_plus = converter.expr_to_dict("08 +V神")
    if json.dumps(d_plain, sort_keys=True) != json.dumps(d_plus, sort_keys=True):
        logger.mesg(f"{ok} '08 +V神' differs from '08 V神' (constraint vs must)")
        passed += 1
    else:
        logger.fail(f"{fail} '08 +V神' should differ from '08 V神'")
        failed += 1

    d_plus_str = json.dumps(d_plus, ensure_ascii=False)
    if (
        "es_tok_query_string" in d_plus_str
        and "08 +V神" in d_plus_str
        and "es_tok_constraints" not in d_plus_str
    ):
        logger.mesg(f"{ok} '08 +V神' keeps +V神 inline in the query text")
        passed += 1
    else:
        logger.fail(f"{fail} '08 +V神': unexpected output: {d_plus_str}")
        failed += 1

    info = rewriter.get_query_info("08 +V神")
    kb = info.get("keywords_body", [])
    ct = info.get("constraint_texts", [])
    if kb == ["08"] and ct == ["V神"]:
        logger.mesg(f"{ok} '08 +V神': keywords_body={kb}, constraint_texts={ct}")
        passed += 1
    else:
        logger.fail(f"{fail} '08 +V神': keywords_body={kb}, constraint_texts={ct}")
        failed += 1

    info2 = rewriter.get_query_info("+08 V神 -85")
    kb2 = info2.get("keywords_body", [])
    ct2 = info2.get("constraint_texts", [])
    if kb2 == ["V神"] and ct2 == ["08"]:
        logger.mesg(f"{ok} '+08 V神 -85': keywords_body={kb2}, constraint_texts={ct2}")
        passed += 1
    else:
        logger.fail(
            f"{fail} '+08 V神 -85': keywords_body={kb2}, constraint_texts={ct2}"
        )
        failed += 1

    info3 = rewriter.get_query_info("+达芬奇 -广告")
    cf = info3.get("constraint_filter", {})
    cf_str = json.dumps(cf, ensure_ascii=False)
    if (
        "es_tok_query_string" in cf_str
        and "+达芬奇" in cf_str
        and "-广告" in cf_str
        and "es_tok_constraints" not in cf_str
    ):
        logger.mesg(
            f"{ok} get_query_info('+达芬奇 -广告') returns exact-segment filter"
        )
        passed += 1
    else:
        logger.fail(f"{fail} unexpected constraint_filter: {cf_str}")
        failed += 1

    d_both_plus = converter.expr_to_dict("+08 +V神")
    d_both_plain = converter.expr_to_dict("08 V神")
    if json.dumps(d_both_plus, sort_keys=True) != json.dumps(
        d_both_plain, sort_keys=True
    ):
        logger.mesg(f"{ok} '+08 +V神' differs from '08 V神' (AND vs OR)")
        passed += 1
    else:
        logger.fail(f"{fail} '+08 +V神' should differ from '08 V神'")
        failed += 1

    both_str = json.dumps(d_both_plus, ensure_ascii=False)
    if (
        "es_tok_query_string" in both_str
        and "+08 +V神" in both_str
        and "es_tok_constraints" not in both_str
    ):
        logger.mesg(f"{ok} '+08 +V神' keeps both exact-segment tokens inline")
        passed += 1
    else:
        logger.fail(f"{fail} '+08 +V神': unexpected output: {both_str}")
        failed += 1

    d_exact = converter.expr_to_dict("+达芬奇 +影视飓风")
    exact_str = json.dumps(d_exact, ensure_ascii=False)
    if (
        "es_tok_query_string" in exact_str
        and "+达芬奇 +影视飓风" in exact_str
        and "es_tok_constraints" not in exact_str
    ):
        logger.mesg(f"{ok} '+达芬奇 +影视飓风' uses exact-segment query syntax")
        passed += 1
    else:
        logger.fail(f"{fail} '+达芬奇 +影视飓风': unexpected output: {exact_str}")
        failed += 1

    info_or = rewriter.get_query_info("(+若生命将于明日落幕 || +工藤晴香) 游戏音乐")
    cf_or_str = json.dumps(info_or.get("constraint_filter", {}), ensure_ascii=False)
    if (
        "minimum_should_match" in cf_or_str
        and "+若生命将于明日落幕" in cf_or_str
        and "+工藤晴香" in cf_or_str
        and "游戏音乐" not in cf_or_str
    ):
        logger.mesg(
            f"{ok} OR exact-segment constraint filter keeps only constraint terms"
        )
        passed += 1
    else:
        logger.fail(f"{fail} unexpected OR constraint_filter: {cf_or_str}")
        failed += 1

    info_case = rewriter.get_query_info("+红警HBK08 一块地")
    cf_case_str = json.dumps(info_case.get("constraint_filter", {}), ensure_ascii=False)
    kb_case = info_case.get("keywords_body", [])
    ct_case = info_case.get("constraint_texts", [])
    if (
        "+红警HBK08" in cf_case_str
        and "一块地" not in cf_case_str
        and "红警HBK08" in ct_case
        and kb_case == ["一块地"]
    ):
        logger.mesg(
            f"{ok} mixed CJK+ASCII constraint text stays out of keywords_body: {ct_case}"
        )
        passed += 1
    else:
        logger.fail(
            f"{fail} mixed CJK+ASCII constraint handling wrong: cf={cf_case_str}, "
            f"keywords_body={kb_case}, constraint_texts={ct_case}"
        )
        failed += 1

    info4 = rewriter.get_query_info("影视飓风")
    cf4 = info4.get("constraint_filter", {})
    if not cf4:
        logger.mesg(f"{ok} No constraint_filter for query without +/-")
        passed += 1
    else:
        logger.fail(f"{fail} Unexpected constraint_filter: {cf4}")
        failed += 1

    logger.note(f"Constraint regression: {passed} passed, {failed} failed")
    assert failed == 0
