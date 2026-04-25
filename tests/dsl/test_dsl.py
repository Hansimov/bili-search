"""Comprehensive tests for the DSL module.

Tests cover:
1. Grammar parsing (atom expressions, boolean expressions)
2. Elastic conversion (word, filter, user, constraint)
3. User exclusion bug fix (must_not preserved in KNN filters)
4. Token constraints (+/- prefix preserved in es_tok_query_string)
5. Complex boolean exact-segment expressions
6. Filter merger functionality
7. Query rewriting
8. Edge cases and regression tests
"""

import json
from collections import defaultdict
from tclogger import logger, logstr, dict_to_str, brk

# ============================================================================
# Test Data
# ============================================================================

# --- Atom expression test cases (from dsl/test.py + new constraint cases) ---
date_queries = [
    ["date=1d", ("date_expr", 1)],
    ["dt= =1h", ("date_expr", 1)],
    ["d< = 2wk", ("date_expr", 1)],
    ["d= 2024", ("date_expr", 1)],
    ["d=2024-01/01", ("date_expr", 1)],
    ["dt==[2024, 2025-01]", ("date_expr", 1)],
    ["date==[,2024, 3d,,)", ("date_expr", 1)],
    ["d=this_d", ("date_expr", 1)],
    ["d=t d", ("date_expr", 1)],
    ["d=this_h", ("date_expr", 1)],
    ["d=th", ("date_expr", 1)],
    ["d=past_week", ("date_expr", 1)],
    ["d=pw", ("date_expr", 1)],
    ["d=[last_d,]", ("date_expr", 1)],
]

user_queries = [
    ["u=影视飓风", ("user_expr", 1)],
    ["u==咬人猫=", ("user_expr", 1)],
    ["user!=(飓多多StormCrew,何同学，影视飓风)", ("user_expr", 1)],
    ['@!["-LKs-",  ，红警HBK08，，红警月亮3,,]', ("user_expr", 1)],
    ['戈壁 u="中国国家地理"', (("user_expr", 1), ("word_expr", 1))],
]

uid_queries = [
    ["uid=1234", ("uid_expr", 1)],
    ["uid=[123,456,789)", ("uid_expr", 1)],
    ["mid! =[123,456]", ("uid_expr", 1)],
]

stat_queries = [
    ["bf<1000", ("stat_expr", 1)],
    [":v>=10k", ("stat_expr", 1)],
    [":dz= = [1k,10k]", ("stat_expr", 1)],
    [":vw <= [ 1w,10w )", ("stat_expr", 1)],
    [":lk>= = [ 1w,10w )", ("stat_expr", 1)],
]

dura_queries = [
    ["dura>30", ("dura_expr", 1)],
    ["t<=1h", ("dura_expr", 1)],
    ["t=[30,1m30s]", ("dura_expr", 1)],
    ["d=1d t>5h v>1w", (("date_expr", 1), ("dura_expr", 1), ("stat_expr", 1))],
]

region_queries = [
    ["rg = 动画", ("region_expr", 1)],
    ["region=(影视,动画,音乐)", ("region_expr", 1)],
    ["rid ! = [1,24, 153]", ("region_expr", 1)],
    ["rg- = =(影视,动画,153]", ("region_expr", 1)],
]

word_queries = [
    ["k=你好", ("word_expr", 1)],
    ['k="世界,你好"', ("word_expr", 1)],
    ["k!=[你好，世界]", ("word_expr", 1)],
    ["k-=[你好,世界]", ("word_expr", 1)],
    ['k=["你好,世界","再见，故乡"]', ("word_expr", 1)],
    ["k=3-0", ("word_expr", 1)],
    ['你好 世界 "再见，故乡"', ("word_expr", 3)],
    ['"你好，故乡"', ("word_expr", 1)],
]

constraint_queries = [
    ["+影视飓风", ("word_expr", 1)],
    ["-广告", ("word_expr", 1)],
    ["+影视飓风 -小米", ("word_expr", 2)],
    ['+"影视飓风" !"李四维"', ("word_expr", 2)],
]

# --- Boolean expression test cases ---
bool_queries = [
    ["你好 这是 世界 u=hello", [("user_expr", 1), ("word_expr", 3)], [("co", 1)]],
    ["hello && world", [("word_expr", 2)], [("and", 1)]],
    ["hello | | world & & nothing", [("word_expr", 3)], [("and", 1), ("or", 1)]],
    ["(hello || world)", [("word_expr", 2)], [("or", 1), ("pa", 1)]],
    ["(hello || world) 你好", [("word_expr", 3)], [("co", 1), ("or", 1), ("pa", 1)]],
    [
        "(hello || world) && nothing",
        [("word_expr", 3)],
        [("and", 1), ("or", 1), ("pa", 1)],
    ],
    [
        "find nothing && ((hello | world) && anything)",
        [("word_expr", 5)],
        [("and", 2), ("co", 1), ("or", 1), ("pa", 2)],
    ],
    [
        "(find nothing) || ((hello | world) && anything)",
        [("word_expr", 5)],
        [("or", 2), ("co", 1), ("and", 1), ("pa", 3)],
    ],
    [
        "(hello world) (find nothing) (((",
        [("word_expr", 4)],
        [("co", 3), ("pa", 2)],
    ],
    ["hello || world || boy", [("word_expr", 3)], [("or", 2)]],
    ["( ( find nothing ) )", [("word_expr", 2)], [("co", 1), ("pa", 2)]],
    ["你好 这是 世界 ()", [("word_expr", 3)], [("co", 1), ("pa", 1)]],
    ["( ( (", [], []],
    ["( ( find nothing", [("word_expr", 2)], [("co", 1)]],  # FLAT PASSED
    ["你好 这是 (( 世界 (", [("word_expr", 3)], [("co", 2)]],  # FLAT PASSED
]

# --- Component expression test cases ---
comp_queries = [
    [
        "影视飓风 v>10k :coin>=25 u=[飓多多StormCrew, 亿点点不一样] 风光摄影",
        [("user_expr", 1), ("word_expr", 2), ("stat_expr", 2)],
        [("co", 1)],
    ],
    ['+"影视飓风" !"李四维"', [("word_expr", 2)], [("co", 1)]],
    [
        "影视飓风 v>10k :coin>=25 u=[飓多多StormCrew, 亿点点不一样 ,, 影视飓风]",
        [("user_expr", 1), ("word_expr", 1), ("stat_expr", 2)],
        [("co", 1)],
    ],
    [
        "影视飓风 v>10k :coin>=25 u=[,) , 何同学",
        [("user_expr", 1), ("word_expr", 2), ("stat_expr", 2)],
        [("co", 1)],
    ],
    [
        "(影视飓风 || 飓多多 || TIM 李四维 && 青青 && k-=LKS) (v>=1w || :coin>=25)",
        [("word_expr", 6), ("stat_expr", 2)],
        [("co", 2), ("and", 2), ("or", 3), ("pa", 2)],
    ],
    [
        "(影视飓风 || 飓多多 || TIM )",
        [("word_expr", 3)],
        [("or", 2), ("pa", 1)],
    ],
    [
        ":date=2024-01 :view>=1w",
        [("date_expr", 1), ("stat_expr", 1)],
        [("co", 1)],
    ],
    [
        ":date=2024-01-01 yingshi",
        [("date_expr", 1), ("word_expr", 1)],
        [("co", 1)],
    ],
    [
        "《影视飓风》 :date=2024-01/01 :view>=1w ",
        [("date_expr", 1), ("word_expr", 1), ("stat_expr", 1)],
        [("co", 1)],
    ],
    [
        '(+"雷军" || +"小米") (+"影视飓风" || +"tim")',
        [("word_expr", 4)],
        [("co", 1), ("or", 2), ("pa", 2)],
    ],
    ['"deep learning"~', [("word_expr", 1)], []],
]


queries_of_atoms = [
    *date_queries,
    *user_queries,
    *uid_queries,
    *stat_queries,
    *dura_queries,
    *region_queries,
    *word_queries,
    *constraint_queries,
]

queries_of_bools = [
    *bool_queries,
    *comp_queries,
]


# ============================================================================
# Helper Functions
# ============================================================================


def list_to_tuple(lst: list) -> tuple:
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0]
    else:
        lst = sorted(lst, key=lambda x: x[0])
        return tuple(lst)


def get_atoms_info(node) -> tuple:
    atom_expr_keys = node.get_all_atom_childs_expr_keys()
    atoms_info = defaultdict(int)
    for key in atom_expr_keys:
        atoms_info[key] += 1
    atoms_info = sorted(atoms_info.items(), key=lambda x: x[0])
    return list_to_tuple(atoms_info)


def get_bools_info(node) -> tuple:
    bool_expr_keys = node.get_all_bool_childs_keys()
    bools_info = defaultdict(int)
    for key in bool_expr_keys:
        bools_info[key] += 1
    bools_info = sorted(bools_info.items(), key=lambda x: x[0])
    return list_to_tuple(bools_info)


# ============================================================================
# Test Functions
# ============================================================================


def test_atom_parsing():
    """Test that atom expressions parse into correct expr types."""
    logger.note("TEST: Atom Expression Parsing")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter

    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    for query, correct_atoms_info in queries_of_atoms:
        expr_tree = converter.construct_expr_tree(query)
        atoms_info = get_atoms_info(expr_tree)
        if atoms_info == correct_atoms_info:
            logger.mesg(f"{ok} {query}: {atoms_info}")
            passed += 1
        else:
            logger.fail(f"{fail} {query}:")
            logger.okay(f"  expected: {correct_atoms_info}")
            logger.fail(f"  got:      {atoms_info}")
            failed += 1

    logger.note(f"Atom parsing: {passed} passed, {failed} failed")
    assert failed == 0


def test_bool_parsing():
    """Test boolean expression parsing with correct grouping."""
    logger.note("TEST: Boolean Expression Parsing")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter

    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    for query, correct_atoms_info, correct_bools_info in queries_of_bools:
        correct_atoms_info = list_to_tuple(correct_atoms_info)
        correct_bools_info = list_to_tuple(correct_bools_info)
        expr_tree = converter.construct_expr_tree(query)
        atoms_info = get_atoms_info(expr_tree)
        bools_info = get_bools_info(expr_tree)
        flat_tree = converter.flatter.flatten(expr_tree)
        atoms_finfo = get_atoms_info(flat_tree)
        bools_finfo = get_bools_info(flat_tree)

        if (atoms_info == correct_atoms_info and bools_info == correct_bools_info) or (
            atoms_finfo == correct_atoms_info and bools_finfo == correct_bools_info
        ):
            logger.mesg(f"{ok} {brk(query)}: atoms={atoms_finfo}, bools={bools_finfo}")
            passed += 1
        else:
            logger.fail(f"{fail} {brk(query)}:")
            logger.okay(f"  expected atoms: {correct_atoms_info}")
            logger.fail(f"  got atoms:      {atoms_info} (flat: {atoms_finfo})")
            logger.okay(f"  expected bools: {correct_bools_info}")
            logger.fail(f"  got bools:      {bools_info} (flat: {bools_finfo})")
            failed += 1

    logger.note(f"Bool parsing: {passed} passed, {failed} failed")
    assert failed == 0


def test_elastic_conversion():
    """Test that DSL queries produce valid Elasticsearch dictionaries."""
    logger.note("TEST: Elastic Conversion")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter

    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    test_cases = [
        # (query, description, check_func)
        (
            "影视飓风",
            "word → must with query_string",
            lambda d: "bool" in d and "must" in d["bool"],
        ),
        (
            "u=影视飓风",
            "user → filter with term",
            lambda d: (
                "bool" in d
                and "filter" in d["bool"]
                and ("term" in str(d) or "terms" in str(d))
            ),
        ),
        (
            "v>10k",
            "stat → filter with range",
            lambda d: "bool" in d and "filter" in d["bool"] and "range" in str(d),
        ),
        (
            "影视飓风 v>10k u=[飓多多,何同学]",
            "word+stat+user → must+filter",
            lambda d: ("bool" in d and "must" in d["bool"] and "filter" in d["bool"]),
        ),
        (
            "hello || world",
            "or → should with minimum_should_match",
            lambda d: "bool" in d and "should" in d["bool"],
        ),
        (
            "hello && world",
            "and → must/filter clauses",
            lambda d: "bool" in d,
        ),
        (
            '(hello || world) && "nothing"',
            "complex bool → nested bool",
            lambda d: "bool" in d and "should" in str(d),
        ),
    ]

    for query, desc, check_fn in test_cases:
        try:
            elastic_dict = converter.expr_to_dict(query)
            if check_fn(elastic_dict):
                logger.mesg(f"{ok} {query}: {desc}")
                passed += 1
            else:
                logger.fail(f"{fail} {query}: {desc}")
                logger.fail(f"  got: {dict_to_str(elastic_dict, add_quotes=True)}")
                failed += 1
        except Exception as e:
            logger.fail(f"{fail} {query}: {desc} — Exception: {e}")
            failed += 1

    logger.note(f"Elastic conversion: {passed} passed, {failed} failed")
    assert failed == 0


def test_user_exclusion():
    """Test that user exclusion (u!=[...]) generates correct must_not clauses."""
    logger.note("TEST: User Exclusion (must_not)")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter

    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    test_cases = [
        (
            "u!=[影视飓风,何同学]",
            "has must_not with terms",
            lambda d: (
                "bool" in d
                and "must_not" in d["bool"]
                and "terms" in d["bool"]["must_not"]
            ),
        ),
        (
            "搜索词 u!=[影视飓风,何同学]",
            "has must + must_not",
            lambda d: ("bool" in d and "must" in d["bool"] and "must_not" in d["bool"]),
        ),
        (
            '@!["-LKs-",红警HBK08]',
            "at_neq produces must_not",
            lambda d: "bool" in d and "must_not" in d["bool"],
        ),
        (
            "u=影视飓风",
            "positive user match has filter (not must_not)",
            lambda d: (
                "bool" in d
                and "filter" in d["bool"]
                and "must_not" not in d.get("bool", {})
            ),
        ),
        (
            "u==[影视飓风,何同学]",
            "positive user list match has filter",
            lambda d: "bool" in d and "filter" in d["bool"],
        ),
    ]

    for query, desc, check_fn in test_cases:
        elastic_dict = converter.expr_to_dict(query)
        if check_fn(elastic_dict):
            logger.mesg(f"{ok} {query}: {desc}")
            passed += 1
        else:
            logger.fail(f"{fail} {query}: {desc}")
            logger.fail(f"  got: {dict_to_str(elastic_dict, add_quotes=True)}")
            failed += 1

    logger.note(f"User exclusion: {passed} passed, {failed} failed")
    assert failed == 0


def test_knn_filter_extraction():
    """Test that user exclusion (must_not) is preserved in KNN filter extraction.

    This is a regression test for the bug where u!=[...] exclusions were lost
    when constructing KNN pre-filters because get_query_filters_from_query_dsl_dict()
    only extracted bool.filter clauses, ignoring bool.must_not.
    """
    logger.note("TEST: KNN Filter — User Exclusion Preserved")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter
    from dsl.filter import QueryDslDictFilterMerger

    converter = DslExprToElasticConverter()
    merger = QueryDslDictFilterMerger()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    # --- Test old method (get_query_filters) misses must_not ---
    exclusion_dict = converter.expr_to_dict("u!=[影视飓风,何同学]")
    old_clauses = merger.get_query_filters_from_query_dsl_dict(exclusion_dict)
    new_clauses = merger.get_all_bool_clauses_from_query_dsl_dict(exclusion_dict)

    if len(old_clauses) == 0:
        logger.mesg(f"{ok} Old method correctly returns [] for must_not-only dict")
        passed += 1
    else:
        logger.fail(
            f"{fail} Old method should return [] for must_not-only dict, got: {old_clauses}"
        )
        failed += 1

    if len(new_clauses) > 0 and any(
        "bool" in c and "must_not" in c.get("bool", {}) for c in new_clauses
    ):
        logger.mesg(f"{ok} New method preserves must_not clauses")
        passed += 1
    else:
        logger.fail(f"{fail} New method should preserve must_not: {new_clauses}")
        failed += 1

    # --- Test combined query with exclusion + filter ---
    combined_dict = converter.expr_to_dict("搜索词 u!=[影视飓风] v>10k")
    filter_only_tree = converter.construct_expr_tree("搜索词 u!=[影视飓风] v>10k")
    filter_only_tree = filter_only_tree.filter_atoms_by_keys(exclude_keys=["word_expr"])
    filter_dict = converter.expr_tree_to_dict(filter_only_tree)
    all_clauses = merger.get_all_bool_clauses_from_query_dsl_dict(filter_dict)

    has_must_not = any(
        "bool" in c and "must_not" in c.get("bool", {}) for c in all_clauses
    )
    has_range = any("range" in c for c in all_clauses)

    if has_must_not and has_range:
        logger.mesg(f"{ok} Combined: must_not + range filter both preserved")
        passed += 1
    else:
        logger.fail(f"{fail} Combined: must_not={has_must_not}, range={has_range}")
        logger.fail(f"  clauses: {all_clauses}")
        failed += 1

    # --- Test positive user filter is still correct ---
    positive_dict = converter.expr_to_dict("搜索词 u=影视飓风")
    pos_filter_tree = converter.construct_expr_tree("搜索词 u=影视飓风")
    pos_filter_tree = pos_filter_tree.filter_atoms_by_keys(exclude_keys=["word_expr"])
    pos_filter_dict = converter.expr_tree_to_dict(pos_filter_tree)
    pos_clauses = merger.get_all_bool_clauses_from_query_dsl_dict(pos_filter_dict)

    if any("term" in c or "terms" in c for c in pos_clauses):
        logger.mesg(f"{ok} Positive user filter: preserved as term/terms")
        passed += 1
    else:
        logger.fail(f"{fail} Positive user filter lost: {pos_clauses}")
        failed += 1

    logger.note(f"KNN filter: {passed} passed, {failed} failed")
    assert failed == 0


def test_filter_merger():
    """Test the QueryDslDictFilterMerger functionality."""
    logger.note("TEST: Filter Merger")
    logger.note("=" * 60)

    from dsl.filter import QueryDslDictFilterMerger

    merger = QueryDslDictFilterMerger()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    # --- Range merging (union) ---
    v1 = {"gt": 1000, "lte": 2000}
    v2 = {"gte": 1500, "lt": 2500}
    union = merger.merge_range_values(v1, v2, "u")
    if union.get("gt") == 1000 and union.get("lt") == 2500:
        logger.mesg(f"{ok} Range union: {union}")
        passed += 1
    else:
        logger.fail(f"{fail} Range union: expected gt=1000,lt=2500, got {union}")
        failed += 1

    # --- Range merging (intersect) ---
    intersect = merger.merge_range_values(v1, v2, "i")
    if intersect.get("gte") == 1500 and intersect.get("lte") == 2000:
        logger.mesg(f"{ok} Range intersect: {intersect}")
        passed += 1
    else:
        logger.fail(
            f"{fail} Range intersect: expected gte=1500,lte=2000, got {intersect}"
        )
        failed += 1

    # --- Term merging (union) ---
    t1 = ["红警HBK08", "红警月亮3"]
    t2 = "红警HBK08"
    term_union = merger.merge_term_values(t1, t2, "u")
    if set(term_union) == {"红警HBK08", "红警月亮3"}:
        logger.mesg(f"{ok} Term union: {term_union}")
        passed += 1
    else:
        logger.fail(f"{fail} Term union: {term_union}")
        failed += 1

    # --- Term merging (intersect) ---
    term_intersect = merger.merge_term_values(t1, t2, "i")
    if term_intersect == ["红警HBK08"]:
        logger.mesg(f"{ok} Term intersect: {term_intersect}")
        passed += 1
    else:
        logger.fail(f"{fail} Term intersect: {term_intersect}")
        failed += 1

    # --- get_all_bool_clauses preserves filter + must_not ---
    query_dict = {
        "bool": {
            "must": {"es_tok_query_string": {"query": "test"}},
            "must_not": {"terms": {"owner.name.keyword": ["foo", "bar"]}},
            "filter": {"range": {"stat.view": {"gte": 100}}},
        }
    }
    all_clauses = merger.get_all_bool_clauses_from_query_dsl_dict(query_dict)
    has_filter = any("range" in c for c in all_clauses)
    has_must_not = any(
        "bool" in c and "must_not" in c.get("bool", {}) for c in all_clauses
    )
    if has_filter and has_must_not:
        logger.mesg(f"{ok} get_all_bool_clauses: filter + must_not both present")
        passed += 1
    else:
        logger.fail(
            f"{fail} get_all_bool_clauses: filter={has_filter}, must_not={has_must_not}"
        )
        failed += 1

    # --- get_all_bool_clauses with only filter (no must_not) ---
    filter_only_dict = {"bool": {"filter": [{"range": {"stat.view": {"gte": 100}}}]}}
    filter_clauses = merger.get_all_bool_clauses_from_query_dsl_dict(filter_only_dict)
    if len(filter_clauses) == 1 and "range" in filter_clauses[0]:
        logger.mesg(f"{ok} get_all_bool_clauses: filter-only case works")
        passed += 1
    else:
        logger.fail(f"{fail} get_all_bool_clauses filter-only: {filter_clauses}")
        failed += 1

    # --- get_all_bool_clauses with empty dict ---
    empty_clauses = merger.get_all_bool_clauses_from_query_dsl_dict({})
    if empty_clauses == []:
        logger.mesg(f"{ok} get_all_bool_clauses: empty dict → []")
        passed += 1
    else:
        logger.fail(f"{fail} get_all_bool_clauses empty: {empty_clauses}")
        failed += 1

    # --- filter_maps_to_list: single-element term → {"term": {field: val}} ---
    filter_maps = {("terms", "owner.name.keyword"): ["影视飓风"]}
    result = merger.filter_maps_to_list(filter_maps)
    if (
        len(result) == 1
        and "term" in result[0]
        and "owner.name.keyword" in result[0]["term"]
    ):
        logger.mesg(f"{ok} Single term filter: correct {result[0]}")
        passed += 1
    else:
        logger.fail(f"{fail} Single term filter: got {result}")
        failed += 1

    # --- filter_maps_to_list: multi-element terms ---
    filter_maps_multi = {("terms", "owner.mid"): [123, 456, 789]}
    result_multi = merger.filter_maps_to_list(filter_maps_multi)
    if (
        len(result_multi) == 1
        and "terms" in result_multi[0]
        and result_multi[0]["terms"]["owner.mid"] == [123, 456, 789]
    ):
        logger.mesg(f"{ok} Multi term filter: correct")
        passed += 1
    else:
        logger.fail(f"{fail} Multi term filter: got {result_multi}")
        failed += 1

    # --- filter_maps_to_list: range ---
    filter_maps_range = {("range", "stat.view"): {"gte": 1000}}
    result_range = merger.filter_maps_to_list(filter_maps_range)
    if (
        len(result_range) == 1
        and "range" in result_range[0]
        and result_range[0]["range"]["stat.view"]["gte"] == 1000
    ):
        logger.mesg(f"{ok} Range filter: correct")
        passed += 1
    else:
        logger.fail(f"{fail} Range filter: got {result_range}")
        failed += 1

    logger.note(f"Filter merger: {passed} passed, {failed} failed")
    assert failed == 0


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


def test_edge_cases():
    """Test edge cases and potential error scenarios."""
    logger.note("TEST: Edge Cases")
    logger.note("=" * 60)

    from dsl.elastic import DslExprToElasticConverter

    converter = DslExprToElasticConverter()
    ok = logstr.okay("✓")
    fail = logstr.fail("×")
    passed = 0
    failed = 0

    edge_cases = [
        ("( ( (", "unbalanced parens only"),
        ('""', "empty quoted string"),
        ("k=", "key with no value"),
        ("v>0", "zero stat threshold"),
        ("  hello  ", "extra whitespace"),
        ("d=2024 d=2025", "duplicate date filters"),
    ]

    for query, desc in edge_cases:
        try:
            converter.expr_to_dict(query)
            logger.mesg(f"{ok} {brk(query)}: {desc} → no crash")
            passed += 1
        except Exception as e:
            logger.fail(f"{fail} {brk(query)}: {desc} → {type(e).__name__}: {e}")
            failed += 1

    try:
        converter.expr_to_dict("")
        logger.fail(f"{fail} empty query: should raise parse error")
        failed += 1
    except Exception:
        logger.mesg(f"{ok} empty query: correctly raises parse error")
        passed += 1

    logger.note(f"Edge cases: {passed} passed, {failed} failed")
    assert failed == 0
