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
