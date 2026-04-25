from fastapi.encoders import jsonable_encoder
from tclogger import logger, logstr, dict_to_str, brk

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.explorer import VideoExplorer
from converters.query.dsl import MultiMatchQueryDSLConstructor
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.query.filter import QueryFilterExtractor
from elastics.tests.test_videos import make_searcher, make_explorer, author_values, author_items


def test_author_ordering():
    """Test that author ordering follows video appearance order when sort_field="first_appear_order"."""
    from tclogger import logger, logstr, brk, dict_to_str

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    test_queries = [
        "红警08 小块地",
        "影视飓风 罗永浩",
    ]

    for query in test_queries:
        logger.note(f"\n> Testing author ordering for: [{query}]")

        # Get explore result
        explore_res = explorer.explore_v2(
            query=query,
            rank_method="stats",
            rank_top_k=50,
            group_owner_limit=25,
            verbose=False,
        )

        # Find the step with hits and the group_hits_by_owner step
        hits_result = None
        authors_result = None
        for step in explore_res.get("data", []):
            if step.get("name") == "most_relevant_search":
                hits_result = step.get("output", {})
            elif step.get("name") == "group_hits_by_owner":
                authors_result = step.get("output", {}).get("authors", [])

        if not hits_result or not authors_result:
            logger.warn("× Missing hits or authors result")
            continue

        hits = hits_result.get("hits", [])

        # Track first appearance order from hits
        expected_first_appear_order = {}
        for idx, hit in enumerate(hits):
            mid = hit.get("owner", {}).get("mid")
            if mid and mid not in expected_first_appear_order:
                expected_first_appear_order[mid] = idx

        # Verify authors dict preserves first_appear_order
        logger.hint("  Authors returned (from backend):")
        for i, (mid, author_info) in enumerate(author_items(authors_result)):
            author_name = author_info.get("name", "")
            first_appear = author_info.get("first_appear_order", -1)
            sum_rank_score = author_info.get("sum_rank_score", 0)
            expected_appear = expected_first_appear_order.get(int(mid), -1)
            match_status = "✓" if first_appear == expected_appear else "×"
            logger.mesg(
                f"    [{i}] {author_name:20} first_appear={first_appear:3} "
                f"(expected={expected_appear:3}) {match_status} sum_rank_score={sum_rank_score:.2f}"
            )

        # IMPORTANT: The issue is that the backend returns authors sorted by sum_rank_score,
        # not by first_appear_order. Let's verify:
        author_list = author_values(authors_result)

        # Check if authors are sorted by sum_rank_score (current backend behavior)
        is_sorted_by_sum_rank = all(
            author_list[i].get("sum_rank_score", 0)
            >= author_list[i + 1].get("sum_rank_score", 0)
            for i in range(len(author_list) - 1)
        )

        # Check if authors are sorted by first_appear_order (what frontend expects for "综合排序")
        is_sorted_by_first_appear = all(
            author_list[i].get("first_appear_order", 0)
            <= author_list[i + 1].get("first_appear_order", 0)
            for i in range(len(author_list) - 1)
        )

        logger.hint("  Sorting analysis:")
        logger.mesg(f"    Sorted by sum_rank_score: {is_sorted_by_sum_rank}")
        logger.mesg(f"    Sorted by first_appear_order: {is_sorted_by_first_appear}")

        # Show first 5 videos and their owners
        logger.hint("  First 5 hits and their owners (expected author order):")
        seen_owners = set()
        for idx, hit in enumerate(hits[:15]):
            owner = hit.get("owner", {})
            mid = owner.get("mid")
            name = owner.get("name", "")
            rank_score = hit.get("rank_score", 0)
            if mid not in seen_owners:
                seen_owners.add(mid)
                logger.mesg(
                    f"    [{len(seen_owners)}] {name:20} (first video at position {idx})"
                )
                if len(seen_owners) >= 5:
                    break


def test_author_grouper_unit():
    """Unit test for AuthorGrouper - no ES connection needed"""
    from ranks.grouper import AuthorGrouper
    from tclogger import logger

    # Create mock hits data that simulates the structure from ES
    mock_hits = [
        # First video from AuthorA (should appear first by order)
        {
            "bvid": "BV001",
            "owner": {"mid": 1001, "name": "AuthorA", "face": "face_a.jpg"},
            "rank_score": 0.9,
            "title": "Video 1 by A",
        },
        # First video from AuthorB (should appear second by order)
        {
            "bvid": "BV002",
            "owner": {"mid": 1002, "name": "AuthorB", "face": "face_b.jpg"},
            "rank_score": 0.95,  # Higher score but appears later in hits
            "title": "Video 1 by B",
        },
        # Second video from AuthorA
        {
            "bvid": "BV003",
            "owner": {"mid": 1001, "name": "AuthorA", "face": "face_a.jpg"},
            "rank_score": 0.85,
            "title": "Video 2 by A",
        },
        # First video from AuthorC (should appear third by order)
        {
            "bvid": "BV004",
            "owner": {"mid": 1003, "name": "AuthorC", "face": "face_c.jpg"},
            "rank_score": 0.99,  # Highest score but appears last in hits
            "title": "Video 1 by C",
        },
    ]

    grouper = AuthorGrouper()

    # Test 1: Sort by first_appear_order (should preserve video appearance order)
    logger.note("> Test 1: AuthorGrouper with sort_field='first_appear_order'")
    authors_by_order = grouper.group(mock_hits, sort_field="first_appear_order")

    logger.mesg(f"  Authors count: {len(authors_by_order)}")
    for i, (mid, author) in enumerate(authors_by_order.items()):
        logger.mesg(
            f"  [{i}] mid={mid} {author['name']:10}: "
            f"first_appear={author['first_appear_order']}, "
            f"sum_score={author.get('sum_rank_score', 0):.2f}, "
            f"count={author['sum_count']}"
        )

    # Verify order: should be A(0), B(1), C(3) by first appearance
    expected_order = [1001, 1002, 1003]
    actual_order = list(authors_by_order.keys())

    if actual_order == expected_order:
        logger.success(f"  ✓ Correct order by first_appear_order: {actual_order}")
    else:
        logger.err(f"  ✗ Wrong order! Expected: {expected_order}, Got: {actual_order}")

    # Test 2: Sort by sum_rank_score
    logger.note("> Test 2: AuthorGrouper with sort_field='sum_rank_score'")
    authors_by_score = grouper.group(mock_hits, sort_field="sum_rank_score")

    for i, (mid, author) in enumerate(authors_by_score.items()):
        logger.mesg(
            f"  [{i}] mid={mid} {author['name']:10}: "
            f"first_appear={author['first_appear_order']}, "
            f"sum_score={author.get('sum_rank_score', 0):.2f}"
        )

    # A: 0.9 + 0.85 = 1.75, B: 0.95, C: 0.99
    # Order by sum_rank_score desc: A (1.75), C (0.99), B (0.95)
    expected_by_score = [1001, 1003, 1002]
    actual_by_score = list(authors_by_score.keys())

    if actual_by_score == expected_by_score:
        logger.success(f"  ✓ Correct order by sum_rank_score: {actual_by_score}")
    else:
        logger.warn(
            f"  Order by sum_rank_score: {actual_by_score} (expected: {expected_by_score})"
        )

    logger.success("\n✓ AuthorGrouper unit tests completed!")


def test_author_grouper_list():
    """Test that AuthorGrouper.group_as_list returns list (for JSON transport)"""
    from ranks.grouper import AuthorGrouper
    from tclogger import logger

    # Create mock hits data
    mock_hits = [
        {
            "bvid": "BV001",
            "owner": {"mid": 1001, "name": "AuthorA", "face": "face_a.jpg"},
            "rank_score": 0.9,
            "title": "Video 1 by A",
        },
        {
            "bvid": "BV002",
            "owner": {"mid": 1002, "name": "AuthorB", "face": "face_b.jpg"},
            "rank_score": 0.95,
            "title": "Video 1 by B",
        },
        {
            "bvid": "BV003",
            "owner": {"mid": 1001, "name": "AuthorA", "face": "face_a.jpg"},
            "rank_score": 0.85,
            "title": "Video 2 by A",
        },
    ]

    grouper = AuthorGrouper()

    # Test group_as_list returns a list
    logger.note("> Test: AuthorGrouper.group_as_list() returns list")
    authors_list = grouper.group_as_list(mock_hits, sort_field="first_appear_order")

    assert isinstance(authors_list, list), "Should return list, not dict"
    logger.success(f"  ✓ Returned type: {type(authors_list).__name__}")

    # Verify order is preserved in list
    expected_mids = [1001, 1002]
    actual_mids = [a["mid"] for a in authors_list]
    assert actual_mids == expected_mids, f"Expected {expected_mids}, got {actual_mids}"
    logger.success(f"  ✓ List order preserved: {actual_mids}")

    # Test that JSON serialization preserves order
    import json

    json_str = json.dumps(authors_list)
    restored_list = json.loads(json_str)
    restored_mids = [a["mid"] for a in restored_list]
    assert restored_mids == expected_mids, "JSON should preserve list order"
    logger.success(f"  ✓ JSON transport preserves order: {restored_mids}")

    logger.success("\n✓ AuthorGrouper list tests completed!")


def test_ranks_imports():
    """Test that ranks module imports work correctly"""
    from tclogger import logger

    logger.note("> Testing ranks module imports...")

    # Test importing from submodules directly
    from ranks.constants import (
        RANK_METHOD_TYPE,
        RANK_METHOD,  # renamed from RANK_METHOD_DEFAULT
        RANK_TOP_K,
        AUTHOR_SORT_FIELD_TYPE,
        AUTHOR_SORT_FIELD,  # renamed from AUTHOR_SORT_FIELD_DEFAULT
    )
    from ranks.ranker import VideoHitsRanker
    from ranks.reranker import get_reranker
    from ranks.grouper import AuthorGrouper
    from ranks.scorers import StatsScorer, PubdateScorer
    from ranks.fusion import ScoreFuser

    logger.success("  ✓ All submodule imports work")

    # Verify renamed constants
    assert RANK_TOP_K == 50, f"RANK_TOP_K should be 50, got {RANK_TOP_K}"
    assert (
        RANK_METHOD == "diversified"
    ), f"RANK_METHOD should be 'diversified', got {RANK_METHOD}"
    assert (
        AUTHOR_SORT_FIELD == "first_appear_order"
    ), f"AUTHOR_SORT_FIELD should be 'first_appear_order', got {AUTHOR_SORT_FIELD}"
    logger.success(f"  ✓ RANK_TOP_K = {RANK_TOP_K}")
    logger.success(f"  ✓ RANK_METHOD = {RANK_METHOD}")
    logger.success(f"  ✓ AUTHOR_SORT_FIELD = {AUTHOR_SORT_FIELD}")

    # Test classes can be instantiated
    ranker = VideoHitsRanker()
    grouper = AuthorGrouper()
    stats_scorer = StatsScorer()
    logger.success("  ✓ All classes can be instantiated")

    logger.success("\n✓ Ranks module import tests completed!")


def test_constants_refactoring():
    """Test that constants refactoring is correct - _DEFAULT suffixes removed"""
    from tclogger import logger

    logger.note("> Testing constants refactoring...")

    # Test ranks.constants renamed constants
    from ranks.constants import (
        RANK_METHOD,  # was RANK_METHOD_DEFAULT
        AUTHOR_SORT_FIELD,  # was AUTHOR_SORT_FIELD_DEFAULT
    )

    assert RANK_METHOD == "diversified"
    assert AUTHOR_SORT_FIELD == "first_appear_order"
    logger.success("  ✓ ranks.constants: RANK_METHOD, AUTHOR_SORT_FIELD")

    # Test elastics.videos.constants renamed constants
    from elastics.videos.constants import (
        USE_SCRIPT_SCORE,  # was USE_SCRIPT_SCORE_DEFAULT
        QMOD,  # was QMOD_DEFAULT
        KNN_SIMILARITY,  # was KNN_SIMILARITY_DEFAULT
    )

    assert USE_SCRIPT_SCORE == False
    assert QMOD == ["word", "vector"]
    assert KNN_SIMILARITY == "hamming"
    logger.success(
        "  ✓ elastics.videos.constants: USE_SCRIPT_SCORE, QMOD, KNN_SIMILARITY"
    )

    # Test dsl.fields.qmod renamed constants
    from dsl.fields.qmod import QMOD as QMOD_FROM_QMOD

    assert QMOD_FROM_QMOD == ["word", "vector"]
    logger.success("  ✓ dsl.fields.qmod: QMOD")

    # Test that old names no longer exist
    try:
        from ranks.constants import RANK_METHOD_DEFAULT

        assert False, "RANK_METHOD_DEFAULT should not exist"
    except ImportError:
        logger.success("  ✓ RANK_METHOD_DEFAULT correctly removed")

    try:
        from ranks.constants import AUTHOR_SORT_FIELD_DEFAULT

        assert False, "AUTHOR_SORT_FIELD_DEFAULT should not exist"
    except ImportError:
        logger.success("  ✓ AUTHOR_SORT_FIELD_DEFAULT correctly removed")

    # Test that elastics.videos.constants no longer re-exports from ranks
    from elastics.videos import constants as ev_constants

    # These should NOT be attributes of ev_constants anymore
    assert not hasattr(ev_constants, "RANK_TOP_K") or "RANK_TOP_K" not in dir(
        ev_constants
    ), "RANK_TOP_K should not be re-exported from elastics.videos.constants"
    logger.success("  ✓ elastics.videos.constants cleaned of ranks re-exports")

    logger.success("\n✓ Constants refactoring tests completed!")


def test_filter_only_search():
    """Test filter-only search functionality (no keywords, just filters)."""
    logger.note("> Testing filter-only search...")

    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Test 1: Single user filter
    logger.hint('\n[Test 1] Single user filter: u="影视飓风"')
    res = searcher.filter_only_search(
        query='u="影视飓风"',
        limit=50,
        verbose=False,
    )
    total_hits = res.get("total_hits", 0)
    return_hits = res.get("return_hits", 0)
    narrow_filter = res.get("narrow_filter", False)

    logger.mesg(f"  total_hits: {total_hits}")
    logger.mesg(f"  return_hits: {return_hits}")
    logger.mesg(f"  narrow_filter: {narrow_filter}")

    assert narrow_filter == True, "Should detect narrow filter for user query"
    assert return_hits > 0, "Should return results"
    logger.success("  ✓ Single user filter test passed")

    # Test 2: Multiple user filter
    logger.hint('\n[Test 2] Multiple user filter: u=["影视飓风","修电脑的张哥"]')
    res = searcher.filter_only_search(
        query='u=["影视飓风","修电脑的张哥"]',
        limit=1000,
        verbose=False,
    )
    total_hits = res.get("total_hits", 0)
    return_hits = res.get("return_hits", 0)
    narrow_filter = res.get("narrow_filter", False)

    logger.mesg(f"  total_hits: {total_hits}")
    logger.mesg(f"  return_hits: {return_hits}")
    logger.mesg(f"  narrow_filter: {narrow_filter}")

    assert narrow_filter == True, "Should detect narrow filter for multiple users"
    # For narrow filter, return_hits should equal total_hits (up to limit)
    expected_return = min(total_hits, 1000)
    assert (
        return_hits == expected_return
    ), f"Expected {expected_return} return_hits, got {return_hits}"
    logger.success("  ✓ Multiple user filter test passed")

    # Test 3: Range filter only (NOT narrow)
    logger.hint("\n[Test 3] Range filter only: v>=1000 (should NOT be narrow)")
    res = searcher.filter_only_search(
        query="v>=1000",
        limit=50,
        verbose=False,
    )
    narrow_filter = res.get("narrow_filter", True)  # Default True to fail if not set

    logger.mesg(f"  narrow_filter: {narrow_filter}")

    assert narrow_filter == False, "Range-only filter should NOT be narrow"
    logger.success("  ✓ Range filter test passed")

    # Test 4: Negative user filter (NOT narrow)
    logger.hint('\n[Test 4] Negative user filter: u!="影视飓风" (should NOT be narrow)')
    res = searcher.filter_only_search(
        query='u!="影视飓风"',
        limit=50,
        verbose=False,
    )
    narrow_filter = res.get("narrow_filter", True)

    logger.mesg(f"  narrow_filter: {narrow_filter}")

    assert narrow_filter == False, "Negative user filter should NOT be narrow"
    logger.success("  ✓ Negative user filter test passed")

    logger.success("\n✓ Filter-only search tests completed!")


def test_compute_passage():
    """Test compute_passage function for reranking."""
    logger.note("> Testing compute_passage function...")

    from ranks.reranker import compute_passage

    # Test 1: Complete hit with all fields
    logger.hint("\n[Test 1] Complete hit with all fields")
    hit1 = {
        "bvid": "BV123",
        "title": "Test Video Title",
        "owner": {"name": "TestAuthor", "mid": 12345},
        "tags": "tag1, tag2, tag3",
        "desc": "This is a test description.",
    }
    passage1 = compute_passage(hit1)
    logger.mesg(f"  Passage: {passage1}")

    assert "【TestAuthor】" in passage1, "Should contain author in 【】 brackets"
    assert "Test Video Title" in passage1, "Should contain title"
    assert "(tag1, tag2, tag3)" in passage1, "Should contain tags in ()"
    assert "This is a test description." in passage1, "Should contain description"
    logger.success("  ✓ Complete hit test passed")

    # Test 2: Hit with nested owner structure
    logger.hint("\n[Test 2] Hit with nested owner structure")
    hit2 = {
        "bvid": "BV456",
        "title": "Another Video",
        "owner": {"name": "影视飓风", "mid": 946974},
        "tags": "",  # Empty tags
        "desc": "-",  # Placeholder desc
    }
    passage2 = compute_passage(hit2)
    logger.mesg(f"  Passage: {passage2}")

    assert "【影视飓风】" in passage2, "Should contain Chinese author name"
    assert "Another Video" in passage2, "Should contain title"
    assert "()" not in passage2, "Should not include empty tags"
    assert "-" not in passage2, "Should not include placeholder desc"
    logger.success("  ✓ Nested owner test passed")

    # Test 3: Hit with missing fields
    logger.hint("\n[Test 3] Hit with missing fields")
    hit3 = {"bvid": "BV789", "title": "Minimal Video"}
    passage3 = compute_passage(hit3)
    logger.mesg(f"  Passage: {passage3}")

    assert "Minimal Video" in passage3, "Should contain title"
    assert len(passage3) > 0, "Should produce non-empty passage"
    logger.success("  ✓ Missing fields test passed")

    # Test 4: Passage truncation
    logger.hint("\n[Test 4] Passage truncation")
    hit4 = {
        "bvid": "BV999",
        "title": "X" * 3000,  # Very long title
        "owner": {"name": "Author"},
        "tags": "Y" * 1000,
        "desc": "Z" * 1000,
    }
    passage4 = compute_passage(hit4, max_passage_len=500)
    logger.mesg(f"  Passage length: {len(passage4)} (max: 500)")

    assert len(passage4) <= 500, "Should truncate to max_passage_len"
    logger.success("  ✓ Truncation test passed")

    logger.success("\n✓ compute_passage tests completed!")


def test_narrow_filter_detection():
    """Test has_narrow_filters detection with various filter combinations."""
    logger.note("> Testing narrow filter detection...")

    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Test cases: (filter_clauses, expected_is_narrow, description)
    test_cases = [
        # Positive user filter - IS narrow
        (
            [{"term": {"owner.name.keyword": "影视飓风"}}],
            True,
            "term owner.name.keyword (positive)",
        ),
        (
            [{"terms": {"owner.name.keyword": ["影视飓风", "修电脑的张哥"]}}],
            True,
            "terms owner.name.keyword (positive)",
        ),
        # Positive bvid filter - IS narrow
        ([{"term": {"bvid.keyword": "BV123"}}], True, "term bvid.keyword (positive)"),
        (
            [{"terms": {"bvid.keyword": ["BV123", "BV456"]}}],
            True,
            "terms bvid.keyword (positive)",
        ),
        # Positive mid filter - IS narrow
        ([{"term": {"owner.mid": 946974}}], True, "term owner.mid (positive)"),
        # Range filters - NOT narrow
        ([{"range": {"stat.view": {"gte": 1000}}}], False, "range stat.view"),
        ([{"range": {"pubdate": {"gte": 1700000000}}}], False, "range pubdate"),
        # Negative user filter (must_not wrapped) - NOT narrow
        # Note: must_not doesn't appear in filter_clauses from get_filters_from_query
        (
            [{"bool": {"must_not": [{"term": {"owner.name.keyword": "影视飓风"}}]}}],
            False,
            "bool.must_not owner.name.keyword (negative)",
        ),
        # Mixed positive user + range - IS narrow
        (
            [
                {"term": {"owner.name.keyword": "影视飓风"}},
                {"range": {"stat.view": {"gte": 1000}}},
            ],
            True,
            "positive user + range",
        ),
        # Empty filter - NOT narrow
        ([], False, "empty filters"),
        # Nested bool.filter with narrow - IS narrow
        (
            [{"bool": {"filter": {"term": {"owner.name.keyword": "影视飓风"}}}}],
            True,
            "nested bool.filter with narrow term",
        ),
    ]

    all_passed = True
    for filter_clauses, expected, description in test_cases:
        actual = searcher.has_narrow_filters(filter_clauses)
        status = "✓" if actual == expected else "×"
        if actual != expected:
            all_passed = False
        logger.mesg(f"  {status} {description}: expected={expected}, actual={actual}")

    assert all_passed, "Some narrow filter detection tests failed"
    logger.success("\n✓ Narrow filter detection tests completed!")


def test_filter_only_explore():
    """Test filter-only explore through unified_explore."""
    logger.note("> Testing filter-only explore...")

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX,
        elastic_env_name=ELASTIC_DEV,
    )

    # Test: Multiple authors with q=w (filter-only)
    logger.hint('\n[Test] Multiple authors: u=["影视飓风","修电脑的张哥"] q=w')

    result = explorer.unified_explore(
        query='u=["影视飓风","修电脑的张哥"] q=w',
        verbose=False,
        rank_top_k=1000,
    )

    logger.mesg(f"  Status: {result.get('status')}")

    # Find relevant step info
    for step in result.get("data", []):
        step_name = step.get("name")
        output = step.get("output", {})

        if step_name == "most_relevant_search":
            total_hits = output.get("total_hits", 0)
            return_hits = output.get("return_hits", 0)
            filter_only = output.get("filter_only", False)
            narrow_filter = output.get("narrow_filter", False)

            logger.mesg(f"  Step: {step_name}")
            logger.mesg(f"    total_hits: {total_hits}")
            logger.mesg(f"    return_hits: {return_hits}")
            logger.mesg(f"    filter_only: {filter_only}")
            logger.mesg(f"    narrow_filter: {narrow_filter}")

            assert filter_only == True, "Should be filter_only search"
            assert narrow_filter == True, "Should detect narrow filter"
            # For narrow filter, should return all hits
            if total_hits <= 1000:
                assert return_hits == total_hits, f"Should return all {total_hits} hits"

        elif step_name == "group_hits_by_owner":
            authors = output.get("authors", [])
            logger.mesg(f"  Step: {step_name}")
            logger.mesg(f"    authors count: {len(authors)}")
            if authors:
                # authors is a list of author dicts
                for author_data in authors[:2]:
                    author_name = author_data.get("name", "Unknown")
                    hits = author_data.get("hits", [])
                    logger.mesg(f"    - {author_name}: {len(hits)} videos")

    logger.success("\n✓ Filter-only explore test completed!")


def test_owner_name_keyword_boost():
    """Test that keywords matching owner.name get boosted in reranking.

    This tests the fix for: query `张哥 u=["修电脑的张哥","靓女维修佬"] q=vr`
    should boost hits from "修电脑的张哥" because "张哥" matches their owner.name.
    """
    logger.note("> Testing owner.name keyword boost in reranking...")

    from ranks.reranker import check_keyword_match, EmbeddingReranker

    # Test 1: check_keyword_match function
    logger.hint("\n[Test 1] check_keyword_match with owner.name")

    test_cases = [
        ("修电脑的张哥", ["张哥"], True, 1),
        ("影视飓风", ["张哥"], False, 0),
        ("靓女维修佬", ["张哥"], False, 0),
        ("修电脑的张哥", ["修电脑", "张哥"], True, 2),
        ("TestAuthor", ["test"], True, 1),  # case-insensitive
    ]

    for text, keywords, expected_match, expected_count in test_cases:
        has_match, match_count = check_keyword_match(text, keywords)
        status = (
            "✓"
            if (has_match == expected_match and match_count == expected_count)
            else "×"
        )
        logger.mesg(
            f"  {status} '{text}' + {keywords} -> match={has_match}, count={match_count}"
        )

    # Test 2: Simulated reranking with owner.name boost
    logger.hint("\n[Test 2] Simulated hits with owner.name keyword boost")

    # Create mock hits - one with matching owner, one without
    mock_hits = [
        {
            "bvid": "BV001",
            "title": "显卡维修教程",
            "owner": {"name": "靓女维修佬", "mid": 1001},
            "tags": "显卡, 维修",
            "desc": "显卡维修视频",
        },
        {
            "bvid": "BV002",
            "title": "显卡维修入门",
            "owner": {"name": "修电脑的张哥", "mid": 1002},
            "tags": "显卡, 维修",
            "desc": "显卡维修基础",
        },
    ]

    # Keywords: "张哥"
    keywords = ["张哥"]

    # Check which hit should get boosted
    from tclogger import dict_get

    for hit in mock_hits:
        owner_name = dict_get(hit, "owner.name", default="", sep=".")
        has_match, match_count = check_keyword_match(owner_name, keywords)
        logger.mesg(
            f"  {hit['bvid']} ({owner_name}): keyword_match={has_match}, count={match_count}"
        )

        if owner_name == "修电脑的张哥":
            assert has_match == True, "张哥 should match 修电脑的张哥"
            assert match_count == 1, "Should have 1 match"
        else:
            assert has_match == False, "张哥 should NOT match 靓女维修佬"

    logger.success("  ✓ Owner.name keyword matching works correctly")

    # Test 3: Real search with q=vr
    logger.hint(
        '\n[Test 3] Real search: 张哥 u=["修电脑的张哥","靓女维修佬"] q=vr d=1m'
    )

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX,
        elastic_env_name=ELASTIC_DEV,
    )
    if not explorer.embed_client.is_available():
        import pytest

        pytest.skip(
            "embed client unavailable; owner.name keyword boost test requires live embeddings"
        )

    result = explorer.unified_explore(
        query='张哥 u=["修电脑的张哥","靓女维修佬"] q=vr d=1m',
        verbose=False,
        rank_top_k=50,
    )

    # Find knn_search step to check keyword_boost
    for step in result.get("data", []):
        if step.get("name") == "knn_search":
            hits = step.get("output", {}).get("hits", [])

            if not hits:
                logger.warn("  No hits returned")
                continue

            # Check keyword_boost values
            logger.mesg(f"  Total hits: {len(hits)}")

            zhang_ge_boosted = 0
            liang_nv_boosted = 0

            for hit in hits[:10]:  # Check first 10 hits
                owner_name = dict_get(hit, "owner.name", default="", sep=".")
                keyword_boost = hit.get("keyword_boost", 1)
                cosine_sim = hit.get("cosine_similarity", 0)
                rerank_score = hit.get("rerank_score", 0)

                if "张哥" in owner_name:
                    if keyword_boost > 1:
                        zhang_ge_boosted += 1
                else:
                    if keyword_boost > 1:
                        liang_nv_boosted += 1

                logger.mesg(
                    f"    {hit['bvid'][:13]} ({owner_name[:10]:10}) "
                    f"cosine={cosine_sim:.4f} boost={keyword_boost:.2f} "
                    f"score={rerank_score:.4f}"
                )

            # Verify 修电脑的张哥 hits have keyword_boost > 1
            logger.mesg(f"\n  修电脑的张哥 boosted (in top 10): {zhang_ge_boosted}")
            logger.mesg(f"  靓女维修佬 boosted (in top 10): {liang_nv_boosted}")

            # The fix should make 修电脑的张哥 hits have boost > 1
            assert (
                zhang_ge_boosted > 0
            ), "修电脑的张哥 hits should have keyword_boost > 1"
            logger.success("  ✓ 修电脑的张哥 hits correctly boosted by keyword '张哥'")

    logger.success("\n✓ Owner.name keyword boost test completed!")
