from fastapi.encoders import jsonable_encoder
from tclogger import logger, logstr, dict_to_str, brk

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.explorer import VideoExplorer
from converters.query.dsl import MultiMatchQueryDSLConstructor
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.query.filter import QueryFilterExtractor
from elastics.tests.test_videos import make_searcher, make_explorer, author_values, author_items


def test_knn_filter_bug():
    """Test for the KNN filter bug reported by user.

    Bug description:
    - Query `u="红警HBK08" q=vr` returns 94 docs (all docs from user "红警HBK08")
    - Query `一小块地 u="红警HBK08" q=vr` returns only 3 docs regardless of keywords

    Root cause analysis:
    When KNN search is done with filters, ES returns docs that are both:
    1. Within the top-k nearest neighbors to the query vector
    2. Match the filter criteria

    If a user has 94 documents but only 3 are in the global top-k nearest neighbors
    to the keyword embedding, only 3 are returned.

    Solution:
    For narrow filters (small result set), we should:
    1. First fetch all matching doc IDs using filter-only search
    2. Then compute vector similarity within that filtered set
    """
    from tclogger import dict_to_str

    logger.note("> Testing KNN filter bug...")

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Test 1: Filter only (no keywords) - should return all 94 docs
    query1 = 'u="红警HBK08" q=vr'
    logger.hint(f"\nTest 1: Filter only - [{query1}]")
    res1 = explorer.unified_explore(query1, rank_top_k=100, verbose=False)
    step1 = next(
        (
            s
            for s in res1.get("data", [])
            if s["name"] in ["knn_search", "most_relevant_search"]
        ),
        None,
    )
    if step1:
        hits1 = step1.get("output", {}).get("return_hits", 0)
        total1 = step1.get("output", {}).get("total_hits", 0)
        filter_only1 = step1.get("output", {}).get("filter_only", False)
        logger.mesg(
            f"  total_hits={total1}, return_hits={hits1}, filter_only={filter_only1}"
        )
    else:
        logger.warn("  No search step found")
        hits1 = 0
        total1 = 0

    # Test 2: With keywords - currently returns only 3 docs (BUG)
    query2 = '一小块地 u="红警HBK08" q=vr'
    logger.hint(f"\nTest 2: With keywords - [{query2}]")
    res2 = explorer.unified_explore(query2, rank_top_k=100, verbose=True)
    step2 = next(
        (
            s
            for s in res2.get("data", [])
            if s["name"] in ["knn_search", "most_relevant_search"]
        ),
        None,
    )
    if step2:
        hits2 = step2.get("output", {}).get("return_hits", 0)
        total2 = step2.get("output", {}).get("total_hits", 0)
        filter_only2 = step2.get("output", {}).get("filter_only", False)
        logger.mesg(
            f"  total_hits={total2}, return_hits={hits2}, filter_only={filter_only2}"
        )
    else:
        logger.warn("  No search step found")
        hits2 = 0
        total2 = 0

    # Test 3: With different keywords - should also return 3 docs if bug exists
    query3 = '红警08 u="红警HBK08" q=vr'
    logger.hint(f"\nTest 3: Different keywords - [{query3}]")
    res3 = explorer.unified_explore(query3, rank_top_k=100, verbose=False)
    step3 = next(
        (
            s
            for s in res3.get("data", [])
            if s["name"] in ["knn_search", "most_relevant_search"]
        ),
        None,
    )
    if step3:
        hits3 = step3.get("output", {}).get("return_hits", 0)
        total3 = step3.get("output", {}).get("total_hits", 0)
        logger.mesg(f"  total_hits={total3}, return_hits={hits3}")
    else:
        logger.warn("  No search step found")
        hits3 = 0
        total3 = 0

    # Summary and verification
    logger.hint("\n> Summary:")
    logger.mesg(f"  Filter only (no keywords): {total1} total hits, {hits1} returned")
    logger.mesg(f"  With keywords '一小块地': {total2} total hits, {hits2} returned")
    logger.mesg(f"  With keywords '红警08': {total3} total hits, {hits3} returned")

    # The bug is confirmed if:
    # 1. Filter only returns many docs (94+)
    # 2. With keywords returns very few (3)
    if total1 > 10 and total2 <= 10 and total1 != total2:
        logger.warn("  × BUG CONFIRMED: Keywords cause significant result reduction")
        logger.warn(f"    Expected: ~{total1} hits, Got: {total2} hits")
    elif total1 > 10 and total2 > 10:
        logger.success("  ✓ Bug appears to be fixed!")
    else:
        logger.mesg("  ⓘ Inconclusive - user may not have enough docs")


def test_knn_explore_rerank_debug():
    """Debug test for KNN explore with rerank (q=vr mode).

    Tests different queries to identify memory/performance issues.
    """
    import time
    import gc
    import tracemalloc

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Queries to test - some might cause issues
    test_queries = [
        # These work fine
        # "影视飓风 v>100",
        # "马其顿 v>100",
        # This causes memory issues
        "圣甲虫 v>100",
    ]

    for query in test_queries:
        logger.note(f"\n{'='*60}")
        logger.note(f"> Testing q=vr mode with query: [{query}]")
        logger.note(f"{'='*60}")

        # Start memory tracking
        gc.collect()
        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            explore_res = explorer.knn_explore(
                query=query,
                enable_rerank=True,
                rerank_max_hits=1000,
                rank_top_k=50,
                group_owner_limit=10,
                verbose=True,
            )

            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            logger.success(f"\n> Query completed!")
            logger.mesg(f"  Status: {explore_res.get('status', 'N/A')}")
            logger.mesg(f"  Total time: {end_time - start_time:.2f}s")
            logger.mesg(f"  Memory current: {current / 1024 / 1024:.2f} MB")
            logger.mesg(f"  Memory peak: {peak / 1024 / 1024:.2f} MB")

            # Print step timings
            for step_res in explore_res.get("data", []):
                stage_name = step_res.get("name", "unknown")
                output = step_res.get("output", {})
                if isinstance(output, dict) and "perf" in output:
                    perf = output["perf"]
                    logger.hint(f"  {stage_name} perf: {perf}")

        except Exception as e:
            tracemalloc.stop()
            logger.warn(f"× Query failed: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

        # Force garbage collection between queries
        gc.collect()
        logger.mesg("  GC collected")


def test_rerank_step_by_step():
    """Step-by-step test of the rerank process to identify bottleneck."""
    import time
    import gc

    from converters.embed.embed_client import get_embed_client
    from ranks.reranker import get_reranker, compute_passage

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    test_query = "圣甲虫 v>100"

    logger.note(f"> Step-by-step rerank test for: [{test_query}]")

    # Step 1: Get query info and filters
    logger.hint("\n[Step 1] Parse query and get filters...")
    start = time.perf_counter()
    query_info, filter_clauses = explorer.get_filters_from_query(
        query=test_query,
        extra_filters=[],
    )
    logger.mesg(f"  Time: {(time.perf_counter() - start)*1000:.2f}ms")

    words_expr = query_info.get("words_expr", "")
    keywords_body = query_info.get("keywords_body", [])
    embed_text = " ".join(keywords_body) if keywords_body else words_expr or test_query

    logger.mesg(f"  words_expr: {words_expr}")
    logger.mesg(f"  keywords_body: {keywords_body}")
    logger.mesg(f"  embed_text: {embed_text}")
    logger.mesg(f"  filter_clauses: {len(filter_clauses)}")

    # Step 2: Get embedding vector
    logger.hint("\n[Step 2] Get embedding vector...")
    start = time.perf_counter()
    embed_client = get_embed_client()
    query_hex = embed_client.text_to_hex(embed_text)
    logger.mesg(f"  Time: {(time.perf_counter() - start)*1000:.2f}ms")
    logger.mesg(f"  Hex length: {len(query_hex) if query_hex else 0}")

    if not query_hex:
        logger.warn("× Failed to get embedding")
        return

    query_vector = embed_client.hex_to_byte_array(query_hex)
    logger.mesg(f"  Vector length: {len(query_vector)}")

    # Step 3: KNN search
    logger.hint("\n[Step 3] KNN search...")
    start = time.perf_counter()
    knn_res = explorer.knn_search(
        query=test_query,
        limit=1000,
        rank_top_k=1000,
        skip_ranking=True,
        verbose=False,
    )
    logger.mesg(f"  Time: {(time.perf_counter() - start)*1000:.2f}ms")

    knn_hits = knn_res.get("hits", [])
    logger.mesg(f"  Total hits: {knn_res.get('total_hits', 0)}")
    logger.mesg(f"  Return hits: {len(knn_hits)}")

    if not knn_hits:
        logger.warn("× No KNN results")
        return

    # Step 4: Prepare passages
    logger.hint("\n[Step 4] Prepare passages...")
    start = time.perf_counter()
    valid_passages = []
    valid_indices = []
    for i, hit in enumerate(knn_hits):
        passage = compute_passage(hit)
        if passage:
            valid_indices.append(i)
            valid_passages.append(passage)
    logger.mesg(f"  Time: {(time.perf_counter() - start)*1000:.2f}ms")
    logger.mesg(f"  Valid passages: {len(valid_passages)}")

    # Show passage length distribution
    if valid_passages:
        lengths = [len(p) for p in valid_passages]
        logger.mesg(
            f"  Passage length - min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)/len(lengths):.0f}"
        )
        total_chars = sum(lengths)
        logger.mesg(f"  Total chars: {total_chars}")

    # Step 5: Call rerank API
    logger.hint("\n[Step 5] Call rerank API (THIS MAY HANG)...")
    logger.warn("  If this hangs, the issue is in tfmx.rerank()")
    start = time.perf_counter()

    try:
        rankings = embed_client.rerank(embed_text, valid_passages)
        logger.mesg(f"  Time: {(time.perf_counter() - start)*1000:.2f}ms")
        logger.mesg(f"  Rankings count: {len(rankings) if rankings else 0}")

        if rankings:
            # Show some sample rankings
            logger.mesg(f"  First 3 rankings: {rankings[:3]}")
    except Exception as e:
        logger.warn(f"× Rerank failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    del valid_passages
    del knn_hits
    gc.collect()

    logger.success("\n> Step-by-step test completed!")


def test_highlight_bug():
    """Test for the highlighting bug with short keywords.

    Bug description:
    - Query `go u="影视飓风" q=vr` returns results containing "GoPro"
    - But "GoPro" is not highlighted with the "go" keyword

    Root cause:
    - The CharMatchHighlighter uses min_alpha_match=3 by default
    - For keyword "go" (2 chars), prefix match with "gopro" only matches 2 chars
    - 2 < 3 (min_alpha_match), so no match is found

    Expected behavior:
    - If keyword is shorter than min_alpha_match, and the keyword is a complete
      prefix of the text token, it should still match
    """
    from tclogger import dict_to_str
    from converters.highlight.char_match import CharMatchHighlighter, tokenize_to_units

    logger.note("> Testing highlight bug with short keywords...")

    highlighter = CharMatchHighlighter()

    # Test 1: Short keyword "go" should match "GoPro"
    test_cases = [
        # (text, keywords_query, expected_contains, description)
        ("GoPro运动相机", "go", "<hit>Go</hit>", "short keyword 'go' prefix match"),
        ("GoPro Hero 12", ["go"], "<hit>Go</hit>", "list input with short keyword"),
        ("gopro测试", "GO", "<hit>go</hit>", "case insensitive match"),
        ("一个好相机GoPro", "go", "<hit>Go</hit>", "match in middle of text"),
        ("hello world", "he", "<hit>he</hit>", "'he' should match 'hello'"),
        (
            "testing",
            "test",
            "<hit>test</hit>",
            "'test' should match 'testing' (4 chars)",
        ),
    ]

    passed = 0
    failed = 0

    for text, keywords, expected, description in test_cases:
        result = highlighter.highlight(text, keywords)
        if expected in result:
            logger.success(f"  ✓ {description}")
            logger.mesg(f"    Input: [{text}] + keywords={keywords}")
            logger.mesg(f"    Output: {result}")
            passed += 1
        else:
            logger.warn(f"  ✗ {description}")
            logger.mesg(f"    Input: [{text}] + keywords={keywords}")
            logger.mesg(f"    Output: {result}")
            logger.mesg(f"    Expected to contain: {expected}")
            failed += 1

    # Test 2: Integrated test with knn_explore
    logger.hint('\n> Testing with knn_explore (go u="影视飓风" q=vr)...')

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    query = 'go u="影视飓风" q=vr'
    res = explorer.unified_explore(query, rank_top_k=50, verbose=False)

    # Find the group_hits_by_owner step to check highlights
    group_step = next(
        (s for s in res.get("data", []) if s["name"] == "group_hits_by_owner"),
        None,
    )

    if group_step:
        authors = group_step.get("output", {}).get("authors", [])
        gopro_found = False
        gopro_highlighted = False

        for author_name, author_data in author_items(authors):
            hits = author_data.get("hits", [])
            for hit in hits:
                title = hit.get("title", "")
                # Check if GoPro is in title
                if "gopro" in title.lower() or "go pro" in title.lower():
                    gopro_found = True
                    # Check if it's highlighted
                    highlights = hit.get("highlights", {})
                    merged = highlights.get("merged", {})
                    title_highlight = merged.get("title", [""])[0]
                    if "<hit>" in title_highlight.lower():
                        gopro_highlighted = True
                        logger.mesg(f"    Found highlighted: {title_highlight}")
                    else:
                        logger.mesg(f"    Found but NOT highlighted: {title}")

        if gopro_found:
            if gopro_highlighted:
                logger.success("  ✓ GoPro found and highlighted correctly!")
                passed += 1
            else:
                logger.warn("  ✗ GoPro found but NOT highlighted (BUG!)")
                failed += 1
        else:
            logger.hint("  ⓘ No GoPro videos found in results (can't verify)")
    else:
        logger.warn("  ✗ No group_hits_by_owner step found")
        failed += 1

    # Summary
    logger.hint(f"\n> Summary: {passed} passed, {failed} failed")
    if failed == 0:
        logger.success("  ✓ All highlight tests passed!")
    else:
        logger.warn(f"  ✗ {failed} tests failed - bug needs fixing")


def test_user_filter_coverage():
    """Test for user filter coverage with dual-sort approach.

    User request:
    - For u=... filters, return 2000*N most recent (by pubdate) AND
      2000*N highest views (by stat.view), where N = number of owners
    - Deduplicate results since there's overlap

    This test verifies that the filter-first approach returns enough videos
    from a user to provide good coverage for semantic search.

    Note: Dual-sort is only used when there ARE keywords in the query,
    because that's when KNN search would be used and could miss documents.
    """
    from tclogger import dict_to_str

    logger.note("> Testing user filter coverage with dual-sort approach...")

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Test with a query that has KEYWORDS + user filter
    # This triggers KNN search which uses dual_sort_filter_search for narrow filters
    query = 'go u="影视飓风" q=vr'
    logger.hint(f"\nTest query WITH keywords: [{query}]")

    res = explorer.unified_explore(query, rank_top_k=100, verbose=True)

    # Find knn_search step
    knn_step = next(
        (s for s in res.get("data", []) if s["name"] == "knn_search"),
        None,
    )

    if knn_step:
        output = knn_step.get("output", {})
        total_hits = output.get("total_hits", 0)
        return_hits = output.get("return_hits", 0)
        narrow_filter_used = output.get("narrow_filter_used", False)
        dual_sort_used = output.get("dual_sort_used", False)
        dual_sort_info = output.get("dual_sort_info", {})

        logger.mesg(f"  total_hits: {total_hits}")
        logger.mesg(f"  return_hits: {return_hits}")
        logger.mesg(f"  narrow_filter_used: {narrow_filter_used}")
        logger.mesg(f"  dual_sort_used: {dual_sort_used}")
        if dual_sort_info:
            logger.mesg(f"  dual_sort_info: {dual_sort_info}")
            # The real coverage is merged_unique / total_hits
            merged_unique = dual_sort_info.get("merged_unique", 0)
            popular_skipped = dual_sort_info.get("popular_skipped", False)
            if total_hits > 0:
                fetch_coverage = merged_unique / total_hits * 100
                logger.mesg(
                    f"  fetch_coverage: {fetch_coverage:.1f}% ({merged_unique}/{total_hits})"
                )
                if popular_skipped:
                    logger.hint("  ⓘ Popular query skipped (all docs from recent)")
                if fetch_coverage >= 90:
                    logger.success(f"  ✓ Good fetch coverage ({fetch_coverage:.1f}%)")
                elif fetch_coverage >= 50:
                    logger.hint(f"  ⓘ Medium fetch coverage ({fetch_coverage:.1f}%)")
                else:
                    logger.warn(f"  ✗ Low fetch coverage ({fetch_coverage:.1f}%)")

        if total_hits > 0:
            # For narrow filters, return_hits should equal total_hits (no rank_top_k limit)
            coverage = return_hits / total_hits * 100
            if narrow_filter_used:
                logger.mesg(
                    f"  return_coverage: {coverage:.1f}% ({return_hits}/{total_hits})"
                )
                if return_hits == total_hits:
                    logger.success(
                        f"  ✓ All {total_hits} docs returned (no rank_top_k limit)"
                    )
                else:
                    logger.warn(f"  ✗ Expected {total_hits} docs, got {return_hits}")
            else:
                logger.mesg(f"  display_coverage: {coverage:.1f}% (rank_top_k limited)")
    else:
        logger.warn("  ✗ No knn_search step found")

    # Test with multiple users (also needs keywords to trigger KNN + dual-sort)
    # Note: Use comma separator for multiple users in DSL, not pipe (|)
    query2 = "相机 u=(影视飓风,老师好我叫何同学) q=vr"
    logger.hint(f"\nTest query with multiple users: [{query2}]")

    res2 = explorer.unified_explore(query2, rank_top_k=100, verbose=False)

    knn_step2 = next(
        (s for s in res2.get("data", []) if s["name"] == "knn_search"),
        None,
    )

    if knn_step2:
        output2 = knn_step2.get("output", {})
        total_hits2 = output2.get("total_hits", 0)
        return_hits2 = output2.get("return_hits", 0)
        dual_sort_info2 = output2.get("dual_sort_info", {})
        narrow_filter_used2 = output2.get("narrow_filter_used", False)
        logger.mesg(f"  total_hits: {total_hits2}")
        logger.mesg(f"  return_hits: {return_hits2}")
        if dual_sort_info2:
            logger.mesg(f"  dual_sort_info: {dual_sort_info2}")
            owner_count = dual_sort_info2.get("owner_count", 0)
            if owner_count == 2:
                logger.success(f"  ✓ Correctly detected 2 owners, limits scaled")
            else:
                logger.warn(f"  ✗ Expected 2 owners, got {owner_count}")
        # Verify all docs returned
        if narrow_filter_used2 and return_hits2 == total_hits2:
            logger.success(f"  ✓ All {total_hits2} docs returned for multi-user query")
        elif narrow_filter_used2:
            logger.warn(f"  ✗ Expected {total_hits2} docs, got {return_hits2}")
    else:
        logger.warn("  ✗ No knn_search step found for multi-user query")

    logger.success("\n> User filter coverage test completed!")


def test_vr_performance_with_narrow_filters():
    """Test q=vr performance with narrow filters (user/bvid).

    This test validates that:
    1. LSH embedding is SKIPPED for narrow filter queries (saves ~50-100ms)
    2. Performance tracking is properly captured at each step
    3. The rerank step timing is captured correctly

    The sample query `a7m3 u=["影视飓风","老师好我叫何同学"] q=vr` should:
    - Skip LSH (lsh_embedding_ms should be 0)
    - Use filter-first approach (dual_sort_filter_search)
    - Rerank with float embeddings for semantic similarity
    """
    from tclogger import dict_to_str

    logger.note("> Testing q=vr performance with narrow filters...")
    logger.note("> This test validates LSH skip optimization for narrow filters")

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Test queries with narrow filters and q=vr
    test_queries = [
        # (query, description, expect_lsh_skip)
        (
            'a7m3 u=["影视飓风","老师好我叫何同学"] q=vr',
            "narrow filter with multiple users",
            True,
        ),
        ('相机 u="影视飓风" q=vr', "narrow filter with single user", True),
        ("相机 q=vr", "broad query without user filter", False),
        ('u="红警HBK08" q=vr', "only user filter (no keywords)", True),
        ("红警08 q=vr", "broad query with keywords", False),
    ]

    for query, description, expect_lsh_skip in test_queries:
        logger.hint(f"\n> Test: {description}")
        logger.mesg(f"  Query: [{query}]")
        logger.mesg(f"  Expected LSH skip: {expect_lsh_skip}")

        result = explorer.unified_explore(
            query=query,
            rank_top_k=100,
            group_owner_limit=10,
            verbose=True,
        )

        status = result.get("status", "unknown")
        perf = result.get("perf", {})

        logger.mesg(f"  Status: {status}")
        logger.mesg(f"  Overall perf: {perf}")

        # Check if LSH was skipped for narrow filters
        lsh_ms = perf.get("lsh_embedding_ms", 0)
        filter_search_ms = perf.get("filter_search_ms", 0)
        knn_search_ms = perf.get("knn_search_ms", 0)
        rerank_ms = perf.get("rerank_ms", 0)
        total_ms = perf.get("total_ms", 0)

        if expect_lsh_skip:
            if lsh_ms == 0:
                logger.success(f"  ✓ LSH correctly skipped (0ms)")
            else:
                logger.warn(f"  ✗ LSH should be skipped but took {lsh_ms}ms")

            if filter_search_ms > 0:
                logger.success(f"  ✓ Filter-first search used ({filter_search_ms}ms)")
            else:
                logger.warn(f"  ✗ Filter-first search should be used")
        else:
            if lsh_ms > 0:
                logger.success(f"  ✓ LSH computed as expected ({lsh_ms}ms)")
            else:
                logger.warn(f"  ⓘ LSH was 0ms (possibly cached)")

            if knn_search_ms > 0:
                logger.success(f"  ✓ KNN search used ({knn_search_ms}ms)")

        # Check rerank timing
        if rerank_ms > 0:
            logger.mesg(f"  Rerank time: {rerank_ms}ms")
        else:
            logger.warn(f"  ⓘ Rerank time not captured or 0ms")

        logger.mesg(f"  Total time: {total_ms}ms")

        # Find the construct_knn_query step to check lsh_skipped flag
        for step in result.get("data", []):
            if step.get("name") == "construct_knn_query":
                output = step.get("output", {})
                lsh_skipped = output.get("lsh_skipped", False)
                narrow_filters = output.get("narrow_filters", False)

                if expect_lsh_skip:
                    if lsh_skipped:
                        logger.success(f"  ✓ lsh_skipped flag is True")
                    else:
                        logger.warn(f"  ✗ lsh_skipped flag should be True")

                if narrow_filters == expect_lsh_skip:
                    logger.success(
                        f"  ✓ narrow_filters flag matches expectation: {narrow_filters}"
                    )
                else:
                    logger.warn(
                        f"  ✗ narrow_filters={narrow_filters}, expected={expect_lsh_skip}"
                    )

                step_perf = output.get("perf", {})
                logger.mesg(f"  Step perf: {step_perf}")

            # Check rerank step for detailed timing
            elif step.get("name") == "rerank":
                output = step.get("output", {})
                rerank_perf = output.get("perf", {})
                reranked_count = output.get("reranked_count", 0)

                logger.mesg(f"  Rerank details:")
                logger.mesg(f"    - Reranked count: {reranked_count}")
                logger.mesg(f"    - Detailed perf: {rerank_perf}")

                # Check for detailed timing breakdown
                if rerank_perf:
                    passage_prep = rerank_perf.get("passage_prep_ms", 0)
                    rerank_call = rerank_perf.get("rerank_call_ms", 0)
                    keyword_scoring = rerank_perf.get("keyword_scoring_ms", 0)

                    logger.mesg(f"    - Passage prep: {passage_prep}ms")
                    logger.mesg(f"    - Rerank API call: {rerank_call}ms")
                    logger.mesg(f"    - Keyword scoring: {keyword_scoring}ms")

    logger.success("\n> q=vr performance test completed!")


def test_rerank_client_diagnostic():
    """Diagnostic test to trace exactly where time is spent in rerank calls.

    This test helps identify the 30x performance discrepancy:
    - Server side: ~659ms for /rerank
    - Client side: ~19,000ms for rerank_call_ms

    Traces:
    1. TEIClients initialization time
    2. First call (cold) vs subsequent calls (warm)
    3. Detailed timing within each call
    """
    import time

    logger.note("> Rerank client diagnostic test...")

    # Step 1: Test TEIClients initialization
    logger.hint("\n> Step 1: Testing TEIClients initialization time")
    from configs.envs import TEI_CLIENTS_ENDPOINTS

    logger.mesg(f"  Endpoints: {TEI_CLIENTS_ENDPOINTS}")

    t0 = time.perf_counter()
    from tfmx import TEIClients

    t1 = time.perf_counter()
    logger.mesg(f"  Import time: {(t1 - t0) * 1000:.2f}ms")

    t0 = time.perf_counter()
    clients = TEIClients(endpoints=TEI_CLIENTS_ENDPOINTS)
    t1 = time.perf_counter()
    logger.mesg(f"  TEIClients init time: {(t1 - t0) * 1000:.2f}ms")

    # Step 2: Test first rerank call (cold)
    logger.hint("\n> Step 2: First rerank call (cold)")

    # Create some sample passages
    passages = [f"这是测试文本 {i}" for i in range(100)]
    query = "测试查询"

    t0 = time.perf_counter()
    results = clients.rerank([query], passages)
    t1 = time.perf_counter()
    logger.mesg(f"  First rerank call (100 passages): {(t1 - t0) * 1000:.2f}ms")
    logger.mesg(f"  Results count: {len(results[0]) if results else 0}")

    # Step 3: Test second rerank call (warm)
    logger.hint("\n> Step 3: Second rerank call (warm)")

    t0 = time.perf_counter()
    results = clients.rerank([query], passages)
    t1 = time.perf_counter()
    logger.mesg(f"  Second rerank call (100 passages): {(t1 - t0) * 1000:.2f}ms")

    # Step 4: Test with more passages (like the real case)
    logger.hint("\n> Step 4: Larger rerank (1000 passages)")

    passages_large = [
        f"这是测试文本，包含更多内容用于模拟真实场景 {i}" for i in range(1000)
    ]

    t0 = time.perf_counter()
    results = clients.rerank([query], passages_large)
    t1 = time.perf_counter()
    logger.mesg(f"  Large rerank call (1000 passages): {(t1 - t0) * 1000:.2f}ms")

    # Step 5: Test with TextEmbedClient wrapper
    logger.hint("\n> Step 5: Using TextEmbedClient wrapper")
    from converters.embed.embed_client import TextEmbedClient

    t0 = time.perf_counter()
    client = TextEmbedClient(lazy_init=False)
    t1 = time.perf_counter()
    logger.mesg(f"  TextEmbedClient init (lazy=False): {(t1 - t0) * 1000:.2f}ms")

    t0 = time.perf_counter()
    rankings = client.rerank(query, passages, verbose=True)
    t1 = time.perf_counter()
    logger.mesg(f"  TextEmbedClient.rerank (100 passages): {(t1 - t0) * 1000:.2f}ms")

    # Step 6: Test LSH for comparison
    logger.hint("\n> Step 6: LSH timing for comparison")

    t0 = time.perf_counter()
    hex_vec = client.text_to_hex("测试查询")
    t1 = time.perf_counter()
    logger.mesg(f"  LSH first call: {(t1 - t0) * 1000:.2f}ms")

    t0 = time.perf_counter()
    hex_vec = client.text_to_hex("另一个查询")
    t1 = time.perf_counter()
    logger.mesg(f"  LSH second call: {(t1 - t0) * 1000:.2f}ms")

    logger.success("\n> Diagnostic test completed!")


def test_embed_client_keepalive():
    """Test the keepalive functionality of TextEmbedClient.

    This test verifies:
    1. warmup() correctly initializes the connection
    2. start_keepalive() starts the background thread
    3. Activity time tracking works correctly
    4. refresh_if_stale() detects stale connections
    """
    import time
    from converters.embed.embed_client import (
        TextEmbedClient,
        KEEPALIVE_TIMEOUT,
    )

    logger.note("> Testing TextEmbedClient keepalive functionality...")

    # Create a fresh client (not the singleton)
    client = TextEmbedClient(lazy_init=True)

    # Test 1: Warmup
    logger.hint("\n> Test 1: Warmup")
    t0 = time.time()
    success = client.warmup(verbose=True)
    t1 = time.time()
    logger.mesg(f"  Warmup success: {success}, took {(t1-t0)*1000:.0f}ms")

    # Test 2: Check stale detection
    logger.hint("\n> Test 2: Stale detection")
    is_stale = client._is_connection_stale()
    logger.mesg(f"  Is stale immediately after warmup: {is_stale} (expected: False)")

    # Manually make it look stale by setting old timestamp
    client._last_activity_time = time.time() - KEEPALIVE_TIMEOUT - 10
    is_stale = client._is_connection_stale()
    logger.mesg(f"  Is stale after timeout: {is_stale} (expected: True)")

    # Test 3: Refresh if stale
    logger.hint("\n> Test 3: Refresh if stale")
    t0 = time.time()
    result = client.refresh_if_stale(verbose=True)
    t1 = time.time()
    logger.mesg(f"  Refresh result: {result}, took {(t1-t0)*1000:.0f}ms")

    # Check if it's no longer stale
    is_stale = client._is_connection_stale()
    logger.mesg(f"  Is stale after refresh: {is_stale} (expected: False)")

    # Test 4: Start keepalive thread
    logger.hint("\n> Test 4: Keepalive thread")
    client.start_keepalive()
    logger.mesg("  Keepalive started")
    logger.mesg(f"  Thread alive: {client._keepalive_thread.is_alive()}")

    # Wait a bit and check if thread is still running
    time.sleep(0.5)
    logger.mesg(
        f"  Thread still alive after 0.5s: {client._keepalive_thread.is_alive()}"
    )

    # Test 5: Stop keepalive
    logger.hint("\n> Test 5: Stop keepalive")
    client.stop_keepalive()
    logger.mesg(f"  Thread stopped: {client._keepalive_thread is None}")

    # Cleanup
    client.close()
    logger.success("\n> Keepalive test completed!")
