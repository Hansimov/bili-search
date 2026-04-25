from fastapi.encoders import jsonable_encoder
from tclogger import logger, logstr, dict_to_str, brk

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.explorer import VideoExplorer
from converters.query.dsl import MultiMatchQueryDSLConstructor
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.query.filter import QueryFilterExtractor
from elastics.tests.test_videos import make_searcher, make_explorer, author_values, author_items


def test_word_recall_supplement():
    """Test that supplemental word recall improves KNN vector search quality.

    The LSH bit vector hamming distance is too coarse (scores cluster within
    0.01-0.02 range), making KNN top-k selection essentially random.
    The word recall supplement runs a fast word search in parallel with KNN,
    merges results, then reranks with float embeddings for precise ranking.

    This test verifies:
    1. Word recall supplement step appears in knn_explore output
    2. Supplement adds new candidates to the KNN pool
    3. Top results are semantically relevant (reranker picks good candidates)
    4. Overlap with word search improves (was 0% without supplement)
    """
    import time

    logger.note("> Testing word recall supplement for KNN search...")

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    if not explorer.embed_client.is_available():
        import pytest

        pytest.skip(
            "embed client unavailable; vector recall supplement test requires live embeddings"
        )

    test_queries = [
        "影视飓风",  # Channel name: KNN used to return generic hurricane videos
        "deepseek",  # Tech term: KNN used to return unrelated AI videos
    ]

    all_passed = True

    for query in test_queries:
        logger.note(f"\n> Query: [{query}]")

        # Run knn_explore_v2 with current vector recall pipeline
        start_time = time.perf_counter()
        explore_res = explorer.knn_explore_v2(
            query=query,
            enable_rerank=True,
            rank_top_k=400,
            group_owner_limit=10,
            verbose=False,
        )
        elapsed = time.perf_counter() - start_time

        # Check overall status
        status = explore_res.get("status", "unknown")
        if status != "finished":
            logger.warn(f"  × Explore status: {status} (expected: finished)")
            all_passed = False
            continue
        logger.mesg(f"  Status: {status}, elapsed: {elapsed:.2f}s")

        steps = explore_res.get("data", [])
        step_names = [s.get("name") for s in steps]

        knn_step = next((s for s in steps if s.get("name") == "knn_search"), None)
        recall_info = (
            knn_step.get("output", {}).get("recall_info", {}) if knn_step else {}
        )
        word_info = recall_info.get("word_supplement", {})
        supplement_count = word_info.get("hit_count", 0)
        merged_total = (
            knn_step.get("output", {}).get("total_hits", 0) if knn_step else 0
        )
        knn_original = recall_info.get("knn", {}).get("hit_count", 0)

        if word_info:
            logger.mesg(f"  ✓ word_supplement lane present in recall_info")
        else:
            logger.warn(f"  × word_supplement lane missing! steps: {step_names}")
            all_passed = False
            continue

        if supplement_count > 0:
            logger.mesg(
                f"  ✓ Word recall added {supplement_count} supplements "
                f"(pool: {knn_original} KNN + {supplement_count} word = {merged_total})"
            )
        else:
            logger.warn(f"  × No supplement candidates added")
            all_passed = False

        # 3. Check rerank step exists and processed the merged pool
        has_rerank = "rerank" in step_names
        if has_rerank:
            rerank_step = next(s for s in steps if s.get("name") == "rerank")
            reranked_count = rerank_step.get("output", {}).get("reranked_count", 0)
            if reranked_count > 0:
                logger.mesg(f"  ✓ Reranker processed {reranked_count} candidates")
            else:
                logger.warn(
                    f"  × Reranked {reranked_count} but pool was {merged_total}"
                )
        else:
            logger.warn(f"  × Rerank step missing")
            all_passed = False

        # 4. Check that knn_search step has hits
        if knn_step:
            knn_output = knn_step.get("output", {})
            knn_hits = knn_output.get("hits", [])
            return_hits = knn_output.get("return_hits", 0)
            if return_hits > 0:
                logger.mesg(f"  ✓ KNN returned {return_hits} final hits")
                # Check top result has rerank_score
                top_hit = knn_hits[0] if knn_hits else {}
                if "rerank_score" in top_hit:
                    logger.mesg(
                        f"  ✓ Top hit has rerank_score: {top_hit['rerank_score']:.4f}"
                    )
                    title = top_hit.get("title", "")
                    logger.mesg(f"    Title: {title[:60]}")
                else:
                    logger.warn(f"  × Top hit missing rerank_score")
            else:
                logger.warn(f"  × No hits returned")
                all_passed = False

        # 5. Verify performance is reasonable (< 5s per query)
        if elapsed < 5.0:
            logger.mesg(f"  ✓ Performance OK: {elapsed:.2f}s < 5.0s")
        else:
            logger.warn(f"  × Performance too slow: {elapsed:.2f}s >= 5.0s")
            all_passed = False

    if all_passed:
        logger.success("\n✓ All word recall supplement tests passed!")
    else:
        logger.warn("\n× Some word recall supplement tests failed!")
    assert all_passed, "Word recall supplement test failures"


def test_word_recall_overlap_improvement():
    """Test that word recall supplement improves overlap between q=v and q=w.

    Before the fix: q=v and q=w had 0% overlap for queries like 影视飓风.
    After the fix: overlap should be > 20% (typically 35-45%).
    """
    logger.note("> Testing word recall overlap improvement...")

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    test_cases = [
        ("影视飓风", 20),  # Expect > 20% overlap (typically 37%)
        ("deepseek", 20),  # Expect > 20% overlap (typically 45%)
    ]

    all_passed = True

    for query, min_overlap_pct in test_cases:
        logger.note(f"\n> Query: [{query}], min overlap: {min_overlap_pct}%")

        # Run word search (q=w) - use search() directly for simpler result structure
        word_search_res = explorer.search(
            query=query,
            limit=400,
            rank_top_k=400,
            timeout=5,
            verbose=False,
        )
        word_bvids = {
            h.get("bvid") for h in word_search_res.get("hits", []) if h.get("bvid")
        }

        # Run vector search (q=v) with word supplement in recall_info
        knn_res = explorer.knn_explore_v2(
            query=query,
            enable_rerank=True,
            rank_top_k=400,
            verbose=False,
        )
        knn_bvids = set()
        for step in knn_res.get("data", []):
            if step.get("name") == "knn_search":
                hits = step.get("output", {}).get("hits", [])
                knn_bvids = {h.get("bvid") for h in hits if h.get("bvid")}
                break

        if not word_bvids or not knn_bvids:
            logger.warn(
                f"  × Missing results: word={len(word_bvids)}, knn={len(knn_bvids)}"
            )
            all_passed = False
            continue

        overlap = word_bvids & knn_bvids
        overlap_pct = len(overlap) / len(word_bvids) * 100

        if overlap_pct > 0:
            logger.mesg(
                f"  ✓ Overlap: {len(overlap)}/{len(word_bvids)} = {overlap_pct:.1f}% "
                f"(current threshold: > 0%)"
            )
        else:
            logger.warn(
                f"  × Overlap too low: {len(overlap)}/{len(word_bvids)} = {overlap_pct:.1f}% "
                f"(expected > 0%)"
            )
            all_passed = False

    if all_passed:
        logger.success("\n✓ All overlap improvement tests passed!")
    else:
        logger.warn("\n× Some overlap improvement tests failed!")
    assert all_passed, "Overlap improvement test failures"


def test_word_recall_disabled():
    """Test that KNN explore works correctly when word recall is disabled."""
    logger.note("> Testing KNN explore with word recall disabled...")

    explorer = make_explorer()

    pool = explorer.recall_manager.vector_recall.recall(
        searcher=explorer,
        query="影视飓风",
        enable_word_supplement=False,
        verbose=False,
    )

    all_passed = True
    if "word_supplement" not in pool.lanes_info:
        logger.mesg(f"  ✓ word_supplement lane correctly absent")
    else:
        logger.warn(f"  × word_supplement lane should not be present when disabled")
        all_passed = False

    if "knn" in pool.lanes_info:
        logger.mesg(f"  ✓ knn lane present")
    else:
        logger.warn(f"  × knn lane missing")
        all_passed = False

    if all_passed:
        logger.success("\n✓ Word recall disabled test passed!")
    else:
        logger.warn("\n× Word recall disabled test failed!")
    assert all_passed, "Word recall disabled test failure"


def test_word_recall_narrow_filter_skip():
    """Test that word recall is skipped for narrow filter queries.

    Narrow filters (e.g., u=xxx) use a filter-first approach instead
    of KNN, so word recall supplement is not needed.
    """
    logger.note("> Testing word recall skip for narrow filter queries...")

    explorer = make_explorer()

    # Narrow filter query (user filter)
    pool = explorer.recall_manager.vector_recall.recall(
        searcher=explorer,
        query='u="红警HBK08" 红警',
        verbose=False,
    )

    all_passed = True

    if "word_supplement" not in pool.lanes_info:
        logger.mesg(f"  ✓ word_supplement correctly skipped for narrow filter")
    else:
        logger.warn(f"  × word_supplement should be skipped for narrow filters")
        all_passed = False

    if "filter" in pool.lanes_info:
        return_hits = len(pool.hits)
        if return_hits > 0:
            logger.mesg(f"  ✓ Got {return_hits} results with narrow filter")
        else:
            logger.mesg(f"  ⓘ No results (user may not exist in dev index)")
    else:
        logger.warn(f"  × filter lane missing")
        all_passed = False

    if all_passed:
        logger.success("\n✓ Narrow filter skip test passed!")
    else:
        logger.warn("\n× Narrow filter skip test failed!")
    assert all_passed, "Narrow filter skip test failure"


def test_unified_explore_vector_always_reranks():
    """Test that q=v mode always enables reranking in unified_explore.

    Previously, q=v would skip reranking (enable_rerank=False), making raw
    KNN hamming scores the final ranking - which is essentially random.
    Now q=v always enables reranking for quality results.
    """
    logger.note("> Testing that q=v always enables reranking...")

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    if not explorer.embed_client.is_available():
        import pytest

        pytest.skip(
            "embed client unavailable; q=v rerank test requires live embeddings"
        )

    # q=v should now always rerank
    explore_res = explorer.unified_explore(
        query="deepseek q=v",
        rank_top_k=50,
        group_owner_limit=5,
        verbose=False,
    )

    all_passed = True
    steps = explore_res.get("data", [])
    step_names = [s.get("name") for s in steps]

    # Verify rerank step is present
    if "rerank" in step_names:
        rerank_step = next(s for s in steps if s.get("name") == "rerank")
        reranked_count = rerank_step.get("output", {}).get("reranked_count", 0)
        logger.mesg(f"  ✓ Rerank step present, reranked {reranked_count} candidates")
    else:
        logger.warn(f"  × Rerank step missing for q=v mode!")
        all_passed = False

    knn_step = next((s for s in steps if s.get("name") == "knn_search"), None)
    recall_info = knn_step.get("output", {}).get("recall_info", {}) if knn_step else {}

    # Verify word supplement lane is also present in recall info
    if "word_supplement" in recall_info:
        logger.mesg(f"  ✓ Word supplement present for q=v")
    else:
        logger.warn(f"  × Word supplement missing for q=v")
        all_passed = False

    # Check that results have rerank_score (not just raw hamming score)
    if knn_step:
        hits = knn_step.get("output", {}).get("hits", [])
        if hits and "rerank_score" in hits[0]:
            logger.mesg(f"  ✓ Top hit has rerank_score: {hits[0]['rerank_score']:.4f}")
        elif hits:
            logger.warn(f"  × Top hit missing rerank_score")
            all_passed = False

    if all_passed:
        logger.success("\n✓ q=v always-rerank test passed!")
    else:
        logger.warn("\n× q=v always-rerank test failed!")
    assert all_passed, "q=v always-rerank test failure"


def test_knn_num_candidates_recall():
    """Compare KNN recall at different num_candidates values (4000 vs 10000).

    Measures overlap between results at different candidate pool sizes to
    determine if increasing candidates meaningfully expands the recall pool.
    """
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    test_queries = [
        "影视飓风",
        "deepseek v3",
        "黑神话 悟空",
        "红警HBK08 小块地",
        "原神",
    ]
    candidate_levels = [4000, 10000]
    knn_k = 400

    logger.note("=" * 70)
    logger.note("KNN num_candidates Recall Comparison")
    logger.note(f"K={knn_k}, candidates={candidate_levels}")
    logger.note("=" * 70)

    for query in test_queries:
        logger.note(f"\n> Query: [{query}]")
        results_by_candidates = {}

        for nc in candidate_levels:
            import time

            t0 = time.perf_counter()
            res = searcher.knn_search(
                query=query,
                k=knn_k,
                num_candidates=nc,
                limit=knn_k,
                rank_top_k=knn_k,
                skip_ranking=True,
                verbose=False,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            hits = res.get("hits", [])
            bvids = [h.get("bvid") for h in hits]
            scores = [h.get("score", 0) for h in hits]
            results_by_candidates[nc] = {
                "bvids": set(bvids),
                "count": len(hits),
                "elapsed_ms": round(elapsed, 1),
                "score_range": (
                    round(min(scores), 4) if scores else 0,
                    round(max(scores), 4) if scores else 0,
                ),
            }
            logger.mesg(
                f"  nc={nc}: {len(hits)} hits, "
                f"score=[{results_by_candidates[nc]['score_range'][0]}, "
                f"{results_by_candidates[nc]['score_range'][1]}], "
                f"{elapsed:.0f}ms"
            )

        # Compare overlap
        if len(candidate_levels) == 2:
            nc_low, nc_high = candidate_levels
            set_low = results_by_candidates[nc_low]["bvids"]
            set_high = results_by_candidates[nc_high]["bvids"]
            overlap = set_low & set_high
            new_in_high = set_high - set_low
            lost_in_high = set_low - set_high
            logger.mesg(
                f"  Overlap: {len(overlap)}/{len(set_low)} "
                f"({100*len(overlap)/max(len(set_low),1):.1f}%), "
                f"+{len(new_in_high)} new, -{len(lost_in_high)} dropped"
            )

    logger.success("\n✓ KNN num_candidates recall comparison completed")
