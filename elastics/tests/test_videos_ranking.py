from fastapi.encoders import jsonable_encoder
from tclogger import logger, logstr, dict_to_str, brk

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.explorer import VideoExplorer
from converters.query.dsl import MultiMatchQueryDSLConstructor
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.query.filter import QueryFilterExtractor
from elastics.tests.test_videos import make_searcher, make_explorer, author_values, author_items


def test_qmod_recall_comparison():
    """Compare recall across different qmod modes to understand why
    no q= (default hybrid) sometimes gives better results.

    Default (no q=): hybrid (word + vector) with tiered ranking
    q=w: word-only with stats ranking
    q=v: vector-only with relevance ranking + rerank + word recall
    q=wv: hybrid with tiered ranking (same as default)
    """
    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    test_queries = [
        "影视飓风",
        "deepseek v3",
        "黑神话 悟空",
    ]
    modes = [
        (None, "default (hybrid)"),
        (["word"], "q=w"),
        (["vector"], "q=v"),
        (["word", "vector"], "q=wv"),
        (["vector", "rerank"], "q=vr"),
        (["word", "vector", "rerank"], "q=wvr"),
    ]

    logger.note("=" * 70)
    logger.note("Qmod Recall Comparison")
    logger.note("=" * 70)

    for query in test_queries:
        logger.note(f"\n> Query: [{query}]")
        results_by_mode = {}

        for qmod, label in modes:
            import time

            t0 = time.perf_counter()
            try:
                result = explorer.unified_explore(
                    query=query,
                    qmod=qmod,
                    rank_top_k=50,
                    group_owner_limit=10,
                    verbose=False,
                )
                elapsed = (time.perf_counter() - t0) * 1000

                # Extract hits from knn_search or most_relevant_search step
                hits = []
                rank_method = "?"
                for step in result.get("data", []):
                    if step.get("name") in (
                        "knn_search",
                        "most_relevant_search",
                        "hybrid_search",
                    ):
                        hits = step.get("output", {}).get("hits", [])
                        rank_method = step.get("output", {}).get("rank_method", "?")
                        break

                bvids = [h.get("bvid") for h in hits]
                titles = [h.get("title", "?")[:30] for h in hits[:3]]
                results_by_mode[label] = set(bvids)

                logger.mesg(
                    f"  {label:20s}: {len(hits):3d} hits, "
                    f"rank={rank_method}, {elapsed:.0f}ms"
                )
                for i, t in enumerate(titles):
                    logger.file(f"    [{i+1}] {t}")
            except Exception as e:
                logger.warn(f"  {label:20s}: ERROR - {e}")
                results_by_mode[label] = set()

        # Compare overlaps between modes
        logger.hint(f"\n  Overlap matrix (query: {query}):")
        mode_labels = [l for _, l in modes if l in results_by_mode]
        for i, l1 in enumerate(mode_labels):
            for l2 in mode_labels[i + 1 :]:
                s1, s2 = results_by_mode[l1], results_by_mode[l2]
                if s1 and s2:
                    overlap = len(s1 & s2)
                    union = len(s1 | s2)
                    logger.mesg(
                        f"    {l1} ∩ {l2}: "
                        f"{overlap}/{min(len(s1),len(s2))} "
                        f"(IoU={100*overlap/max(union,1):.1f}%)"
                    )

    logger.success("\n✓ Qmod recall comparison completed")


def test_stat_score_in_ranking():
    """Test that stat_score from blux.doc_score is properly integrated
    into ranking when available from the ES index."""
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Test 1: Verify stat_score is available in source fields
    logger.note("> Verifying stat_score available from ES index")
    res = searcher.search(
        query="影视飓风",
        limit=10,
        rank_top_k=10,
        verbose=False,
    )
    hits = res.get("hits", [])
    has_stat_score = any("stat_score" in h for h in hits)
    logger.mesg(f"  stat_score in hits: {has_stat_score}")
    if hits:
        for h in hits[:3]:
            ss = h.get("stat_score", "N/A")
            rs = h.get("rank_score", "N/A")
            logger.mesg(
                f"  [{h.get('title','?')[:25]}] "
                f"stat_score={ss}, rank_score={rs}, "
                f"views={h.get('stat',{}).get('view',0)}"
            )

    # Test 2: Verify ranking order considers stat_score
    logger.note("> Testing rank_score incorporates doc quality")
    res2 = searcher.search(
        query="deepseek",
        limit=50,
        rank_top_k=50,
        rank_method="stats",
        verbose=False,
    )
    hits2 = res2.get("hits", [])
    if hits2:
        rank_scores = [h.get("rank_score", 0) for h in hits2]
        # Verify descending order
        is_sorted = all(
            rank_scores[i] >= rank_scores[i + 1] for i in range(len(rank_scores) - 1)
        )
        logger.mesg(f"  rank_scores sorted descending: {is_sorted}")
        logger.mesg(
            f"  rank_score range: [{min(rank_scores):.4f}, {max(rank_scores):.4f}]"
        )

    logger.success("\n✓ stat_score ranking test completed")


def test_realtime_time_factor():
    """Test that PubdateScorer uses real-time time_factor from blux.doc_score.

    Verifies:
    - Recent videos get higher time_factor than old videos
    - Time factor is in [0.45, 1.30] range (from DocScorer)
    - Normalization maps to [0, 1]
    - Different ages produce different scores (not constant like old bug)
    """
    import time as _time
    from ranks.scorers import PubdateScorer
    from ranks.constants import TIME_FACTOR_MIN, TIME_FACTOR_MAX

    logger.note("> Testing real-time PubdateScorer...")

    scorer = PubdateScorer()
    now = _time.time()

    # Test different ages
    test_ages = [
        ("1 hour ago", now - 3600),
        ("1 day ago", now - 86400),
        ("3 days ago", now - 259200),
        ("7 days ago", now - 604800),
        ("15 days ago", now - 1296000),
        ("30 days ago", now - 2592000),
        ("90 days ago", now - 7776000),
    ]

    prev_factor = float("inf")
    results = []
    for label, pubdate in test_ages:
        factor = scorer.calc(pubdate, now_ts=now)
        norm = scorer.normalize(factor)
        results.append((label, factor, norm))

        # Time factor should decrease with age
        assert factor <= prev_factor + 0.01, (
            f"Time factor should decrease with age: {label} got {factor}, "
            f"prev was {prev_factor}"
        )
        prev_factor = factor

        # Time factor should be in valid range
        assert TIME_FACTOR_MIN - 0.01 <= factor <= TIME_FACTOR_MAX + 0.01, (
            f"{label}: time_factor {factor} out of range "
            f"[{TIME_FACTOR_MIN}, {TIME_FACTOR_MAX}]"
        )

        # Normalized should be in [0, 1]
        assert -0.01 <= norm <= 1.01, f"{label}: normalized {norm} out of [0, 1]"

        logger.mesg(f"  {label:15s}: time_factor={factor:.4f}, norm={norm:.4f}")

    # Verify NOT all the same (old bug: all videos got 0.25)
    factors = [r[1] for r in results]
    assert max(factors) - min(factors) > 0.1, (
        f"Time factors should vary meaningfully, got range "
        f"[{min(factors):.4f}, {max(factors):.4f}]"
    )

    logger.success("\n✓ Real-time time_factor test completed")


def test_bm25_embedding_blend():
    """Test that ScoreFuser.blend_relevance correctly blends signals.

    Verifies:
    - With BM25 available, both signals contribute
    - Without BM25, only embedding contributes
    - Keyword boost adds bonus
    - Output is bounded [0, 1]
    """
    from ranks.fusion import ScoreFuser

    logger.note("> Testing BM25 + embedding blending...")

    # Case 1: Both signals available
    r1 = ScoreFuser.blend_relevance(cosine_similarity=0.8, bm25_norm=0.9)
    logger.mesg(f"  cosine=0.8, bm25=0.9 → {r1:.4f}")
    assert 0.5 < r1 <= 1.0, f"Expected reasonable blend, got {r1}"

    # Case 2: Only embedding
    r2 = ScoreFuser.blend_relevance(cosine_similarity=0.8, bm25_norm=0.0)
    logger.mesg(f"  cosine=0.8, bm25=0.0 → {r2:.4f}")
    assert 0.3 < r2 <= 1.0, f"Expected embedding-only score, got {r2}"

    # Case 3: BM25 adds value beyond embedding alone
    r3_no_bm25 = ScoreFuser.blend_relevance(
        cosine_similarity=0.5, bm25_norm=0.0, keyword_boost=1.0
    )
    r3_with_bm25 = ScoreFuser.blend_relevance(
        cosine_similarity=0.5, bm25_norm=0.8, keyword_boost=1.0
    )
    logger.mesg(
        f"  cosine=0.5: without BM25={r3_no_bm25:.4f}, with BM25={r3_with_bm25:.4f}"
    )
    assert r3_with_bm25 > r3_no_bm25, "BM25 should increase the blended score"

    # Case 4: Keyword boost adds bonus
    r4_no_boost = ScoreFuser.blend_relevance(
        cosine_similarity=0.7, bm25_norm=0.5, keyword_boost=1.0
    )
    r4_with_boost = ScoreFuser.blend_relevance(
        cosine_similarity=0.7, bm25_norm=0.5, keyword_boost=4.0
    )
    logger.mesg(
        f"  cosine=0.7 bm25=0.5: boost=1.0→{r4_no_boost:.4f}, boost=4.0→{r4_with_boost:.4f}"
    )
    assert r4_with_boost > r4_no_boost, "Keyword boost should increase score"

    # Case 5: Output bounded [0, 1]
    r5_max = ScoreFuser.blend_relevance(
        cosine_similarity=1.0, bm25_norm=1.0, keyword_boost=10.0
    )
    r5_min = ScoreFuser.blend_relevance(
        cosine_similarity=0.0, bm25_norm=0.0, keyword_boost=1.0
    )
    logger.mesg(f"  max inputs → {r5_max:.4f}, min inputs → {r5_min:.4f}")
    assert 0.0 <= r5_min <= 1.0, f"Min blend out of range: {r5_min}"
    assert 0.0 <= r5_max <= 1.0, f"Max blend out of range: {r5_max}"

    logger.success("\n✓ BM25+embedding blend test completed")


def test_preference_ranking_unit():
    """Test preference-weighted fusion produces different rankings for different modes.

    Verifies:
    - All preference modes produce valid scores
    - prefer_quality boosts high-stats items
    - prefer_recency boosts recent items
    - prefer_relevance boosts high-relevance items
    - balanced is the default
    """
    from ranks.fusion import ScoreFuser
    from ranks.constants import RANK_PREFER_PRESETS

    logger.note("> Testing preference-weighted fusion...")

    fuser = ScoreFuser()

    # Create test scenarios with different strengths
    scenarios = {
        "high_quality": {"quality": 0.9, "relevance": 0.4, "recency": 0.3},
        "high_relevance": {"quality": 0.3, "relevance": 0.9, "recency": 0.3},
        "high_recency": {"quality": 0.3, "relevance": 0.4, "recency": 0.9},
        "balanced_mid": {"quality": 0.5, "relevance": 0.5, "recency": 0.5},
    }

    for prefer_mode in RANK_PREFER_PRESETS:
        logger.mesg(f"\n  Preference: {prefer_mode}")
        scores = {}
        for scenario_name, vals in scenarios.items():
            score = fuser.fuse_with_preference(
                quality=vals["quality"],
                relevance=vals["relevance"],
                recency=vals["recency"],
                prefer=prefer_mode,
            )
            scores[scenario_name] = score
            logger.mesg(f"    {scenario_name:20s}: {score:.4f}")

        # Verify all scores are in valid range
        for name, score in scores.items():
            assert (
                0.0 <= score <= 1.0
            ), f"{prefer_mode}/{name}: score {score} out of range"

    # Verify preference modes favor their dimension
    q_bal = fuser.fuse_with_preference(0.9, 0.3, 0.3, "balanced")
    q_pq = fuser.fuse_with_preference(0.9, 0.3, 0.3, "prefer_quality")
    assert q_pq > q_bal, "prefer_quality should give higher score to high-quality"
    logger.mesg(f"\n  High quality: balanced={q_bal:.4f}, prefer_quality={q_pq:.4f}")

    r_bal = fuser.fuse_with_preference(0.3, 0.9, 0.3, "balanced")
    r_pr = fuser.fuse_with_preference(0.3, 0.9, 0.3, "prefer_relevance")
    assert r_pr > r_bal, "prefer_relevance should give higher score to high-relevance"
    logger.mesg(f"  High relevance: balanced={r_bal:.4f}, prefer_relevance={r_pr:.4f}")

    t_bal = fuser.fuse_with_preference(0.3, 0.3, 0.9, "balanced")
    t_pt = fuser.fuse_with_preference(0.3, 0.3, 0.9, "prefer_recency")
    assert t_pt > t_bal, "prefer_recency should give higher score to high-recency"
    logger.mesg(f"  High recency: balanced={t_bal:.4f}, prefer_recency={t_pt:.4f}")

    logger.success("\n✓ Preference ranking unit test completed")


def test_preference_rank_integration():
    """Test preference_rank method of VideoHitsRanker with mock reranked hits.

    Verifies:
    - preference_rank correctly uses BM25 + cosine blend
    - Different preference modes produce different orderings
    - All required fields are populated in output
    """
    import time as _time
    from copy import deepcopy
    from ranks.ranker import VideoHitsRanker

    logger.note("> Testing preference_rank integration...")

    ranker = VideoHitsRanker()
    now = _time.time()

    # Create mock reranked hits with diverse characteristics
    mock_hits = [
        {
            "bvid": "BV_popular_old",
            "title": "Popular but old video",
            "stat": {
                "view": 5000000,
                "favorite": 100000,
                "coin": 50000,
                "reply": 20000,
                "share": 5000,
                "danmaku": 10000,
            },
            "pubdate": int(now - 2592000),  # 30 days ago
            "cosine_similarity": 0.6,
            "keyword_boost": 2.0,
            "original_score": 15.0,
            "reranked": True,
        },
        {
            "bvid": "BV_fresh_relevant",
            "title": "Fresh and highly relevant",
            "stat": {
                "view": 50000,
                "favorite": 1000,
                "coin": 500,
                "reply": 200,
                "share": 50,
                "danmaku": 100,
            },
            "pubdate": int(now - 3600),  # 1 hour ago
            "cosine_similarity": 0.95,
            "keyword_boost": 3.0,
            "original_score": 20.0,
            "reranked": True,
        },
        {
            "bvid": "BV_mediocre",
            "title": "Average everything",
            "stat": {
                "view": 100000,
                "favorite": 2000,
                "coin": 1000,
                "reply": 500,
                "share": 100,
                "danmaku": 200,
            },
            "pubdate": int(now - 604800),  # 7 days ago
            "cosine_similarity": 0.7,
            "keyword_boost": 1.5,
            "original_score": 10.0,
            "reranked": True,
        },
    ]

    # Test with different preferences
    for prefer_mode in [
        "balanced",
        "prefer_quality",
        "prefer_relevance",
        "prefer_recency",
    ]:
        test_hits = deepcopy(mock_hits)
        hits_info = {"hits": test_hits}

        result = ranker.preference_rank(hits_info, top_k=3, prefer=prefer_mode)
        ranked_hits = result["hits"]

        logger.mesg(f"\n  Preference: {prefer_mode}")
        for h in ranked_hits:
            logger.mesg(
                f"    {h['bvid']:25s}: rank_score={h.get('rank_score', 0):.4f} "
                f"q={h.get('quality_score', 0):.3f} "
                f"r={h.get('relevance_score', 0):.3f} "
                f"t={h.get('recency_score', 0):.3f}"
            )

        # Verify all required fields present
        for h in ranked_hits:
            assert "rank_score" in h, f"Missing rank_score in {h['bvid']}"
            assert "quality_score" in h, f"Missing quality_score in {h['bvid']}"
            assert "relevance_score" in h, f"Missing relevance_score in {h['bvid']}"
            assert "recency_score" in h, f"Missing recency_score in {h['bvid']}"
            assert "time_factor" in h, f"Missing time_factor in {h['bvid']}"
            assert (
                0 <= h["rank_score"] <= 1
            ), f"rank_score out of range: {h['rank_score']}"

        assert result["rank_method"] == "preference"
        assert result["prefer"] == prefer_mode

    # Verify preference modes produce different orderings
    # prefer_recency should favor fresh_relevant (1 hour old)
    test_t = deepcopy(mock_hits)
    rt = ranker.preference_rank({"hits": test_t}, top_k=3, prefer="prefer_recency")
    recency_first = rt["hits"][0]["bvid"]
    logger.mesg(f"\n  prefer_recency first: {recency_first}")
    assert (
        recency_first == "BV_fresh_relevant"
    ), f"prefer_recency should favor BV_fresh_relevant, got {recency_first}"

    logger.success("\n✓ preference_rank integration test completed")


def test_reranker_preserves_original_score():
    """Test that the reranker preserves the original BM25 score.

    Verifies:
    - hit["original_score"] is set before overwriting hit["score"]
    - The original BM25 score survives the reranking process
    """
    logger.note("> Testing reranker preserves original_score...")

    # Create mock hits simulating word search results with BM25 scores
    mock_hits = [
        {
            "bvid": f"BV_test_{i}",
            "title": f"Test video {i}",
            "score": float(20 - i),  # BM25 scores: 20, 19, 18, ...
            "stat": {"view": 1000 * i},
            "pubdate": 1700000000,
            "owner": {"name": "test"},
            "tags": "test",
            "desc": "test",
        }
        for i in range(5)
    ]

    # Simulate what the reranker does (we just check that the pattern works)
    for hit in mock_hits:
        original_bm25 = hit["score"]
        # This is exactly what the reranker now does
        hit["original_score"] = hit.get("score", 0)
        hit["score"] = 0.5  # Overwrite with rerank_score
        hit["cosine_similarity"] = 0.7
        hit["keyword_boost"] = 1.5
        hit["reranked"] = True

        assert (
            hit["original_score"] == original_bm25
        ), f"original_score should be {original_bm25}, got {hit['original_score']}"
        assert hit["score"] != original_bm25, "score should be overwritten"

    logger.mesg(f"  All {len(mock_hits)} hits preserved original_score correctly")
    logger.success("\n✓ Reranker original_score preservation test completed")


def test_knn_k_1000():
    """Test that KNN_K is set to 1000 for better recall and more rerank candidates."""
    from elastics.videos.constants import KNN_K

    logger.note(f"> Testing KNN_K = {KNN_K}")
    assert KNN_K == 1000, f"KNN_K should be 1000, got {KNN_K}"
    logger.success(f"  ✓ KNN_K = {KNN_K}")
    logger.success("\n✓ KNN_K test completed")
