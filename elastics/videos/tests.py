from btok import SentenceCategorizer
from fastapi.encoders import jsonable_encoder
from tclogger import logger, logstr, dict_to_str, brk

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.explorer import VideoExplorer
from converters.query.dsl import MultiMatchQueryDSLConstructor
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.query.filter import QueryFilterExtractor


def test_random():
    searcher = VideoSearcherV2(ELASTIC_VIDEOS_DEV_INDEX)
    logger.note("> Getting random results ...")
    res = searcher.random(limit=3)
    logger.mesg(dict_to_str(res))


filter_queries = [
    "Hansimov 2018",
    "黑神话 2024 :coin>1000 :view<100000",
]


def test_filter():
    searcher = VideoSearcherV2(ELASTIC_VIDEOS_DEV_INDEX)
    for query in filter_queries:
        match_fields = ["title^2.5", "owner.name^2", "desc", "pubdate_str^2.5"]
        date_match_fields = [
            "title^0.5",
            "owner.name^0.25",
            "desc^0.2",
            "pubdate_str^2.5",
        ]
        filter_extractor = QueryFilterExtractor()
        query_keywords, filters = filter_extractor.construct(query)
        query_without_filters = " ".join(query_keywords)
        query_constructor = MultiMatchQueryDSLConstructor()
        query_dsl_dict = query_constructor.construct(
            query=query_without_filters,
            match_fields=match_fields,
            date_match_fields=date_match_fields,
        )

        if filters:
            query_dsl_dict["bool"]["filter"] = filters

        logger.note(f"> Construct DSL for query:", end=" ")
        logger.mesg(f"[{query}]")
        logger.success(dict_to_str(query_dsl_dict))
        script_query_dsl_dict = ScriptScoreQueryDSLConstructor().construct(
            query_dsl_dict
        )
        logger.note(dict_to_str(script_query_dsl_dict))
        logger.mesg(ScriptScoreQueryDSLConstructor().get_script_source_by_stats())

        search_res = searcher.search(
            query,
            source_fields=["title", "owner.name", "desc", "pubdate_str", "stat"],
            boost=True,
            use_script_score=True,
            detail_level=1,
            limit=3,
            timeout=2,
            verbose=True,
        )
        if search_res["took"] < 0:
            logger.warn(dict_to_str(search_res))


suggest_queries = [
    # "影视飓feng",
    # "yingshiju",
    # "yingshi ju",
    # "影视ju",
    # "hongjing 08",
    # "Hongjing 08? 2024",
    # "Hongjing (08 || 月亮3) 2024",
    # "hongjing 小快地? (08 || 月亮3) d=15d u=红警最top",
    # "hongjing (xiaokuaidi || bingtian)"
    # "俞利均",
    # "liangwenfeng",
    "deepseek v3 0324",
    # "xiaomi",
    # "d=3d",
    # "u=雷军",
    # "d=2d v>1w",
    # "v=[1k,10000] d=2d",
    # "d=2d v=[1k,10000]",
    # "v=[1k,10000]",
    # "bv=BV1LP4y1H7TJ",
    # "d=[1d,3d]",
    # "bv=[BV1Pt411o7Zb,BV1Xx411c7cH]",
    # "av=[1,2,4]",
    # "bv=[BV1Pt411o7Zb,2]",
    # "Hongjing 08 xiaokuaidi 2024",
    # "5197",
    # "影视 ju :date<=7d 2024",
    # ":date<=7d 2024",
    # "edmundd",
    # "gpt-sovits huaer",
    # "XIAOMI su7",
    # "Niko",
    # "shanghai major hanlengde",
    # "after effects jiaocheng",
    # "damoxing weidiao",
    # "大模型 微调",
    # "d=1d",
    # "llm 大模型",
    # "are you ok",
    # "changcheng",
    # "10-22",
    # "Python 教程",
    # ":date=2024-10-31 :view>=1w",
]


def test_suggest():
    searcher = VideoSearcherV1(ELASTIC_VIDEOS_DEV_INDEX)
    for query in suggest_queries:
        query_str = logstr.mesg(brk(query))
        logger.note(f"> Query: {query_str}")
        res = searcher.multi_level_suggest(query, limit=20, verbose=True)
        hits = res.pop("hits")
        # for idx, hit in enumerate(hits[:3]):
        #     logger.note(f"* Hit {idx}:")
        #     logger.file(dict_to_str(hit, align_list=False), indent=4)
        logger.success(f"✓ Suggest results:")
        logger.success(dict_to_str(res, align_list=False), indent=2)


multi_level_search_queries = [
    "are you ok",
    "游戏星",
    "10-22",
]


def test_multi_level_search():
    searcher = VideoSearcherV1(ELASTIC_VIDEOS_DEV_INDEX)
    for query in multi_level_search_queries:
        logger.note("> Searching results:", end=" ")
        logger.file(f"[{query}]")
        res = searcher.multi_level_search(
            query, limit=500, use_script_score=True, verbose=True
        )
        hits = res.pop("hits")
        logger.success(dict_to_str(res))
        # for idx, hit in enumerate(hits[:3]):
        #     logger.note(f"* Hit {idx}:")
        #     logger.file(dict_to_str(hit, align_list=False), indent=4)


search_queries = [
    # "影视飓风^3 巴黎",
    # "雷军 AI (锐评 || 吐槽^3 || 游戏^4)",
    # "deepseek v3 2024",
    # "deepseek v3 0324 d=1d",
    # "day by day",
    # "v>1k 原神",
    # "v>1 k=原神",
    # '+"影视飓风" +“李四维”',
    # '+"影视飓风" +“-lks-”',
    # '《"你的名字"》',
    # '"何同学" v>1k',
    # '"何同学"',
    # "影视飓风",
    # '"影视飓风"',
    # "影视",
    # "雪茄 教程",
    '影视飓风 "罗永浩"',
]


def test_search():
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    for query in search_queries:
        logger.note("> Searching results:", end=" ")
        logger.file(f"[{query}]")
        res = searcher.search(query, limit=50, verbose=True)
        hits = res.pop("hits")
        logger.success(dict_to_str(res))
        for idx, hit in enumerate(hits[:3]):
            logger.note(f"* Hit {idx}:")
            logger.file(dict_to_str(hit, align_list=False), indent=4)


def test_agg():
    """DEPRECATD. Use VideoExplorer instead."""
    searcher = VideoSearcherV2(ELASTIC_VIDEOS_DEV_INDEX)
    for query in search_queries:
        logger.note("> Agg results:", end=" ")
        logger.file(f"[{query}]")
        res = searcher.agg(query, verbose=False)
        logger.success(dict_to_str(res, align_list=False), indent=2)


def test_explore():
    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    for query in search_queries:
        logger.note("> Explore results:", end=" ")
        logger.file(f"[{query}]")
        explore_res = explorer.explore(
            query, rank_method="stats", rank_top_k=3, verbose=True
        )
        for step_res in explore_res:
            stage_name = step_res["name"]
            logger.hint(f"* stage result of {(logstr.mesg(brk(stage_name)))}:")
            if stage_name != "group_hits_by_owner":
                logger.mesg(dict_to_str(step_res, align_list=False), indent=2)


split_queries = [
    # "外場最速伝説と飛翔の雄鷹！髮型監一郎です！",
    # "《我的世界》是一款沙盒游戏",
    # "GTA6新预告有哪些细节",
    # "s1mple即将加入新战队的新闻",
    # "能不能给我推荐点原神新出的角色的皮肤",
    # "鸣潮 皮肤和壁纸",
    # "我想知道鸣潮的公式是啥意思",
    "原神,启动!是什么梗",
]


def test_categorize():
    categorizer = SentenceCategorizer()
    for query in split_queries:
        logger.note("> Categorizing:", end=" ")
        logger.file(f"[{query}]")
        res = categorizer.categorize(query)
        logger.success(dict_to_str(res, add_quotes=True), indent=2)


knn_queries = [
    "红警08 小块地",
    # "deepseek v3 0324",
    # "影视飓风 罗永浩",
]


def test_knn_search():
    """Test KNN search functionality."""
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    for query in knn_queries:
        logger.note("> KNN searching:", end=" ")
        logger.file(f"[{query}]")
        res = searcher.knn_search(
            query=query,
            limit=10,
            rank_top_k=10,
            verbose=True,
        )
        hits = res.pop("hits", [])
        logger.success(f"Total hits: {res.get('total_hits', 0)}")
        logger.success(f"Return hits: {res.get('return_hits', 0)}")
        for idx, hit in enumerate(hits[:3]):
            logger.note(f"* Hit {idx}:")
            logger.file(dict_to_str(hit, align_list=False), indent=4)


knn_queries_with_filter = [
    "donk 高光集锦 d>2024-01-01 v>1w",
]


def test_knn_search_with_filters():
    """Test KNN search with DSL filters."""
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    # KNN search with DSL filters
    query = knn_queries_with_filter[0]
    logger.note("> KNN searching with filters:", end=" ")
    logger.file(f"[{query}]")
    res = searcher.knn_search(
        query=query,
        limit=10,
        rank_top_k=10,
        verbose=True,
    )
    hits = res.pop("hits", [])
    logger.success(f"Total hits: {res.get('total_hits', 0)}")
    logger.success(f"Return hits: {res.get('return_hits', 0)}")
    for idx, hit in enumerate(hits[:3]):
        logger.note(f"* Hit {idx}:")
        logger.file(dict_to_str(hit, align_list=False), indent=4)


def test_knn_explore():
    """Test KNN explore functionality."""
    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    for query in knn_queries:
        logger.note("> KNN exploring:", end=" ")
        logger.file(f"[{query}]")
        explore_res = explorer.knn_explore(
            query=query,
            rank_top_k=50,
            group_owner_limit=10,
            verbose=True,
        )
        logger.success(f"Status: {explore_res.get('status', 'N/A')}")
        for step_res in explore_res.get("data", []):
            stage_name = step_res["name"]
            logger.hint(f"* stage result of {(logstr.mesg(brk(stage_name)))}:")
            if stage_name != "group_hits_by_owner":
                logger.mesg(dict_to_str(step_res, align_list=False), indent=2)


def test_embed_client():
    """Test embed client functionality."""
    from converters.embed import TextEmbedSearchClient

    logger.note("> Testing embed client...")
    client = TextEmbedSearchClient()

    if not client.is_available():
        logger.warn("× Embed client not available, skipping test")
        return

    # Test single text embedding
    logger.hint("Test: Single text embedding")
    hex_str = client.text_to_hex("红警HBK08 游戏视频")
    if hex_str:
        logger.okay(f"Hex string length: {len(hex_str)}")
        logger.mesg(f"First 64 chars: {hex_str[:64]}...")
    else:
        logger.warn("× Failed to get hex string")

    # Test hex to byte array conversion
    logger.hint("Test: Hex to byte array")
    byte_array = client.hex_to_byte_array(hex_str)
    if byte_array:
        logger.okay(f"Byte array length: {len(byte_array)}")
        logger.mesg(f"First 10 bytes: {byte_array[:10]}")
    else:
        logger.warn("× Failed to convert hex to byte array")

    client.close()


# Query mode / hybrid search test queries
hybrid_queries = [
    "黑神话 q=wv",
    "影视飓风 q=v d>2024",
    "deepseek q=wv v>1k",
    "q=wv 游戏",
]


def test_hybrid_search():
    """Test hybrid search functionality."""
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    for query in hybrid_queries:
        logger.note("> Hybrid searching:", end=" ")
        logger.file(f"[{query}]")
        res = searcher.hybrid_search(query, limit=50, verbose=True)
        hits = res.get("hits", [])
        logger.success(f"Total hits: {res.get('total_hits', 0)}")
        logger.success(f"Word hits: {res.get('word_hits_count', 0)}")
        logger.success(f"KNN hits: {res.get('knn_hits_count', 0)}")
        logger.success(f"Fusion method: {res.get('fusion_method', 'N/A')}")
        for idx, hit in enumerate(hits[:3]):
            logger.note(f"* Hit {idx}:")
            hybrid_score = hit.get("hybrid_score", 0)
            logger.mesg(f"  hybrid_score: {hybrid_score:.4f}")
            logger.file(dict_to_str(hit, align_list=False), indent=4)


def test_unified_explore():
    """Test unified explore with automatic query mode detection."""
    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )
    test_queries = [
        ("黑神话 悟空", None),  # Should default to ["vector"]
        ("黑神话 q=v", None),  # Should use ["vector"] from query
        ("影视飓风 q=wv d>2024", None),  # Should use ["word", "vector"] from query
        ("deepseek", ["vector"]),  # Explicit mode override
        ("游戏", ["word", "vector"]),  # Explicit mode override
    ]
    for query, mode in test_queries:
        logger.note(f"> Unified exploring: [{query}] qmod={mode}")
        explore_res = explorer.unified_explore(
            query=query,
            qmod=mode,
            rank_top_k=50,
            group_owner_limit=10,
            verbose=True,
        )
        logger.success(f"Status: {explore_res.get('status', 'N/A')}")
        # qmod is now in the first step's output
        first_step = explore_res.get("data", [{}])[0]
        qmod_from_output = first_step.get("output", {}).get("qmod", "N/A")
        logger.success(f"Query mode (from step output): {qmod_from_output}")
        for step_res in explore_res.get("data", []):
            stage_name = step_res["name"]
            logger.hint(f"* stage: {logstr.mesg(brk(stage_name))}")


def test_qmod_parser():
    """Test qmod parsing from DSL."""
    from converters.dsl.fields.qmod import (
        extract_qmod_from_expr_tree,
        QMOD_DEFAULT,
    )
    from converters.dsl.elastic import DslExprToElasticConverter

    converter = DslExprToElasticConverter()
    test_cases = [
        ("q=w", ["word"]),
        ("q=v", ["vector"]),
        ("q=wv", ["word", "vector"]),
        ("q=vw", ["word", "vector"]),  # normalized order
        ("qm=w", ["word"]),
        ("qmod=v", ["vector"]),
        ("黑神话 q=v", ["vector"]),
        ("黑神话 q=wv v>1w", ["word", "vector"]),
        ("黑神话 悟空", ["vector"]),  # default is vector
    ]

    logger.note("> Testing qmod parser...")
    for query, expected in test_cases:
        try:
            expr_tree = converter.construct_expr_tree(query)
            mode = extract_qmod_from_expr_tree(expr_tree)
            status = "✓" if mode == expected else "×"
            logger.mesg(f"  {status} [{query}] -> {mode} (expected: {expected})")
        except Exception as e:
            logger.warn(f"  × [{query}] -> ERROR: {e}")


def test_rrf_fusion_fill():
    """Test fill-and-supplement RRF fusion logic to ensure 400 results."""
    logger.note("> Testing RRF fusion fill-and-supplement strategy...")

    # Use DEV index and environment
    searcher = VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Test Case 1: Simulate scenario with sufficient data
    # When word + knn have enough unique bvids, fusion should return limit results
    logger.note("\n> Test Case 1: Simulating word=300, knn=300, overlap=50")

    # Create 300 word hits
    word_hits = [{"bvid": f"BV_word_{i}", "score": 30 - i * 0.05} for i in range(300)]

    # Create 300 knn hits with 50 overlapping bvids
    knn_hits = []
    # First 50 overlap with word
    for i in range(50):
        knn_hits.append({"bvid": f"BV_word_{i}", "score": 0.95 - i * 0.005})
    # Remaining 250 are unique
    for i in range(250):
        knn_hits.append({"bvid": f"BV_knn_{i}", "score": 0.90 - i * 0.003})

    # Total unique = 300 + 250 = 550 > 400
    # Call _rrf_fusion with limit=400
    fused = searcher._rrf_fusion(word_hits, knn_hits, limit=400)

    # Verify result count
    expected = 400
    actual = len(fused)
    status = "✓" if actual == expected else "×"
    logger.mesg(f"  {status} Result count: {actual} (expected: {expected})")

    # Check selection tiers
    tier_counts = {}
    for hit in fused:
        tier = hit.get("selection_tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    logger.mesg(f"  Selection tiers: {tier_counts}")
    logger.mesg(f"    - word_top: {tier_counts.get('word_top', 0)}")
    logger.mesg(f"    - knn_top: {tier_counts.get('knn_top', 0)}")
    logger.mesg(f"    - word_knn_top: {tier_counts.get('word_knn_top', 0)}")
    logger.mesg(f"    - fusion_fill: {tier_counts.get('fusion_fill', 0)}")

    # Test Case 2: Real search with q=vw
    logger.note("\n> Test Case 2: Real hybrid search with q=vw")
    search_res = searcher.hybrid_search(
        "红警08 q=vw",
        limit=400,
        rank_top_k=400,
        timeout=10,
        verbose=True,
    )

    result_count = search_res.get("return_hits", 0)
    word_count = search_res.get("word_hits_count", 0)
    knn_count = search_res.get("knn_hits_count", 0)

    status = "✓" if result_count == 400 else "×"
    logger.mesg(
        f"  {status} Hybrid search returned: {result_count} results (expected: 400)"
    )
    logger.mesg(f"    - Word hits: {word_count}")
    logger.mesg(f"    - KNN hits: {knn_count}")

    if search_res.get("hits"):
        # Check selection tier distribution
        tier_counts = {}
        for hit in search_res["hits"]:
            tier = hit.get("selection_tier", "unknown")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        logger.mesg(f"  Selection tier distribution:")
        for tier, count in sorted(tier_counts.items()):
            logger.mesg(f"    - {tier}: {count}")

        # Show first few results
        logger.mesg(f"\n  First 5 results:")
        for i, hit in enumerate(search_res["hits"][:5]):
            logger.mesg(
                f"    [{i+1}] {hit.get('bvid')} "
                f"hybrid={hit.get('hybrid_score', 0):.2f} "
                f"tier={hit.get('selection_tier')} "
                f"word_rank={hit.get('word_rank')} "
                f"knn_rank={hit.get('knn_rank')}"
            )

    # Test Case 3: Compare with q=w and q=v
    logger.note("\n> Test Case 3: Comparing q=w, q=v, q=vw")

    queries = [
        ("红警08 q=w", "word-only"),
        ("红警08 q=v", "vector-only"),
        ("红警08 q=vw", "hybrid"),
    ]

    for query, desc in queries:
        if "q=w" in query:
            res = searcher.search(query, limit=400, rank_top_k=400, timeout=10)
        elif "q=v" in query:
            res = searcher.knn_search(
                query, k=400, limit=400, rank_top_k=400, timeout=10
            )
        else:
            res = searcher.hybrid_search(query, limit=400, rank_top_k=400, timeout=10)

        count = res.get("return_hits", 0)
        status = "✓" if count == 400 else "×"
        logger.mesg(f"  {status} {desc:12} [{query}] -> {count} results")

    logger.success("> RRF fusion test completed!")


def test_hybrid_explore_count():
    """Test that hybrid_explore returns 400 results correctly."""
    logger.note("> Testing hybrid_explore result count...")

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Test unified_explore with q=vw
    result = explorer.unified_explore(
        query="苏神神了 q=vw",
        verbose=True,
        rank_top_k=400,
    )

    logger.mesg(f"Status: {result.get('status')}")

    # Find the hybrid_search step
    for step in result.get("data", []):
        step_name = step.get("name")
        output = step.get("output", {})

        if step_name == "hybrid_search":
            return_hits = output.get("return_hits", 0)
            word_hits = output.get("word_hits_count", 0)
            knn_hits = output.get("knn_hits_count", 0)
            status = "✓" if return_hits == 400 else "×"
            logger.mesg(
                f"  {status} hybrid_search: return_hits={return_hits} (expected: 400)"
            )
            logger.mesg(f"      word_hits_count={word_hits}, knn_hits_count={knn_hits}")

        elif step_name == "group_hits_by_owner":
            authors = output.get("authors", {})
            total_videos = sum(
                len(author.get("videos", [])) for author in authors.values()
            )
            logger.mesg(
                f"  group_hits_by_owner: {len(authors)} authors, {total_videos} total videos"
            )


# Filter-only search test queries (no keywords, only filters)
filter_only_queries = [
    'u="红警HBK08"',
    'u="红警HBK08" q=v',
    'u="红警HBK08" q=w',
    'u="红警HBK08" q=wv',
    "d>2024-01-01 v>10000",
    "u=影视飓风 d>2024-06-01",
]


def test_filter_only_search(searcher: VideoSearcherV2 = None):
    """Test that filter-only queries (no keywords) work correctly.

    When a query contains only filter expressions (like u=xxx, d>xxx, v>xxx)
    without any search keywords, the search should:
    1. Not attempt KNN/vector search (which requires text for embedding)
    2. Not attempt word matching
    3. Use match_all + filters to return all matching results
    """
    logger.note("> Testing filter-only search...")

    if searcher is None:
        searcher = VideoSearcherV2(
            index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
        )

    for query in filter_only_queries:
        logger.note(f"\n> Query: [{query}]")

        # Check if query has keywords
        has_keywords = searcher.has_search_keywords(query)
        logger.mesg(f"  has_search_keywords: {has_keywords}")

        # Test with different query modes
        qmod = searcher.get_qmod_from_query(query)
        logger.mesg(f"  qmod from query: {qmod}")

        # Test filter_only_search directly
        logger.hint("  Direct filter_only_search:")
        res = searcher.filter_only_search(
            query=query,
            limit=50,
            rank_top_k=50,
            verbose=False,
        )
        total_hits = res.get("total_hits", 0)
        return_hits = res.get("return_hits", 0)
        filter_only_flag = res.get("filter_only", False)
        logger.mesg(
            f"    total_hits={total_hits}, return_hits={return_hits}, filter_only={filter_only_flag}"
        )

        # Test through knn_search (should fallback to filter_only_search)
        if "q=v" in query or (not any(x in query for x in ["q=w", "q=wv", "q=vw"])):
            logger.hint("  Via knn_search (should fallback):")
            knn_res = searcher.knn_search(
                query=query,
                limit=50,
                rank_top_k=50,
                verbose=False,
            )
            knn_total = knn_res.get("total_hits", 0)
            knn_return = knn_res.get("return_hits", 0)
            knn_filter_only = knn_res.get("filter_only", False)
            status = "✓" if knn_filter_only or knn_return > 3 else "×"
            logger.mesg(
                f"    {status} total_hits={knn_total}, return_hits={knn_return}, filter_only={knn_filter_only}"
            )

        # Test through hybrid_search (should fallback to filter_only_search)
        if "q=wv" in query or "q=vw" in query:
            logger.hint("  Via hybrid_search (should fallback):")
            hybrid_res = searcher.hybrid_search(
                query=query,
                limit=50,
                rank_top_k=50,
                verbose=False,
            )
            hybrid_total = hybrid_res.get("total_hits", 0)
            hybrid_return = hybrid_res.get("return_hits", 0)
            hybrid_filter_only = hybrid_res.get("filter_only", False)
            status = "✓" if hybrid_filter_only or hybrid_return > 3 else "×"
            logger.mesg(
                f"    {status} total_hits={hybrid_total}, return_hits={hybrid_return}, filter_only={hybrid_filter_only}"
            )

        # Verify that filter_only search returns more than 3 results
        if total_hits >= 3:
            status = "✓" if return_hits > 3 else "×"
            logger.mesg(
                f"  {status} Filter-only search returned {return_hits} results (expected > 3)"
            )
        else:
            logger.mesg(f"  ⓘ Query matches only {total_hits} documents")

    logger.success("> Filter-only search test completed!")


def test_filter_only_vs_regular(searcher: VideoSearcherV2 = None):
    """Compare filter-only search (q=v/q=wv without keywords) vs regular search."""
    logger.note("> Comparing filter-only vs regular search...")

    if searcher is None:
        searcher = VideoSearcherV2(
            index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
        )

    # Test: u="红警HBK08" should return same results regardless of q=w/v/wv
    test_cases = [
        ('u="红警HBK08"', "default"),
        ('u="红警HBK08" q=w', "word mode"),
        ('u="红警HBK08" q=v', "vector mode"),
        ('u="红警HBK08" q=wv', "hybrid mode"),
    ]

    results = {}
    for query, mode in test_cases:
        qmod = searcher.get_qmod_from_query(query)
        has_keywords = searcher.has_search_keywords(query)

        if "word" in qmod and "vector" in qmod:
            res = searcher.hybrid_search(query, limit=50, verbose=False)
        elif "vector" in qmod:
            res = searcher.knn_search(query, limit=50, verbose=False)
        else:
            res = searcher.search(query, limit=50, verbose=False)

        total_hits = res.get("total_hits", 0)
        return_hits = res.get("return_hits", 0)
        filter_only = res.get("filter_only", False)

        results[mode] = {
            "total_hits": total_hits,
            "return_hits": return_hits,
            "filter_only": filter_only,
            "has_keywords": has_keywords,
        }

        logger.mesg(
            f"  [{mode:12}] qmod={qmod}, has_keywords={has_keywords}, "
            f"total={total_hits}, return={return_hits}, filter_only={filter_only}"
        )

    # All modes should return the same number of results for filter-only queries
    return_counts = [r["return_hits"] for r in results.values()]
    all_same = len(set(return_counts)) == 1 or all(c > 3 for c in return_counts)

    if all_same:
        logger.success(
            "  ✓ All query modes return consistent results for filter-only query"
        )
    else:
        logger.warn(f"  × Inconsistent results across modes: {return_counts}")


def test_filter_only_explore(explorer: VideoExplorer = None):
    """Test that filter-only queries work correctly in explore methods.

    When a query contains only filter expressions without search keywords,
    the explore methods should still return correct results.
    """
    logger.note("> Testing filter-only explore...")

    if explorer is None:
        explorer = VideoExplorer(
            index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
        )

    # Test queries with only filters (no search keywords)
    test_cases = [
        ('u="红警HBK08"', "default (no qmod)"),
        ('u="红警HBK08" q=v', "vector mode"),
        ('u="红警HBK08" q=w', "word mode"),
        ('u="红警HBK08" q=wv', "hybrid mode"),
        ("红警HBK08 q=v", "with keywords + vector"),  # This should use KNN
        ("红警HBK08 q=wv", "with keywords + hybrid"),  # This should use hybrid
    ]

    for query, desc in test_cases:
        logger.note(f"\n> Query: [{query}] ({desc})")

        has_keywords = explorer.has_search_keywords(query)
        qmod = explorer.get_qmod_from_query(query)
        logger.mesg(f"  has_search_keywords: {has_keywords}, qmod: {qmod}")

        try:
            result = explorer.unified_explore(
                query=query,
                rank_top_k=50,
                group_owner_limit=10,
                verbose=False,
            )

            status = result.get("status", "unknown")
            data = result.get("data", [])

            # Find the search step result
            search_step = None
            for step in data:
                if step.get("name") in [
                    "most_relevant_search",
                    "knn_search",
                    "hybrid_search",
                ]:
                    search_step = step
                    break

            if search_step:
                output = search_step.get("output", {})
                total_hits = output.get("total_hits", 0)
                return_hits = output.get("return_hits", 0)
                filter_only = output.get("filter_only", False)
                step_name = search_step.get("name")

                # For filter-only queries, we expect return_hits > 3
                if not has_keywords:
                    check = "✓" if return_hits > 3 or total_hits <= 3 else "×"
                else:
                    check = "✓"  # With keywords, any result is fine

                logger.mesg(
                    f"  {check} step={step_name}, total_hits={total_hits}, "
                    f"return_hits={return_hits}, filter_only={filter_only}"
                )
            else:
                logger.warn(f"  × No search step found in result")

            # Check group_hits_by_owner
            group_step = None
            for step in data:
                if step.get("name") == "group_hits_by_owner":
                    group_step = step
                    break

            if group_step:
                authors = group_step.get("output", {}).get("authors", {})
                logger.mesg(f"  Authors found: {len(authors)}")

        except Exception as e:
            logger.warn(f"  × Error: {e}")
            import traceback

            traceback.print_exc()

    logger.success("> Filter-only explore test completed!")


def test_json_serialization(explorer: VideoExplorer = None):
    """Test that filter_only_search results can be serialized by FastAPI."""
    if explorer is None:
        explorer = VideoExplorer(ELASTIC_VIDEOS_DEV_INDEX)

    test_queries = [
        'u="红警HBK08"',
        'u="红警HBK08" q=v',
        "d>2024-01-01 v>10000",
    ]

    logger.note("> Testing JSON serialization (simulating FastAPI)...")
    for query in test_queries:
        logger.hint(f"\n  Query: [{query}]")
        try:
            res = explorer.filter_only_search(
                query, limit=10, rank_top_k=10, verbose=False
            )
            # This is what FastAPI does internally
            encoded = jsonable_encoder(res)
            logger.success("    ✓ jsonable_encoder succeeded")
            logger.mesg(
                f'      total_hits={res.get("total_hits")}, filter_only={res.get("filter_only")}'
            )
            has_tree = "query_expr_tree" in res.get("query_info", {})
            logger.mesg(f"      query_expr_tree in query_info: {has_tree}")
        except Exception as e:
            logger.warn(f"    ✗ Error: {type(e).__name__}: {e}")

    logger.success("\n> JSON serialization test completed!")


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

    from converters.embed import get_embed_client, get_reranker
    from converters.embed.reranker import compute_passage

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


if __name__ == "__main__":
    # test_random()
    # test_filter()
    # test_suggest()
    # test_multi_level_search()
    # test_search()
    # test_agg()
    # test_explore()
    # test_split()
    # test_categorize()
    # test_embed_client()
    # test_knn_search()
    # test_knn_search_with_filters()
    # test_knn_explore()
    # test_hybrid_search()
    # test_unified_explore()
    # test_qmod_parser()
    # test_rrf_fusion_fill()
    # test_hybrid_explore_count()

    # Debug tests for memory issues
    test_knn_explore_rerank_debug()
    # test_rerank_step_by_step()

    # python -m elastics.videos.tests
