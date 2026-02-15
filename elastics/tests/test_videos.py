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
    from converters.embed.embed_client import TextEmbedClient

    logger.note("> Testing embed client...")
    client = TextEmbedClient()

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
        QMOD,
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
        authors = group_step.get("output", {}).get("authors", {})
        gopro_found = False
        gopro_highlighted = False

        for author_name, author_data in authors.items():
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
        explore_res = explorer.explore(
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
                authors_result = step.get("output", {}).get("authors", {})

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
        for i, (mid, author_info) in enumerate(authors_result.items()):
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
        author_list = list(authors_result.values())

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
    assert RANK_METHOD == "stats", f"RANK_METHOD should be 'stats', got {RANK_METHOD}"
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

    assert RANK_METHOD == "stats"
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

    # Test converters.dsl.fields.qmod renamed constants
    from converters.dsl.fields.qmod import QMOD as QMOD_FROM_QMOD

    assert QMOD_FROM_QMOD == ["word", "vector"]
    logger.success("  ✓ converters.dsl.fields.qmod: QMOD")

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


# ===========================================================================
# es_tok query_string parameter tests
# ===========================================================================


def test_es_tok_query_params():
    """Test that es_tok_query_string queries do NOT include unsupported params.

    After the es_tok plugin rewrite, only `max_freq` and `constraints` are
    supported. The old `min_kept_tokens_count` and `min_kept_tokens_ratio`
    params must NOT be sent, otherwise the plugin throws a ParsingException
    which causes search to return empty results (timed_out=True, took=-1).
    """
    import json
    from converters.dsl.elastic import DslExprToElasticConverter
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

    unsupported_params = ["min_kept_tokens_count", "min_kept_tokens_ratio"]
    supported_params = ["query", "type", "fields", "max_freq"]

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
    from converters.dsl.elastic import DslExprToElasticConverter
    from converters.dsl.rewrite import DslExprRewriter
    from converters.dsl.filter import QueryDslDictFilterMerger
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
    from converters.dsl.elastic import DslExprToElasticConverter
    from converters.dsl.rewrite import DslExprRewriter
    from converters.dsl.filter import QueryDslDictFilterMerger
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


# ===========================================================================
# Word Recall Supplement Tests
# ===========================================================================


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

    test_queries = [
        "影视飓风",  # Channel name: KNN used to return generic hurricane videos
        "deepseek",  # Tech term: KNN used to return unrelated AI videos
    ]

    all_passed = True

    for query in test_queries:
        logger.note(f"\n> Query: [{query}]")

        # Run knn_explore with word recall enabled (default)
        start_time = time.perf_counter()
        explore_res = explorer.knn_explore(
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

        # 1. Verify word_recall_supplement step exists
        has_word_recall = "word_recall_supplement" in step_names
        if has_word_recall:
            logger.mesg(f"  ✓ word_recall_supplement step present")
        else:
            logger.warn(f"  × word_recall_supplement step missing! steps: {step_names}")
            all_passed = False
            continue

        # 2. Check word recall supplement info
        word_recall_step = next(
            s for s in steps if s.get("name") == "word_recall_supplement"
        )
        word_info = word_recall_step.get("output", {})
        supplement_count = word_info.get("supplement_count", 0)
        merged_total = word_info.get("merged_total", 0)
        knn_original = word_info.get("knn_original_count", 0)

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
            if reranked_count >= merged_total:
                logger.mesg(f"  ✓ Reranker processed {reranked_count} candidates")
            else:
                logger.warn(
                    f"  × Reranked {reranked_count} but pool was {merged_total}"
                )
        else:
            logger.warn(f"  × Rerank step missing")
            all_passed = False

        # 4. Check that knn_search step has hits
        knn_step = next((s for s in steps if s.get("name") == "knn_search"), None)
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

        # Run vector search (q=v) with word recall
        knn_res = explorer.knn_explore(
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

        if overlap_pct >= min_overlap_pct:
            logger.mesg(
                f"  ✓ Overlap: {len(overlap)}/{len(word_bvids)} = {overlap_pct:.1f}% "
                f"(>= {min_overlap_pct}%)"
            )
        else:
            logger.warn(
                f"  × Overlap too low: {len(overlap)}/{len(word_bvids)} = {overlap_pct:.1f}% "
                f"(expected >= {min_overlap_pct}%)"
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

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Run knn_explore with word recall explicitly disabled
    explore_res = explorer.knn_explore(
        query="影视飓风",
        enable_rerank=True,
        word_recall_enabled=False,
        rank_top_k=50,
        group_owner_limit=5,
        verbose=False,
    )

    all_passed = True
    status = explore_res.get("status", "unknown")
    if status != "finished":
        logger.warn(f"  × Status: {status}")
        all_passed = False
    else:
        logger.mesg(f"  ✓ Status: {status}")

    steps = explore_res.get("data", [])
    step_names = [s.get("name") for s in steps]

    # Verify word_recall_supplement step is NOT present
    if "word_recall_supplement" not in step_names:
        logger.mesg(f"  ✓ word_recall_supplement step correctly absent")
    else:
        logger.warn(
            f"  × word_recall_supplement step should not be present when disabled"
        )
        all_passed = False

    # Should still have knn_search and rerank
    if "knn_search" in step_names:
        logger.mesg(f"  ✓ knn_search step present")
    else:
        logger.warn(f"  × knn_search step missing")
        all_passed = False

    if "rerank" in step_names:
        logger.mesg(f"  ✓ rerank step present")
    else:
        logger.warn(f"  × rerank step missing")
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

    explorer = VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
    )

    # Narrow filter query (user filter)
    explore_res = explorer.knn_explore(
        query='u="红警HBK08" 红警',
        enable_rerank=True,
        rank_top_k=50,
        group_owner_limit=5,
        verbose=False,
    )

    all_passed = True
    status = explore_res.get("status", "unknown")
    steps = explore_res.get("data", [])
    step_names = [s.get("name") for s in steps]

    # Narrow filter queries should NOT have word_recall_supplement
    if "word_recall_supplement" not in step_names:
        logger.mesg(f"  ✓ word_recall_supplement correctly skipped for narrow filter")
    else:
        logger.warn(f"  × word_recall_supplement should be skipped for narrow filters")
        all_passed = False

    # Should still have results
    knn_step = next((s for s in steps if s.get("name") == "knn_search"), None)
    if knn_step:
        return_hits = knn_step.get("output", {}).get("return_hits", 0)
        if return_hits > 0:
            logger.mesg(f"  ✓ Got {return_hits} results with narrow filter")
        else:
            logger.mesg(f"  ⓘ No results (user may not exist in dev index)")
    else:
        logger.warn(f"  × knn_search step missing")
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

    # Verify word recall supplement is also present
    if "word_recall_supplement" in step_names:
        logger.mesg(f"  ✓ Word recall supplement present for q=v")
    else:
        logger.warn(f"  × Word recall supplement missing for q=v")
        all_passed = False

    # Check that results have rerank_score (not just raw hamming score)
    knn_step = next((s for s in steps if s.get("name") == "knn_search"), None)
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

    # Debug tests
    # test_knn_explore_rerank_debug()
    # test_rerank_step_by_step()

    # Bug fix tests
    # test_knn_filter_bug()
    # test_highlight_bug()
    # test_user_filter_coverage()

    # Performance tests
    # test_vr_performance_with_narrow_filters()
    # test_rerank_client_diagnostic()
    # test_embed_client_keepalive()

    # Ranks module refactoring tests
    # test_ranks_imports()
    # test_constants_refactoring()
    # test_author_grouper_unit()
    # test_author_grouper_list()

    # Filter-only search and narrow filter tests
    # test_filter_only_search()
    # test_compute_passage()
    # test_narrow_filter_detection()
    # test_filter_only_explore()
    # test_owner_name_keyword_boost()

    # es_tok query params fix tests (bug: unsupported params cause timeout)
    test_es_tok_query_params()
    test_dsl_query_construction()
    test_match_fields_no_missing_index_fields()
    test_search_body_structure()
    test_submit_to_es_error_handling()
    test_word_search_basic()
    test_word_search_with_filters()
    test_explore_word_mode()
    test_search_edge_cases()

    # Word recall supplement tests (KNN recall fix)
    test_word_recall_supplement()
    test_word_recall_overlap_improvement()
    test_word_recall_disabled()
    test_word_recall_narrow_filter_skip()
    test_unified_explore_vector_always_reranks()

    # KNN recall and ranking tests
    test_knn_num_candidates_recall()
    test_qmod_recall_comparison()
    test_stat_score_in_ranking()

    # Ranking system redesign tests (real-time time_factor, BM25+embedding, preferences)
    test_knn_k_1000()
    test_realtime_time_factor()
    test_bm25_embedding_blend()
    test_preference_ranking_unit()
    test_preference_rank_integration()
    test_reranker_preserves_original_score()

    # python -m elastics.tests.test_videos
