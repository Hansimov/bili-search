from btok import SentenceCategorizer
from tclogger import logger, logstr, dict_to_str, brk

from elastics.videos.constants import (
    ELASTIC_VIDEOS_DEV_INDEX,
    ELASTIC_VIDEOS_PRO_INDEX,
    ELASTIC_DEV,
)
from elastics.videos.searcher import VideoSearcherV1
from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.explorer import VideoExplorer
from elastics.videos.splitter import QuerySplitter
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


def test_split():
    splitter = QuerySplitter(ELASTIC_VIDEOS_DEV_INDEX)
    for query in split_queries:
        logger.note("> Splitting:", end=" ")
        logger.file(f"[{query}]")
        res = splitter.split(query)
        logger.success(dict_to_str(res, add_quotes=True), indent=2)


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
    test_rrf_fusion_fill()
    test_hybrid_explore_count()

    # python -m elastics.videos.tests
