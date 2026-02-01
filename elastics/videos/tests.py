from btok import SentenceCategorizer
from tclogger import logger, logstr, dict_to_str, brk

from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
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
    test_knn_search_with_filters()
    # test_knn_explore()

    # python -m elastics.videos.tests
