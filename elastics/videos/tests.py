from tclogger import logger, logstr, dict_to_str, brk

from elastics.videos.searcher import VideoSearcherV2
from elastics.videos.explorer import VideoExplorer
from elastics.videos.constants import VIDEOS_INDEX_DEFAULT
from converters.query.dsl import MultiMatchQueryDSLConstructor
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.query.filter import QueryFilterExtractor

searcher = VideoSearcherV2(VIDEOS_INDEX_DEFAULT)
explorer = VideoExplorer(VIDEOS_INDEX_DEFAULT)


def test_random():
    logger.note("> Getting random results ...")
    res = searcher.random(limit=3)
    logger.mesg(dict_to_str(res))


filter_queries = [
    "Hansimov 2018",
    "黑神话 2024 :coin>1000 :view<100000",
]


def test_filter():
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
    '"影视飓风"'
]


def test_search():
    for query in search_queries:
        logger.note("> Searching results:", end=" ")
        logger.file(f"[{query}]")
        res = searcher.search(
            query,
            limit=50,
            detail_level=1,
            use_script_score=True,
            verbose=True,
        )
        # hits = res.pop("hits")
        # logger.success(dict_to_str(res))
        # for idx, hit in enumerate(hits[:3]):
        #     logger.note(f"* Hit {idx}:")
        #     logger.file(dict_to_str(hit, align_list=False), indent=4)


def test_agg():
    for query in search_queries:
        logger.note("> Agg results:", end=" ")
        logger.file(f"[{query}]")
        res = searcher.agg(query, verbose=False)
        logger.success(dict_to_str(res, align_list=False), indent=2)


def test_explore():
    for query in search_queries:
        logger.note("> Explore results:", end=" ")
        logger.file(f"[{query}]")
        explore_res = explorer.explore(query, verbose=True)
        for step_res in explore_res:
            stage_name = step_res["name"]
            logger.hint(f"* stage result of {(logstr.mesg(brk(stage_name)))}:")
            logger.mesg(dict_to_str(step_res, align_list=False), indent=2)


if __name__ == "__main__":
    # test_random()
    # test_filter()
    # test_suggest()
    # test_multi_level_search()
    # test_search()
    # test_agg()
    test_explore()

    # python -m elastics.videos.tests
