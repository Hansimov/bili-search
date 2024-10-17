from tclogger import logger, logstr, dict_to_str

from elastics.videos.searcher import VideoSearcher
from converters.query.dsl import MultiMatchQueryDSLConstructor
from converters.query.dsl import ScriptScoreQueryDSLConstructor
from converters.query.filter import QueryFilterExtractor

searcher = VideoSearcher("bili_videos_dev3")


def test_random():
    logger.note("> Getting random results ...")
    res = searcher.random(limit=3)
    logger.mesg(dict_to_str(res))


def test_suggest():
    query = "影视飓feng"
    logger.note(f"> Query: [{logstr.mesg(query)}]")
    res = searcher.suggest(query, limit=50, verbose=True)
    hits = res.pop("hits")
    logger.success(f"✓ Suggest results:")
    logger.success(dict_to_str(res), indent=2)
    # for idx, hit in enumerate(hits):
    #     logger.note(f"* Hit {idx}:")
    #     logger.file(dict_to_str(hit, align_list=False), indent=4)


def test_multi_level_search():
    query = "影视"
    logger.note("> Searching results:", end=" ")
    logger.file(f"[{query}]")
    res = searcher.multi_level_search(
        query, limit=500, use_script_score=True, verbose=True
    )
    hits = res.pop("hits")
    logger.success(dict_to_str(res))
    # for idx, hit in enumerate(hits):
    #     logger.note(f"* Hit {idx}:")
    #     logger.file(dict_to_str(hit, align_list=False), indent=4)


def test_search():
    query = "影视飓feng"
    logger.note("> Searching results:", end=" ")
    logger.file(f"[{query}]")
    res = searcher.search(
        query, limit=50, detail_level=3, use_script_score=True, verbose=True
    )
    hits = res.pop("hits")
    logger.success(dict_to_str(res))
    # for idx, hit in enumerate(hits):
    #     logger.note(f"* Hit {idx}:")
    #     logger.file(dict_to_str(hit, align_list=False), indent=4)


def test_filter():
    query = "Hansimov 2018"
    query = "黑神话 2024 :coin>1000 :view<100000"
    match_fields = ["title^2.5", "owner.name^2", "desc", "pubdate_str^2.5"]
    date_match_fields = ["title^0.5", "owner.name^0.25", "desc^0.2", "pubdate_str^2.5"]
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
    script_query_dsl_dict = ScriptScoreQueryDSLConstructor().construct(query_dsl_dict)
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
    searcher.suggest("yingshi", limit=3, verbose=True, timeout=2)


if __name__ == "__main__":
    test_random()
    test_suggest()
    test_multi_level_search()
    test_search()
    test_filter()

    # python -m elastics.videos.tests
