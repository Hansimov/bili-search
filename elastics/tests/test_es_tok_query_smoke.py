"""Fast smoke tests for es_tok_query_string exact-segment behavior.

Usage:
    pytest elastics/tests/test_es_tok_query_smoke.py -q
"""

from elastics.videos.constants import ELASTIC_DEV, ELASTIC_VIDEOS_DEV_INDEX
from elastics.videos.searcher_v2 import VideoSearcherV2


def make_searcher() -> VideoSearcherV2:
    return VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX,
        elastic_env_name=ELASTIC_DEV,
    )


def extract_bvids(res: dict) -> set[str]:
    return {hit.get("bvid") for hit in res.get("hits", []) if hit.get("bvid")}


def live_search(searcher: VideoSearcherV2, query: str, limit: int = 10) -> dict:
    return searcher.search(
        query,
        source_fields=["bvid", "title"],
        add_highlights_info=False,
        limit=limit,
        timeout=5,
        verbose=False,
    )


def test_smoke_required_split_chinese_segment():
    searcher = make_searcher()
    res = live_search(searcher, "+若生命将于明日落幕")
    assert "BV1v8w8zwEBQ" in extract_bvids(res)


def test_smoke_quoted_split_chinese_segment():
    searcher = make_searcher()
    res = live_search(searcher, '"若生命将于明日落幕"')
    assert "BV1v8w8zwEBQ" in extract_bvids(res)


def test_smoke_keyword_with_excluded_exact_segment():
    searcher = make_searcher()
    res = live_search(searcher, "游戏音乐 -若生命将于明日落幕", limit=20)
    assert "BV1v8w8zwEBQ" not in extract_bvids(res)


def test_smoke_quoted_named_segment():
    searcher = make_searcher()
    res = live_search(searcher, '"工藤晴香"')
    assert "BV1gcwuzhEaX" in extract_bvids(res)
