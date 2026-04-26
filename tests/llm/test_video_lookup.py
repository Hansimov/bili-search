from llms.tools.video_lookup import coerce_search_video_lookup_arguments
from llms.tools.video_lookup import normalize_search_video_lookup_arguments
from llms.tools.video_lookup import parse_search_video_lookup_query


def test_parse_search_video_lookup_query_supports_mid_date_window():
    assert parse_search_video_lookup_query(":uid=39627524 :date<=30d") == (
        "mid",
        "39627524",
        "30d",
    )
    assert parse_search_video_lookup_query(":mid = 39627524 :date <= 30d") == (
        "mid",
        "39627524",
        "30d",
    )


def test_coerce_search_video_lookup_arguments_rewrites_explicit_bvid_queries():
    assert coerce_search_video_lookup_arguments({"queries": ["BV1e9cfz5EKj"]}) == {
        "mode": "lookup",
        "bv": "BV1e9cfz5EKj",
    }


def test_coerce_search_video_lookup_arguments_rewrites_author_only_query():
    assert coerce_search_video_lookup_arguments(
        {"queries": [":uid=39627524 :date <= 30d"], "limit": 3}
    ) == {
        "mode": "lookup",
        "limit": 3,
        "date_window": "30d",
        "mid": "39627524",
    }


def test_coerce_search_video_lookup_arguments_accepts_numeric_mid():
    assert coerce_search_video_lookup_arguments({"mid": 39627524, "limit": 3}) == {
        "mid": "39627524",
        "limit": 3,
        "mode": "lookup",
    }


def test_normalize_search_video_lookup_arguments_keeps_mixed_query_unchanged():
    arguments = {"queries": ["BV1e9cfz5EKj", "黑神话悟空"]}

    assert normalize_search_video_lookup_arguments(arguments) == arguments
