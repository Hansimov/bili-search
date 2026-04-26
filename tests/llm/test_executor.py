from unittest.mock import MagicMock

from llms.tools.executor import ToolExecutor


def test_search_videos_coerces_explicit_bvid_query_to_lookup_request():
    mock_search = MagicMock()
    mock_search.lookup_videos.return_value = {
        "lookup_by": "bvid",
        "hits": [
            {
                "bvid": "BV1e9cfz5EKj",
                "title": "ComfyUI 入门教程",
                "owner": {"name": "作者A", "mid": 12345},
                "link": "https://www.bilibili.com/video/BV1e9cfz5EKj",
            }
        ],
        "total_hits": 1,
        "source_counts": {"lookup_videos": 1},
    }

    executor = ToolExecutor(search_client=mock_search, google_client=MagicMock())

    result = executor._search_videos({"queries": ["BV1e9cfz5EKj"]})

    mock_search.lookup_videos.assert_called_once_with(
        bvids=["BV1e9cfz5EKj"],
        mids=None,
        limit=10,
        date_window=None,
        exclude_bvids=None,
        verbose=False,
    )
    assert result["mode"] == "lookup"
    assert result["bvids"] == ["BV1e9cfz5EKj"]
    assert result["total_hits"] == 1


def test_search_videos_coerces_author_only_query_to_lookup_request():
    mock_search = MagicMock()
    mock_search.lookup_videos.return_value = {
        "lookup_by": "mid",
        "hits": [
            {
                "bvid": "BV1abcabcab1",
                "title": "近期投稿",
                "owner": {"name": "作者A", "mid": 39627524},
                "link": "https://www.bilibili.com/video/BV1abcabcab1",
            }
        ],
        "total_hits": 1,
        "source_counts": {"mongo": 1},
    }

    executor = ToolExecutor(search_client=mock_search, google_client=MagicMock())

    result = executor._search_videos(
        {"queries": [":uid=39627524 :date <= 30d"], "limit": 3}
    )

    mock_search.lookup_videos.assert_called_once_with(
        bvids=None,
        mids=["39627524"],
        limit=3,
        date_window="30d",
        exclude_bvids=None,
        verbose=False,
    )
    assert result["mode"] == "lookup"
    assert result["mids"] == ["39627524"]
    assert result["date_window"] == "30d"
    assert result["source_counts"] == {"mongo": 1}


def test_search_videos_accepts_numeric_mid_lookup_request():
    mock_search = MagicMock()
    mock_search.lookup_videos.return_value = {
        "lookup_by": "mid",
        "hits": [],
        "total_hits": 0,
        "source_counts": {"mongo": 0},
    }

    executor = ToolExecutor(search_client=mock_search, google_client=MagicMock())

    result = executor._search_videos({"mid": 39627524, "limit": 3})

    mock_search.lookup_videos.assert_called_once_with(
        bvids=None,
        mids=["39627524"],
        limit=3,
        date_window=None,
        exclude_bvids=None,
        verbose=False,
    )
    assert result["mode"] == "lookup"
    assert result["mids"] == ["39627524"]


def test_search_videos_keeps_mixed_query_out_of_lookup_mode():
    mock_search = MagicMock()
    executor = ToolExecutor(search_client=mock_search, google_client=MagicMock())

    result = executor._search_videos({"queries": ["BV1e9cfz5EKj", "黑神话"]})

    mock_search.lookup_videos.assert_not_called()
    assert result == {
        "results": [
            {
                "query": "BV1e9cfz5EKj",
                "error": "Search explore unavailable",
                "hits": [],
                "total_hits": 0,
            },
            {
                "query": "黑神话",
                "error": "Search explore unavailable",
                "hits": [],
                "total_hits": 0,
            },
        ]
    }
