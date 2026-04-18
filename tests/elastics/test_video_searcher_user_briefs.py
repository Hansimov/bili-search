from unittest.mock import MagicMock

from elastics.videos.searcher_v2 import VideoSearcherV2


def test_get_user_briefs_projects_face_and_video_count_only():
    searcher = VideoSearcherV2.__new__(VideoSearcherV2)
    searcher.mongo = MagicMock()
    searcher.mongo.get_agg_cursor.return_value = iter(
        [
            {
                "mid": 946974,
                "name": "影视飓风",
                "face": "https://example.com/face.jpg",
                "video_count": 321,
            }
        ]
    )

    result = searcher.get_user_briefs(["946974", "bad-mid", 946974])

    assert result == [
        {
            "mid": 946974,
            "name": "影视飓风",
            "face": "https://example.com/face.jpg",
            "video_count": 321,
        }
    ]

    searcher.mongo.get_agg_cursor.assert_called_once()
    collection_name, pipeline = searcher.mongo.get_agg_cursor.call_args.args[:2]
    assert collection_name == "users"
    assert pipeline[0] == {"$match": {"mid": {"$in": [946974]}}}
    assert pipeline[1]["$project"] == {
        "_id": 0,
        "mid": 1,
        "name": 1,
        "face": 1,
        "video_count": {"$size": {"$ifNull": ["$videos", []]}},
    }


def test_lookup_videos_by_bvids_prefers_mongo_and_backfills_es_fields():
    searcher = VideoSearcherV2.__new__(VideoSearcherV2)
    searcher.mongo = MagicMock()
    searcher.fetch_docs_by_bvids = MagicMock(
        return_value={
            "hits": [
                {
                    "bvid": "BV1ABC",
                    "title": "ES 标题",
                    "owner": {"mid": 1, "name": "ES 作者"},
                    "stat": {"view": 321},
                }
            ]
        }
    )
    searcher.mongo.get_agg_cursor.return_value = iter(
        [
            {
                "bvid": "BV1ABC",
                "title": "Mongo 标题",
                "desc": "Mongo 描述",
                "pubdate": 1700000000,
                "owner": {"mid": 1, "name": "Mongo 作者"},
            }
        ]
    )

    result = searcher.lookup_videos(bvids=["BV1abc"])

    assert result["lookup_by"] == "bvids"
    assert result["bvids"] == ["BV1abc"]
    assert result["source_counts"] == {"mongo": 1, "es": 1}
    assert result["hits"][0]["title"] == "Mongo 标题"
    assert result["hits"][0]["desc"] == "Mongo 描述"
    assert result["hits"][0]["stat"]["view"] == 321


def test_lookup_videos_by_mids_uses_recent_mongo_pipeline_with_excludes():
    searcher = VideoSearcherV2.__new__(VideoSearcherV2)
    searcher.mongo = MagicMock()
    searcher.fetch_docs_by_bvids = MagicMock(return_value={"hits": []})
    searcher.mongo.get_agg_cursor.return_value = iter(
        [
            {
                "bvid": "BV1MID2",
                "title": "最近视频",
                "pubdate": 1700000000,
                "owner": {"mid": 39627524, "name": "食贫道"},
            }
        ]
    )

    result = searcher.lookup_videos(
        mids=["39627524"],
        limit=3,
        date_window="30d",
        exclude_bvids=["BV1OLD"],
    )

    assert result["lookup_by"] == "mids"
    assert result["mids"] == [39627524]
    assert result["total_hits"] == 1

    collection_name, pipeline = searcher.mongo.get_agg_cursor.call_args.args[:2]
    assert collection_name == "videos"
    assert pipeline[0]["$match"]["owner.mid"] == {"$in": [39627524]}
    assert pipeline[0]["$match"]["bvid"] == {"$nin": ["BV1OLD"]}
    assert "$gte" in pipeline[0]["$match"]["pubdate"]
    assert pipeline[1] == {"$sort": {"pubdate": -1}}
    assert pipeline[2] == {"$limit": 3}
