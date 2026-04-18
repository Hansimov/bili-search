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
