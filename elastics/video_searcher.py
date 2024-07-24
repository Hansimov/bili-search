from typing import Literal

from elastics.client import ElasticSearchClient
from elastics.video_details_searcher import VideoDetailsSearcher


class VideoSearcher(VideoDetailsSearcher):
    SOURCE_FIELDS = [
        "title",
        "bvid",
        "owner",
        "pic",
        "duration",
        "desc",
        "stat",
        "pubdate_str",
        "insert_at_str",
    ]
    SUGGEST_MATCH_FIELDS = ["title", "title.pinyin"]
    SEARCH_MATCH_FIELDS = [
        "title",
        "title.pinyin",
        "owner.name",
        "owner.name.pinyin",
        "desc",
        "desc.pinyin",
        "pubdate_str",
    ]
    BOOSTED_FIELDS = {
        "title": 2.5,
        "owner.name": 2,
        "desc": 1,
        "pubdate_str": 2.5,
    }
    DOC_EXCLUDED_SOURCE_FIELDS = []

    MATCH_TYPE = Literal[
        "best_fields",
        "most_fields",
        "cross_fields",
        "phrase",
        "phrase_prefix",
        "bool_prefix",
    ]
    MATCH_BOOL = Literal["must", "should", "must_not", "filter"]
    MATCH_OPERATOR = Literal["or", "and"]

    SUGGEST_MATCH_TYPE = "phrase_prefix"
    SEARCH_MATCH_TYPE = "phrase_prefix"
    SEARCH_MATCH_BOOL = "must"
    SEARCH_MATCH_OPERATOR = "or"

    SEARCH_DETAIL_LEVELS = {
        1: {"match_type": "phrase_prefix", "bool": "must", "pinyin": False},
        2: {"match_type": "cross_fields", "bool": "must", "operator": "and"},
        3: {"match_type": "cross_fields", "bool": "must"},
        4: {"match_type": "most_fields", "bool": "must"},
        5: {"match_type": "most_fields", "bool": "should"},
    }
    MAX_SEARCH_DETAIL_LEVEL = 4

    SUGGEST_LIMIT = 10
    SEARCH_LIMIT = 50
    # This constant is to contain more hits for redundance,
    # as drop_no_highlights would drop some hits
    NO_HIGHLIGHT_REDUNDANCE_RATIO = 2

    def __init__(self, index_name: str = "bili_videos"):
        self.index_name = index_name
        self.es = ElasticSearchClient()
        self.es.connect()


if __name__ == "__main__":
    searcher = VideoSearcher("bili_videos_dev")
    searcher.search("碧诗", boost=True, detail_level=1, limit=10, verbose=True)

    # python -m elastics.video_searcher
