from typing import Literal

STATS_FIELDS_FOR_LLM_SUGGEST = ["stat.view", "stat.coin", "stat.danmaku"]
STATS_FIELDS_FOR_LLM_SEARCH = ["stat.view", "stat.coin", "stat.danmaku"]
STATS_FIELDS_FOR_LLM_EXPLORE = ["stat.view", "stat.coin", "stat.danmaku"]

SOURCE_FIELDS_FOR_LLM_SUGGEST = [
    *["title", "bvid", "owner", "pubdate"],
    *STATS_FIELDS_FOR_LLM_SUGGEST,
]

SOURCE_FIELDS_FOR_LLM_SEARCH = [
    *["title", "desc", "tags"],
    *["bvid", "pic", "owner", "pubdate"],
    *STATS_FIELDS_FOR_LLM_SEARCH,
]

SOURCE_FIELDS_FOR_LLM_EXPLORE = [
    *["title", "desc", "tags"],
    *["bvid", "pic", "owner", "pubdate"],
    *STATS_FIELDS_FOR_LLM_EXPLORE,
]
