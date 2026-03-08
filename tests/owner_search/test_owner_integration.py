import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llms.chat.handler import ChatHandler
from elastics.videos.explorer import VideoExplorer


class DummyOwnerSearcher:
    def get_owners(self, mids: list[int]) -> dict[int, dict]:
        return {
            1: {
                "mid": 1,
                "name": "影视飓风",
                "total_videos": 598,
                "total_view": 2530000000,
                "influence_score": 0.92,
                "quality_score": 0.79,
                "activity_score": 0.68,
                "profile_domain_ready": True,
                "core_tokenizer_version": "coretok-dev",
            }
        }


def test_group_hits_by_owner_enriches_owner_profile_and_face():
    explorer = VideoExplorer.__new__(VideoExplorer)
    explorer.owner_searcher = DummyOwnerSearcher()
    explorer.get_user_docs = lambda mids: {1: {"mid": 1, "face": "face-1.jpg"}}

    search_res = {
        "hits": [
            {
                "bvid": "BV1",
                "owner": {"mid": 1, "name": "旧名字"},
                "pubdate": 1700000000,
                "stat": {"view": 100},
                "sort_score": 1.0,
                "rank_score": 2.0,
            },
            {
                "bvid": "BV2",
                "owner": {"mid": 1, "name": "旧名字"},
                "pubdate": 1700000100,
                "stat": {"view": 200},
                "sort_score": 1.5,
                "rank_score": 2.5,
            },
        ]
    }

    authors = explorer.group_hits_by_owner(search_res, limit=5)

    assert len(authors) == 1
    author = authors[0]
    assert author["name"] == "影视飓风"
    assert author["sum_count"] == 2
    assert author["total_videos"] == 598
    assert author["influence_score"] == 0.92
    assert author["profile_domain_ready"] is True
    assert author["core_tokenizer_version"] == "coretok-dev"
    assert author["face"] == "face-1.jpg"


def test_chat_handler_parses_search_owners_command():
    commands = ChatHandler._parse_tool_commands(
        '我来找做黑神话的UP主。\n<search_owners query="黑神话悟空" sort_by="influence"/>'
    )

    assert commands == [
        {
            "type": "search_owners",
            "args": {"query": "黑神话悟空", "sort_by": "influence"},
        }
    ]
