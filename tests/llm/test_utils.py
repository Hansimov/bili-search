"""Tests for llms.tools.utils — hit formatting and analysis utilities.

These tests are pure unit tests with no external dependencies.

Run:
    python -m tests.llm.test_utils
"""

from tclogger import logger

from llms.tools.utils import (
    extract_field,
    shrink_hit,
    add_link,
    format_hit_for_llm,
    format_hits_for_llm,
    extract_explore_hits,
    extract_explore_authors,
    analyze_suggest_for_authors,
    format_view_count,
)


# ============================================================
# Test data
# ============================================================

SAMPLE_HIT = {
    "bvid": "BV1abc123",
    "title": "测试视频标题",
    "desc": "这是一个测试描述",
    "tags": "标签1,标签2",
    "pic": "http://example.com/pic.jpg",
    "owner": {"mid": 12345, "name": "测试UP主", "face": "http://example.com/face.jpg"},
    "pubdate": 1708700000,
    "stat": {
        "view": 123456,
        "coin": 5678,
        "danmaku": 1234,
        "like": 9999,
        "reply": 456,
        "favorite": 789,
        "share": 123,
    },
    "duration": 360,
    "score": 0.95,
}

SAMPLE_EXPLORE_RESULT = {
    "query": "黑神话",
    "status": "finished",
    "data": [
        {
            "step": 0,
            "name": "most_relevant_search",
            "output": {
                "hits": [
                    {
                        "bvid": "BV1a",
                        "title": "Video 1",
                        "owner": {"mid": 1, "name": "A"},
                    },
                    {
                        "bvid": "BV1b",
                        "title": "Video 2",
                        "owner": {"mid": 2, "name": "B"},
                    },
                ],
                "total_hits": 42,
            },
        },
        {
            "step": 1,
            "name": "group_hits_by_owner",
            "output": {
                "authors": [
                    {"mid": 1, "name": "A", "count": 3},
                    {"mid": 2, "name": "B", "count": 2},
                ],
            },
        },
    ],
}

SAMPLE_SUGGEST_RESULT = {
    "query": "影视飓风",
    "total_hits": 25,
    "hits": [
        {
            "bvid": "BV1x",
            "title": "影视飓风的视频1",
            "owner": {"mid": 946974, "name": "影视飓风"},
            "highlights": {"merged": ["<em>影视飓风</em>的视频1"]},
        },
        {
            "bvid": "BV1y",
            "title": "影视飓风评测",
            "owner": {"mid": 946974, "name": "影视飓风"},
            "highlights": {"merged": ["<em>影视飓风</em>评测"]},
        },
        {
            "bvid": "BV1z",
            "title": "飓风来了",
            "owner": {"mid": 1780480185, "name": "飓多多StormCrew"},
            "highlights": {"merged": ["<em>飓风</em>来了"]},
        },
    ],
}


# ============================================================
# Tests
# ============================================================


def test_extract_field():
    """Test nested field extraction."""
    logger.note("=" * 60)
    logger.note("[TEST] extract_field")

    assert extract_field(SAMPLE_HIT, "bvid") == "BV1abc123"
    assert extract_field(SAMPLE_HIT, "stat.view") == 123456
    assert extract_field(SAMPLE_HIT, "owner.name") == "测试UP主"
    assert extract_field(SAMPLE_HIT, "nonexistent") is None
    assert extract_field(SAMPLE_HIT, "stat.nonexistent") is None

    logger.success("[PASS] extract_field")


def test_shrink_hit():
    """Test hit shrinking to specified fields."""
    logger.note("=" * 60)
    logger.note("[TEST] shrink_hit")

    fields = ["title", "bvid", "stat.view", "owner"]
    result = shrink_hit(SAMPLE_HIT, fields)

    assert result["title"] == "测试视频标题"
    assert result["bvid"] == "BV1abc123"
    assert result["stat"]["view"] == 123456
    assert result["owner"]["mid"] == 12345
    assert "desc" not in result
    assert "score" not in result
    assert "duration" not in result

    logger.success("[PASS] shrink_hit")


def test_add_link():
    """Test adding bilibili link."""
    logger.note("=" * 60)
    logger.note("[TEST] add_link")

    hit = {"bvid": "BV1test"}
    add_link(hit)
    assert hit["link"] == "https://www.bilibili.com/video/BV1test"

    # No bvid
    hit2 = {"title": "no bvid"}
    add_link(hit2)
    assert "link" not in hit2

    logger.success("[PASS] add_link")


def test_format_hit_for_llm():
    """Test full hit formatting for LLM."""
    logger.note("=" * 60)
    logger.note("[TEST] format_hit_for_llm")

    result = format_hit_for_llm(SAMPLE_HIT)

    assert "link" in result
    assert "bvid" in result
    assert "title" in result
    assert "owner" in result
    # Should NOT have internal fields
    assert "score" not in result
    assert "duration" not in result

    logger.success("[PASS] format_hit_for_llm")


def test_format_hits_max():
    """Test max_hits limit in format_hits_for_llm."""
    logger.note("=" * 60)
    logger.note("[TEST] format_hits_for_llm max_hits")

    hits = [{"bvid": f"BV{i}", "title": f"Video {i}"} for i in range(30)]
    result = format_hits_for_llm(hits, max_hits=10)
    assert len(result) == 10

    result_all = format_hits_for_llm(hits, max_hits=50)
    assert len(result_all) == 30

    logger.success("[PASS] format_hits_for_llm max_hits")


def test_extract_explore_hits():
    """Test extracting hits from explore response."""
    logger.note("=" * 60)
    logger.note("[TEST] extract_explore_hits")

    hits, total = extract_explore_hits(SAMPLE_EXPLORE_RESULT)
    assert len(hits) == 2
    assert total == 42
    assert hits[0]["bvid"] == "BV1a"

    # Empty result
    hits_empty, total_empty = extract_explore_hits({"data": []})
    assert len(hits_empty) == 0
    assert total_empty == 0

    # No data key
    hits_none, total_none = extract_explore_hits({})
    assert len(hits_none) == 0
    assert total_none == 0

    logger.success("[PASS] extract_explore_hits")


def test_extract_explore_authors():
    """Test extracting authors from explore response."""
    logger.note("=" * 60)
    logger.note("[TEST] extract_explore_authors")

    authors = extract_explore_authors(SAMPLE_EXPLORE_RESULT)
    assert len(authors) == 2
    assert authors[0]["name"] == "A"

    # No group step
    authors_empty = extract_explore_authors(
        {"data": [{"name": "search", "output": {}}]}
    )
    assert len(authors_empty) == 0

    logger.success("[PASS] extract_explore_authors")


def test_analyze_suggest_for_authors():
    """Test suggest result analysis for author detection."""
    logger.note("=" * 60)
    logger.note("[TEST] analyze_suggest_for_authors")

    result = analyze_suggest_for_authors(SAMPLE_SUGGEST_RESULT, "影视飓风")

    assert result["query"] == "影视飓风"
    assert result["total_hits"] == 25
    assert "影视飓风" in result["highlighted_keywords"]
    assert "影视飓风" in result["related_authors"]

    author = result["related_authors"]["影视飓风"]
    assert author["uid"] == 946974
    assert author["ratio"] > 0.5  # Dominates results
    assert author.get("highlighted") == True

    logger.success(f"  Author ratio: {author['ratio']}")
    logger.success(f"  Keywords: {result['highlighted_keywords']}")
    logger.success("[PASS] analyze_suggest_for_authors")


def test_analyze_suggest_empty():
    """Test suggest analysis with empty results."""
    logger.note("=" * 60)
    logger.note("[TEST] analyze_suggest_for_authors (empty)")

    result = analyze_suggest_for_authors({"hits": [], "total_hits": 0}, "nothing")
    assert result["total_hits"] == 0
    assert result["highlighted_keywords"] == {}
    assert result["related_authors"] == {}

    logger.success("[PASS] analyze_suggest_for_authors (empty)")


def test_format_view_count():
    """Test view count formatting."""
    logger.note("=" * 60)
    logger.note("[TEST] format_view_count")

    assert format_view_count(500) == "500"
    assert format_view_count(10000) == "1.0万"
    assert format_view_count(123456) == "12.3万"
    assert format_view_count(1500000) == "150.0万"
    assert format_view_count(100000000) == "1.0亿"
    assert format_view_count(None) == ""

    logger.success("[PASS] format_view_count")


if __name__ == "__main__":
    tests = [
        ("extract_field", test_extract_field),
        ("shrink_hit", test_shrink_hit),
        ("add_link", test_add_link),
        ("format_hit_for_llm", test_format_hit_for_llm),
        ("format_hits_max", test_format_hits_max),
        ("extract_explore_hits", test_extract_explore_hits),
        ("extract_explore_authors", test_extract_explore_authors),
        ("analyze_suggest_for_authors", test_analyze_suggest_for_authors),
        ("analyze_suggest_empty", test_analyze_suggest_empty),
        ("format_view_count", test_format_view_count),
    ]

    results = {}
    for name, test_func in tests:
        try:
            test_func()
            results[name] = "PASS"
        except Exception as e:
            logger.warn(f"\n[FAIL] {name}: {e}")
            results[name] = f"FAIL: {e}"

    logger.note("\n" + "=" * 60)
    logger.note("[SUMMARY]")
    logger.note("=" * 60)
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    for name, result in results.items():
        status = logger.success if result == "PASS" else logger.warn
        status(f"  {name}: {result}")
    logger.note(f"\n  {passed}/{total} tests passed")

    # python -m tests.llm.test_utils
