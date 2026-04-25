"""Title-match and relevance-gating tests for diversified ranking."""

import time

from tclogger import logger

from ranks.diversified import DiversifiedRanker


def _make_hit(bvid, score=1.0, view=1000, pubdate=None, stat_score=0.5, **kw):
    """Create a test hit dict with required fields."""
    if pubdate is None:
        pubdate = int(time.time()) - 86400
    return {
        "bvid": bvid,
        "score": score,
        "stat": {
            "view": view,
            "coin": view // 10,
            "like": view // 5,
            "favorite": view // 20,
            "danmaku": view // 50,
            "reply": view // 50,
        },
        "pubdate": pubdate,
        "stat_score": stat_score,
        **kw,
    }


def test_title_match_tagging_title_only():
    """_tag_title_matches should tag docs when query appears in title."""
    logger.note("> Test: title-match tagging on title field")
    from recalls.word import MultiLaneWordRecall

    hits = [
        {"bvid": "BV001", "title": "通义实验室发布新模型", "tags": "AI"},
        {"bvid": "BV002", "title": "AI模型测试", "tags": "通义实验室,测试"},
        {"bvid": "BV003", "title": "无关视频", "tags": "无关标签"},
    ]

    MultiLaneWordRecall._tag_title_matches(hits, "通义实验室")

    # BV001: query in title → should be tagged
    assert hits[0].get("_title_matched") is True, "Query in title should be tagged"
    # BV002: query not in title, but in tags → should be tagged
    assert hits[1].get("_title_matched") is True, "Query in tags should be tagged"
    # BV003: query in neither → should not be tagged
    assert not hits[2].get("_title_matched"), "Non-matching doc should not be tagged"

    logger.success("  PASSED")


def test_title_match_tagging_tags_only():
    """_tag_title_matches should tag docs when query appears only in tags."""
    logger.note("> Test: title-match tagging on tags field")
    from recalls.word import MultiLaneWordRecall

    hits = [
        {"bvid": "BV001", "title": "一个普通视频", "tags": "飓风营救,动作,电影"},
        {"bvid": "BV002", "title": "飓风营救完整版", "tags": "电影"},
        {"bvid": "BV003", "title": "天气预报", "tags": "气象"},
    ]

    MultiLaneWordRecall._tag_title_matches(hits, "飓风营救")

    assert hits[0].get("_title_matched") is True, "Query in tags should be tagged"
    assert hits[1].get("_title_matched") is True, "Query in title should be tagged"
    assert not hits[2].get("_title_matched"), "Non-matching should not be tagged"

    logger.success("  PASSED")


def test_title_match_tagging_short_query():
    """Short queries (<=2 chars) should match as substrings in title or tags."""
    logger.note("> Test: title-match tagging with short query")
    from recalls.word import MultiLaneWordRecall

    hits = [
        {"bvid": "BV001", "title": "红警08对战", "tags": "游戏"},
        {"bvid": "BV002", "title": "新版本更新", "tags": "红警,攻略"},
        {"bvid": "BV003", "title": "完全无关", "tags": "其他"},
    ]

    MultiLaneWordRecall._tag_title_matches(hits, "红警")

    assert hits[0].get("_title_matched") is True, "Short query in title"
    assert hits[1].get("_title_matched") is True, "Short query in tags"
    assert not hits[2].get("_title_matched"), "Short query not in either"

    logger.success("  PASSED")


def test_title_match_tagging_preserves_existing():
    """Already-tagged docs should not be overwritten by _tag_title_matches."""
    logger.note("> Test: title-match tagging preserves existing tags")
    from recalls.word import MultiLaneWordRecall

    hits = [
        {
            "bvid": "BV001",
            "title": "无关标题",
            "tags": "无关标签",
            "_title_matched": True,
        },
    ]

    MultiLaneWordRecall._tag_title_matches(hits, "其他查询")

    # Should preserve existing True tag even though query doesn't match
    assert hits[0]["_title_matched"] is True

    logger.success("  PASSED")


def test_title_match_bonus():
    """Docs with _title_matched should get boosted relevance_score."""
    logger.note("> Test: title-match bonus in scoring")

    now = int(time.time())
    # One high-score doc to set max_score, then two docs with identical BM25
    hits = [
        _make_hit("ANCHOR", score=40, view=5000, pubdate=now - 3600, stat_score=0.5),
        _make_hit("MATCH", score=20, view=5000, pubdate=now - 3600, stat_score=0.5),
        _make_hit("NO_MATCH", score=20, view=5000, pubdate=now - 3600, stat_score=0.5),
    ]
    hits[1]["_title_matched"] = True

    ranker = DiversifiedRanker()
    ranker._score_all_dimensions(hits, now_ts=now)

    # Title-matched doc should have higher relevance_score than non-matched
    assert hits[1]["relevance_score"] > hits[2]["relevance_score"], (
        f"Title-matched doc ({hits[1]['relevance_score']}) should have higher "
        f"relevance than non-matched ({hits[2]['relevance_score']})"
    )

    # The bonus should be TITLE_MATCH_BONUS (0.15)
    from ranks.constants import TITLE_MATCH_BONUS

    diff = hits[1]["relevance_score"] - hits[2]["relevance_score"]
    assert (
        abs(diff - TITLE_MATCH_BONUS) < 0.01
    ), f"Expected bonus ≈{TITLE_MATCH_BONUS}, got {diff}"

    # Title-matched doc should also have higher headline_score
    assert hits[1]["headline_score"] > hits[2]["headline_score"]

    logger.success("  PASSED")


def test_title_match_helps_ranking():
    """Title-matched doc should rank higher than non-matched with same BM25."""
    logger.note("> Test: title-match impact on full ranking")

    now = int(time.time())
    hits = []

    # 5 non-title-matched docs with high BM25
    for i in range(5):
        hits.append(
            _make_hit(
                f"PLAIN{i}",
                score=25 - i,
                view=10000,
                pubdate=now - 86400 * 7,
                stat_score=0.5,
            )
        )

    # 3 title-matched docs with moderate BM25 (slightly lower than PLAIN docs)
    for i in range(3):
        h = _make_hit(
            f"TITLE{i}",
            score=20 - i,
            view=8000,
            pubdate=now - 86400 * 7,
            stat_score=0.5,
        )
        h["_title_matched"] = True
        hits.append(h)

    # 10 filler docs
    for i in range(10):
        hits.append(_make_hit(f"FILL{i}", score=5, view=1000, stat_score=0.2))

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank_with_fused_fallback(
        hits_info={"hits": hits},
        top_k=10,
        diversify_top_n=10,
    )

    ranked = result["hits"]
    top_5_bvids = {h["bvid"] for h in ranked[:5]}

    # At least 2 title-matched docs should be in top 5
    title_in_top5 = sum(1 for bvid in top_5_bvids if bvid.startswith("TITLE"))
    assert title_in_top5 >= 2, (
        f"Expected ≥2 title-matched docs in top-5, got {title_in_top5}. "
        f"Top 5: {[h['bvid'] for h in ranked[:5]]}"
    )

    logger.success("  PASSED")


def test_relevance_gating_in_slots():
    """Relevance gating should prevent irrelevant docs from taking dimension slots."""
    logger.note("> Test: relevance gating in slot allocation")

    now = int(time.time())
    hits = []

    # 5 relevant docs with moderate stats
    for i in range(5):
        hits.append(
            _make_hit(
                f"REL{i}",
                score=40 - i,
                view=5000,
                pubdate=now - 86400 * 30,
                stat_score=0.4,
            )
        )

    # 10 irrelevant but popular/recent/quality docs
    for i in range(10):
        hits.append(
            _make_hit(
                f"IRR{i}",
                score=1.0,  # Very low BM25 → low relevance
                view=2000000,  # Very high views
                pubdate=now - 3600,  # Very recent
                stat_score=0.95,  # Very high quality
            )
        )

    ranker = DiversifiedRanker()
    result = ranker.diversified_rank(
        hits_info={"hits": hits},
        top_k=10,
        prefer="balanced",
    )

    ranked = result["hits"]

    # With relevance gating, dimension slots should be dominated by relevant docs
    # because IRR docs have low relevance_factor, gating their dimension scores.
    # At minimum, the relevance slots (2) + most other dimension slots should be REL docs
    rel_in_top = sum(1 for h in ranked if h["bvid"].startswith("REL"))
    assert rel_in_top >= 4, (
        f"Expected ≥4 relevant docs in top-10 (with gating), got {rel_in_top}. "
        f"Irrelevant docs may be bypassing relevance gating. "
        f"Ranked: {[(h['bvid'], h.get('_slot_dimension')) for h in ranked]}"
    )

    logger.success("  PASSED")


def test_graduated_threshold_relaxation():
    """When too few pass the relevance floor, threshold should relax gradually."""
    logger.note("> Test: graduated threshold relaxation")

    now = int(time.time())
    hits = []

    # Only 3 docs with high relevance (pass SLOT_MIN_RELEVANCE=0.30)
    for i in range(3):
        hits.append(
            _make_hit(
                f"HIGH{i}",
                score=30 - i,
                view=5000,
                pubdate=now - 86400,
                stat_score=0.5,
            )
        )

    # 7 docs with moderate relevance (would pass 0.15 but not 0.30)
    for i in range(7):
        hits.append(
            _make_hit(
                f"MED{i}",
                score=8 - i * 0.5,
                view=3000,
                pubdate=now - 86400 * 7,
                stat_score=0.4,
            )
        )

    ranker = DiversifiedRanker()

    # Score dimensions
    ranker._score_all_dimensions(hits, now_ts=now)

    # Allocate with top_k=10 — only 3 pass strict floor, should relax
    from ranks.diversified import SLOT_PRESETS

    slots = SLOT_PRESETS["balanced"]
    result = ranker._allocate_slots(hits, slots, top_k=10)

    # Should return 10 results (relaxation allows more candidates)
    assert len(result) == 10, f"Expected 10 results, got {len(result)}"

    # No duplicates
    bvids = [h["bvid"] for h in result]
    assert len(set(bvids)) == len(bvids), f"Duplicate bvids: {bvids}"

    logger.success("  PASSED")


def test_gated_scores_cleaned_up():
    """Temporary _gated_* fields should be cleaned up after slot allocation."""
    logger.note("> Test: gated score field cleanup")

    now = int(time.time())
    hits = [_make_hit(f"BV{i}", score=20 - i, view=1000) for i in range(15)]

    ranker = DiversifiedRanker()
    ranker._score_all_dimensions(hits, now_ts=now)

    from ranks.diversified import SLOT_PRESETS

    slots = SLOT_PRESETS["balanced"]
    ranker._allocate_slots(hits, slots, top_k=10)

    # No _gated_* fields should remain on any hit
    for h in hits:
        gated_fields = [k for k in h.keys() if k.startswith("_gated_")]
        assert not gated_fields, f"Leftover gated fields on {h['bvid']}: {gated_fields}"

    logger.success("  PASSED")
