from llms.orchestration.video_queries import VideoQueryNormalizer


def test_build_video_followup_focus_query_uses_structured_focus_only():
    assert (
        VideoQueryNormalizer.build_video_followup_focus_query(
            "",
            explicit_entities=["h20"],
            explicit_topics=["显卡"],
        )
        == "h20 显卡"
    )


def test_extract_explicit_dsl_query_preserves_q_vr_mode():
    assert (
        VideoQueryNormalizer.extract_explicit_dsl_query(
            "h20 显卡 q=vr"
        )
        == "h20 显卡 q=vr"
    )


def test_normalize_title_like_video_search_arguments_keeps_user_q_mode():
    args = {"queries": ["h20 英伟达 vr 相关视频 :view>=500"]}

    normalized = VideoQueryNormalizer.normalize_title_like_video_search_arguments(
        args,
        "h20 显卡 q=vr",
    )

    assert normalized == {"queries": ["h20 显卡 q=vr"]}


def test_video_query_normalizer_does_not_own_semantic_question_rewrite():
    assert (
        VideoQueryNormalizer.extract_title_like_video_query(
            "有没有讲 【AI爱音】Pieces 点歌请看我的置顶动态喵 的高质量视频？"
        )
        == ""
    )


def test_extract_title_like_video_query_keeps_quoted_title_tail():
    assert (
        VideoQueryNormalizer.extract_title_like_video_query(
            '"心脏骤停"『OverThink』- 时光代理人ED 翻唱【'
        )
        == "心脏骤停 『OverThink』- 时光代理人ED 翻唱"
    )
