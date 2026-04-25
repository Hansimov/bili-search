from llms.orchestration.video_queries import VideoQueryNormalizer


def test_build_video_followup_focus_query_skips_pure_meta_request():
    assert VideoQueryNormalizer.build_video_followup_focus_query("请总结一下") == ""


def test_extract_explicit_dsl_query_preserves_q_vr_mode():
    assert (
        VideoQueryNormalizer.extract_explicit_dsl_query(
            "帮我找 h20 显卡 q=vr 的相关视频，列出最相关的几个"
        )
        == "h20 显卡 q=vr"
    )


def test_normalize_title_like_video_search_arguments_keeps_user_q_mode():
    args = {"queries": ["h20 英伟达 vr 相关视频 :view>=500"]}

    normalized = VideoQueryNormalizer.normalize_title_like_video_search_arguments(
        args,
        "帮我找 h20 显卡 q=vr 的相关视频，列出最相关的几个",
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
            '忽略口播和套话，帮我找和 "心脏骤停"『OverThink』- 时光代理人ED 翻唱【 真正相关的视频。'
        )
        == "心脏骤停 『OverThink』- 时光代理人ED 翻唱"
    )


def test_extract_title_like_video_query_from_typo_prefix_without_brackets():
    assert (
        VideoQueryNormalizer.extract_title_like_video_query(
            "我可能打错了字，想找 冻人狐音翻唱超洗脑咖喱情歌「Sajna」｜“妆只为你一人点， 相关的视频。"
        )
        == "冻人狐音翻唱超洗脑咖喱情歌「Sajna」｜“妆只为你一人点"
    )
