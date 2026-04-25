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
