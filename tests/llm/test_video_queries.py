from llms.orchestration.video_queries import VideoQueryNormalizer


def test_build_video_followup_focus_query_skips_pure_meta_request():
    assert VideoQueryNormalizer.build_video_followup_focus_query("请总结一下") == ""
