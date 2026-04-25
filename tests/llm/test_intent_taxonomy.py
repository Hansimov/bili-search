from llms.intent.taxonomy import detect_final_target, detect_task_mode


def test_detect_final_target_prefers_mixed_route():
    text = "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？"
    assert detect_final_target(text) == "mixed"


def test_detect_final_target_distinguishes_external_and_owner_queries():
    assert detect_final_target("Gemini 2.5 最近有哪些官方更新？") == "external"
    assert detect_final_target("推荐几个做黑神话悟空内容的UP主") == "owners"
    assert detect_final_target("何同学有哪些关联账号？") == "relations"
    assert detect_final_target("推荐几个 ComfyUI 入门教程视频") == "videos"


def test_detect_task_mode_handles_recent_video_and_followup_lookup():
    assert detect_task_mode("影视飓风最近有什么新视频？", "videos") == "repeat"
    assert detect_task_mode("何同学最近是谁？", "owners") == "lookup_entity"
    assert detect_task_mode("推荐几条黑神话悟空剧情解析视频", "videos") == "exploration"
