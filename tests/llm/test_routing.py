from llms.routing import build_intent_profile, select_prompt_asset_ids


def test_build_intent_profile_for_abstract_video_query():
    profile = build_intent_profile([{"role": "user", "content": "来点让我开心的视频"}])

    assert profile.final_target == "videos"
    assert profile.task_mode == "exploration"
    assert profile.needs_keyword_expansion is True
    assert "funny" in profile.top_labels("expected_payoff")
    assert "promise.cover_hook" in profile.doc_signal_hints
    assert "route.videos.brief" in select_prompt_asset_ids(profile)


def test_build_intent_profile_for_relation_query():
    profile = build_intent_profile(
        [{"role": "user", "content": "何同学有哪些关联账号？"}]
    )

    assert profile.final_target == "relations"
    assert profile.task_mode == "lookup_entity"
    assert profile.needs_owner_resolution is False
    assert "route.relations.brief" in select_prompt_asset_ids(profile)


def test_build_intent_profile_for_mixed_official_and_bili_query():
    profile = build_intent_profile(
        [
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            }
        ]
    )

    assert profile.final_target == "mixed"
    assert profile.needs_external_search is True
    assert profile.complexity_score >= 0.5
    assert "route.mixed.brief" in select_prompt_asset_ids(profile)
