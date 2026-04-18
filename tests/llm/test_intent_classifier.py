from llms.intent import build_intent_profile, select_prompt_asset_ids


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


def test_build_intent_profile_marks_alias_like_tutorial_query_for_expansion():
    profile = build_intent_profile(
        [{"role": "user", "content": "康夫UI 有什么入门教程？"}]
    )

    assert profile.final_target == "videos"
    assert profile.needs_keyword_expansion is True
    assert profile.needs_term_normalization is True
    assert profile.needs_owner_resolution is False
    asset_ids = select_prompt_asset_ids(profile)
    assert "semantic.expansion.brief" in asset_ids
    assert "tool.expand_query.brief" in asset_ids
    assert "tool.expand_query.detailed" in asset_ids
    assert "tool.expand_query.examples" in asset_ids
    assert "tool.search_videos.detailed" in asset_ids
    assert "tool.search_videos.examples" in asset_ids


def test_build_intent_profile_extracts_clean_topic_for_creator_discovery():
    profile = build_intent_profile(
        [{"role": "user", "content": "推荐几个做黑神话悟空内容的UP主"}]
    )

    assert profile.final_target == "owners"
    assert "黑神话悟空" in profile.explicit_topics
    assert all(topic != "推荐几个做黑神话悟空" for topic in profile.explicit_topics)


def test_build_intent_profile_carries_owner_context_for_representative_followup():
    profile = build_intent_profile(
        [
            {"role": "user", "content": "何同学有哪些关联账号？"},
            {"role": "assistant", "content": "我先帮你找相关作者线索。"},
            {"role": "user", "content": "那他的代表作有哪些？"},
        ]
    )

    assert profile.final_target == "videos"
    assert profile.needs_owner_resolution is True
    assert "何同学" in profile.explicit_entities or "何同学" in profile.explicit_topics
    assert "那他的" not in profile.explicit_entities
    asset_ids = select_prompt_asset_ids(profile)
    assert "tool.search_videos.detailed" in asset_ids
    assert "tool.search_videos.examples" in asset_ids


def test_build_intent_profile_preserves_explicit_bv_anchor_for_author_recent_query():
    profile = build_intent_profile(
        [
            {
                "role": "user",
                "content": "BV1e9cfz5EKj 这期视频的作者是谁。他最近还发了哪些视频。",
            }
        ]
    )

    assert profile.final_target == "videos"
    assert profile.needs_keyword_expansion is False
    assert profile.needs_owner_resolution is True
    assert "BV1e9cfz5EKj" in profile.explicit_entities
    assert all("作者是谁" not in topic for topic in profile.explicit_topics)
    assert all("最近还发了哪些视频" not in topic for topic in profile.explicit_topics)
