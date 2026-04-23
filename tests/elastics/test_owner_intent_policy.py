from elastics.videos.owner_intent_policy import get_owner_intent_policy


def test_owner_intent_policy_loads_asset_defaults():
    policy = get_owner_intent_policy()

    assert policy.resolve_size == 8
    assert policy.score_min == 180.0
    assert policy.filter_gap_min == 30.0
    assert policy.multi_owner_source_labels == frozenset({"topic", "relation"})


def test_owner_intent_policy_detects_model_code_queries():
    policy = get_owner_intent_policy()

    assert policy.looks_like_model_code_query("b200") is True
    assert policy.looks_like_model_code_query("v3-0324") is True
    assert policy.looks_like_model_code_query("黑神话") is False
    assert policy.looks_like_model_code_query("袁启 采访") is False
