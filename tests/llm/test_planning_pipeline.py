from llms.planning.pipeline import ToolPlanningContext
from llms.planning.pipeline import select_tool_planning_plugins
from llms.protocol import IntentProfile


def _intent(**kwargs) -> IntentProfile:
    return IntentProfile(
        raw_query=kwargs.pop("raw_query", "test"),
        normalized_query=kwargs.pop("normalized_query", "test"),
        **kwargs,
    )


def test_select_tool_planning_plugins_uses_intent_for_video_bootstrap():
    context = ToolPlanningContext(
        commands=[{"type": "search_videos", "args": {"queries": ["Gemini 2.5"]}}],
        messages=[{"role": "user", "content": "找点 Gemini 2.5 解读视频"}],
        last_tool_results=[],
        owner_result_scope=None,
        intent=_intent(
            final_target="videos",
            task_mode="exploration",
            needs_keyword_expansion=True,
        ),
    )

    selected = [plugin.name for plugin in select_tool_planning_plugins(context)]

    assert "token_assisted_normalization" in selected
    assert "google_keyword_bootstrap" in selected
    assert "google_creator_bootstrap" not in selected


def test_select_tool_planning_plugins_uses_owner_route_for_creator_bootstrap():
    context = ToolPlanningContext(
        commands=[],
        messages=[{"role": "user", "content": "推荐几个做黑神话悟空内容的UP主"}],
        last_tool_results=[],
        owner_result_scope=None,
        intent=_intent(
            final_target="owners",
            task_mode="exploration",
        ),
    )

    selected = [plugin.name for plugin in select_tool_planning_plugins(context)]

    assert "google_creator_bootstrap" in selected
    assert "google_keyword_bootstrap" not in selected


def test_select_tool_planning_plugins_continues_intermediate_results_only_when_needed():
    context = ToolPlanningContext(
        commands=[],
        messages=[{"role": "user", "content": "继续"}],
        last_tool_results=[
            {"type": "search_owners", "result": {"owners": []}},
        ],
        owner_result_scope=None,
        intent=_intent(final_target="owners", task_mode="lookup_entity"),
    )

    selected = [plugin.name for plugin in select_tool_planning_plugins(context)]

    assert selected == ["continue_intermediate_plan"]
