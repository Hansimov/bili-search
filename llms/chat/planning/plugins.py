from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolPlanningContext:
    commands: list[dict]
    messages: list[dict]
    last_tool_results: list[dict] | None
    owner_result_scope: list[dict] | None


class ToolPlanningPlugin(Protocol):
    name: str

    def apply(self, handler_cls, context: ToolPlanningContext) -> list[dict]: ...


@dataclass(frozen=True)
class HandlerMethodPlanningPlugin:
    name: str
    method_name: str
    use_owner_result_scope: bool = False

    def apply(self, handler_cls, context: ToolPlanningContext) -> list[dict]:
        method = getattr(handler_cls, self.method_name)
        tool_results = (
            context.owner_result_scope
            if self.use_owner_result_scope
            else context.last_tool_results
        )
        return method(context.commands, context.messages, tool_results)


DEFAULT_TOOL_PLANNING_PLUGINS: tuple[ToolPlanningPlugin, ...] = (
    HandlerMethodPlanningPlugin(
        name="token_assisted_normalization",
        method_name="_normalize_token_assisted_search_commands",
    ),
    HandlerMethodPlanningPlugin(
        name="google_keyword_bootstrap",
        method_name="_build_google_keyword_bootstrap_commands",
    ),
    HandlerMethodPlanningPlugin(
        name="google_creator_bootstrap",
        method_name="_build_google_creator_bootstrap_commands",
    ),
    HandlerMethodPlanningPlugin(
        name="google_space_owner_followup",
        method_name="_build_google_space_owner_followup_commands",
    ),
    HandlerMethodPlanningPlugin(
        name="continue_intermediate_plan",
        method_name="_continue_intermediate_plan",
    ),
    HandlerMethodPlanningPlugin(
        name="owner_assisted_video_search",
        method_name="_build_owner_assisted_video_search_commands",
        use_owner_result_scope=True,
    ),
    HandlerMethodPlanningPlugin(
        name="fallback_token_assisted_search",
        method_name="_fallback_token_assisted_search_commands",
    ),
)


def apply_tool_planning_plugins(
    handler_cls,
    context: ToolPlanningContext,
    plugins: tuple[ToolPlanningPlugin, ...] = DEFAULT_TOOL_PLANNING_PLUGINS,
) -> list[dict]:
    commands = list(context.commands or [])
    for plugin in plugins:
        commands = plugin.apply(
            handler_cls,
            ToolPlanningContext(
                commands=commands,
                messages=context.messages,
                last_tool_results=context.last_tool_results,
                owner_result_scope=context.owner_result_scope,
            ),
        )
    return commands
