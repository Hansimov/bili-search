"""Planning pipeline and plugin selection.

Plugin activation must depend on structured intent and execution signals, not on
scattered keyword checks. Individual rewriters may still normalize text, but the
decision to run a plugin belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from llms.contracts import IntentProfile
from llms.tools.names import canonical_tool_name


ToolResultScope = Literal["last_tool_results", "owner_result_scope"]


@dataclass(frozen=True, slots=True)
class ToolPlanningContext:
    commands: list[dict]
    messages: list[dict]
    last_tool_results: list[dict] | None
    owner_result_scope: list[dict] | None
    intent: IntentProfile | None = None

    def with_commands(self, commands: list[dict]) -> "ToolPlanningContext":
        return ToolPlanningContext(
            commands=list(commands),
            messages=self.messages,
            last_tool_results=self.last_tool_results,
            owner_result_scope=self.owner_result_scope,
            intent=self.intent,
        )


@dataclass(frozen=True, slots=True)
class PlanningSignals:
    command_types: frozenset[str]
    last_result_types: frozenset[str]
    owner_scope_types: frozenset[str]
    has_search_videos_command: bool
    has_search_google_command: bool
    has_owner_scope: bool
    has_google_results: bool
    has_owner_results: bool
    has_token_results: bool
    only_intermediate_results: bool


class ToolPlanningPlugin(Protocol):
    name: str
    priority: int

    def should_apply(
        self,
        context: ToolPlanningContext,
        signals: PlanningSignals,
    ) -> bool: ...

    def apply(self, handler_cls, context: ToolPlanningContext) -> list[dict]: ...


@dataclass(frozen=True, slots=True)
class HandlerMethodPlanningPlugin:
    name: str
    method_name: str
    priority: int
    predicate: Callable[[ToolPlanningContext, PlanningSignals], bool]
    tool_result_scope: ToolResultScope = "last_tool_results"

    def should_apply(
        self,
        context: ToolPlanningContext,
        signals: PlanningSignals,
    ) -> bool:
        return bool(self.predicate(context, signals))

    def apply(self, handler_cls, context: ToolPlanningContext) -> list[dict]:
        method = getattr(handler_cls, self.method_name)
        tool_results = (
            context.owner_result_scope
            if self.tool_result_scope == "owner_result_scope"
            else context.last_tool_results
        )
        return method(
            context.commands,
            context.messages,
            tool_results,
            context.intent,
        )


def _types_of(items: list[dict] | None) -> frozenset[str]:
    return frozenset(
        canonical_tool_name(str(item.get("type") or ""))
        for item in items or []
        if item.get("type")
    )


def _has_results_of_type(items: list[dict] | None, tool_name: str) -> bool:
    expected = canonical_tool_name(tool_name)
    return any(
        canonical_tool_name(str(item.get("type") or "")) == expected
        for item in items or []
    )


def build_planning_signals(context: ToolPlanningContext) -> PlanningSignals:
    command_types = frozenset(
        canonical_tool_name(str(command.get("type") or ""))
        for command in context.commands or []
        if command.get("type")
    )
    last_result_types = _types_of(context.last_tool_results)
    owner_scope_types = _types_of(context.owner_result_scope)
    only_intermediate_results = bool(last_result_types) and last_result_types.issubset(
        {"search_owners", "expand_query"}
    )
    return PlanningSignals(
        command_types=command_types,
        last_result_types=last_result_types,
        owner_scope_types=owner_scope_types,
        has_search_videos_command="search_videos" in command_types,
        has_search_google_command="search_google" in command_types,
        has_owner_scope=bool(context.owner_result_scope),
        has_google_results=_has_results_of_type(
            context.last_tool_results, "search_google"
        ),
        has_owner_results=_has_results_of_type(
            context.last_tool_results, "search_owners"
        ),
        has_token_results=_has_results_of_type(
            context.last_tool_results, "expand_query"
        ),
        only_intermediate_results=only_intermediate_results,
    )


DEFAULT_TOOL_PLANNING_PLUGINS: tuple[ToolPlanningPlugin, ...] = (
    HandlerMethodPlanningPlugin(
        name="token_assisted_normalization",
        method_name="_normalize_token_assisted_search_commands",
        priority=10,
        predicate=lambda context, signals: (
            signals.has_search_videos_command or signals.has_token_results
        ),
    ),
    HandlerMethodPlanningPlugin(
        name="google_keyword_bootstrap",
        method_name="_build_google_keyword_bootstrap_commands",
        priority=20,
        predicate=lambda context, signals: (
            bool(context.intent)
            and context.intent.final_target == "videos"
            and context.intent.needs_keyword_expansion
            and not context.intent.needs_term_normalization
            and not signals.has_search_google_command
            and not signals.has_google_results
        ),
    ),
    HandlerMethodPlanningPlugin(
        name="google_creator_bootstrap",
        method_name="_build_google_creator_bootstrap_commands",
        priority=30,
        predicate=lambda context, signals: (
            bool(context.intent)
            and context.intent.final_target in {"owners", "relations"}
            and not signals.only_intermediate_results
            and not signals.has_search_google_command
            and not signals.has_google_results
        ),
    ),
    HandlerMethodPlanningPlugin(
        name="google_space_owner_followup",
        method_name="_build_google_space_owner_followup_commands",
        priority=40,
        predicate=lambda context, signals: (
            bool(context.intent)
            and context.intent.final_target in {"owners", "relations"}
            and signals.has_google_results
            and not signals.has_owner_results
            and "search_owners" not in signals.command_types
        ),
    ),
    HandlerMethodPlanningPlugin(
        name="continue_intermediate_plan",
        method_name="_continue_intermediate_plan",
        priority=50,
        predicate=lambda context, signals: (
            not context.commands and signals.only_intermediate_results
        ),
    ),
    HandlerMethodPlanningPlugin(
        name="owner_assisted_video_search",
        method_name="_build_owner_assisted_video_search_commands",
        priority=60,
        predicate=lambda context, signals: (
            signals.has_owner_scope
            and (
                signals.has_search_videos_command
                or not context.intent
                or context.intent.final_target in {"videos", "mixed"}
            )
        ),
        tool_result_scope="owner_result_scope",
    ),
    HandlerMethodPlanningPlugin(
        name="fallback_token_assisted_search",
        method_name="_fallback_token_assisted_search_commands",
        priority=70,
        predicate=lambda context, signals: (
            signals.has_search_videos_command and signals.has_token_results
        ),
    ),
)


def select_tool_planning_plugins(
    context: ToolPlanningContext,
    plugins: tuple[ToolPlanningPlugin, ...] = DEFAULT_TOOL_PLANNING_PLUGINS,
) -> tuple[ToolPlanningPlugin, ...]:
    signals = build_planning_signals(context)
    selected = [plugin for plugin in plugins if plugin.should_apply(context, signals)]
    return tuple(sorted(selected, key=lambda plugin: plugin.priority))


def apply_tool_planning_plugins(
    handler_cls,
    context: ToolPlanningContext,
    plugins: tuple[ToolPlanningPlugin, ...] = DEFAULT_TOOL_PLANNING_PLUGINS,
) -> list[dict]:
    commands = list(context.commands or [])
    for plugin in sorted(plugins, key=lambda item: item.priority):
        current_context = context.with_commands(commands)
        signals = build_planning_signals(current_context)
        if not plugin.should_apply(current_context, signals):
            continue
        commands = plugin.apply(handler_cls, current_context)
    return commands
