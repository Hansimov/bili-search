"""Shared data contracts for the llms package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


PromptLevel = Literal["brief", "detailed", "examples"]
ToolVisibility = Literal["user", "internal"]
ModelRole = Literal["small", "large"]


@dataclass(slots=True)
class FacetScore:
    label: str
    score: float = 1.0


@dataclass(slots=True)
class IntentProfile:
    raw_query: str
    normalized_query: str
    final_target: str = "videos"
    task_mode: str = "exploration"
    motivation: list[FacetScore] = field(default_factory=list)
    consumption_mode: list[FacetScore] = field(default_factory=list)
    expected_payoff: list[FacetScore] = field(default_factory=list)
    constraints: list[FacetScore] = field(default_factory=list)
    visual_intent_hints: list[FacetScore] = field(default_factory=list)
    explicit_entities: list[str] = field(default_factory=list)
    explicit_topics: list[str] = field(default_factory=list)
    doc_signal_hints: dict[str, list[str]] = field(default_factory=dict)
    facet_weights: dict[str, float] = field(
        default_factory=lambda: {"promise": 0.45, "evidence": 0.20, "payoff": 0.35}
    )
    ambiguity: float = 0.0
    complexity_score: float = 0.0
    needs_keyword_expansion: bool = False
    needs_term_normalization: bool = False
    needs_owner_resolution: bool = False
    needs_external_search: bool = False
    is_followup: bool = False
    route_reason: str = ""

    def top_labels(self, attr_name: str, limit: int = 3) -> list[str]:
        values = getattr(self, attr_name, []) or []
        sorted_values = sorted(values, key=lambda item: item.score, reverse=True)
        return [item.label for item in sorted_values[:limit]]

    def to_prompt_lines(self) -> list[str]:
        lines = [
            f"- final_target: {self.final_target}",
            f"- task_mode: {self.task_mode}",
            f"- ambiguity: {self.ambiguity:.2f}",
            f"- complexity: {self.complexity_score:.2f}",
        ]
        if self.top_labels("motivation"):
            lines.append(f"- motivation: {', '.join(self.top_labels('motivation'))}")
        if self.top_labels("expected_payoff"):
            lines.append(
                f"- expected_payoff: {', '.join(self.top_labels('expected_payoff'))}"
            )
        if self.top_labels("consumption_mode"):
            lines.append(
                f"- consumption_mode: {', '.join(self.top_labels('consumption_mode'))}"
            )
        if self.top_labels("constraints"):
            lines.append(f"- constraints: {', '.join(self.top_labels('constraints'))}")
        if self.explicit_entities:
            lines.append(
                f"- explicit_entities: {', '.join(self.explicit_entities[:5])}"
            )
        if self.explicit_topics:
            lines.append(f"- explicit_topics: {', '.join(self.explicit_topics[:5])}")
        if self.doc_signal_hints:
            compact = []
            for key, values in list(self.doc_signal_hints.items())[:4]:
                compact.append(f"{key}={','.join(values[:4])}")
            lines.append(f"- doc_signal_hints: {'; '.join(compact)}")
        if self.needs_keyword_expansion:
            lines.append("- route_flag: needs_keyword_expansion")
        if self.needs_term_normalization:
            lines.append("- route_flag: needs_term_normalization")
        if self.needs_owner_resolution:
            lines.append("- route_flag: needs_owner_resolution")
        if self.needs_external_search:
            lines.append("- route_flag: needs_external_search")
        if self.is_followup:
            lines.append("- route_flag: followup_dialogue")
        if self.route_reason:
            lines.append(f"- route_reason: {self.route_reason}")
        return lines


@dataclass(slots=True)
class PromptAsset:
    asset_id: str
    title: str
    section: str
    level: PromptLevel
    content: str
    tags: tuple[str, ...] = ()
    tool_name: str | None = None


@dataclass(slots=True)
class PromptSelection:
    prompt: str
    assets: list[PromptAsset] = field(default_factory=list)
    section_chars: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ModelSpec:
    config_name: str
    model_name: str
    role: ModelRole
    provider: str = ""
    api_format: str = "openai"
    thinking_adapter: str = "auto"
    description: str = ""
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_multimodal: bool = False
    supports_reasoning: bool = False
    max_iterations: int = 4


@dataclass(slots=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]
    visibility: ToolVisibility = "user"
    source: str = "function_call"


@dataclass(slots=True)
class ToolExecutionRecord:
    result_id: str
    request: ToolCallRequest
    result: dict[str, Any]
    summary: dict[str, Any]
    visibility: ToolVisibility = "user"

    def to_tool_event_call(self, *, status: str = "completed") -> dict[str, Any]:
        payload = {
            "type": self.request.name,
            "args": self.request.arguments,
            "status": status,
            "visibility": self.visibility,
            "result_id": self.result_id,
        }
        if status == "completed":
            payload["result"] = self.result
            payload["summary"] = self.summary
        return payload


@dataclass(slots=True)
class OrchestrationResult:
    content: str
    reasoning_content: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    usage_trace: dict[str, Any] = field(default_factory=dict)
    prompt_profile: dict[str, Any] = field(default_factory=dict)
    thinking: bool = False
    content_streamed: bool = False


__all__ = [
    "FacetScore",
    "IntentProfile",
    "ModelRole",
    "ModelSpec",
    "OrchestrationResult",
    "PromptAsset",
    "PromptLevel",
    "PromptSelection",
    "ToolCallRequest",
    "ToolExecutionRecord",
    "ToolVisibility",
]
