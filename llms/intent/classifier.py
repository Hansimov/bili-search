"""Intent classifier built on top of the shared taxonomy.

The classifier now focuses on orchestration: taxonomy ranking, structured
signal derivation, and IntentProfile assembly. Signal heuristics and prompt
asset selection live in dedicated helpers so this module stays small and
auditable.
"""

from __future__ import annotations

from llms.contracts import IntentProfile
from llms.intent.prompt_selection import select_prompt_asset_ids
from llms.intent.signals import build_conversation_window
from llms.intent.signals import derive_intent_signals
from llms.intent.taxonomy import detect_final_target
from llms.intent.taxonomy import detect_task_mode
from llms.intent.taxonomy import rank_final_target_matches
from llms.intent.taxonomy import rank_task_mode_matches


def build_intent_profile(messages: list[dict]) -> IntentProfile:
    window = build_conversation_window(messages)
    final_target_matches = rank_final_target_matches(
        window.normalized_query,
        history_text=window.history_text,
    )
    final_target = detect_final_target(
        window.normalized_query,
        history_text=window.history_text,
    )
    task_mode_matches = rank_task_mode_matches(
        window.normalized_query,
        final_target,
        history_text=window.history_text,
    )
    task_mode = detect_task_mode(
        window.normalized_query,
        final_target,
        history_text=window.history_text,
    )
    signals = derive_intent_signals(
        messages=messages,
        window=window,
        final_target=final_target,
        task_mode=task_mode,
        final_target_matches=final_target_matches,
        task_mode_matches=task_mode_matches,
    )

    return IntentProfile(
        raw_query=window.latest_user_text,
        normalized_query=window.normalized_query,
        final_target=final_target,
        task_mode=task_mode,
        motivation=signals.motivation,
        consumption_mode=signals.consumption_mode,
        expected_payoff=signals.expected_payoff,
        constraints=signals.constraints,
        visual_intent_hints=signals.visual_intent_hints,
        explicit_entities=signals.explicit_entities,
        explicit_topics=signals.explicit_topics,
        doc_signal_hints=signals.doc_signal_hints,
        ambiguity=signals.ambiguity,
        complexity_score=signals.complexity_score,
        needs_keyword_expansion=signals.needs_keyword_expansion,
        needs_term_normalization=signals.needs_term_normalization,
        needs_owner_resolution=signals.needs_owner_resolution,
        needs_external_search=signals.needs_external_search,
        is_followup=window.is_followup,
        route_reason=", ".join(signals.route_reasons),
    )


__all__ = ["build_intent_profile", "select_prompt_asset_ids"]
