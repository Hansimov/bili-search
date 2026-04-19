"""Inspect llm routing, prompt assets, and model selection for a query.

Usage:
    conda run -n ai python debugs/inspect_llm_routing.py "帮我找最近的 Gemini 2.5 解读视频"
"""

from __future__ import annotations

import argparse

from tclogger import logger

from llms.orchestration import ChatOrchestrator
from llms.models import DEFAULT_LARGE_MODEL_CONFIG, DEFAULT_SMALL_MODEL_CONFIG
from llms.models import ModelRegistry
from llms.prompts.copilot import build_prompt_selection
from llms.intent import build_intent_profile


class _DummyClient:
    def __init__(self, model: str):
        self.model = model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="user query to inspect")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="simulate thinking mode when selecting planner/response models",
    )
    parser.add_argument(
        "--large-config",
        default=DEFAULT_LARGE_MODEL_CONFIG,
        help="primary large-model config to inspect",
    )
    parser.add_argument(
        "--small-config",
        default=DEFAULT_SMALL_MODEL_CONFIG,
        help="primary small-model config to inspect",
    )
    parser.add_argument(
        "--supports-transcript-lookup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="simulate whether transcript lookup is available",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    messages = [{"role": "user", "content": args.query}]
    intent = build_intent_profile(messages)
    registry = ModelRegistry.from_envs(
        primary_large_config=args.large_config,
        primary_small_config=args.small_config,
    )
    orchestrator = ChatOrchestrator(
        llm_client=_DummyClient(registry.primary("large").model_name),
        small_llm_client=_DummyClient(registry.primary("small").model_name),
        tool_executor=None,
        model_registry=registry,
    )
    selection = build_prompt_selection(intent=intent)
    search_capabilities = {
        "supports_transcript_lookup": args.supports_transcript_lookup,
        "default_query_mode": "wv",
        "rerank_query_mode": "vwr",
        "supports_multi_query": True,
        "supports_owner_search": True,
        "supports_google_search": False,
        "relation_endpoints": [],
        "docs": ["search_syntax"],
    }
    extra_asset_ids = orchestrator._extra_prompt_assets(
        messages,
        intent,
        search_capabilities,
    )
    prefer_small = bool(extra_asset_ids)

    logger.note("INTENT")
    for line in intent.to_prompt_lines():
        logger.mesg(line)

    logger.note("\nPROMPT ASSETS")
    for asset in selection.assets:
        logger.mesg(f"- {asset.asset_id} [{asset.section}/{asset.level}]")

    logger.note("\nMODEL SELECTION")
    logger.mesg(
        f"- prefer_small: {prefer_small} (extra_assets={extra_asset_ids or ['<none>']})"
    )
    for stage in ("planner", "response", "delegate"):
        decision = orchestrator._select_model(
            intent,
            stage=stage,
            thinking=args.thinking,
            prefer_small=prefer_small,
        )
        factors = ", ".join(decision.factors) if decision.factors else "-"
        logger.mesg(
            f"- {stage}: {decision.spec.config_name} [{decision.spec.provider}] ({decision.spec.role})"
        )
        logger.mesg(f"  reason: {decision.reason}")
        logger.mesg(f"  factors: {factors}")

    logger.note("\nPROMPT PROFILE")
    logger.mesg(f"- total_chars: {len(selection.prompt)}")
    for section, char_count in sorted(selection.section_chars.items()):
        logger.mesg(f"- {section}: {char_count}")


if __name__ == "__main__":
    main()
