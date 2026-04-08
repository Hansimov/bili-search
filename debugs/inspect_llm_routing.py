"""Inspect llm routing, prompt assets, and model selection for a query.

Usage:
    conda run -n ai python debugs/inspect_llm_routing.py "帮我找最近的 Gemini 2.5 解读视频"
"""

from __future__ import annotations

import argparse

from tclogger import logger

from llms.chat.orchestrator import ChatOrchestrator
from llms.config import ModelRegistry
from llms.prompts.copilot import build_prompt_selection
from llms.routing import build_intent_profile


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    messages = [{"role": "user", "content": args.query}]
    intent = build_intent_profile(messages)
    registry = ModelRegistry.from_envs()
    orchestrator = ChatOrchestrator(
        llm_client=_DummyClient(registry.primary("large").model_name),
        small_llm_client=_DummyClient(registry.primary("small").model_name),
        tool_executor=None,
        model_registry=registry,
    )
    selection = build_prompt_selection(intent=intent)

    logger.note("INTENT")
    for line in intent.to_prompt_lines():
        logger.mesg(line)

    logger.note("\nPROMPT ASSETS")
    for asset in selection.assets:
        logger.mesg(f"- {asset.asset_id} [{asset.section}/{asset.level}]")

    logger.note("\nMODEL SELECTION")
    for stage in ("planner", "response", "delegate"):
        _, spec = orchestrator._select_model(
            intent, stage=stage, thinking=args.thinking
        )
        logger.mesg(f"- {stage}: {spec.config_name} ({spec.role})")

    logger.note("\nPROMPT PROFILE")
    logger.mesg(f"- total_chars: {len(selection.prompt)}")
    for section, char_count in sorted(selection.section_chars.items()):
        logger.mesg(f"- {section}: {char_count}")


if __name__ == "__main__":
    main()
