"""Inspect prompt asset selection and assembled system prompt for a query.

Usage:
    conda run -n ai python debugs/inspect_llm_prompt_selection.py "来点让我开心的视频"
    conda run -n ai python debugs/inspect_llm_prompt_selection.py "何同学有哪些关联账号" --full-prompt
"""

from __future__ import annotations

import argparse

from tclogger import logger

from llms.prompts.copilot import build_prompt_selection
from llms.intent import build_intent_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="user query to inspect")
    parser.add_argument(
        "--full-prompt",
        action="store_true",
        help="print the full assembled system prompt",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    messages = [{"role": "user", "content": args.query}]
    intent = build_intent_profile(messages)
    selection = build_prompt_selection(intent=intent)

    logger.note("SELECTED ASSETS")
    for asset in selection.assets:
        logger.mesg(
            f"- {asset.asset_id} | section={asset.section} | level={asset.level}"
        )

    logger.note("\nSECTION SIZES")
    for section, char_count in sorted(selection.section_chars.items()):
        logger.mesg(f"- {section}: {char_count}")
    logger.mesg(f"- total_chars: {len(selection.prompt)}")

    if args.full_prompt:
        logger.note("\nFULL PROMPT")
        logger.file(selection.prompt)


if __name__ == "__main__":
    main()
