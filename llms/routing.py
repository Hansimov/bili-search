"""Compatibility shim for the intent package.

Keep all intent logic in llms.intent.*. Do not add regex routing here.
"""

from llms.intent.classifier import build_intent_profile
from llms.intent.classifier import select_prompt_asset_ids

__all__ = ["build_intent_profile", "select_prompt_asset_ids"]
