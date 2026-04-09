"""Compatibility shim for the intent taxonomy package.

Keep taxonomy examples in llms.intent.taxonomy. Do not add keyword routing here.
"""

from llms.intent.taxonomy import detect_final_target
from llms.intent.taxonomy import detect_task_mode

__all__ = ["detect_final_target", "detect_task_mode"]
