"""Compatibility shim for the orchestration engine.

New code should import ChatOrchestrator from llms.orchestration.engine.
"""

from llms.orchestration.engine import ChatOrchestrator

__all__ = ["ChatOrchestrator"]
