"""Compatibility shim for llms.models.registry."""

from llms.models.registry import DEFAULT_LARGE_MODEL_CONFIG
from llms.models.registry import DEFAULT_SMALL_MODEL_CONFIG
from llms.models.registry import ModelRegistry
from llms.models.registry import create_model_clients

__all__ = [
    "DEFAULT_LARGE_MODEL_CONFIG",
    "DEFAULT_SMALL_MODEL_CONFIG",
    "ModelRegistry",
    "create_model_clients",
]
