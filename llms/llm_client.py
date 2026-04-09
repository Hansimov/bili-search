"""Compatibility shim for llms.models.client."""

from llms.models.client import ChatResponse
from llms.models.client import LLMClient
from llms.models.client import ToolCall
from llms.models.client import create_llm_client

__all__ = ["ChatResponse", "LLMClient", "ToolCall", "create_llm_client"]
