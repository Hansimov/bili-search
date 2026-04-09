"""Intelligent search chat service for Bilibili videos.

Provides an AI copilot that analyzes user intent, executes searches,
and generates structured responses via an OpenAI-compatible API.

Integrated into SearchApp (apps/search_app.py) as /chat/completions endpoint.

Module structure:
    - contracts:       Shared data contracts for intent, prompts, and orchestration
    - models:          Model registry and LLM client package
    - tools:           Tool definitions and execution (with direct search)
    - prompts:         System prompts and syntax reference
    - runtime:         CLI and other local runtime entrypoints
    - intent:          Taxonomy, classifier, and intent signal extraction
    - orchestration:   Planner/response loop and result store helpers
    - chat:            Chat handler with tool-calling loop
    - cli:             Backward-compatible shim for llms.runtime.cli
"""
