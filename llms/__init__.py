"""Intelligent search chat service for Bilibili videos.

Provides an AI copilot that analyzes user intent, executes searches,
and generates structured responses via an OpenAI-compatible API.

Integrated into SearchApp (apps/search_app.py) as /chat/completions endpoint.

Module structure:
    - llm_client:      LLM API client with function calling support
    - tools:           Tool definitions and execution (with direct search)
    - prompts:         System prompts and syntax reference
    - chat:            Chat handler with tool-calling loop
    - cli:             CLI interactive mode for testing
"""
