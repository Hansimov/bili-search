"""Intelligent search chat service for Bilibili videos.

Provides an AI copilot that analyzes user intent, executes searches,
and generates structured responses via an OpenAI-compatible API.

Module structure:
    - llm_client:      LLM API client with function calling support
    - search_service:  HTTP client for the Search App service
    - tools:           Tool definitions and execution
    - prompts:         System prompts and syntax reference
    - chat:            Chat handler with tool-calling loop
    - app:             FastAPI app (/chat/completions endpoint)
    - cli:             CLI interactive mode for testing
"""
