"""FastAPI app for the chat service.

Exposes an OpenAI-compatible /chat/completions endpoint that:
1. Receives user messages
2. Internally uses LLM + search tools to analyze intent and retrieve results
3. Returns structured responses in OpenAI chat completion format

Supports both streaming (SSE) and non-streaming modes.

Usage:
    # Production mode (default):
    python -m llms.app

    # Development mode:
    python -m llms.app -m dev

    # Custom port:
    python -m llms.app -p 20002

    # Custom LLM model:
    python -m llms.app --llm-config deepseek
"""

import argparse
import asyncio
import json
import sys
import uvicorn

from copy import deepcopy
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from tclogger import TCLogger, dict_to_str
from typing import Optional

from configs.envs import SECRETS
from llms.llm_client import create_llm_client
from llms.search_service import SearchServiceClient
from llms.chat.handler import ChatHandler

logger = TCLogger()

# Default app configuration
DEFAULT_APP_ENVS = {
    "app_name": "Bili Chat App",
    "host": "0.0.0.0",
    "port": 20005,
    "search_app_url": "http://localhost:20001",
    "llm_config": "deepseek",
    "version": "0.1.0",
}


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Message role: system/user/assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    stream: Optional[bool] = Field(False, description="Enable SSE streaming")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    model: Optional[str] = Field(
        None, description="Model override (unused, for compatibility)"
    )


class ChatApp:
    """FastAPI application for the AI-powered search chat service.

    Provides an OpenAI-compatible /chat/completions endpoint that integrates
    LLM reasoning with the Bili Search engine.
    """

    def __init__(self, app_envs: dict = None):
        envs = app_envs or DEFAULT_APP_ENVS
        self.title = envs.get("app_name", "Bili Chat App")
        self.version = envs.get("version", "0.1.0")
        self.app = FastAPI(
            docs_url="/",
            title=self.title,
            version=self.version,
            swagger_ui_parameters={"defaultModelsExpandDepth": -1},
        )

        # Initialize components
        llm_config = envs.get("llm_config", "deepseek")
        search_url = envs.get("search_app_url", "http://localhost:20001")

        self.llm_client = create_llm_client(
            model_config=llm_config,
            verbose=True,
        )
        self.search_client = SearchServiceClient(
            base_url=search_url,
            verbose=True,
        )
        self.handler = ChatHandler(
            llm_client=self.llm_client,
            search_client=self.search_client,
            verbose=True,
        )

        self.allow_cors()
        self.setup_routes()
        logger.success(f"> {self.title} - v{self.version}")
        logger.mesg(f"  LLM: {llm_config} ({self.llm_client.model})")
        logger.mesg(f"  Search: {search_url}")

    def allow_cors(self):
        """Enable CORS for cross-origin requests."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def chat_completions(self, request: ChatCompletionRequest):
        """OpenAI-compatible chat completion endpoint.

        Processes the conversation, internally executing search tools as needed,
        and returns the final response.
        """
        messages = [msg.model_dump() for msg in request.messages]

        if request.stream:
            return EventSourceResponse(
                self._stream_response(messages, request.temperature),
                media_type="text/event-stream",
            )
        else:
            result = await asyncio.to_thread(
                self.handler.handle,
                messages=messages,
                temperature=request.temperature,
            )
            return result

    async def _stream_response(
        self,
        messages: list[dict],
        temperature: float = None,
    ):
        """Generate SSE events for streaming response."""
        # Run the handler in a thread to not block the event loop
        chunks = await asyncio.to_thread(
            lambda: list(
                self.handler.handle_stream(
                    messages=messages,
                    temperature=temperature,
                )
            )
        )
        for chunk in chunks:
            yield {"data": chunk}

    async def health(self):
        """Health check endpoint."""
        search_available = await asyncio.to_thread(self.search_client.is_available)
        return {
            "status": "ok",
            "search_service": "available" if search_available else "unavailable",
            "llm_model": self.llm_client.model,
        }

    def setup_routes(self):
        self.app.post(
            "/chat/completions",
            summary="Chat completion (OpenAI-compatible)",
            description="Send messages and receive AI-generated responses with integrated video search.",
        )(self.chat_completions)

        # Also support /v1/chat/completions for OpenAI SDK compatibility
        self.app.post(
            "/v1/chat/completions",
            summary="Chat completion (OpenAI SDK compatible path)",
            include_in_schema=False,
        )(self.chat_completions)

        self.app.get(
            "/health",
            summary="Health check",
        )(self.health)


class ChatAppArgParser(argparse.ArgumentParser):
    """CLI argument parser for the chat app."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_arguments()

    def add_arguments(self):
        self.add_argument("-s", "--host", type=str, help="Host to bind")
        self.add_argument("-p", "--port", type=int, help="Port to bind")
        self.add_argument(
            "-m", "--mode", type=str, default="prod", help="Running mode (prod/dev)"
        )
        self.add_argument(
            "--search-url",
            type=str,
            help="Search App URL (e.g. http://localhost:20001)",
        )
        self.add_argument(
            "--llm-config",
            type=str,
            default="deepseek",
            help="LLM config name from secrets.json (e.g. deepseek, volcengine)",
        )
        self.add_argument(
            "--reload", action="store_true", default=False, help="Enable auto-reload"
        )
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])

    def build_app_envs(self) -> dict:
        """Build app environment config from CLI args and defaults."""
        envs = deepcopy(DEFAULT_APP_ENVS)

        args = self.args
        mode = args.mode

        # Mode-specific defaults
        if mode == "dev":
            envs["port"] = 21005
            envs["search_app_url"] = "http://localhost:21001"

        # CLI overrides
        if args.host:
            envs["host"] = args.host
        if args.port:
            envs["port"] = args.port
        if args.search_url:
            envs["search_app_url"] = args.search_url
        if args.llm_config:
            envs["llm_config"] = args.llm_config

        envs["mode"] = mode

        logger.note("App Envs:")
        logger.mesg(dict_to_str(envs), indent=2)

        return envs


if __name__ == "__main__":
    arg_parser = ChatAppArgParser()
    app_envs = arg_parser.build_app_envs()
    chat_app = ChatApp(app_envs)
    app = chat_app.app

    uvicorn.run(
        "__main__:app",
        host=app_envs["host"],
        port=app_envs["port"],
        reload=arg_parser.args.reload,
    )

    # python -m llms.app
    # python -m llms.app -m dev
    # python -m llms.app -p 20002 --llm-config deepseek
    # python -m llms.app --search-url http://localhost:21001
