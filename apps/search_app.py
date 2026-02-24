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
from typing import Optional, List, Union

from configs.envs import SEARCH_APP_ENVS
from converters.embed.embed_client import init_embed_client_with_keepalive
from elastics.videos.constants import SOURCE_FIELDS, DOC_EXCLUDED_SOURCE_FIELDS
from elastics.videos.constants import SEARCH_MATCH_FIELDS
from elastics.videos.constants import SUGGEST_MATCH_FIELDS
from elastics.videos.constants import SEARCH_MATCH_TYPE, SUGGEST_MATCH_TYPE
from elastics.videos.constants import MAX_SEARCH_DETAIL_LEVEL
from elastics.videos.constants import MAX_SUGGEST_DETAIL_LEVEL
from elastics.videos.constants import SUGGEST_LIMIT, SEARCH_LIMIT
from elastics.videos.constants import USE_SCRIPT_SCORE
from elastics.videos.constants import QMOD_SINGLE_TYPE, QMOD
from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.explorer import VideoExplorer
from ranks.constants import RANK_METHOD_TYPE, RANK_METHOD

logger = TCLogger()


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


class SearchApp:
    def __init__(self, app_envs: dict = {}):
        self.title = app_envs.get("app_name")
        self.version = app_envs.get("version")
        self.app = FastAPI(
            docs_url="/",
            title=self.title,
            version=self.version,
            swagger_ui_parameters={"defaultModelsExpandDepth": -1},
        )
        self.app_envs = app_envs
        self.init_searchers()
        self.init_embed_client()
        self.init_chat_handler()
        self.allow_cors()
        self.setup_routes()
        logger.okay(f"> {self.title} - v{self.version}")

    def init_searchers(self):
        self.mode = self.app_envs.get("mode", "prod")
        self.elastic_videos_index = self.app_envs["elastic_index"]
        self.elastic_env_name = self.app_envs.get("elastic_env_name", None)
        self.video_searcher = VideoSearcherV2(
            self.elastic_videos_index, elastic_env_name=self.elastic_env_name
        )
        self.video_explorer = VideoExplorer(
            self.elastic_videos_index, elastic_env_name=self.elastic_env_name
        )

    def init_embed_client(self):
        """Initialize embed client with keepalive for long-running service.

        This prevents connection timeouts during idle periods by:
        1. Warming up the connection at startup
        2. Starting a background keepalive thread
        """
        logger.hint("> Initializing embed client with keepalive...")
        init_embed_client_with_keepalive()

    def init_chat_handler(self):
        """Initialize the LLM chat handler for /chat/completions.

        Uses the video_searcher and video_explorer directly (no HTTP),
        wrapping them in a SearchService for the tool executor.
        """
        llm_config = self.app_envs.get("llm_config", "")
        if not llm_config:
            self.chat_handler = None
            logger.hint("> Chat handler disabled (no llm_config)")
            return

        from llms.llm_client import create_llm_client
        from llms.tools.executor import SearchService
        from llms.chat.handler import ChatHandler

        self.llm_client = create_llm_client(
            model_config=llm_config,
            verbose=True,
        )
        search_service = SearchService(
            video_searcher=self.video_searcher,
            video_explorer=self.video_explorer,
            verbose=True,
        )
        self.chat_handler = ChatHandler(
            llm_client=self.llm_client,
            search_client=search_service,
            verbose=True,
        )
        logger.okay(f"  Chat: LLM={llm_config} ({self.llm_client.model})")

    def allow_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def search(
        self,
        query: str = Body(...),
        suggest_info: Optional[dict] = Body({}),
        match_fields: Optional[list[str]] = Body(SEARCH_MATCH_FIELDS),
        source_fields: Optional[list[str]] = Body(SOURCE_FIELDS),
        match_type: Optional[str] = Body(SEARCH_MATCH_TYPE),
        use_script_score: Optional[bool] = Body(USE_SCRIPT_SCORE),
        rank_method: Optional[RANK_METHOD_TYPE] = Body(RANK_METHOD),
        detail_level: int = Body(-1),
        max_detail_level: int = Body(MAX_SEARCH_DETAIL_LEVEL),
        limit: Optional[int] = Body(SEARCH_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        results = self.video_searcher.search(
            query,
            match_fields=match_fields,
            source_fields=source_fields,
            match_type=match_type,
            suggest_info=suggest_info,
            use_script_score=use_script_score,
            rank_method=rank_method,
            detail_level=detail_level,
            max_detail_level=max_detail_level,
            limit=limit,
            verbose=verbose,
        )
        return results

    def explore(
        self,
        query: str = Body(...),
        qmod: Optional[Union[str, list[str]]] = Body(None),
        suggest_info: Optional[dict] = Body({}),
        verbose: Optional[bool] = Body(False),
    ):
        """Explore videos with automatic query mode detection.

        Query mode (qmod) can be:
        - "w" or "word" or ["word"]: Word-based search
        - "v" or "vector" or ["vector"]: Vector-based KNN search (default)
        - "wv" or ["word", "vector"]: Hybrid search (word + vector)

        The mode can be specified via:
        1. The qmod parameter
        2. DSL in query (e.g., "黑神话 q=v" or "q=wv")
        """
        result = self.video_explorer.unified_explore(
            query=query,
            qmod=qmod,
            suggest_info=suggest_info,
            verbose=verbose,
        )
        return result

    def suggest(
        self,
        query: str = Body(...),
        match_fields: Optional[list[str]] = Body(SUGGEST_MATCH_FIELDS),
        source_fields: Optional[list[str]] = Body(SOURCE_FIELDS),
        match_type: Optional[str] = Body(SUGGEST_MATCH_TYPE),
        use_script_score: Optional[bool] = Body(USE_SCRIPT_SCORE),
        rank_method: Optional[RANK_METHOD_TYPE] = Body(RANK_METHOD),
        use_pinyin: Optional[bool] = Body(True),
        detail_level: int = Body(-1),
        max_detail_level: int = Body(MAX_SUGGEST_DETAIL_LEVEL),
        limit: Optional[int] = Body(SUGGEST_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        results = self.video_searcher.suggest(
            query,
            match_fields=match_fields,
            source_fields=source_fields,
            match_type=match_type,
            use_script_score=use_script_score,
            rank_method=rank_method,
            use_pinyin=use_pinyin,
            detail_level=detail_level,
            max_detail_level=max_detail_level,
            limit=limit,
            verbose=verbose,
        )
        return results

    def random(
        self,
        seed_update_seconds: Optional[int] = Body(SUGGEST_LIMIT),
        limit: Optional[int] = Body(SUGGEST_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        results = self.video_searcher.random(
            seed_update_seconds=seed_update_seconds, limit=limit, verbose=verbose
        )
        return results

    def latest(
        self,
        limit: int = Body(SUGGEST_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        results = self.video_searcher.latest(limit=limit, verbose=verbose)
        return results

    def doc(
        self,
        bvid: str = Body(...),
        included_source_fields: Optional[List[str]] = Body([]),
        excluded_source_fields: Optional[List[str]] = Body(DOC_EXCLUDED_SOURCE_FIELDS),
        verbose: Optional[bool] = Body(False),
    ):
        doc = self.video_searcher.doc(
            bvid,
            included_source_fields=included_source_fields,
            excluded_source_fields=excluded_source_fields,
            verbose=verbose,
        )
        return doc

    def knn_search(
        self,
        query: str = Body(...),
        source_fields: Optional[list[str]] = Body(SOURCE_FIELDS),
        rank_method: Optional[RANK_METHOD_TYPE] = Body(RANK_METHOD),
        limit: Optional[int] = Body(SEARCH_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        """Perform KNN vector search using text embeddings."""
        results = self.video_searcher.knn_search(
            query=query,
            source_fields=source_fields,
            rank_method=rank_method,
            limit=limit,
            verbose=verbose,
        )
        return results

    def hybrid_search(
        self,
        query: str = Body(...),
        source_fields: Optional[list[str]] = Body(SOURCE_FIELDS),
        suggest_info: Optional[dict] = Body({}),
        rank_method: Optional[RANK_METHOD_TYPE] = Body(RANK_METHOD),
        limit: Optional[int] = Body(SEARCH_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        """Perform hybrid search combining word-based and vector-based retrieval."""
        results = self.video_searcher.hybrid_search(
            query=query,
            source_fields=source_fields,
            suggest_info=suggest_info,
            rank_method=rank_method,
            limit=limit,
            verbose=verbose,
        )
        return results

    def setup_routes(self):
        self.app.post(
            "/suggest",
            summary="Get suggestions by query",
        )(self.suggest)

        self.app.post(
            "/search",
            summary="Get search results by query",
        )(self.search)

        self.app.post(
            "/explore",
            summary="Get explore results by query",
        )(self.explore)

        self.app.post(
            "/random",
            summary="Get random suggestions",
        )(self.random)

        self.app.post(
            "/latest",
            summary="Get latest suggestions",
        )(self.latest)

        self.app.post(
            "/doc",
            summary="Get video details by bvid",
        )(self.doc)

        self.app.post(
            "/knn_search",
            summary="KNN vector search using text embeddings",
        )(self.knn_search)

        self.app.post(
            "/hybrid_search",
            summary="Hybrid search combining word and vector retrieval",
        )(self.hybrid_search)

        # Chat endpoints (only if chat handler is initialized)
        if self.chat_handler is not None:
            self.app.post(
                "/chat/completions",
                summary="Chat completion (OpenAI-compatible)",
                description="Send messages and receive AI-generated responses with integrated video search.",
            )(self.chat_completions)

            self.app.post(
                "/v1/chat/completions",
                summary="Chat completion (OpenAI SDK compatible path)",
                include_in_schema=False,
            )(self.chat_completions)

            self.app.get(
                "/health",
                summary="Health check",
            )(self.health)

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
                self.chat_handler.handle,
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
        chunks = await asyncio.to_thread(
            lambda: list(
                self.chat_handler.handle_stream(
                    messages=messages,
                    temperature=temperature,
                )
            )
        )
        for chunk in chunks:
            yield {"data": chunk}

    async def health(self):
        """Health check endpoint."""
        status = {
            "status": "ok",
            "search_service": "integrated",
        }
        if self.chat_handler is not None:
            status["llm_model"] = self.llm_client.model
        return status


class SearchAppArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_arguments()

    def add_arguments(self, arg_dict: list = []):
        self.add_argument(
            "-s",
            "--host",
            type=str,
            help=f"Host of app",
        )
        self.add_argument(
            "-p",
            "--port",
            type=int,
            help=f"Port of app",
        )
        self.add_argument(
            "-m",
            "--mode",
            type=str,
            default="prod",
            help=f"Running mode of app",
        )
        self.add_argument(
            "-ei",
            "--elastic-index",
            type=str,
            help=f"Elastic videos index name",
        )
        self.add_argument(
            "-ev",
            "--elastic-env-name",
            type=str,
            help=f"Elastic env name in secrets.json",
        )
        self.add_argument(
            "-lc",
            "--llm-config",
            type=str,
            default="",
            help="LLM config name from secrets.json (e.g. deepseek, volcengine). "
            "Enables /chat/completions endpoint when set.",
        )

        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])

    def update_app_envs(self, app_envs: dict):
        new_app_envs = deepcopy(app_envs)
        mode = self.args.mode
        new_app_envs["mode"] = mode
        for key, val in app_envs.items():
            if isinstance(val, dict) and mode in val.keys():
                new_app_envs[key] = val[mode]

        if self.args.host:
            new_app_envs["host"] = self.args.host
        if self.args.port:
            new_app_envs["port"] = self.args.port
        if self.args.elastic_index:
            new_app_envs["elastic_index"] = self.args.elastic_index
        if self.args.elastic_env_name:
            new_app_envs["elastic_env_name"] = self.args.elastic_env_name
        if self.args.llm_config:
            new_app_envs["llm_config"] = self.args.llm_config

        self.new_app_envs = new_app_envs

        logger.note(f"App Envs:")
        logger.mesg(dict_to_str(new_app_envs))

        return new_app_envs


if __name__ == "__main__":
    app_envs = SEARCH_APP_ENVS
    arg_parser = SearchAppArgParser()
    new_app_envs = arg_parser.update_app_envs(app_envs)
    app = SearchApp(new_app_envs).app
    uvicorn.run("__main__:app", host=new_app_envs["host"], port=new_app_envs["port"])

    # Production mode by default:
    # python -m apps.search_app
    # python -m apps.search_app -ei bili_videos_pro1 -ev elastic_pro

    # Development mode:
    # python -m apps.search_app -m dev -ei bili_videos_dev6 -ev elastic_dev
    # python -m apps.search_app -m dev -ei bili_videos_dev6 -ev elastic_dev -p 21001

    # With LLM config for chat:
    # python -m apps.search_app -m dev -ei bili_videos_dev6 -ev elastic_dev -lc deepseek
