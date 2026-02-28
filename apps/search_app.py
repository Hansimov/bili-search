import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import uuid
import uvicorn

from copy import deepcopy
from fastapi import FastAPI, Body, Request
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
    thinking: Optional[bool] = Field(
        False, description="Enable thinking/reasoning mode for deeper analysis"
    )
    max_iterations: Optional[int] = Field(
        None,
        description="Override max tool-calling iterations (default: 5, thinking: 10)",
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
        # Registry of active chat streams: stream_id → threading.Event
        # Used by /chat/abort to cancel in-flight requests
        self._active_streams: dict[str, threading.Event] = {}
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

            self.app.post(
                "/chat/abort",
                summary="Abort an active chat stream",
                description="Cancel an in-flight streaming chat request by stream_id.",
            )(self.abort_stream)

            self.app.get(
                "/health",
                summary="Health check",
            )(self.health)

    async def chat_completions(
        self, request: ChatCompletionRequest, http_request: Request
    ):
        """OpenAI-compatible chat completion endpoint.

        Processes the conversation, internally executing search tools as needed,
        and returns the final response.

        When thinking=True:
        - Increases max tool-calling iterations (default 10 vs 5)
        - Enables deeper analysis prompting
        """
        messages = [msg.model_dump() for msg in request.messages]

        if request.stream:
            return EventSourceResponse(
                self._stream_response(
                    messages,
                    request.temperature,
                    thinking=request.thinking or False,
                    max_iterations=request.max_iterations,
                    http_request=http_request,
                ),
                media_type="text/event-stream",
            )
        else:
            result = await asyncio.to_thread(
                self.chat_handler.handle,
                messages=messages,
                temperature=request.temperature,
                thinking=request.thinking or False,
                max_iterations=request.max_iterations,
            )
            return result

    async def abort_stream(self, stream_id: str = Body(..., embed=True)):
        """Abort an active chat stream by stream_id.

        Called by the frontend when the user presses the stop button.
        This is more reliable than relying on TCP disconnect detection
        through proxies.
        """
        event = self._active_streams.get(stream_id)
        if event:
            event.set()
            logger.warn(f"> Stream {stream_id} aborted via /chat/abort")
            return {"status": "aborted", "stream_id": stream_id}
        else:
            logger.warn(f"> Stream {stream_id} not found for abort")
            return {"status": "not_found", "stream_id": stream_id}

    async def _stream_response(
        self,
        messages: list[dict],
        temperature: float = None,
        thinking: bool = False,
        max_iterations: int = None,
        http_request: Request = None,
    ):
        """Generate SSE events for streaming response.

        Uses asyncio.Queue to forward chunks from the synchronous
        chat handler generator to the async SSE response in real-time.

        Cancellation is supported via three mechanisms:
        1. Background monitor task: checks http_request.is_disconnected()
           every 0.5s, independently of generator iteration.
        2. Explicit /chat/abort endpoint: sets cancelled via _active_streams.
        3. Generator cleanup: try/finally ensures cancelled is set when
           the generator is closed (e.g. EventSourceResponse calls aclose).
        """
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()

        # Register this stream for explicit abort
        stream_id = uuid.uuid4().hex[:12]
        self._active_streams[stream_id] = cancelled
        logger.note(f"> Stream started: {stream_id}")

        # Background task: monitor client disconnection independently
        # This runs on the event loop even if the generator is suspended at yield.
        async def _monitor_disconnect():
            while not cancelled.is_set():
                try:
                    if http_request and await http_request.is_disconnected():
                        logger.warn(
                            f"> Stream {stream_id}: "
                            "client disconnect detected by monitor"
                        )
                        cancelled.set()
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.5)

        monitor_task = asyncio.create_task(_monitor_disconnect())

        def _produce():
            try:
                for chunk in self.chat_handler.handle_stream(
                    messages=messages,
                    temperature=temperature,
                    thinking=thinking,
                    max_iterations=max_iterations,
                    cancelled=cancelled,
                ):
                    if cancelled.is_set():
                        logger.warn(
                            f"> Stream {stream_id}: " "producer stopping (cancelled)"
                        )
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("__error__", str(e)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        fut = loop.run_in_executor(None, _produce)

        try:
            # Send stream_id as first SSE event so frontend can use it for /chat/abort
            yield {"data": json.dumps({"stream_id": stream_id}, ensure_ascii=False)}

            while True:
                if cancelled.is_set():
                    logger.warn(
                        f"> Stream {stream_id}: " "consumer stopping (cancelled)"
                    )
                    break

                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if chunk is None:
                    break
                if isinstance(chunk, tuple) and chunk[0] == "__error__":
                    break
                yield {"data": chunk}
        finally:
            # Ensure cancellation on any exit path, including GeneratorExit
            # when EventSourceResponse stops iterating (client disconnect).
            if not cancelled.is_set():
                logger.warn(
                    f"> Stream {stream_id}: "
                    "cancelled via generator cleanup (finally)"
                )
            cancelled.set()
            monitor_task.cancel()
            # Unregister from active streams
            self._active_streams.pop(stream_id, None)
            logger.note(f"> Stream ended: {stream_id}")

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
            help="LLM config name from secrets.json (e.g. deepseek, gpt, volcengine). "
            "Enables /chat/completions endpoint when set.",
        )
        self.add_argument(
            "-k",
            "--kill",
            action="store_true",
            default=False,
            help="Kill any existing process that is listening on the target port before starting.",
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

    if arg_parser.args.kill:
        _port = new_app_envs["port"]
        try:
            import subprocess

            result = subprocess.run(
                ["lsof", "-t", f"-i:{_port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
            )
            pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
            for pid in pids:
                if pid != os.getpid():
                    logger.warn(f"> Port {_port} in use by PID {pid} — killing it")
                    os.kill(pid, signal.SIGKILL)
        except Exception as _e:
            logger.warn(f"> Could not auto-kill port {_port}: {_e}")

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
    # python -m apps.search_app -m dev -ei bili_videos_dev6 -ev elastic_dev -lc gpt
