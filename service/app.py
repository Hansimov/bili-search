from __future__ import annotations

import asyncio
import json
import threading
import uuid

from copy import deepcopy
from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from tclogger import TCLogger
from typing import List, Optional, Union

from converters.embed.embed_client import init_embed_client_with_keepalive
from elastics.owners import OwnerSearcher
from elastics.relations import RELATED_ENDPOINTS, RelationsClient
from elastics.videos.constants import DOC_EXCLUDED_SOURCE_FIELDS
from elastics.videos.constants import MAX_SEARCH_DETAIL_LEVEL
from elastics.videos.constants import MAX_SUGGEST_DETAIL_LEVEL
from elastics.videos.constants import QMOD_SINGLE_TYPE, QMOD
from elastics.videos.constants import SEARCH_LIMIT, SUGGEST_LIMIT
from elastics.videos.constants import SEARCH_MATCH_FIELDS
from elastics.videos.constants import SEARCH_MATCH_TYPE
from elastics.videos.constants import SOURCE_FIELDS
from elastics.videos.constants import SUGGEST_MATCH_FIELDS
from elastics.videos.constants import SUGGEST_MATCH_TYPE
from elastics.videos.constants import USE_SCRIPT_SCORE
from elastics.videos.explorer import VideoExplorer
from elastics.videos.searcher_v2 import VideoSearcherV2
from ranks.constants import RANK_METHOD, RANK_METHOD_TYPE
from service.envs import SEARCH_APP_ENV_KEYS
from service.envs import apply_search_app_envs_to_environment
from service.envs import get_search_app_env_overrides_from_env
from service.envs import resolve_search_app_envs


logger = TCLogger()


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system/user/assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    stream: Optional[bool] = Field(False, description="Enable SSE streaming")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    model: Optional[str] = Field(
        None,
        description="Model override (unused, for compatibility)",
    )
    thinking: Optional[bool] = Field(
        False,
        description="Enable thinking/reasoning mode for deeper analysis",
    )
    max_iterations: Optional[int] = Field(
        None,
        description="Override max tool-calling iterations (default: 5, thinking: 10)",
    )


class SearchApp:
    def __init__(self, app_envs: dict | None = None):
        resolved_envs = app_envs or resolve_search_app_envs()
        self.title = resolved_envs.get("app_name")
        self.version = resolved_envs.get("version")
        self.app = FastAPI(
            docs_url="/",
            title=self.title,
            version=self.version,
            swagger_ui_parameters={"defaultModelsExpandDepth": -1},
        )
        self.app_envs = resolved_envs
        self._active_streams: dict[str, threading.Event] = {}
        self.init_searchers()
        self.init_embed_client()
        self.init_chat_handler()
        self.allow_cors()
        self.setup_routes()
        logger.okay(f"> {self.title} - v{self.version}")

    def init_searchers(self):
        self.elastic_videos_index = self.app_envs["elastic_index"]
        self.elastic_env_name = self.app_envs.get("elastic_env_name", None)
        self.video_searcher = VideoSearcherV2(
            self.elastic_videos_index,
            elastic_env_name=self.elastic_env_name,
        )
        self.video_explorer = VideoExplorer(
            self.elastic_videos_index,
            elastic_env_name=self.elastic_env_name,
        )
        self.relations_client = RelationsClient(
            self.elastic_videos_index,
            elastic_env_name=self.elastic_env_name,
        )
        self.owner_searcher = OwnerSearcher(
            self.elastic_videos_index,
            elastic_env_name=self.elastic_env_name,
            relations_client=self.relations_client,
        )

    def init_chat_handler(self):
        llm_config = self.app_envs.get("llm_config", "")
        if not llm_config:
            self.chat_handler = None
            logger.hint("> Chat handler disabled (no llm_config)")
            return

        from llms.chat.handler import ChatHandler
        from llms.llm_client import create_llm_client
        from llms.tools.executor import SearchService

        self.llm_client = create_llm_client(model_config=llm_config, verbose=True)
        search_service = SearchService(
            video_searcher=self.video_searcher,
            video_explorer=self.video_explorer,
            owner_searcher=self.owner_searcher,
            relations_client=self.relations_client,
            verbose=True,
        )
        self.chat_handler = ChatHandler(
            llm_client=self.llm_client,
            search_client=search_service,
            verbose=True,
        )
        logger.okay(f"  Chat: LLM={llm_config} ({self.llm_client.model})")

    def init_embed_client(self):
        logger.hint("> Initializing embed client with keepalive...")
        init_embed_client_with_keepalive()

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
        return self.video_searcher.search(
            query,
            match_fields=match_fields,
            source_fields=source_fields,
            match_type=match_type,
            suggest_info=suggest_info,
            use_script_score=use_script_score,
            rank_method=rank_method,
            detail_level=detail_level,
            limit=limit,
            verbose=verbose,
        )

    def explore(
        self,
        query: str = Body(...),
        qmod: Optional[Union[str, list[str]]] = Body(None),
        suggest_info: Optional[dict] = Body({}),
        verbose: Optional[bool] = Body(False),
    ):
        return self.video_explorer.unified_explore(
            query=query,
            qmod=qmod,
            suggest_info=suggest_info,
            verbose=verbose,
        )

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
        return self.video_searcher.suggest(
            query,
            match_fields=match_fields,
            source_fields=source_fields,
            match_type=match_type,
            use_script_score=use_script_score,
            use_pinyin=use_pinyin,
            detail_level=detail_level,
            limit=limit,
            verbose=verbose,
        )

    def random(
        self,
        seed_update_seconds: Optional[int] = Body(SUGGEST_LIMIT),
        limit: Optional[int] = Body(SUGGEST_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        return self.video_searcher.random(
            seed_update_seconds=seed_update_seconds,
            limit=limit,
            verbose=verbose,
        )

    def latest(
        self,
        limit: int = Body(SUGGEST_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        return self.video_searcher.latest(limit=limit, verbose=verbose)

    def doc(
        self,
        bvid: str = Body(...),
        included_source_fields: Optional[List[str]] = Body([]),
        excluded_source_fields: Optional[List[str]] = Body(DOC_EXCLUDED_SOURCE_FIELDS),
        verbose: Optional[bool] = Body(False),
    ):
        return self.video_searcher.doc(
            bvid,
            included_source_fields=included_source_fields,
            excluded_source_fields=excluded_source_fields,
            verbose=verbose,
        )

    def knn_search(
        self,
        query: str = Body(...),
        source_fields: Optional[list[str]] = Body(SOURCE_FIELDS),
        rank_method: Optional[RANK_METHOD_TYPE] = Body(RANK_METHOD),
        limit: Optional[int] = Body(SEARCH_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        return self.video_searcher.knn_search(
            query=query,
            source_fields=source_fields,
            rank_method=rank_method,
            limit=limit,
            verbose=verbose,
        )

    def hybrid_search(
        self,
        query: str = Body(...),
        source_fields: Optional[list[str]] = Body(SOURCE_FIELDS),
        suggest_info: Optional[dict] = Body({}),
        rank_method: Optional[RANK_METHOD_TYPE] = Body(RANK_METHOD),
        limit: Optional[int] = Body(SEARCH_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        return self.video_searcher.hybrid_search(
            query=query,
            source_fields=source_fields,
            suggest_info=suggest_info,
            rank_method=rank_method,
            limit=limit,
            verbose=verbose,
        )

    def related_tokens_by_tokens(
        self,
        text: str = Body(...),
        fields: Optional[list[str]] = Body(None),
        mode: str = Body("auto"),
        size: int = Body(8),
        scan_limit: int = Body(128),
        use_pinyin: bool = Body(True),
    ):
        return self.relations_client.related_tokens_by_tokens(
            text=text,
            fields=fields,
            mode=mode,
            size=size,
            scan_limit=scan_limit,
            use_pinyin=use_pinyin,
        )

    def search_owners(
        self,
        text: str = Body(...),
        mode: str = Body("auto"),
        size: int = Body(8),
    ):
        return self.owner_searcher.search(
            text=text,
            mode=mode,
            size=size,
        )

    def related_owners_by_tokens(
        self,
        text: str = Body(...),
        fields: Optional[list[str]] = Body(None),
        size: int = Body(8),
        scan_limit: int = Body(128),
        use_pinyin: bool = Body(True),
    ):
        return self.relations_client.related_owners_by_tokens(
            text=text,
            fields=fields,
            size=size,
            scan_limit=scan_limit,
            use_pinyin=use_pinyin,
        )

    def related_videos_by_videos(
        self,
        bvid: Optional[str] = Body(None),
        bvids: Optional[list[str]] = Body(None),
        size: int = Body(10),
        scan_limit: int = Body(128),
    ):
        return self.relations_client.related_videos_by_videos(
            bvid=bvid,
            bvids=bvids,
            size=size,
            scan_limit=scan_limit,
        )

    def related_owners_by_videos(
        self,
        bvid: Optional[str] = Body(None),
        bvids: Optional[list[str]] = Body(None),
        size: int = Body(10),
        scan_limit: int = Body(128),
    ):
        return self.relations_client.related_owners_by_videos(
            bvid=bvid,
            bvids=bvids,
            size=size,
            scan_limit=scan_limit,
        )

    def related_videos_by_owners(
        self,
        mid: Optional[int] = Body(None),
        mids: Optional[list[int]] = Body(None),
        size: int = Body(10),
        scan_limit: int = Body(128),
    ):
        return self.relations_client.related_videos_by_owners(
            mid=mid,
            mids=mids,
            size=size,
            scan_limit=scan_limit,
        )

    def related_owners_by_owners(
        self,
        mid: Optional[int] = Body(None),
        mids: Optional[list[int]] = Body(None),
        size: int = Body(10),
        scan_limit: int = Body(128),
    ):
        return self.relations_client.related_owners_by_owners(
            mid=mid,
            mids=mids,
            size=size,
            scan_limit=scan_limit,
        )

    def setup_routes(self):
        self.app.post("/suggest", summary="Get suggestions by query")(self.suggest)
        self.app.post("/search", summary="Get search results by query")(self.search)
        self.app.post("/explore", summary="Get explore results by query")(self.explore)
        self.app.post("/random", summary="Get random suggestions")(self.random)
        self.app.post("/latest", summary="Get latest suggestions")(self.latest)
        self.app.post("/doc", summary="Get video details by bvid")(self.doc)
        self.app.post(
            "/knn_search",
            summary="KNN vector search using text embeddings",
        )(self.knn_search)
        self.app.post(
            "/hybrid_search",
            summary="Hybrid search combining word and vector retrieval",
        )(self.hybrid_search)
        self.app.post(
            "/related_tokens_by_tokens",
            summary="Find related tokens by tokens",
        )(self.related_tokens_by_tokens)
        self.app.post(
            "/search_owners",
            summary="Search owners by name, topic, or relation",
        )(self.search_owners)
        self.app.post(
            "/related_owners_by_tokens",
            summary="Find related owners by topic tokens",
        )(self.related_owners_by_tokens)
        self.app.post(
            "/related_videos_by_videos",
            summary="Find related videos by seed videos",
        )(self.related_videos_by_videos)
        self.app.post(
            "/related_owners_by_videos",
            summary="Find related owners by seed videos",
        )(self.related_owners_by_videos)
        self.app.post(
            "/related_videos_by_owners",
            summary="Find related videos by seed owners",
        )(self.related_videos_by_owners)
        self.app.post(
            "/related_owners_by_owners",
            summary="Find related owners by seed owners",
        )(self.related_owners_by_owners)

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

        self.app.get("/health", summary="Health check")(self.health)
        self.app.get(
            "/capabilities",
            summary="Runtime service capabilities",
        )(self.capabilities)

    async def chat_completions(
        self,
        request: ChatCompletionRequest,
        http_request: Request,
    ):
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

        return await asyncio.to_thread(
            self.chat_handler.handle,
            messages=messages,
            temperature=request.temperature,
            thinking=request.thinking or False,
            max_iterations=request.max_iterations,
        )

    async def abort_stream(self, stream_id: str = Body(..., embed=True)):
        event = self._active_streams.get(stream_id)
        if event:
            event.set()
            logger.warn(f"> Stream {stream_id} aborted via /chat/abort")
            return {"status": "aborted", "stream_id": stream_id}

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
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()
        stream_id = uuid.uuid4().hex[:12]
        self._active_streams[stream_id] = cancelled
        logger.note(f"> Stream started: {stream_id}")

        async def _monitor_disconnect():
            while not cancelled.is_set():
                try:
                    if http_request and await http_request.is_disconnected():
                        logger.warn(
                            f"> Stream {stream_id}: client disconnect detected by monitor"
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
                            f"> Stream {stream_id}: producer stopping (cancelled)"
                        )
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("__error__", str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        fut = loop.run_in_executor(None, _produce)

        try:
            yield {"data": json.dumps({"stream_id": stream_id}, ensure_ascii=False)}
            while True:
                if cancelled.is_set():
                    logger.warn(f"> Stream {stream_id}: consumer stopping (cancelled)")
                    break

                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if chunk is None:
                    break
                if isinstance(chunk, tuple) and chunk[0] == "__error__":
                    yield {"event": "error", "data": chunk[1]}
                    break
                yield {"data": chunk}
        finally:
            cancelled.set()
            self._active_streams.pop(stream_id, None)
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            await fut

    async def health(self):
        result = {
            "status": "ok",
            "search_service": "integrated",
        }
        if self.chat_handler is not None:
            result["llm_model"] = self.llm_client.model
        return result

    async def capabilities(self):
        result = {
            "service_name": self.title,
            "service_type": "remote",
            "version": self.version,
            "elastic_index": self.elastic_videos_index,
            "default_query_mode": "wv",
            "rerank_query_mode": "vwr",
            "supports_multi_query": True,
            "supports_author_check": False,
            "supports_owner_search": True,
            "supports_google_search": False,
            "relation_endpoints": list(RELATED_ENDPOINTS),
            "available_endpoints": [
                "/health",
                "/capabilities",
                "/suggest",
                "/search",
                "/explore",
                "/random",
                "/latest",
                "/doc",
                "/knn_search",
                "/hybrid_search",
                "/search_owners",
                *[f"/{endpoint}" for endpoint in RELATED_ENDPOINTS],
            ],
            "docs": ["search_syntax"],
            "chat_enabled": self.chat_handler is not None,
        }
        if self.chat_handler is not None:
            result["llm_model"] = self.llm_client.model
        return result


def create_app(app_envs: dict | None = None) -> FastAPI:
    return SearchApp(app_envs or resolve_search_app_envs()).app


def create_app_from_env() -> FastAPI:
    overrides = get_search_app_env_overrides_from_env()
    return create_app(resolve_search_app_envs(overrides=overrides))
