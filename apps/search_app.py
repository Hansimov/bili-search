import argparse
import sys
import uvicorn

from copy import deepcopy
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
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
        # self.allow_cors()
        self.setup_routes()
        logger.success(f"> {self.title} - v{self.version}")

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
