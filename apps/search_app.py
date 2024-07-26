import uvicorn

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tclogger import logger
from typing import Optional, List

from apps.arg_parser import ArgParser
from configs.envs import SEARCH_APP_ENVS
from elastics.video_searcher import VideoSearcher


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
        self.mode = app_envs.get("mode", "prod")
        # self.allow_cors()
        self.video_searcher = VideoSearcher(app_envs["bili_videos_index"])
        self.setup_routes()
        logger.success(f"> {self.title} - v{self.version}")

    def allow_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def suggest(
        self,
        query: str = Body(...),
        match_fields: Optional[list[str]] = Body(VideoSearcher.SUGGEST_MATCH_FIELDS),
        match_type: Optional[str] = Body(VideoSearcher.SUGGEST_MATCH_TYPE),
        limit: Optional[int] = Body(VideoSearcher.SUGGEST_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        suggestions = self.video_searcher.suggest(
            query,
            match_fields=match_fields,
            match_type=match_type,
            limit=limit,
            verbose=verbose,
        )
        return suggestions

    def search(
        self,
        query: str = Body(...),
        match_fields: Optional[list[str]] = Body(VideoSearcher.SEARCH_MATCH_FIELDS),
        match_type: Optional[str] = Body(VideoSearcher.SEARCH_MATCH_TYPE),
        boost: bool = Body(True),
        boosted_fields: dict = Body(VideoSearcher.BOOSTED_FIELDS),
        detail_level: int = Body(-1),
        max_detail_level: int = Body(VideoSearcher.MAX_SEARCH_DETAIL_LEVEL),
        limit: Optional[int] = Body(VideoSearcher.SEARCH_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        suggestions = self.video_searcher.detailed_search(
            query,
            match_fields=match_fields,
            match_type=match_type,
            boost=boost,
            boosted_fields=boosted_fields,
            detail_level=detail_level,
            max_detail_level=max_detail_level,
            limit=limit,
            verbose=verbose,
        )
        return suggestions

    def random(
        self,
        seed_update_seconds: Optional[int] = Body(VideoSearcher.SUGGEST_LIMIT),
        limit: Optional[int] = Body(VideoSearcher.SUGGEST_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        suggestions = self.video_searcher.random(
            seed_update_seconds=seed_update_seconds, limit=limit, verbose=verbose
        )
        return suggestions

    def latest(
        self,
        limit: int = Body(VideoSearcher.SUGGEST_LIMIT),
        verbose: Optional[bool] = Body(False),
    ):
        suggestions = self.video_searcher.latest(limit=limit, verbose=verbose)
        return suggestions

    def doc(
        self,
        bvid: str = Body(...),
        included_source_fields: Optional[List[str]] = Body([]),
        excluded_source_fields: Optional[List[str]] = Body(
            VideoSearcher.DOC_EXCLUDED_SOURCE_FIELDS
        ),
        verbose: Optional[bool] = Body(False),
    ):
        doc = self.video_searcher.doc(
            bvid,
            included_source_fields=included_source_fields,
            excluded_source_fields=excluded_source_fields,
            verbose=verbose,
        )
        return doc

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


if __name__ == "__main__":
    app_envs = SEARCH_APP_ENVS
    arg_parser = ArgParser()
    new_app_envs = arg_parser.update_app_envs(app_envs)
    app = SearchApp(new_app_envs).app
    uvicorn.run("__main__:app", host=new_app_envs["host"], port=new_app_envs["port"])

    # Production mode by default:
    # python -m apps.search_app

    # Development mode:
    # python -m apps.search_app -m dev
