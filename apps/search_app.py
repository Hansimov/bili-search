import uvicorn

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tclogger import logger
from typing import Optional, List

from apps.arg_parser import ArgParser
from configs.envs import SEARCH_APP_ENVS
from elastics.video_details_searcher import VideoDetailsSearcher


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
        # self.allow_cors()
        self.video_details_searcher = VideoDetailsSearcher()
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
        match_fields: Optional[list[str]] = Body(["title", "title.pinyin"]),
        limit: Optional[int] = Body(10),
    ):
        suggestions = self.video_details_searcher.suggest(
            query, match_fields=match_fields, limit=limit
        )
        return suggestions

    def random(
        self,
        seed_update_seconds: Optional[int] = Body(10),
        limit: Optional[int] = Body(10),
    ):
        suggestions = self.video_details_searcher.random(
            seed_update_seconds=seed_update_seconds, limit=limit
        )
        return suggestions

    def setup_routes(self):
        self.app.post(
            "/suggest",
            summary="Get suggestions by query",
        )(self.suggest)

        self.app.post(
            "/random",
            summary="Get random suggestions",
        )(self.random)


if __name__ == "__main__":
    app_envs = SEARCH_APP_ENVS
    app = SearchApp(app_envs).app
    app_args = ArgParser(app_envs).args
    if app_args.reload:
        uvicorn.run("__main__:app", host=app_args.host, port=app_args.port, reload=True)
    else:
        uvicorn.run("__main__:app", host=app_args.host, port=app_args.port)

    # python -m apps.search_app
