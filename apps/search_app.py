import concurrent.futures
import requests
import threading
import uvicorn

from fastapi import FastAPI, Body
from pydantic import BaseModel
from tclogger import logger
from typing import Optional, List

from apps.arg_parser import ArgParser
from configs.envs import SEARCH_APP_ENVS
from elastics.video_details_suggest import VideoDetailsSuggester


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
        self.suggester = VideoDetailsSuggester()
        self.setup_routes()
        logger.success(f"> {self.title} - v{self.version}")

    def suggest(self, query: str = Body(...)):
        suggestions = self.suggester.suggest(query)
        return suggestions

    def setup_routes(self):
        self.app.post(
            "/suggest",
            summary="Get suggestions",
        )(self.suggest)


if __name__ == "__main__":
    app_envs = SEARCH_APP_ENVS
    app = SearchApp(app_envs).app
    app_args = ArgParser(app_envs).args
    if app_args.reload:
        uvicorn.run("__main__:app", host=app_args.host, port=app_args.port, reload=True)
    else:
        uvicorn.run("__main__:app", host=app_args.host, port=app_args.port)

    # python -m apps.search_app
