import argparse
import sys
import uvicorn

from copy import deepcopy
from fastapi import FastAPI
from fastapi import WebSocket, WebSocketDisconnect
from tclogger import TCLogger, dict_to_str


logger = TCLogger()

from configs.envs import WEBSOCKET_APP_ENVS
from llms.ws.route import WebsocketRouter


class WebsocketApp:
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
        self.setup_routes()
        logger.success(f"> {self.title} - v{self.version}")

    async def websocket(self, ws: WebSocket):
        try:
            ws_router = WebsocketRouter(ws)
            await ws_router.run()
        # except WebSocketDisconnect:
        #     logger.success("* ws client disconnected")
        except Exception as e:
            logger.warn(f"× ws error: {e}")

    def setup_routes(self):
        self.app.websocket("/ws")(self.websocket)


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_arguments()

    def add_arguments(self):
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
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])

    def update_app_envs(self, app_envs: dict):
        new_app_envs = deepcopy(app_envs)
        new_app_envs["mode"] = self.args.mode
        mode = new_app_envs["mode"]
        for key, val in app_envs.items():
            if isinstance(val, dict) and mode in val.keys():
                new_app_envs[key] = val[mode]

        if self.args.host:
            new_app_envs["host"] = self.args.host
        if self.args.port:
            new_app_envs["port"] = self.args.port
        self.new_app_envs = new_app_envs

        logger.note(f"App Envs:")
        logger.mesg(dict_to_str(new_app_envs))

        return new_app_envs


if __name__ == "__main__":
    app_envs = WEBSOCKET_APP_ENVS
    arg_parser = ArgParser()
    new_app_envs = arg_parser.update_app_envs(app_envs)
    app = WebsocketApp(new_app_envs).app
    uvicorn.run("__main__:app", host=new_app_envs["host"], port=new_app_envs["port"])

    # Production mode by default:
    # python -m apps.websocket_app

    # Development mode:
    # python -m apps.websocket_app -m dev
