import argparse
import sys

from copy import deepcopy
from tclogger import logger
from pprint import pformat


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            "-r",
            "--reload",
            action="store_true",
            help="Reload server on code change",
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
        logger.mesg(pformat(new_app_envs, sort_dicts=False, indent=4))

        return new_app_envs
