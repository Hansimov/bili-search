from __future__ import annotations

import argparse
import sys

from tclogger import logger

from cli.common import add_shared_runtime_args, resolve_runtime_envs_from_args


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        add_shared_runtime_args(self, include_reload=True)
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])

    def update_app_envs(self, app_envs: dict):
        new_app_envs = resolve_runtime_envs_from_args(self.args, base_envs=app_envs)
        self.new_app_envs = new_app_envs
        logger.note("App Envs:")
        logger.mesg(str(new_app_envs))
        return new_app_envs
