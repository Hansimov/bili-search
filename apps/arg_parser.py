import argparse
import sys


class ArgParser(argparse.ArgumentParser):
    def __init__(self, app_envs={}, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.host = app_envs.get("host", "127.0.0.1")
        self.port = app_envs.get("port", 19898)
        self.app_name = app_envs.get("app_name", f"App on {self.host}")

        self.add_argument(
            "-s",
            "--host",
            type=str,
            default=self.host,
            help=f"Host ({self.host}) for {self.app_name}",
        )
        self.add_argument(
            "-p",
            "--port",
            type=int,
            default=app_envs["port"],
            help=f"Port ({self.port}) for {self.app_name}",
        )
        self.add_argument(
            "-r",
            "--reload",
            action="store_true",
            help="Reload server on code change",
        )

        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
