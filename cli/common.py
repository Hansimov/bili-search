from __future__ import annotations

import argparse

from service.envs import get_search_app_env_overrides_from_env, resolve_search_app_envs


def explicit_runtime_filters_from_args(args) -> dict:
    filters = {}
    for key in (
        "port",
        "elastic_index",
        "elastic_env_name",
        "llm_config",
    ):
        value = getattr(args, key, None)
        if value is not None:
            filters[key] = value
    return filters


def format_table(headers: list[str], rows: list[list[object]]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = ["-" * width for width in widths]
    lines = [render_row(headers), render_row(separator)]
    lines.extend(render_row(row) for row in string_rows)
    return "\n".join(lines)


def add_shared_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    include_foreground: bool = False,
    include_reload: bool = False,
    include_kill: bool = False,
):
    parser.add_argument("-s", "--host", type=str, default=None, help="Host override")
    parser.add_argument("-p", "--port", type=int, default=None, help="Port override")
    parser.add_argument(
        "-ei",
        "--elastic-index",
        type=str,
        default=None,
        help="Elastic videos index override",
    )
    parser.add_argument(
        "-ev",
        "--elastic-env-name",
        type=str,
        default=None,
        help="Elastic env name in secrets.json",
    )
    parser.add_argument(
        "-lc",
        "--llm-config",
        type=str,
        default=None,
        help="LLM config override",
    )
    if include_foreground:
        parser.add_argument(
            "--foreground",
            action="store_true",
            default=False,
            help="Run the service in foreground instead of background management mode",
        )
    if include_reload:
        parser.add_argument(
            "-r",
            "--reload",
            action="store_true",
            default=False,
            help="Enable uvicorn auto reload in foreground mode",
        )
    if include_kill:
        parser.add_argument(
            "-k",
            "--kill",
            action="store_true",
            default=False,
            help="Kill any process already listening on the target port before startup",
        )


def add_runtime_mode_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str = "docker",
):
    parser.add_argument(
        "--runtime",
        choices=["local", "docker"],
        default=default,
        help="Run against the local service manager or docker runtime",
    )


def resolve_runtime_envs_from_args(args, base_envs: dict | None = None) -> dict:
    env_overrides = get_search_app_env_overrides_from_env()
    explicit_overrides = {
        "host": getattr(args, "host", None),
        "port": getattr(args, "port", None),
        "elastic_index": getattr(args, "elastic_index", None),
        "elastic_env_name": getattr(args, "elastic_env_name", None),
        "llm_config": getattr(args, "llm_config", None),
    }
    overrides = {
        **env_overrides,
        **{
            key: value for key, value in explicit_overrides.items() if value is not None
        },
    }
    return resolve_search_app_envs(base_envs, overrides=overrides)
