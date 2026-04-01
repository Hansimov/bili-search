from __future__ import annotations

import argparse

from tclogger import chars_len, colored, dict_to_table_str, logstr, rows_to_table_str

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


def _column_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    widths = [chars_len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], chars_len(cell))
    return widths


def _colorize_table_content_lines(
    table_text: str,
    *,
    headers: list[str],
    rows: list[list[str]],
    column_widths: list[int],
    column_colors: list[str],
    col_gap_len: int = 2,
) -> str:
    lines = table_text.splitlines()
    if len(lines) < 4:
        return table_text

    colored_lines = [logstr.note(lines[0])]
    content_indexes = {1, *range(3, len(lines) - 1)}
    gap = " " * col_gap_len
    status_index = next(
        (
            idx
            for idx, header in enumerate(headers)
            if str(header).strip().lower() == "status"
        ),
        None,
    )

    def colorize_status_segment(segment: str, raw_value: str) -> str:
        normalized = raw_value.strip().lower()
        if normalized.startswith("running"):
            return logstr.okay(segment)
        if normalized.startswith(("degraded", "drifted", "orphaned", "warning")):
            return logstr.warn(segment)
        if normalized.startswith(("starting", "created", "restarting", "pending")):
            return logstr.hint(segment)
        if normalized.startswith(
            ("stopped", "dead", "exited", "failed", "error", "terminated")
        ):
            return logstr.warn(segment)
        return segment

    for idx, line in enumerate(lines[1:], start=1):
        if idx not in content_indexes:
            if idx == len(lines) - 1:
                colored_lines.append(logstr.note(line))
            else:
                colored_lines.append(line)
            continue

        offset = 0
        segments: list[str] = []
        row_idx = 0 if idx == 1 else idx - 3
        row_values = headers if idx == 1 else rows[row_idx]
        for col_idx, width in enumerate(column_widths):
            next_offset = offset + width
            segment = line[offset:next_offset]
            colored_segment = colored(segment, column_colors[col_idx])
            if status_index is not None and col_idx == status_index:
                colored_segment = colorize_status_segment(segment, row_values[col_idx])
            elif (
                headers == ["Field", "Value"]
                and len(row_values) >= 2
                and row_values[0].strip().lower() == "status"
                and col_idx == 1
            ):
                colored_segment = colorize_status_segment(segment, row_values[1])
            segments.append(colored_segment)
            offset = next_offset
            if col_idx < len(column_widths) - 1:
                segments.append(gap)
                offset += col_gap_len
        colored_lines.append("".join(segments))
    return "\n".join(colored_lines)


def render_key_value_table(values: dict[str, object]) -> str:
    raw_values = {str(key): str(value) for key, value in values.items()}
    headers = ["Field", "Value"]
    rows = [[key, value] for key, value in raw_values.items()]
    table_text = dict_to_table_str(
        raw_values,
        key_headers=[headers[0]],
        val_headers=[headers[1]],
        header_case="raw",
        header_wsch=" ",
        is_colored=False,
    )
    column_widths = _column_widths(headers, rows)
    return _colorize_table_content_lines(
        table_text,
        headers=headers,
        rows=rows,
        column_widths=column_widths,
        column_colors=["light_cyan", "light_blue"],
    )


def render_list_table(headers: list[str], rows: list[list[object]]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    table_text = rows_to_table_str(
        rows=string_rows,
        headers=headers,
        header_case="raw",
        header_wsch="_",
        is_colored=False,
    )
    column_widths = _column_widths(headers, string_rows)
    column_colors = [
        "light_cyan" if idx % 2 == 0 else "light_blue" for idx in range(len(headers))
    ]
    return _colorize_table_content_lines(
        table_text,
        headers=headers,
        rows=string_rows,
        column_widths=column_widths,
        column_colors=column_colors,
    )


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
    allow_all: bool = False,
):
    choices = ["local", "docker"]
    if allow_all:
        choices.append("all")
    parser.add_argument(
        "--runtime",
        choices=choices,
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
