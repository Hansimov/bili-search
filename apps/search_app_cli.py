from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

import uvicorn

from pathlib import Path
from tclogger import decolored, logger, logstr

from apps.search_app import (
    SEARCH_APP_ENV_KEYS,
    apply_search_app_envs_to_environment,
    resolve_search_app_envs,
)
from webu.cli_support import (
    LocalServiceSpec,
    LocalServiceManager,
    ManagedServiceSpec,
)
from webu.clis.helpers import root_epilog


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE_ROOT / "logs" / "search_app"


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(value) -> str:
    text = str(value or "default")
    chars = [ch.lower() if ch.isalnum() else "-" for ch in text]
    safe = "".join(chars).strip("-")
    return safe or "default"


def _resolve_runtime_envs(args) -> dict:
    overrides = {
        "host": getattr(args, "host", None),
        "port": getattr(args, "port", None),
        "elastic_index": getattr(args, "elastic_index", None),
        "elastic_env_name": getattr(args, "elastic_env_name", None),
        "llm_config": getattr(args, "llm_config", None),
    }
    return resolve_search_app_envs(
        mode=getattr(args, "mode", None),
        overrides=overrides,
    )


def _service_files_for_envs(app_envs: dict) -> tuple[Path, Path]:
    parts = [
        _safe_name(app_envs.get("mode")),
        f"p{int(app_envs['port'])}",
        f"ei-{_safe_name(app_envs.get('elastic_index'))}",
        f"ev-{_safe_name(app_envs.get('elastic_env_name'))}",
        f"lc-{_safe_name(app_envs.get('llm_config'))}",
    ]
    stem = "server." + ".".join(parts)
    return DATA_DIR / f"{stem}.pid", DATA_DIR / f"{stem}.log"


def _build_service_manager(app_envs: dict) -> LocalServiceManager:
    pid_file, log_file = _service_files_for_envs(app_envs)
    service_spec = LocalServiceSpec(
        name="bili_search_app",
        uvicorn_target="apps.search_app_service:create_app_from_env",
        pid_file=pid_file,
        log_file=log_file,
    )
    managed_service = ManagedServiceSpec(
        name="bili_search_app",
        service=service_spec,
        default_host="0.0.0.0",
    )
    return LocalServiceManager(managed_service)


def _sanitize_log_file(log_file: Path):
    if not log_file.exists():
        return
    raw = log_file.read_text(encoding="utf-8", errors="replace")
    cleaned = decolored(raw)
    if cleaned != raw:
        log_file.write_text(cleaned, encoding="utf-8")


def _runtime_env_vars(app_envs: dict) -> dict[str, str]:
    env = {}
    for key, env_name in SEARCH_APP_ENV_KEYS.items():
        value = app_envs.get(key)
        if value is not None:
            env[env_name] = str(value)
    return env


def _health_url(app_envs: dict) -> str:
    return f"http://127.0.0.1:{int(app_envs['port'])}/health"


def _fetch_health(app_envs: dict, timeout: float = 3.0) -> dict:
    request = urllib.request.Request(_health_url(app_envs), method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _print_json(payload: dict):
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _add_runtime_args(parser: argparse.ArgumentParser):
    parser.add_argument("-m", "--mode", type=str, default="prod", help="Running mode")
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


def cmd_start(args):
    app_envs = _resolve_runtime_envs(args)
    service_manager = _build_service_manager(app_envs)
    current = service_manager.status()
    if current["status"] == "running":
        logger.warn(f"  × Search app already running (PID: {current['pid']})")
        return

    _ensure_data_dir()
    logger.note(
        f"> Starting search app on {app_envs['host']}:{app_envs['port']} "
        f"[{app_envs['mode']}] ..."
    )
    result = service_manager.start(
        host=app_envs["host"],
        port=app_envs["port"],
        extra_env=_runtime_env_vars(app_envs),
    )
    pid = result["pid"]
    logger.okay(f"  ✓ Search app started (PID: {pid})")
    logger.mesg(f"  Log: {logstr.file(service_manager.log_file)}")


def cmd_stop(args):
    app_envs = _resolve_runtime_envs(args)
    service_manager = _build_service_manager(app_envs)
    result = service_manager.stop()
    pid = result["pid"]
    if result["status"] == "not_running":
        logger.warn("  × No PID file found — search app not running?")
        return
    if result["status"] == "stale_pid":
        logger.warn(f"  × Process {pid} not found — cleaning up PID file")
        return

    logger.note(f"> Stopping search app (PID: {pid}) ...")
    logger.okay("  ✓ Search app stopped")


def cmd_restart(args):
    app_envs = _resolve_runtime_envs(args)
    service_manager = _build_service_manager(app_envs)
    stop_result = service_manager.stop()
    start_result = service_manager.start(
        host=app_envs["host"],
        port=app_envs["port"],
        extra_env=_runtime_env_vars(app_envs),
    )
    if stop_result["status"] in ("stopped", "stale_pid") and stop_result["pid"]:
        logger.note(f"> Stopping search app (PID: {stop_result['pid']}) ...")
        logger.okay("  ✓ Search app stopped")
    logger.note(
        f"> Starting search app on {app_envs['host']}:{app_envs['port']} "
        f"[{app_envs['mode']}] ..."
    )
    logger.okay(f"  ✓ Search app started (PID: {start_result['pid']})")
    logger.mesg(f"  Log: {logstr.file(service_manager.log_file)}")


def cmd_status(args):
    app_envs = _resolve_runtime_envs(args)
    service_manager = _build_service_manager(app_envs)
    _sanitize_log_file(service_manager.log_file)
    result = service_manager.status()
    pid = result["pid"]
    if result["status"] == "not_running":
        logger.mesg("  Search app: NOT RUNNING (no PID file)")
        logger.mesg(f"  Expected URL: {_health_url(app_envs)}")
        logger.mesg(f"  Expected Log: {logstr.file(service_manager.log_file)}")
        return
    if result["status"] == "dead":
        logger.warn(f"  Search app: DEAD (PID: {pid} not found)")
        logger.mesg("  Cleaned up stale PID file")
        return

    logger.okay(f"  Search app: RUNNING (PID: {pid})")
    logger.mesg(f"  URL: {_health_url(app_envs)}")
    logger.mesg(f"  Log: {logstr.file(service_manager.log_file)}")
    try:
        health = _fetch_health(app_envs, timeout=args.timeout)
        logger.mesg(f"  Health: {json.dumps(health, ensure_ascii=False)}")
    except Exception as exc:
        logger.warn(f"  × Health check failed: {exc}")


def cmd_logs(args):
    app_envs = _resolve_runtime_envs(args)
    service_manager = _build_service_manager(app_envs)
    if not service_manager.log_file.exists():
        logger.warn("  × No log file found")
        return
    _sanitize_log_file(service_manager.log_file)
    if args.follow:
        service_manager.tail_logs()
        return
    print(decolored(service_manager.read_logs(lines=args.lines)), end="")


def cmd_check(args):
    app_envs = _resolve_runtime_envs(args)
    try:
        _print_json(_fetch_health(app_envs, timeout=args.timeout))
    except urllib.error.URLError as exc:
        logger.warn(f"× Health check failed: {exc}")
        raise SystemExit(1) from exc


def cmd_serve(args):
    app_envs = _resolve_runtime_envs(args)
    apply_search_app_envs_to_environment(app_envs)
    uvicorn.run(
        "apps.search_app:create_app_from_env",
        host=app_envs["host"],
        port=app_envs["port"],
        factory=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bili Search App service manager",
        epilog=root_epilog(
            quick_start=[
                "python -m apps.search_app_cli start -m dev -ei bili_videos_dev6 -ev elastic_dev",
                "python -m apps.search_app_cli status -m dev -p 21001",
            ],
            examples=[
                "python -m apps.search_app_cli logs -f",
                "python -m apps.search_app_cli restart -m dev -ei bili_videos_dev6 -ev elastic_dev -lc gpt",
                "python -m apps.search_app_cli serve -m dev -ei bili_videos_dev6 -ev elastic_dev",
            ],
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start search app in background")
    _add_runtime_args(start_parser)
    start_parser.set_defaults(func=cmd_start)

    stop_parser = subparsers.add_parser("stop", help="Stop background search app")
    _add_runtime_args(stop_parser)
    stop_parser.set_defaults(func=cmd_stop)

    restart_parser = subparsers.add_parser(
        "restart", help="Restart background search app"
    )
    _add_runtime_args(restart_parser)
    restart_parser.set_defaults(func=cmd_restart)

    status_parser = subparsers.add_parser(
        "status", help="Show process and health status"
    )
    _add_runtime_args(status_parser)
    status_parser.add_argument(
        "--timeout", type=float, default=3.0, help="Health check timeout in seconds"
    )
    status_parser.set_defaults(func=cmd_status)

    logs_parser = subparsers.add_parser("logs", help="Show search app logs")
    _add_runtime_args(logs_parser)
    logs_parser.add_argument(
        "-n", "--lines", type=int, default=50, help="Number of lines to show"
    )
    logs_parser.add_argument(
        "-f", "--follow", action="store_true", default=False, help="Follow log output"
    )
    logs_parser.set_defaults(func=cmd_logs)

    check_parser = subparsers.add_parser("check", help="Call the health endpoint")
    _add_runtime_args(check_parser)
    check_parser.add_argument(
        "--timeout", type=float, default=3.0, help="Health check timeout in seconds"
    )
    check_parser.set_defaults(func=cmd_check)

    serve_parser = subparsers.add_parser("serve", help="Run in foreground")
    _add_runtime_args(serve_parser)
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
