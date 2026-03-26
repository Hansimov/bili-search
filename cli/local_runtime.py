from __future__ import annotations

import json
import urllib.error

from pathlib import Path
from tclogger import decolored
from tclogger import logger, logstr

from cli.common import explicit_runtime_filters_from_args, format_table
from cli.common import resolve_runtime_envs_from_args
from service.runtime import build_service_manager as _build_service_manager
from service.runtime import ensure_data_dir
from service.runtime import fetch_health as _fetch_health
from service.runtime import health_url as _health_url
from service.runtime import kill_processes_on_port
from service.runtime import list_managed_service_instances
from service.runtime import print_health_json as _print_json
from service.runtime import run_foreground
from service.runtime import runtime_env_vars
from service.runtime import sanitize_log_file as _sanitize_log_file


def cmd_start(args):
    app_envs = resolve_runtime_envs_from_args(args)
    if getattr(args, "foreground", False):
        logger.note(
            f"> Starting search app in foreground on {app_envs['host']}:{app_envs['port']} ..."
        )
        run_foreground(
            app_envs,
            reload=bool(getattr(args, "reload", False)),
            kill=bool(getattr(args, "kill", False)),
        )
        return

    service_manager = _build_service_manager(app_envs)
    current = service_manager.status()
    if current["status"] == "running":
        logger.warn(f"  × Search app already running (PID: {current['pid']})")
        return

    ensure_data_dir()
    if getattr(args, "kill", False):
        kill_processes_on_port(int(app_envs["port"]))
    logger.note(f"> Starting search app on {app_envs['host']}:{app_envs['port']} ...")
    result = service_manager.start(
        host=app_envs["host"],
        port=app_envs["port"],
        extra_env=runtime_env_vars(app_envs),
    )
    logger.okay(f"  ✓ Search app started (PID: {result['pid']})")
    logger.mesg(f"  Log: {logstr.file(service_manager.log_file)}")


def cmd_stop(args):
    app_envs = resolve_runtime_envs_from_args(args)
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
    app_envs = resolve_runtime_envs_from_args(args)
    if getattr(args, "foreground", False):
        cmd_start(args)
        return
    service_manager = _build_service_manager(app_envs)
    stop_result = service_manager.stop()
    start_result = service_manager.start(
        host=app_envs["host"],
        port=app_envs["port"],
        extra_env=runtime_env_vars(app_envs),
    )
    if stop_result["status"] in ("stopped", "stale_pid") and stop_result["pid"]:
        logger.note(f"> Stopping search app (PID: {stop_result['pid']}) ...")
        logger.okay("  ✓ Search app stopped")
    logger.note(f"> Starting search app on {app_envs['host']}:{app_envs['port']} ...")
    logger.okay(f"  ✓ Search app started (PID: {start_result['pid']})")
    logger.mesg(f"  Log: {logstr.file(service_manager.log_file)}")


def cmd_status(args):
    app_envs = resolve_runtime_envs_from_args(args)
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
    app_envs = resolve_runtime_envs_from_args(args)
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
    app_envs = resolve_runtime_envs_from_args(args)
    try:
        _print_json(_fetch_health(app_envs, timeout=args.timeout))
    except urllib.error.URLError as exc:
        logger.warn(f"× Health check failed: {exc}")
        raise SystemExit(1) from exc


def cmd_ps(args):
    instances = list_managed_service_instances(
        filters=explicit_runtime_filters_from_args(args),
        include_all=bool(getattr(args, "all", False)),
    )
    if not instances:
        logger.mesg("  No managed bili-search local services found")
        return

    rows = [
        [
            item["port"],
            item["status"],
            item["started_at"] or "-",
            item["llm_config"],
            item["pid"] or "-",
        ]
        for item in instances
    ]
    print(
        format_table(
            ["PORT", "STATUS", "STARTED_AT", "LLM", "PID"],
            rows,
        )
    )
