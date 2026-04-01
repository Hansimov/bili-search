from __future__ import annotations

import json
import urllib.error

from pathlib import Path
from tclogger import decolored
from tclogger import logger, logstr

from cli.common import explicit_runtime_filters_from_args
from cli.common import render_key_value_table, render_list_table
from cli.common import resolve_runtime_envs_from_args
from service.runtime import build_service_manager as _build_service_manager
from service.runtime import build_local_service_snapshot
from service.runtime import ensure_data_dir
from service.runtime import fetch_health as _fetch_health
from service.runtime import health_url as _health_url
from service.runtime import kill_processes_on_port
from service.runtime import list_managed_service_instances
from service.runtime import print_health_json as _print_json
from service.runtime import run_foreground
from service.runtime import runtime_env_vars
from service.runtime import sanitize_log_file as _sanitize_log_file
from service.runtime import service_files_for_envs


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

    snapshot = build_local_service_snapshot(app_envs, include_health=False)
    if snapshot["runtime_state"] == "running":
        logger.warn(f"  × Search app already running (PID: {snapshot['pid']})")
        return
    if snapshot["worker_status"] == "running" and not getattr(args, "kill", False):
        logger.warn(
            "  × Matching worker is still alive while manager state is stale — use --kill or stop first"
        )
        return

    service_manager = _build_service_manager(app_envs)
    ensure_data_dir()
    if getattr(args, "kill", False):
        if snapshot["worker_status"] == "running":
            logger.note(
                f"> Cleaning up existing search worker on port {app_envs['port']} before startup ..."
            )
        kill_processes_on_port(int(app_envs["port"]))
        pid_file, _log_file = service_files_for_envs(app_envs)
        try:
            pid_file.unlink(missing_ok=True)
        except OSError:
            pass
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
    snapshot = build_local_service_snapshot(app_envs, include_health=False)
    if (
        snapshot["runtime_state"] == "stopped"
        and snapshot["worker_status"] == "running"
    ):
        logger.note(f"> Stopping orphaned search worker on port {app_envs['port']} ...")
        kill_processes_on_port(int(app_envs["port"]))
        pid_file, _log_file = service_files_for_envs(app_envs)
        try:
            pid_file.unlink(missing_ok=True)
        except OSError:
            pass
        logger.okay("  ✓ Orphaned search worker stopped")
        return

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
    snapshot = build_local_service_snapshot(app_envs, include_health=False)

    service_manager = _build_service_manager(app_envs)
    stop_result = None
    if snapshot["runtime_state"] == "running":
        stop_result = service_manager.stop()
    elif snapshot["worker_status"] == "running":
        logger.note(
            f"> Cleaning up orphaned search worker on port {app_envs['port']} before restart ..."
        )
        kill_processes_on_port(int(app_envs["port"]))
        pid_file, _log_file = service_files_for_envs(app_envs)
        try:
            pid_file.unlink(missing_ok=True)
        except OSError:
            pass

    start_result = service_manager.start(
        host=app_envs["host"],
        port=app_envs["port"],
        extra_env=runtime_env_vars(app_envs),
    )
    if (
        stop_result
        and stop_result["status"] in ("stopped", "stale_pid")
        and stop_result["pid"]
    ):
        logger.note(f"> Stopping search app (PID: {stop_result['pid']}) ...")
        logger.okay("  ✓ Search app stopped")
    logger.note(f"> Starting search app on {app_envs['host']}:{app_envs['port']} ...")
    logger.okay(f"  ✓ Search app started (PID: {start_result['pid']})")
    logger.mesg(f"  Log: {logstr.file(service_manager.log_file)}")


def cmd_status(args):
    app_envs = resolve_runtime_envs_from_args(args)
    service_manager = _build_service_manager(app_envs)
    _sanitize_log_file(service_manager.log_file)
    snapshot = build_local_service_snapshot(app_envs, health_timeout=args.timeout)
    if getattr(args, "output", "table") == "json":
        _print_json(snapshot)
        return

    health_value = "-"
    if snapshot["health"] is not None:
        health_value = json.dumps(snapshot["health"], ensure_ascii=False)
    elif snapshot["health_error"]:
        health_value = f"FAILED: {snapshot['health_error']}"

    status_rows = {
        "Status": snapshot["status"],
        "Runtime": snapshot["runtime"],
        "Source": snapshot["source"],
        "Manager State": snapshot["manager_status"],
        "Worker State": snapshot["worker_status"],
        "Manager PID": snapshot["manager_pid"] or "-",
        "Worker PID": snapshot["worker_pid"] or "-",
        "Started At": snapshot["started_at"] or "-",
        "Uptime": snapshot["uptime"] or "-",
        "URL": snapshot["url"],
        "Log": snapshot["log_file"],
        "Health": health_value,
    }
    if snapshot["reason"]:
        status_rows["Reason"] = snapshot["reason"]
    print(render_key_value_table(status_rows))


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

    if getattr(args, "output", "table") == "json":
        _print_json(instances)
        return

    rows = [
        [
            item["port"],
            item["status"],
            item["runtime"],
            item["source"],
            item["started_at"] or "-",
            item["uptime"] or "-",
            item["llm_config"],
            item["name"],
        ]
        for item in instances
    ]
    print(
        render_list_table(
            [
                "PORTS",
                "STATUS",
                "RUNTIME",
                "SOURCE",
                "STARTED_AT",
                "UPTIME",
                "LLM",
                "NAME",
            ],
            rows,
        )
    )
