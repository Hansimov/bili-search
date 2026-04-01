from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request

import uvicorn

from datetime import datetime, timedelta
from pathlib import Path
from tclogger import decolored
from webu.cli_support import LocalServiceManager, LocalServiceSpec, ManagedServiceSpec
from zoneinfo import ZoneInfo

from service.envs import SEARCH_APP_ENV_KEYS
from service.envs import apply_search_app_envs_to_environment
from service.envs import resolve_search_app_envs


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE_ROOT / "logs" / "search_app"
APP_TIMEZONE_NAME = "Asia/Shanghai"
APP_TIMEZONE = ZoneInfo(APP_TIMEZONE_NAME)
LOCAL_RUNTIME = "local"
LOCAL_SOURCE = "workspace"
LOCAL_INSTANCE_NAME_TEMPLATE = "bili-search-p{port}"
LOCAL_UVICORN_TARGETS = (
    "service.uvicorn_factory:create_app_from_env",
    "service.app:create_app_from_env",
)
DEFAULT_LOCAL_STATUS_TIMEOUT_SECONDS = 1.5
_UNSET = object()
SERVICE_FILE_PATTERN = re.compile(
    r"^server\.p(?P<port>\d+)\.ei-(?P<elastic_index>[^.]+)\.ev-(?P<elastic_env_name>[^.]+)\.lc-(?P<llm_config>[^.]+)\.pid$"
)


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_process_timezone():
    os.environ["TZ"] = APP_TIMEZONE_NAME
    tzset = getattr(time, "tzset", None)
    if callable(tzset):
        tzset()


def format_datetime_for_output(value: datetime) -> str:
    local_value = value.astimezone(APP_TIMEZONE)
    return local_value.strftime("%Y-%m-%d %H:%M:%S")


def safe_name(value) -> str:
    text = str(value or "default")
    chars = [ch.lower() if ch.isalnum() else "-" for ch in text]
    safe = "".join(chars).strip("-")
    return safe or "default"


def service_files_for_envs(app_envs: dict) -> tuple[Path, Path]:
    parts = [
        f"p{int(app_envs['port'])}",
        f"ei-{safe_name(app_envs.get('elastic_index'))}",
        f"ev-{safe_name(app_envs.get('elastic_env_name'))}",
        f"lc-{safe_name(app_envs.get('llm_config'))}",
    ]
    stem = "server." + ".".join(parts)
    return DATA_DIR / f"{stem}.pid", DATA_DIR / f"{stem}.log"


def build_service_manager(app_envs: dict) -> LocalServiceManager:
    pid_file, log_file = service_files_for_envs(app_envs)
    service_spec = LocalServiceSpec(
        name="bili_search_app",
        uvicorn_target="service.uvicorn_factory:create_app_from_env",
        pid_file=pid_file,
        log_file=log_file,
    )
    managed_service = ManagedServiceSpec(
        name="bili_search_app",
        service=service_spec,
        default_host="0.0.0.0",
    )
    return LocalServiceManager(managed_service)


def parse_service_file_name(pid_file: Path) -> dict | None:
    match = SERVICE_FILE_PATTERN.match(pid_file.name)
    if not match:
        return None
    parsed = match.groupdict()
    parsed["port"] = int(parsed["port"])
    return parsed


def process_is_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_started_at(pid: int | None) -> str | None:
    elapsed_seconds = process_elapsed_seconds(pid)
    if elapsed_seconds is None:
        return None
    started_at = datetime.now(tz=APP_TIMEZONE) - timedelta(seconds=elapsed_seconds)
    return format_datetime_for_output(started_at)


def process_elapsed_seconds(pid: int | None) -> int | None:
    if not pid or pid <= 0:
        return None
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "etimes="],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(result.stdout.strip())
    except (TypeError, ValueError):
        return None


def format_elapsed_seconds_for_output(elapsed_seconds: int | None) -> str | None:
    if elapsed_seconds is None:
        return None

    days, rem = divmod(max(int(elapsed_seconds), 0), 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}min")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)


def process_uptime(pid: int | None) -> str | None:
    return format_elapsed_seconds_for_output(process_elapsed_seconds(pid))


def local_instance_name(app_envs: dict) -> str:
    return LOCAL_INSTANCE_NAME_TEMPLATE.format(port=int(app_envs["port"]))


def _matches_runtime_filters(entry: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if key == "port":
            if int(entry.get("port", -1)) != int(expected):
                return False
            continue
        if str(entry.get(key) or "") != safe_name(expected):
            return False
    return True


def manager_process_snapshot(app_envs: dict) -> dict:
    pid_file, log_file = service_files_for_envs(app_envs)
    snapshot = {
        "pid_file": str(pid_file),
        "log_file": str(log_file),
        "pid": None,
        "status": "not_running",
        "started_at": None,
        "uptime": None,
    }
    if not pid_file.exists():
        return snapshot

    try:
        raw_pid = pid_file.read_text(encoding="utf-8").strip()
        pid = int(raw_pid)
    except (OSError, ValueError):
        snapshot["status"] = "invalid_pid"
        return snapshot

    snapshot["pid"] = pid
    if process_is_running(pid):
        snapshot["status"] = "running"
        snapshot["started_at"] = process_started_at(pid)
        snapshot["uptime"] = process_uptime(pid)
        return snapshot

    snapshot["status"] = "dead"
    return snapshot


def _iter_process_snapshots() -> list[dict]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,user=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    snapshots = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        raw_pid, raw_ppid, user, args = parts
        if not raw_pid.isdigit() or not raw_ppid.isdigit():
            continue
        snapshots.append(
            {
                "pid": int(raw_pid),
                "ppid": int(raw_ppid),
                "user": user,
                "args": args,
            }
        )
    return snapshots


def match_service_worker_process(app_envs: dict) -> dict | None:
    port = int(app_envs["port"])
    port_pattern = f"--port {port}"

    for snapshot in _iter_process_snapshots():
        args = str(snapshot.get("args") or "")
        if "uvicorn" not in args or port_pattern not in args:
            continue
        if not any(target in args for target in LOCAL_UVICORN_TARGETS):
            continue
        return {
            **snapshot,
            "status": "running",
            "started_at": process_started_at(int(snapshot["pid"])),
            "uptime": process_uptime(int(snapshot["pid"])),
        }

    return None


def _runtime_state_from_manager_status(raw_status: str) -> str:
    if raw_status == "running":
        return "running"
    if raw_status in {"dead", "not_running", "invalid_pid"}:
        return "stopped"
    return "unknown"


def probe_local_health(app_envs: dict, *, timeout: float = 3.0) -> dict:
    try:
        payload = fetch_health(app_envs, timeout=timeout)
    except Exception as exc:
        return {
            "supported": True,
            "ok": False,
            "error": str(exc),
        }
    return {
        "supported": True,
        "ok": True,
        "payload": payload,
    }


def build_local_service_snapshot(
    app_envs: dict,
    *,
    include_health: bool = True,
    health_timeout: float = DEFAULT_LOCAL_STATUS_TIMEOUT_SECONDS,
    manager_snapshot: dict | None = None,
    worker_snapshot: dict | None | object = _UNSET,
) -> dict:
    resolved_envs = resolve_search_app_envs(overrides=app_envs)
    manager = manager_snapshot or manager_process_snapshot(resolved_envs)
    worker = (
        match_service_worker_process(resolved_envs)
        if worker_snapshot is _UNSET
        else worker_snapshot
    )
    runtime_state = _runtime_state_from_manager_status(str(manager["status"]))

    health = None
    if include_health and (runtime_state == "running" or worker is not None):
        health = probe_local_health(resolved_envs, timeout=health_timeout)

    health_ok = bool(health and health.get("ok"))
    worker_running = worker is not None
    status = "unknown"
    if runtime_state == "running":
        status = "degraded" if health and not health_ok else "running"
    elif runtime_state == "stopped":
        status = "degraded" if worker_running or health_ok else "exited"
    elif worker_running or health_ok:
        status = "degraded"

    reasons: list[str] = []
    if runtime_state == "stopped" and worker_running:
        reasons.append(
            "manager state is stopped but a matching worker process is still running"
        )
    elif runtime_state == "running" and not worker_running:
        reasons.append(
            "manager state is running but no matching worker process was found"
        )
    if health and not health_ok and health.get("error"):
        reasons.append(f"health check failed: {health['error']}")

    pid_file, log_file = service_files_for_envs(resolved_envs)
    effective_pid = (worker or {}).get("pid") or manager.get("pid")
    return {
        "name": local_instance_name(resolved_envs),
        "runtime": LOCAL_RUNTIME,
        "source": LOCAL_SOURCE,
        "port": int(resolved_envs["port"]),
        "status": status,
        "state": status,
        "runtime_state": runtime_state,
        "raw_state": str(manager["status"]),
        "manager_status": str(manager["status"]),
        "manager_pid": manager.get("pid"),
        "worker_status": "running" if worker_running else "stopped",
        "worker_pid": (worker or {}).get("pid"),
        "pid": effective_pid,
        "started_at": (worker or {}).get("started_at") or manager.get("started_at"),
        "uptime": (worker or {}).get("uptime") or manager.get("uptime"),
        "elastic_index": resolved_envs.get("elastic_index"),
        "elastic_env_name": resolved_envs.get("elastic_env_name"),
        "llm_config": resolved_envs.get("llm_config"),
        "url": health_url(resolved_envs),
        "pid_file": str(pid_file),
        "log_file": str(log_file),
        "health_ok": health.get("ok") if health else None,
        "health": health.get("payload") if health else None,
        "health_error": health.get("error") if health else None,
        "reason": " ; ".join(reasons) if reasons else None,
    }


def list_managed_service_instances(
    *, filters: dict | None = None, include_all: bool = False
) -> list[dict]:
    ensure_data_dir()
    candidate_entries = []
    for pid_file in sorted(DATA_DIR.glob("server.*.pid")):
        parsed = parse_service_file_name(pid_file)
        if not parsed or not _matches_runtime_filters(parsed, filters):
            continue

        candidate_entries.append(
            {
                "app_envs": {
                    **parsed,
                    "host": "0.0.0.0",
                },
                "manager": manager_process_snapshot(
                    {
                        **parsed,
                        "host": "0.0.0.0",
                    }
                ),
                "mtime": pid_file.stat().st_mtime,
            }
        )

    status_rank = {"running": 0, "invalid_pid": 1, "dead": 2, "not_running": 3}
    worker_by_port: dict[int, dict | None] = {}
    claimed_worker_pids: set[int] = set()
    instances = []

    for candidate in sorted(
        candidate_entries,
        key=lambda item: (
            int(item["app_envs"]["port"]),
            status_rank.get(str(item["manager"]["status"]), 99),
            -float(item["mtime"]),
        ),
    ):
        app_envs = candidate["app_envs"]
        port = int(app_envs["port"])
        if port not in worker_by_port:
            worker_by_port[port] = match_service_worker_process(app_envs)

        worker = worker_by_port[port]
        if worker is not None and int(worker.get("pid") or 0) in claimed_worker_pids:
            worker = None

        snapshot = build_local_service_snapshot(
            app_envs,
            include_health=True,
            health_timeout=DEFAULT_LOCAL_STATUS_TIMEOUT_SECONDS,
            manager_snapshot=candidate["manager"],
            worker_snapshot=worker,
        )
        if not include_all and snapshot["status"] == "exited":
            continue

        if worker is not None and worker.get("pid"):
            claimed_worker_pids.add(int(worker["pid"]))

        instances.append(snapshot)

    return sorted(instances, key=lambda item: (item["port"], item["pid"] or 0))


def sanitize_log_file(log_file: Path):
    if not log_file.exists():
        return
    raw = log_file.read_text(encoding="utf-8", errors="replace")
    cleaned = decolored(raw)
    if cleaned != raw:
        log_file.write_text(cleaned, encoding="utf-8")


def runtime_env_vars(app_envs: dict) -> dict[str, str]:
    env = {"TZ": APP_TIMEZONE_NAME}
    for key, env_name in SEARCH_APP_ENV_KEYS.items():
        value = app_envs.get(key)
        if value is not None:
            env[env_name] = str(value)
    return env


def health_url(app_envs: dict) -> str:
    return f"http://127.0.0.1:{int(app_envs['port'])}/health"


def fetch_health(app_envs: dict, timeout: float = 3.0) -> dict:
    request = urllib.request.Request(health_url(app_envs), method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def print_health_json(payload: dict):
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def kill_processes_on_port(port: int):
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = [int(pid) for pid in result.stdout.split() if pid.strip().isdigit()]
        for pid in pids:
            if pid != os.getpid():
                os.kill(pid, signal.SIGKILL)
    except Exception:
        return


def run_foreground(app_envs: dict, *, reload: bool = False, kill: bool = False):
    if kill:
        kill_processes_on_port(int(app_envs["port"]))
    ensure_process_timezone()
    apply_search_app_envs_to_environment(app_envs)
    uvicorn.run(
        "service.app:create_app_from_env",
        host=app_envs["host"],
        port=app_envs["port"],
        factory=True,
        reload=reload,
    )
