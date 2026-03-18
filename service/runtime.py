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


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE_ROOT / "logs" / "search_app"
APP_TIMEZONE_NAME = "Asia/Shanghai"
APP_TIMEZONE = ZoneInfo(APP_TIMEZONE_NAME)
SERVICE_FILE_PATTERN = re.compile(
    r"^server\.(?P<mode>[^.]+)\.p(?P<port>\d+)\.ei-(?P<elastic_index>[^.]+)\.ev-(?P<elastic_env_name>[^.]+)\.lc-(?P<llm_config>[^.]+)\.pid$"
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
        safe_name(app_envs.get("mode")),
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
    if not pid or pid <= 0:
        return None
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "etimes="],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        elapsed_seconds = int(result.stdout.strip())
    except (TypeError, ValueError):
        return None
    started_at = datetime.now(tz=APP_TIMEZONE) - timedelta(seconds=elapsed_seconds)
    return format_datetime_for_output(started_at)


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


def list_managed_service_instances(
    *, filters: dict | None = None, include_all: bool = False
) -> list[dict]:
    ensure_data_dir()
    instances = []
    for pid_file in sorted(DATA_DIR.glob("server.*.pid")):
        parsed = parse_service_file_name(pid_file)
        if not parsed or not _matches_runtime_filters(parsed, filters):
            continue

        pid = None
        try:
            raw_pid = pid_file.read_text(encoding="utf-8").strip()
            pid = int(raw_pid)
        except (OSError, ValueError):
            pid = None

        status = "running" if process_is_running(pid) else "dead"
        if not include_all and status != "running":
            continue

        log_file = pid_file.with_suffix(".log")
        instances.append(
            {
                **parsed,
                "pid": pid,
                "status": status,
                "started_at": process_started_at(pid) if status == "running" else None,
                "pid_file": str(pid_file),
                "log_file": str(log_file),
                "url": health_url({"port": parsed["port"]}),
            }
        )

    return sorted(
        instances, key=lambda item: (item["port"], item["mode"], item["pid"] or 0)
    )


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
