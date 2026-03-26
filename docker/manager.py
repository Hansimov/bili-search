from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess

from datetime import datetime, timedelta, timezone
from pathlib import Path

from service.runtime import WORKSPACE_ROOT, format_datetime_for_output
from service.runtime import runtime_env_vars, safe_name


DOCKER_DIR = WORKSPACE_ROOT / "docker"
DEFAULT_COMPOSE_FILE = DOCKER_DIR / "docker-compose.yml"
DEFAULT_DOCKERFILE = DOCKER_DIR / "Dockerfile"
DEFAULT_BASE_DOCKERFILE = DOCKER_DIR / "Dockerfile.base"
DEFAULT_ENV_FILE = DOCKER_DIR / ".env"
DEFAULT_STAGE_ROOT = WORKSPACE_ROOT / "logs" / "docker_sources"
DEFAULT_PIP_INDEX_URL = "https://mirrors.ustc.edu.cn/pypi/simple"
DEFAULT_PIP_TRUSTED_HOST = "mirrors.ustc.edu.cn"
DEFAULT_UBUNTU_APT_MIRROR = "https://mirrors.ustc.edu.cn/ubuntu"
DEFAULT_EMPTY_CONTEXT_DIR = DEFAULT_STAGE_ROOT / "empty_context"
CONTAINER_APP_ROOT = "/app"
DEFAULT_SYNC_EXCLUDES = (
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    "configs",
    "logs",
    "build",
    "dist",
    "logs/docker_sources",
)
APP_STATE_FILENAME_TEMPLATE = "container_app_state.p{port}.json"


def default_project_name(app_envs: dict) -> str:
    return f"bili-search-p{int(app_envs['port'])}"


def default_container_name(app_envs: dict) -> str:
    return default_project_name(app_envs)


def default_image_name(app_envs: dict) -> str:
    return f"bili-search:p{int(app_envs['port'])}"


def default_base_image_name() -> str:
    return "bili-search-base:ubuntu24.04-py3"


def ensure_empty_context_dir() -> Path:
    DEFAULT_EMPTY_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_EMPTY_CONTEXT_DIR


def resolve_local_dependency_context(name: str) -> Path:
    candidate = (WORKSPACE_ROOT.parent / name).resolve()
    if candidate.exists():
        return candidate
    return ensure_empty_context_dir()


def stage_name(source_kind: str, ref: str | None) -> str:
    suffix = safe_name(ref or "workspace")
    return f"{safe_name(source_kind)}-{suffix}"


def materialize_source(args) -> Path:
    source_kind = getattr(args, "source", "workspace")
    if source_kind == "workspace":
        return WORKSPACE_ROOT

    DEFAULT_STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    stage_dir = DEFAULT_STAGE_ROOT / stage_name(
        source_kind, getattr(args, "git_ref", None)
    )
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    if source_kind == "local-git":
        git_repo = Path(getattr(args, "git_repo", WORKSPACE_ROOT)).resolve()
        git_ref = getattr(args, "git_ref", None) or "HEAD"
        archive_cmd = (
            f"git -C '{git_repo}' archive --format=tar '{git_ref}' | "
            f"tar -xf - -C '{stage_dir}'"
        )
        subprocess.run(archive_cmd, shell=True, check=True)
        return stage_dir

    if source_kind == "remote-git":
        git_url = getattr(args, "git_url", "").strip()
        if not git_url:
            raise ValueError("--git-url is required when --source=remote-git")
        subprocess.run(["git", "clone", git_url, str(stage_dir)], check=True)
        git_ref = getattr(args, "git_ref", None)
        if git_ref:
            subprocess.run(
                ["git", "-C", str(stage_dir), "checkout", git_ref], check=True
            )
        return stage_dir

    raise ValueError(f"Unsupported source kind: {source_kind}")


def resolve_compose_settings(args, app_envs: dict, build_context: Path) -> dict:
    compose_file = Path(getattr(args, "compose_file", DEFAULT_COMPOSE_FILE)).resolve()
    env_file = Path(getattr(args, "env_file", DEFAULT_ENV_FILE)).resolve()
    dockerfile = Path(getattr(args, "dockerfile", DEFAULT_DOCKERFILE)).resolve()
    base_dockerfile = Path(
        getattr(args, "base_dockerfile", DEFAULT_BASE_DOCKERFILE)
    ).resolve()
    configs_dir = Path(
        getattr(args, "configs_dir", WORKSPACE_ROOT / "configs")
    ).resolve()
    logs_dir = Path(getattr(args, "logs_dir", WORKSPACE_ROOT / "logs")).resolve()
    return {
        "compose_file": compose_file,
        "env_file": env_file,
        "dockerfile": dockerfile,
        "base_dockerfile": base_dockerfile,
        "build_context": build_context.resolve(),
        "configs_dir": configs_dir,
        "logs_dir": logs_dir,
        "project_name": getattr(args, "project_name", None)
        or default_project_name(app_envs),
        "service_name": getattr(args, "service_name", None) or "bili-search",
        "container_name": getattr(args, "container_name", None)
        or default_container_name(app_envs),
        "image": getattr(args, "image", None) or default_image_name(app_envs),
        "base_image": getattr(args, "base_image", None) or default_base_image_name(),
        "pip_index_url": getattr(args, "pip_index_url", None) or DEFAULT_PIP_INDEX_URL,
        "pip_trusted_host": getattr(args, "pip_trusted_host", None)
        or DEFAULT_PIP_TRUSTED_HOST,
        "ubuntu_apt_mirror": getattr(args, "ubuntu_apt_mirror", None)
        or DEFAULT_UBUNTU_APT_MIRROR,
        "sedb_context": Path(
            getattr(args, "sedb_context", None)
            or resolve_local_dependency_context("sedb")
        ).resolve(),
        "tclogger_context": Path(
            getattr(args, "tclogger_context", None)
            or resolve_local_dependency_context("tclogger")
        ).resolve(),
        "webu_context": Path(
            getattr(args, "webu_context", None)
            or resolve_local_dependency_context("webu")
        ).resolve(),
    }


def compose_env(args, app_envs: dict, settings: dict) -> dict[str, str]:
    env = dict(os.environ)
    env.update(runtime_env_vars(app_envs))
    env.update(
        {
            "DOCKER_BUILDKIT": "1",
            "COMPOSE_DOCKER_CLI_BUILD": "1",
            "BSDK_BUILD_CONTEXT": str(settings["build_context"]),
            "BSDK_DOCKERFILE": str(settings["dockerfile"]),
            "BSDK_CONFIGS_DIR": str(settings["configs_dir"]),
            "BSDK_LOGS_DIR": str(settings["logs_dir"]),
            "BSDK_PROJECT_NAME": settings["project_name"],
            "BSDK_SERVICE_NAME": settings["service_name"],
            "BSDK_CONTAINER_NAME": settings["container_name"],
            "BSDK_IMAGE": settings["image"],
            "BSDK_BASE_IMAGE": settings["base_image"],
            "BSDK_PIP_INDEX_URL": settings["pip_index_url"],
            "BSDK_PIP_TRUSTED_HOST": settings["pip_trusted_host"],
            "BSDK_UBUNTU_APT_MIRROR": settings["ubuntu_apt_mirror"],
            "BSDK_SEDB_CONTEXT": str(settings["sedb_context"]),
            "BSDK_TCLOGGER_CONTEXT": str(settings["tclogger_context"]),
            "BSDK_WEBU_CONTEXT": str(settings["webu_context"]),
        }
    )
    return env


def docker_image_exists(image: str, *, env: dict[str, str] | None = None) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def _parse_env_pairs(env_items: list[str] | None) -> dict[str, str]:
    parsed = {}
    for item in env_items or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def _format_started_at(value: str | None) -> str | None:
    if not value or value.startswith("0001-01-01"):
        return None
    text = value.rstrip("Z")
    if "." in text:
        head, frac = text.split(".", 1)
        text = f"{head}.{frac[:6]}"
    dt = datetime.fromisoformat(text)
    if value.endswith("Z"):
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return format_datetime_for_output(dt)


def _parse_container_datetime(value: str | None) -> datetime | None:
    if not value or value.startswith("0001-01-01"):
        return None
    text = value.rstrip("Z")
    if "." in text:
        head, frac = text.split(".", 1)
        text = f"{head}.{frac[:6]}"
    dt = datetime.fromisoformat(text)
    if value.endswith("Z"):
        return dt.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_uptime(value: str | None, *, now: datetime | None = None) -> str | None:
    started_at = _parse_container_datetime(value)
    if started_at is None:
        return None
    current = now or datetime.now(timezone.utc)
    elapsed_seconds = max(int((current - started_at).total_seconds()), 0)
    days, rem = divmod(elapsed_seconds, 24 * 3600)
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


def app_state_path(settings: dict, app_envs: dict) -> Path:
    return (
        settings["logs_dir"]
        / "search_app"
        / APP_STATE_FILENAME_TEMPLATE.format(port=int(app_envs["port"]))
    )


def inspect_container_app_process(args, app_envs: dict) -> dict | None:
    settings = resolve_compose_settings(args, app_envs, WORKSPACE_ROOT)
    env = compose_env(args, app_envs, settings)
    inspect_command = [
        "docker",
        "exec",
        settings["container_name"],
        "python3",
        "-c",
        (
            "import json, subprocess; "
            "lines = subprocess.check_output(['ps', '-eo', 'pid=,etimes=,args='], text=True).splitlines(); "
            "result = None; "
            "\nfor line in lines:\n"
            "    parts = line.strip().split(None, 2)\n"
            "    if len(parts) < 3:\n"
            "        continue\n"
            "    pid, elapsed, command = parts\n"
            "    if 'bsdk start --runtime local --foreground' not in command:\n"
            "        continue\n"
            "    result = {'pid': int(pid), 'elapsed_seconds': int(elapsed)}\n"
            "    break\n"
            "print(json.dumps(result) if result else '')"
        ),
    ]
    result = subprocess.run(
        inspect_command,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        process_state = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None

    elapsed_seconds = int(process_state.get("elapsed_seconds") or 0)
    started_at = datetime.now(timezone.utc) - timedelta(seconds=max(elapsed_seconds, 0))
    started_at = started_at.replace(microsecond=0)
    started_at_value = started_at.isoformat().replace("+00:00", "Z")
    return {
        "pid": process_state.get("pid"),
        "started_at": _format_started_at(started_at_value),
        "uptime": _format_uptime(started_at_value),
        "restart_count": None,
    }


def read_container_app_state(args, app_envs: dict) -> dict | None:
    settings = resolve_compose_settings(args, app_envs, WORKSPACE_ROOT)
    state_path = app_state_path(settings, app_envs)
    if not state_path.exists():
        return inspect_container_app_process(args, app_envs)

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return inspect_container_app_process(args, app_envs)

    started_at_value = str(state.get("started_at") or "").strip()
    return {
        "pid": state.get("pid"),
        "started_at": _format_started_at(started_at_value),
        "uptime": _format_uptime(started_at_value),
        "restart_count": state.get("restart_count"),
    }


def _container_sync_cleanup_command() -> str:
    return (
        f"find {shlex.quote(CONTAINER_APP_ROOT)} -mindepth 1 -maxdepth 1 "
        "! -name configs ! -name logs -exec rm -rf {} +"
    )


def _is_bili_search_container(
    name: str, image: str, labels: dict, env_map: dict
) -> bool:
    project = labels.get("com.docker.compose.project", "")
    service = labels.get("com.docker.compose.service", "")
    search_fields = [name, image, project, service]
    if any("bili-search" in (field or "") for field in search_fields):
        return True
    return "BILI_SEARCH_APP_PORT" in env_map and (
        name.startswith("bili-") or image.startswith("bili-search")
    )


def _matches_runtime_filters(entry: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = entry.get(key)
        if key == "port":
            if int(actual or -1) != int(expected):
                return False
            continue
        if str(actual or "") != str(expected):
            return False
    return True


def list_bili_search_containers(
    *, filters: dict | None = None, include_all: bool = False
) -> list[dict]:
    ps_command = ["docker", "ps", "-q"]
    if include_all:
        ps_command.insert(2, "-a")
    ps_result = subprocess.run(
        ps_command,
        capture_output=True,
        text=True,
        check=False,
    )
    container_ids = [
        line.strip() for line in ps_result.stdout.splitlines() if line.strip()
    ]
    if ps_result.returncode != 0 or not container_ids:
        return []

    inspect_result = subprocess.run(
        ["docker", "inspect", *container_ids],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect_result.returncode != 0:
        return []

    containers = []
    for item in json.loads(inspect_result.stdout or "[]"):
        config = item.get("Config") or {}
        state = item.get("State") or {}
        labels = config.get("Labels") or {}
        env_map = _parse_env_pairs(config.get("Env"))
        name = (item.get("Name") or "").lstrip("/")
        image = config.get("Image") or ""
        if not _is_bili_search_container(name, image, labels, env_map):
            continue

        status = state.get("Status") or (
            "running" if state.get("Running") else "unknown"
        )
        health_status = (state.get("Health") or {}).get("Status")
        display_status = (
            f"{status}/{health_status}"
            if status == "running" and health_status
            else status
        )
        port_value = env_map.get("BILI_SEARCH_APP_PORT")
        port = int(port_value) if port_value and port_value.isdigit() else None
        container = {
            "id": (item.get("Id") or "")[:12],
            "name": name,
            "image": image,
            "project_name": labels.get("com.docker.compose.project", ""),
            "service_name": labels.get("com.docker.compose.service", ""),
            "port": port,
            "elastic_index": env_map.get("BILI_SEARCH_APP_ELASTIC_INDEX"),
            "elastic_env_name": env_map.get("BILI_SEARCH_APP_ELASTIC_ENV_NAME"),
            "llm_config": env_map.get("BILI_SEARCH_APP_LLM_CONFIG"),
            "status": display_status,
            "started_at": _format_started_at(state.get("StartedAt")),
            "uptime": (
                _format_uptime(state.get("StartedAt")) if state.get("Running") else None
            ),
            "network_mode": (item.get("HostConfig") or {}).get("NetworkMode"),
        }
        if not include_all and status != "running":
            continue
        if not _matches_runtime_filters(container, filters):
            continue
        containers.append(container)

    return sorted(
        containers,
        key=lambda item: (
            item["port"] is None,
            item["port"] or 0,
            item["name"],
        ),
    )


def find_bili_search_container_by_port(port: int, *, include_all: bool = True) -> dict:
    matches = list_bili_search_containers(
        filters={"port": int(port)},
        include_all=include_all,
    )
    if not matches:
        raise LookupError(f"No bili-search docker container found on port {port}")

    running_matches = [
        item for item in matches if str(item.get("status") or "").startswith("running")
    ]
    if len(running_matches) == 1:
        return running_matches[0]
    if len(running_matches) > 1:
        raise LookupError(
            f"Multiple running bili-search docker containers found on port {port}"
        )
    if len(matches) == 1:
        return matches[0]
    raise LookupError(f"Multiple bili-search docker containers found on port {port}")


def ensure_base_image(args, app_envs: dict) -> subprocess.CompletedProcess | None:
    settings = resolve_compose_settings(args, app_envs, WORKSPACE_ROOT)
    env = compose_env(args, app_envs, settings)
    if docker_image_exists(settings["base_image"], env=env):
        return None
    return run_base_build(args, app_envs)


def run_base_build(args, app_envs: dict) -> subprocess.CompletedProcess:
    settings = resolve_compose_settings(args, app_envs, WORKSPACE_ROOT)
    env = compose_env(args, app_envs, settings)
    command = [
        "docker",
        "build",
        "-f",
        str(settings["base_dockerfile"]),
        "-t",
        settings["base_image"],
        "--build-arg",
        f"UBUNTU_APT_MIRROR={settings['ubuntu_apt_mirror']}",
        "--build-arg",
        f"PIP_INDEX_URL={settings['pip_index_url']}",
        "--build-arg",
        f"PIP_TRUSTED_HOST={settings['pip_trusted_host']}",
        str(DOCKER_DIR),
    ]
    return subprocess.run(command, env=env, check=False, text=True)


def compose_base_command(settings: dict) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        settings["project_name"],
        "--env-file",
        str(settings["env_file"]),
        "-f",
        str(settings["compose_file"]),
    ]


def _completed_process(
    command: list[str],
    returncode: int,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def sync_source_to_container(
    args,
    app_envs: dict,
    *,
    source_root: Path | None = None,
) -> subprocess.CompletedProcess:
    sync_root = (source_root or materialize_source(args)).resolve()
    settings = resolve_compose_settings(args, app_envs, sync_root)
    env = compose_env(args, app_envs, settings)

    cleanup_command = [
        "docker",
        "exec",
        settings["container_name"],
        "sh",
        "-lc",
        _container_sync_cleanup_command(),
    ]
    cleanup_result = subprocess.run(
        cleanup_command,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if cleanup_result.returncode != 0:
        return _completed_process(
            cleanup_command,
            returncode=cleanup_result.returncode,
            stdout=cleanup_result.stdout or "",
            stderr=cleanup_result.stderr or "",
        )

    tar_command = ["tar", "-C", str(sync_root)]
    for pattern in DEFAULT_SYNC_EXCLUDES:
        tar_command += ["--exclude", pattern]
    tar_command += ["-cf", "-", "."]
    extract_command = [
        "docker",
        "exec",
        "-i",
        settings["container_name"],
        "tar",
        "--no-same-owner",
        "-xf",
        "-",
        "-C",
        CONTAINER_APP_ROOT,
    ]

    tar_process = subprocess.Popen(
        tar_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert tar_process.stdout is not None
    extract_result = subprocess.run(
        extract_command,
        stdin=tar_process.stdout,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    tar_process.stdout.close()
    tar_stderr = tar_process.stderr.read().decode("utf-8", errors="replace")
    tar_returncode = tar_process.wait()

    combined_stdout = "\n".join(
        output for output in [cleanup_result.stdout, extract_result.stdout] if output
    )
    combined_stderr = (extract_result.stderr or "") + tar_stderr
    returncode = extract_result.returncode or tar_returncode
    return _completed_process(
        [*cleanup_command, "&&", *tar_command, "|", *extract_command],
        returncode=returncode,
        stdout=combined_stdout,
        stderr=combined_stderr,
    )


def run_container_app_action(
    args,
    action: str,
    app_envs: dict,
) -> subprocess.CompletedProcess:
    source_root = materialize_source(args)
    settings = resolve_compose_settings(args, app_envs, source_root)
    env = compose_env(args, app_envs, settings)
    outputs: list[str] = []
    errors: list[str] = []

    if action == "restart":
        if bool(getattr(args, "sync_code", True)):
            sync_result = sync_source_to_container(
                args,
                app_envs,
                source_root=source_root,
            )
            if sync_result.stdout:
                outputs.append(sync_result.stdout)
            if sync_result.stderr:
                errors.append(sync_result.stderr)
            if sync_result.returncode != 0:
                return _completed_process(
                    ["docker", "exec", settings["container_name"], "sync"],
                    returncode=sync_result.returncode,
                    stdout="\n".join(outputs),
                    stderr="\n".join(errors),
                )

        signal_result = subprocess.run(
            ["docker", "kill", "--signal", "HUP", settings["container_name"]],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if signal_result.stdout:
            outputs.append(signal_result.stdout)
        if signal_result.stderr:
            errors.append(signal_result.stderr)
        return _completed_process(
            ["docker", "kill", "--signal", "HUP", settings["container_name"]],
            returncode=signal_result.returncode,
            stdout="\n".join(output for output in outputs if output),
            stderr="\n".join(error for error in errors if error),
        )

    raise ValueError(f"Unsupported app action: {action}")


def run_compose(args, action: str, app_envs: dict) -> subprocess.CompletedProcess:
    build_context = materialize_source(args)
    settings = resolve_compose_settings(args, app_envs, build_context)
    env = compose_env(args, app_envs, settings)
    command = compose_base_command(settings)
    service_name = settings["service_name"]
    if action == "build":
        command += ["build", service_name]
    elif action == "start":
        command += ["up", "-d"]
        if not getattr(args, "no_build", False):
            command.append("--build")
        command.append(service_name)
    elif action == "stop":
        command += ["stop", service_name]
    elif action == "down":
        command += ["down", "--remove-orphans"]
    elif action == "restart":
        command += ["restart", service_name]
    elif action == "recreate":
        command += ["up", "-d", "--force-recreate", "--build", service_name]
    elif action == "status":
        command += ["ps", service_name]
    elif action == "logs":
        command += ["logs"]
        if getattr(args, "follow", False):
            command.append("-f")
        if getattr(args, "lines", None):
            command += ["--tail", str(args.lines)]
        command.append(service_name)
    elif action == "config":
        command += ["config"]
    else:
        raise ValueError(f"Unsupported docker action: {action}")
    return subprocess.run(command, env=env, check=False, text=True)
