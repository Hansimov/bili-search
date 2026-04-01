from __future__ import annotations

import argparse
import json

from pathlib import Path
from tclogger import logger
from webu.clis.helpers import root_epilog

from cli import local_runtime as local_runtime_cli
from cli.common import add_runtime_mode_arg, add_shared_runtime_args
from cli.common import explicit_runtime_filters_from_args
from cli.common import render_key_value_table, render_list_table
from cli.common import resolve_runtime_envs_from_args
from docker.manager import DEFAULT_COMPOSE_FILE
from docker.manager import DEFAULT_DOCKERFILE
from docker.manager import DEFAULT_ENV_FILE
from docker.manager import DEFAULT_BASE_DOCKERFILE
from docker.manager import run_compose
from docker.manager import run_container_app_action
from docker.manager import ensure_base_image
from docker.manager import find_bili_search_container_by_port
from docker.manager import list_bili_search_containers
from docker.manager import read_container_app_state
from docker.manager import run_base_build
from service.runtime import ensure_process_timezone, fetch_health, health_url
from service.runtime import list_managed_service_instances


def add_docker_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--compose-file",
        type=str,
        default=str(DEFAULT_COMPOSE_FILE),
        help="docker-compose.yml path",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(DEFAULT_ENV_FILE),
        help="Compose env file path",
    )
    parser.add_argument(
        "--dockerfile",
        type=str,
        default=str(DEFAULT_DOCKERFILE),
        help="Dockerfile path",
    )
    parser.add_argument(
        "--base-dockerfile",
        type=str,
        default=str(DEFAULT_BASE_DOCKERFILE),
        help="Base image Dockerfile path",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=str(Path("configs")),
        help="Host configs directory mounted into the container",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=str(Path("logs")),
        help="Host logs directory mounted into the container",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Compose project name override",
    )
    parser.add_argument(
        "--service-name",
        type=str,
        default=None,
        help="Compose service name override",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default=None,
        help="Container name override",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image name override",
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default=None,
        help="Reusable Ubuntu base image tag override",
    )
    parser.add_argument(
        "--pip-index-url",
        type=str,
        default=None,
        help="Primary pip index URL used by docker builds",
    )
    parser.add_argument(
        "--pip-extra-index-url",
        type=str,
        default=None,
        help="Fallback pip index URL used when the primary mirror is stale or flaky",
    )
    parser.add_argument(
        "--pip-trusted-host",
        type=str,
        default=None,
        help="Trusted host paired with --pip-index-url",
    )
    parser.add_argument(
        "--pip-retries",
        type=int,
        default=None,
        help="Retry count for docker pip downloads before falling back",
    )
    parser.add_argument(
        "--pip-timeout",
        type=int,
        default=None,
        help="Per-request timeout in seconds for docker pip downloads",
    )
    parser.add_argument(
        "--ubuntu-apt-mirror",
        type=str,
        default=None,
        help="Ubuntu apt mirror used by docker base-image builds",
    )
    parser.add_argument(
        "--sedb-context",
        type=str,
        default=None,
        help="Local sedb source tree injected into docker builds when available",
    )
    parser.add_argument(
        "--tclogger-context",
        type=str,
        default=None,
        help="Local tclogger source tree injected into docker builds when available",
    )
    parser.add_argument(
        "--webu-context",
        type=str,
        default=None,
        help="Local webu source tree injected into docker builds when available",
    )
    parser.add_argument(
        "--source",
        choices=["workspace", "local-git", "remote-git"],
        default="workspace",
        help="Source code to build into the container image",
    )
    parser.add_argument(
        "--git-repo",
        type=str,
        default=None,
        help="Local git repository used when --source=local-git",
    )
    parser.add_argument(
        "--git-ref",
        type=str,
        default=None,
        help="Branch, tag, or commit used by git-backed sources",
    )
    parser.add_argument(
        "--git-url",
        type=str,
        default=None,
        help="Remote git URL used when --source=remote-git",
    )


def add_restart_control_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--restart-scope",
        choices=["app", "container"],
        default="app",
        help="Restart only the app process inside the container, or recreate/restart the whole container",
    )
    parser.set_defaults(sync_code=True)
    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument(
        "--sync-code",
        dest="sync_code",
        action="store_true",
        help="Sync the latest source code before restarting the target service (default)",
    )
    sync_group.add_argument(
        "--no-sync-code",
        dest="sync_code",
        action="store_false",
        help="Restart without syncing the latest source code first",
    )


def _runtime(args) -> str:
    return str(getattr(args, "runtime", "docker") or "docker")


def _json_output(args) -> bool:
    return str(getattr(args, "output", "table") or "table") == "json"


def _print_json_payload(payload):
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _ensure_docker_runtime(args, command_name: str):
    if _runtime(args) != "docker":
        raise SystemExit(f"{command_name} only supports --runtime docker")


def _validate_local_only_flags(args, *flag_names: str):
    unsupported = [
        flag_name for flag_name in flag_names if bool(getattr(args, flag_name, False))
    ]
    if unsupported:
        formatted = ", ".join(f"--{name.replace('_', '-')}" for name in unsupported)
        raise SystemExit(f"{formatted} only supports --runtime local")


def _report_compose_result(result):
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _resolve_existing_instance_args(args):
    port = getattr(args, "port", None)
    if port is None:
        return args, resolve_runtime_envs_from_args(args)

    try:
        container = find_bili_search_container_by_port(port, include_all=True)
    except LookupError as exc:
        raise SystemExit(str(exc)) from exc

    args.project_name = container.get("project_name") or getattr(
        args, "project_name", None
    )
    args.service_name = container.get("service_name") or getattr(
        args, "service_name", None
    )
    args.container_name = container.get("name") or getattr(args, "container_name", None)
    args.image = container.get("image") or getattr(args, "image", None)
    app_envs = {
        "host": getattr(args, "host", None) or "0.0.0.0",
        "port": int(container["port"]),
        "elastic_index": container.get("elastic_index"),
        "elastic_env_name": container.get("elastic_env_name"),
        "llm_config": container.get("llm_config"),
    }
    return args, app_envs


def _cmd_docker_start(args):
    app_envs = resolve_runtime_envs_from_args(args)
    logger.note(
        f"> Starting dockerized search service on port {app_envs['port']} from source={args.source} ..."
    )
    if not getattr(args, "no_base_build", False):
        base_result = ensure_base_image(args, app_envs)
        if base_result is not None:
            _report_compose_result(base_result)
    result = run_compose(args, "start", app_envs)
    _report_compose_result(result)


def _cmd_docker_build(args):
    app_envs = resolve_runtime_envs_from_args(args)
    logger.note(
        f"> Building dockerized search service image on port {app_envs['port']} from source={args.source} ..."
    )
    if not getattr(args, "no_base_build", False):
        base_result = ensure_base_image(args, app_envs)
        if base_result is not None:
            _report_compose_result(base_result)
    result = run_compose(args, "build", app_envs)
    _report_compose_result(result)


def _cmd_docker_build_base(args):
    app_envs = resolve_runtime_envs_from_args(args)
    logger.note(
        "> Building reusable Ubuntu base image for bili-search docker runtime ..."
    )
    result = run_base_build(args, app_envs)
    _report_compose_result(result)


def _cmd_docker_stop(args):
    args, app_envs = _resolve_existing_instance_args(args)
    logger.note("> Stopping dockerized search service ...")
    result = run_compose(args, "stop", app_envs)
    _report_compose_result(result)


def _cmd_docker_down(args):
    args, app_envs = _resolve_existing_instance_args(args)
    logger.note("> Removing dockerized search service containers ...")
    result = run_compose(args, "down", app_envs)
    _report_compose_result(result)


def _cmd_docker_restart(args):
    args, app_envs = _resolve_existing_instance_args(args)
    scope = getattr(args, "restart_scope", "app")
    sync_code = bool(getattr(args, "sync_code", True))
    logger.note(
        f"> Restarting dockerized search service (scope={scope}, sync_code={str(sync_code).lower()}) ..."
    )
    if scope == "app":
        result = run_container_app_action(args, "restart", app_envs)
        _report_compose_result(result)
        logger.mesg("  App restarted in-place inside the container")
        logger.mesg("  Container uptime will not change for app-scope restarts")
        return

    if sync_code and not getattr(args, "no_base_build", False):
        base_result = ensure_base_image(args, app_envs)
        if base_result is not None:
            _report_compose_result(base_result)
    action = "recreate" if sync_code else "restart"
    result = run_compose(args, action, app_envs)
    _report_compose_result(result)


def _cmd_docker_status(args):
    args, app_envs = _resolve_existing_instance_args(args)
    container = find_bili_search_container_by_port(app_envs["port"], include_all=True)
    app_state = read_container_app_state(args, app_envs)
    health_value = "-"
    try:
        health = fetch_health(app_envs, timeout=args.timeout)
        health_value = json.dumps(health, ensure_ascii=False)
    except Exception as exc:
        health_value = f"FAILED: {exc}"

    payload = {
        "name": container.get("name") or "-",
        "runtime": container.get("runtime") or "docker",
        "source": container.get("source") or "workspace",
        "project": container.get("project_name") or "-",
        "service": container.get("service_name") or "-",
        "image": container.get("image") or "-",
        "status": container.get("status") or "-",
        "runtime_state": (
            "running"
            if str(container.get("status") or "").startswith("running")
            else "stopped"
        ),
        "ports": container.get("ports") or str(app_envs["port"]),
        "port": int(app_envs["port"]),
        "network": container.get("network_mode") or "-",
        "started_at": container.get("started_at") or "-",
        "uptime": container.get("uptime") or "-",
        "elastic_index": container.get("elastic_index") or "-",
        "elastic_env_name": container.get("elastic_env_name") or "-",
        "llm_config": container.get("llm_config") or "-",
        "app_started_at": app_state["started_at"] if app_state else "-",
        "app_uptime": app_state["uptime"] if app_state else "-",
        "app_restart_count": (
            app_state.get("restart_count")
            if app_state and app_state.get("restart_count") is not None
            else "-"
        ),
        "health_url": health_url(app_envs),
        "health": health_value,
    }
    if _json_output(args):
        _print_json_payload(payload)
        return

    status_rows = {
        "Container": payload["name"],
        "Runtime": payload["runtime"],
        "Source": payload["source"],
        "Project": payload["project"],
        "Service": payload["service"],
        "Image": payload["image"],
        "Status": payload["status"],
        "Ports": payload["ports"],
        "Network": payload["network"],
        "Container Started At": payload["started_at"],
        "Container Uptime": payload["uptime"],
        "Elastic Index": payload["elastic_index"],
        "Elastic Env": payload["elastic_env_name"],
        "LLM": payload["llm_config"],
        "App Started At": payload["app_started_at"],
        "App Uptime": payload["app_uptime"],
        "App Restart Count": payload["app_restart_count"],
        "Health URL": payload["health_url"],
        "Health": payload["health"],
    }
    print(render_key_value_table(status_rows))


def _cmd_docker_logs(args):
    args, app_envs = _resolve_existing_instance_args(args)
    result = run_compose(args, "logs", app_envs)
    _report_compose_result(result)


def _cmd_docker_config(args):
    app_envs = resolve_runtime_envs_from_args(args)
    result = run_compose(args, "config", app_envs)
    _report_compose_result(result)


def _cmd_docker_ps(args):
    containers = list_bili_search_containers(
        filters=explicit_runtime_filters_from_args(args),
        include_all=bool(getattr(args, "all", False)),
    )
    if not containers:
        logger.mesg("  No bili-search docker containers found")
        return

    if _json_output(args):
        _print_json_payload(containers)
        return

    rows = [
        [
            item.get("ports") or item["port"] or "-",
            item["status"],
            item.get("runtime") or "docker",
            item.get("source") or "workspace",
            item["started_at"] or "-",
            item.get("uptime") or "-",
            item["llm_config"] or "-",
            item["name"],
        ]
        for item in containers
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


def _list_local_ps_entries(args):
    return list_managed_service_instances(
        filters=explicit_runtime_filters_from_args(args),
        include_all=bool(getattr(args, "all", False)),
    )


def _list_docker_ps_entries(args):
    return list_bili_search_containers(
        filters=explicit_runtime_filters_from_args(args),
        include_all=bool(getattr(args, "all", False)),
    )


def _sorted_ps_entries(entries: list[dict]) -> list[dict]:
    runtime_rank = {"local": 0, "docker": 1}
    return sorted(
        entries,
        key=lambda item: (
            item.get("port") is None,
            item.get("port") or 0,
            runtime_rank.get(str(item.get("runtime") or ""), 99),
            str(item.get("name") or ""),
        ),
    )


def _render_ps_entries(args, entries: list[dict]):
    if not entries:
        logger.mesg("  No bili-search runtime instances found")
        return
    if _json_output(args):
        _print_json_payload(entries)
        return

    rows = [
        [
            item.get("ports") or item.get("port") or "-",
            item.get("status") or "-",
            item.get("runtime") or "-",
            item.get("source") or "-",
            item.get("started_at") or "-",
            item.get("uptime") or "-",
            item.get("llm_config") or "-",
            item.get("name") or "-",
        ]
        for item in entries
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


def cmd_start(args):
    if _runtime(args) == "local":
        return local_runtime_cli.cmd_start(args)
    _validate_local_only_flags(args, "foreground", "reload", "kill")
    return _cmd_docker_start(args)


def cmd_build(args):
    _ensure_docker_runtime(args, "build")
    return _cmd_docker_build(args)


def cmd_build_base(args):
    _ensure_docker_runtime(args, "build-base")
    return _cmd_docker_build_base(args)


def cmd_stop(args):
    if _runtime(args) == "local":
        return local_runtime_cli.cmd_stop(args)
    return _cmd_docker_stop(args)


def cmd_down(args):
    _ensure_docker_runtime(args, "down")
    return _cmd_docker_down(args)


def cmd_restart(args):
    if _runtime(args) == "local":
        return local_runtime_cli.cmd_restart(args)
    _validate_local_only_flags(args, "foreground")
    return _cmd_docker_restart(args)


def cmd_status(args):
    if _runtime(args) == "local":
        return local_runtime_cli.cmd_status(args)
    return _cmd_docker_status(args)


def cmd_logs(args):
    if _runtime(args) == "local":
        return local_runtime_cli.cmd_logs(args)
    return _cmd_docker_logs(args)


def cmd_check(args):
    return local_runtime_cli.cmd_check(args)


def cmd_config(args):
    _ensure_docker_runtime(args, "config")
    return _cmd_docker_config(args)


def cmd_ps(args):
    runtime = _runtime(args)
    if runtime == "local":
        return _render_ps_entries(
            args, _sorted_ps_entries(_list_local_ps_entries(args))
        )
    if runtime == "docker":
        return _render_ps_entries(
            args, _sorted_ps_entries(_list_docker_ps_entries(args))
        )
    local_entries = _list_local_ps_entries(args)
    docker_entries = _list_docker_ps_entries(args)
    return _render_ps_entries(
        args, _sorted_ps_entries([*local_entries, *docker_entries])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bili Search runtime CLI (bsdk)",
        epilog=root_epilog(
            quick_start=[
                "bsdk start --runtime local --foreground -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt",
                "bsdk start -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt",
            ],
            examples=[
                "bsdk start --runtime local -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt",
                "bsdk restart -p 21001 --restart-scope app",
                "bsdk restart -p 21001 --restart-scope container --no-sync-code",
                "bsdk start --source local-git --git-ref HEAD~1 -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt",
                "bsdk build -p 21001 --pip-index-url https://mirrors.ustc.edu.cn/pypi/simple --pip-extra-index-url https://pypi.org/simple",
                "bsdk status -p 21001",
                "bsdk status --runtime local -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt",
                "bsdk ps --all",
                "bsdk ps --runtime local --all",
                "bsdk config --source remote-git --git-url https://github.com/example/bili-search.git --git-ref main",
            ],
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_runtime_mode_arg(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_base_parser = subparsers.add_parser(
        "build-base", help="Build reusable Ubuntu base image"
    )
    add_runtime_mode_arg(build_base_parser)
    add_shared_runtime_args(build_base_parser)
    add_docker_args(build_base_parser)
    build_base_parser.set_defaults(func=cmd_build_base)

    build_parser = subparsers.add_parser(
        "build", help="Build dockerized service image without starting it"
    )
    add_runtime_mode_arg(build_parser)
    add_shared_runtime_args(build_parser)
    add_docker_args(build_parser)
    build_parser.add_argument(
        "--no-base-build",
        action="store_true",
        default=False,
        help="Skip building the reusable Ubuntu base image before docker compose build",
    )
    build_parser.set_defaults(func=cmd_build)

    start_parser = subparsers.add_parser(
        "start", help="Start bili-search in local or docker runtime"
    )
    add_runtime_mode_arg(start_parser)
    add_shared_runtime_args(
        start_parser,
        include_foreground=True,
        include_reload=True,
        include_kill=True,
    )
    add_docker_args(start_parser)
    start_parser.add_argument(
        "--no-build",
        action="store_true",
        default=False,
        help="Skip docker compose --build during start",
    )
    start_parser.add_argument(
        "--no-base-build",
        action="store_true",
        default=False,
        help="Skip building the reusable Ubuntu base image before compose up",
    )
    start_parser.set_defaults(func=cmd_start)

    stop_parser = subparsers.add_parser("stop", help="Stop dockerized service")
    add_runtime_mode_arg(stop_parser)
    add_shared_runtime_args(stop_parser)
    add_docker_args(stop_parser)
    stop_parser.set_defaults(func=cmd_stop)

    down_parser = subparsers.add_parser(
        "down", help="Remove dockerized service containers"
    )
    add_runtime_mode_arg(down_parser)
    add_shared_runtime_args(down_parser)
    add_docker_args(down_parser)
    down_parser.set_defaults(func=cmd_down)

    restart_parser = subparsers.add_parser("restart", help="Restart dockerized service")
    add_runtime_mode_arg(restart_parser)
    add_shared_runtime_args(restart_parser, include_foreground=True)
    add_docker_args(restart_parser)
    add_restart_control_args(restart_parser)
    restart_parser.add_argument(
        "--no-base-build",
        action="store_true",
        default=False,
        help="Skip building the reusable Ubuntu base image before a synced container recreate",
    )
    restart_parser.set_defaults(func=cmd_restart)

    status_parser = subparsers.add_parser(
        "status", help="Show local or docker runtime status"
    )
    add_runtime_mode_arg(status_parser)
    add_shared_runtime_args(status_parser)
    add_docker_args(status_parser)
    status_parser.add_argument(
        "--timeout", type=float, default=3.0, help="Health check timeout"
    )
    status_parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Render status output as table or JSON",
    )
    status_parser.set_defaults(func=cmd_status)

    logs_parser = subparsers.add_parser("logs", help="Show local or docker logs")
    add_runtime_mode_arg(logs_parser)
    add_shared_runtime_args(logs_parser)
    add_docker_args(logs_parser)
    logs_parser.add_argument(
        "-n", "--lines", type=int, default=50, help="Number of lines to show"
    )
    logs_parser.add_argument(
        "-f", "--follow", action="store_true", default=False, help="Follow logs"
    )
    logs_parser.set_defaults(func=cmd_logs)

    check_parser = subparsers.add_parser("check", help="Call the health endpoint")
    add_runtime_mode_arg(check_parser)
    add_shared_runtime_args(check_parser)
    check_parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Health check timeout in seconds",
    )
    check_parser.set_defaults(func=cmd_check)

    config_parser = subparsers.add_parser("config", help="Render docker compose config")
    add_runtime_mode_arg(config_parser)
    add_shared_runtime_args(config_parser)
    add_docker_args(config_parser)
    config_parser.set_defaults(func=cmd_config)

    ps_parser = subparsers.add_parser(
        "ps",
        aliases=["list"],
        help="List local services or docker containers with runtime metadata",
    )
    add_runtime_mode_arg(ps_parser, default="all", allow_all=True)
    add_shared_runtime_args(ps_parser)
    ps_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Include stopped/exited containers",
    )
    ps_parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Render process list as table or JSON",
    )
    ps_parser.set_defaults(func=cmd_ps)
    return parser


def main(argv: list[str] | None = None):
    ensure_process_timezone()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
