from __future__ import annotations

import argparse

from pathlib import Path
from tclogger import logger
from webu.clis.helpers import root_epilog

from cli.common import add_shared_runtime_args, explicit_runtime_filters_from_args
from cli.common import format_table, resolve_runtime_envs_from_args
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
        help="Pip index URL used by docker builds",
    )
    parser.add_argument(
        "--pip-trusted-host",
        type=str,
        default=None,
        help="Trusted host paired with --pip-index-url",
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


def cmd_start(args):
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


def cmd_build(args):
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


def cmd_build_base(args):
    app_envs = resolve_runtime_envs_from_args(args)
    logger.note(
        "> Building reusable Ubuntu base image for bili-search docker runtime ..."
    )
    result = run_base_build(args, app_envs)
    _report_compose_result(result)


def cmd_stop(args):
    args, app_envs = _resolve_existing_instance_args(args)
    logger.note("> Stopping dockerized search service ...")
    result = run_compose(args, "stop", app_envs)
    _report_compose_result(result)


def cmd_down(args):
    args, app_envs = _resolve_existing_instance_args(args)
    logger.note("> Removing dockerized search service containers ...")
    result = run_compose(args, "down", app_envs)
    _report_compose_result(result)


def cmd_restart(args):
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


def cmd_status(args):
    args, app_envs = _resolve_existing_instance_args(args)
    result = run_compose(args, "status", app_envs)
    _report_compose_result(result)
    container = find_bili_search_container_by_port(app_envs["port"], include_all=True)
    if container.get("network_mode") == "host":
        logger.mesg(
            f"  App Port: {app_envs['port']} (host network mode; docker PORTS column stays empty)"
        )
    app_state = read_container_app_state(args, app_envs)
    if app_state:
        logger.mesg(f"  App Started At: {app_state['started_at'] or '-'}")
        logger.mesg(f"  App Uptime: {app_state['uptime'] or '-'}")
        if app_state.get("restart_count") is not None:
            logger.mesg(f"  App Restart Count: {app_state['restart_count']}")
    try:
        health = fetch_health(app_envs, timeout=args.timeout)
        logger.mesg(f"  Health URL: {health_url(app_envs)}")
        logger.mesg(f"  Health: {health}")
    except Exception as exc:
        logger.warn(f"  × Health check failed: {exc}")


def cmd_logs(args):
    args, app_envs = _resolve_existing_instance_args(args)
    result = run_compose(args, "logs", app_envs)
    _report_compose_result(result)


def cmd_config(args):
    app_envs = resolve_runtime_envs_from_args(args)
    result = run_compose(args, "config", app_envs)
    _report_compose_result(result)


def cmd_ps(args):
    containers = list_bili_search_containers(
        filters=explicit_runtime_filters_from_args(args),
        include_all=bool(getattr(args, "all", False)),
    )
    if not containers:
        logger.mesg("  No bili-search docker containers found")
        return

    rows = [
        [
            item["port"] or "-",
            item["status"],
            item["started_at"] or "-",
            item.get("uptime") or "-",
            item["llm_config"] or "-",
            item["name"],
        ]
        for item in containers
    ]
    print(
        format_table(
            ["PORT", "STATUS", "STARTED_AT", "UPTIME", "LLM", "NAME"],
            rows,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bili Search docker CLI (bsdk)",
        epilog=root_epilog(
            quick_start=[
                "bsdk build-base",
                "bsdk start -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt",
            ],
            examples=[
                "bsdk start --source local-git --git-ref HEAD~1 -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt",
                "bsdk status -p 21001",
                "bsdk ps --all",
                "bsdk config --source remote-git --git-url https://github.com/example/bili-search.git --git-ref main",
            ],
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_base_parser = subparsers.add_parser(
        "build-base", help="Build reusable Ubuntu base image"
    )
    add_shared_runtime_args(build_base_parser)
    add_docker_args(build_base_parser)
    build_base_parser.set_defaults(func=cmd_build_base)

    build_parser = subparsers.add_parser(
        "build", help="Build dockerized service image without starting it"
    )
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
        "start", help="Build and start dockerized service"
    )
    add_shared_runtime_args(start_parser)
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
    add_shared_runtime_args(stop_parser)
    add_docker_args(stop_parser)
    stop_parser.set_defaults(func=cmd_stop)

    down_parser = subparsers.add_parser(
        "down", help="Remove dockerized service containers"
    )
    add_shared_runtime_args(down_parser)
    add_docker_args(down_parser)
    down_parser.set_defaults(func=cmd_down)

    restart_parser = subparsers.add_parser("restart", help="Restart dockerized service")
    add_shared_runtime_args(restart_parser)
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
        "status", help="Show docker and service health status"
    )
    add_shared_runtime_args(status_parser)
    add_docker_args(status_parser)
    status_parser.add_argument(
        "--timeout", type=float, default=3.0, help="Health check timeout"
    )
    status_parser.set_defaults(func=cmd_status)

    logs_parser = subparsers.add_parser("logs", help="Show container logs")
    add_shared_runtime_args(logs_parser)
    add_docker_args(logs_parser)
    logs_parser.add_argument(
        "-n", "--lines", type=int, default=50, help="Number of lines to show"
    )
    logs_parser.add_argument(
        "-f", "--follow", action="store_true", default=False, help="Follow logs"
    )
    logs_parser.set_defaults(func=cmd_logs)

    config_parser = subparsers.add_parser("config", help="Render docker compose config")
    add_shared_runtime_args(config_parser)
    add_docker_args(config_parser)
    config_parser.set_defaults(func=cmd_config)

    ps_parser = subparsers.add_parser(
        "ps",
        aliases=["list"],
        help="List bili-search docker containers with port and startup time",
    )
    add_shared_runtime_args(ps_parser)
    ps_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Include stopped/exited containers",
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
