"""Tests for cli.bsdk docker CLI helpers."""

import json

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from tclogger import decolored


def make_runtime_args(**overrides):
    values = {
        "runtime": "docker",
        "host": None,
        "port": 21001,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "deepseek",
        "compose_file": "docker/docker-compose.yml",
        "env_file": "docker/.env",
        "dockerfile": "docker/Dockerfile",
        "base_dockerfile": "docker/Dockerfile.base",
        "configs_dir": "configs",
        "logs_dir": "logs",
        "project_name": None,
        "service_name": None,
        "container_name": None,
        "image": None,
        "base_image": None,
        "pip_index_url": None,
        "pip_extra_index_url": None,
        "pip_trusted_host": None,
        "pip_retries": None,
        "pip_timeout": None,
        "sedb_context": None,
        "tclogger_context": None,
        "webu_context": None,
        "source": "workspace",
        "git_repo": None,
        "git_ref": None,
        "git_url": None,
        "no_build": False,
        "no_base_build": False,
        "follow": False,
        "lines": 50,
        "timeout": 3.0,
        "restart_scope": "app",
        "sync_code": True,
        "foreground": False,
        "reload": False,
        "kill": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_bsdk_start_uses_compose_runner():
    base_result = MagicMock(stdout="base\n", stderr="", returncode=0)
    result = MagicMock(stdout="ok\n", stderr="", returncode=0)
    with (
        patch("cli.bsdk.ensure_base_image", return_value=base_result) as mock_base,
        patch("cli.bsdk.run_compose", return_value=result) as mock_run,
        patch("cli.bsdk._report_compose_result") as mock_report,
    ):
        from cli.bsdk import cmd_start

        cmd_start(make_runtime_args())

    mock_base.assert_called_once()
    mock_run.assert_called_once()
    assert mock_run.call_args.args[1] == "start"
    assert mock_report.call_args_list[0].args[0] is base_result
    assert mock_report.call_args_list[1].args[0] is result


def test_bsdk_start_dispatches_to_local_runtime():
    with patch("cli.bsdk.local_runtime_cli.cmd_start") as mock_local_start:
        from cli.bsdk import cmd_start

        cmd_start(make_runtime_args(runtime="local"))

    mock_local_start.assert_called_once()


def test_bsdk_start_can_skip_base_build():
    result = MagicMock(stdout="ok\n", stderr="", returncode=0)
    with (
        patch("cli.bsdk.ensure_base_image") as mock_base,
        patch("cli.bsdk.run_compose", return_value=result) as mock_run,
        patch("cli.bsdk._report_compose_result"),
    ):
        from cli.bsdk import cmd_start

        cmd_start(make_runtime_args(no_base_build=True))

    mock_base.assert_not_called()
    mock_run.assert_called_once()


def test_bsdk_build_uses_compose_build_runner():
    base_result = MagicMock(stdout="base\n", stderr="", returncode=0)
    result = MagicMock(stdout="build\n", stderr="", returncode=0)
    with (
        patch("cli.bsdk.ensure_base_image", return_value=base_result) as mock_base,
        patch("cli.bsdk.run_compose", return_value=result) as mock_run,
        patch("cli.bsdk._report_compose_result") as mock_report,
    ):
        from cli.bsdk import cmd_build

        cmd_build(make_runtime_args())

    mock_base.assert_called_once()
    mock_run.assert_called_once()
    assert mock_run.call_args.args[1] == "build"
    assert mock_report.call_args_list[0].args[0] is base_result
    assert mock_report.call_args_list[1].args[0] is result


def test_ensure_base_image_skips_existing_image():
    from docker.manager import ensure_base_image

    app_envs = {
        "port": 21001,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "deepseek",
    }

    with (
        patch("docker.manager.docker_image_exists", return_value=True) as mock_exists,
        patch("docker.manager.run_base_build") as mock_build,
    ):
        result = ensure_base_image(make_runtime_args(), app_envs)

    assert result is None
    mock_exists.assert_called_once()
    mock_build.assert_not_called()


def test_materialize_source_returns_workspace_for_default_source():
    from docker.manager import WORKSPACE_ROOT, materialize_source

    assert materialize_source(make_runtime_args()) == WORKSPACE_ROOT


def test_resolve_compose_settings_uses_defaults():
    from docker.manager import default_base_image_name, default_container_name
    from docker.manager import DEFAULT_PIP_EXTRA_INDEX_URL, DEFAULT_PIP_INDEX_URL
    from docker.manager import DEFAULT_PIP_RETRIES, DEFAULT_PIP_TIMEOUT
    from docker.manager import default_image_name
    from docker.manager import default_project_name, resolve_compose_settings

    app_envs = {
        "port": 21001,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "deepseek",
    }
    settings = resolve_compose_settings(
        make_runtime_args(),
        app_envs,
        build_context=Path("/tmp/build-context"),
    )

    assert settings["project_name"] == default_project_name(app_envs)
    assert settings["container_name"] == default_container_name(app_envs)
    assert settings["image"] == default_image_name(app_envs)
    assert settings["base_image"] == default_base_image_name()
    assert settings["pip_index_url"] == DEFAULT_PIP_INDEX_URL
    assert settings["pip_extra_index_url"] == DEFAULT_PIP_EXTRA_INDEX_URL
    assert settings["pip_retries"] == DEFAULT_PIP_RETRIES
    assert settings["pip_timeout"] == DEFAULT_PIP_TIMEOUT


def test_sync_source_to_container_cleans_before_extract_and_excludes_mounts():
    app_envs = {
        "port": 21001,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "deepseek",
    }
    cleanup_result = MagicMock(returncode=0, stdout="", stderr="")
    extract_result = MagicMock(returncode=0, stdout="synced\n", stderr="")
    tar_process = MagicMock()
    tar_process.stdout = MagicMock()
    tar_process.stderr.read.return_value = b""
    tar_process.wait.return_value = 0

    with (
        patch(
            "docker.manager.subprocess.run",
            side_effect=[cleanup_result, extract_result],
        ) as mock_run,
        patch(
            "docker.manager.subprocess.Popen", return_value=tar_process
        ) as mock_popen,
    ):
        from docker.manager import sync_source_to_container

        result = sync_source_to_container(make_runtime_args(), app_envs)

    cleanup_command = mock_run.call_args_list[0].args[0]
    assert cleanup_command[:4] == ["docker", "exec", "bili-search-p21001", "sh"]
    assert "! -name configs ! -name logs" in cleanup_command[-1]

    tar_command = mock_popen.call_args.args[0]
    assert "configs" in tar_command
    assert "logs" in tar_command
    assert result.returncode == 0


def test_run_container_app_action_syncs_before_hup():
    app_envs = {
        "port": 21001,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "deepseek",
    }
    sync_result = MagicMock(returncode=0, stdout="synced\n", stderr="")
    signal_result = MagicMock(returncode=0, stdout="reloaded\n", stderr="")
    with (
        patch(
            "docker.manager.sync_source_to_container", return_value=sync_result
        ) as mock_sync,
        patch("docker.manager.subprocess.run", return_value=signal_result) as mock_run,
    ):
        from docker.manager import run_container_app_action

        result = run_container_app_action(
            make_runtime_args(sync_code=True), "restart", app_envs
        )

    mock_sync.assert_called_once()
    assert mock_run.call_args.args[0][:3] == ["docker", "kill", "--signal"]
    assert result.returncode == 0


def test_read_container_app_state_formats_supervisor_metadata(tmp_path: Path):
    from docker.manager import read_container_app_state

    app_envs = {
        "port": 21001,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "deepseek",
    }
    state_dir = tmp_path / "logs" / "search_app"
    state_dir.mkdir(parents=True)
    (state_dir / "container_app_state.p21001.json").write_text(
        json.dumps(
            {
                "pid": 321,
                "started_at": "2026-03-18T01:02:03Z",
                "restart_count": 2,
            }
        ),
        encoding="utf-8",
    )

    args = make_runtime_args(logs_dir=str(tmp_path / "logs"))
    state = read_container_app_state(args, app_envs)

    assert state is not None
    assert state["pid"] == 321
    assert state["started_at"] == "2026-03-18 09:02:03"
    assert state["restart_count"] == 2


def test_read_container_app_state_falls_back_to_live_process(tmp_path: Path):
    from docker.manager import read_container_app_state

    app_envs = {
        "port": 21001,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "deepseek",
    }
    args = make_runtime_args(logs_dir=str(tmp_path / "logs"))
    inspect_result = MagicMock(
        returncode=0,
        stdout='{"pid": 654, "elapsed_seconds": 9}\n',
        stderr="",
    )

    with patch("docker.manager.subprocess.run", return_value=inspect_result):
        state = read_container_app_state(args, app_envs)

    assert state is not None
    assert state["pid"] == 654
    assert state["uptime"] is not None
    assert state["restart_count"] is None


def test_list_bili_search_containers_reads_runtime_metadata():
    ps_result = MagicMock(returncode=0, stdout="abc123\n")
    inspect_payload = [
        {
            "Id": "abc123def4567890",
            "Name": "/bili-search-p21001",
            "Config": {
                "Image": "bili-search:p21001",
                "Labels": {
                    "com.docker.compose.project": "bili-search-p21001",
                    "com.docker.compose.service": "bili-search",
                },
                "Env": [
                    "BILI_SEARCH_APP_PORT=21001",
                    "BILI_SEARCH_APP_ELASTIC_INDEX=bili_videos_dev6",
                    "BILI_SEARCH_APP_ELASTIC_ENV_NAME=elastic_dev",
                    "BILI_SEARCH_APP_LLM_CONFIG=deepseek",
                ],
            },
            "State": {
                "Status": "running",
                "Running": True,
                "StartedAt": "2026-03-18T01:02:03.123456789Z",
            },
            "HostConfig": {"NetworkMode": "host"},
            "NetworkSettings": {"Ports": {}},
        }
    ]
    inspect_result = MagicMock(returncode=0, stdout=json.dumps(inspect_payload))

    with patch(
        "docker.manager.subprocess.run",
        side_effect=[ps_result, inspect_result],
    ):
        from docker.manager import list_bili_search_containers

        containers = list_bili_search_containers(filters={"port": 21001})

    assert len(containers) == 1
    assert containers[0]["port"] == 21001
    assert containers[0]["ports"] == "21001"
    assert containers[0]["started_at"] == "2026-03-18 09:02:03"


def test_format_uptime_omits_zero_units():
    from datetime import datetime, timezone

    from docker.manager import _format_uptime

    started_at = "2026-03-18T01:02:03Z"
    now = datetime(2026, 3, 19, 3, 2, 8, tzinfo=timezone.utc)

    assert _format_uptime(started_at, now=now) == "1d 2h 5s"


def test_find_bili_search_container_by_port_prefers_running_instance():
    from docker.manager import find_bili_search_container_by_port

    with patch(
        "docker.manager.list_bili_search_containers",
        return_value=[
            {"port": 21001, "status": "exited", "name": "old"},
            {"port": 21001, "status": "running/healthy", "name": "live"},
        ],
    ):
        container = find_bili_search_container_by_port(21001)

    assert container["name"] == "live"


def test_bsdk_stop_resolves_running_instance_by_port():
    result = MagicMock(stdout="stopped\n", stderr="", returncode=0)
    with (
        patch(
            "cli.bsdk.find_bili_search_container_by_port",
            return_value={
                "project_name": "bili-search-p21001",
                "service_name": "bili-search",
                "name": "bili-search-p21001",
                "image": "bili-search:p21001",
                "port": 21001,
                "elastic_index": "bili_videos_dev6",
                "elastic_env_name": "elastic_dev",
                "llm_config": "deepseek",
            },
        ) as mock_find,
        patch("cli.bsdk.run_compose", return_value=result) as mock_run,
        patch("cli.bsdk._report_compose_result"),
    ):
        from cli.bsdk import cmd_stop

        cmd_stop(
            make_runtime_args(
                port=21001,
                elastic_index=None,
                elastic_env_name=None,
                llm_config=None,
            )
        )

    mock_find.assert_called_once_with(21001, include_all=True)
    resolved_args = mock_run.call_args.args[0]
    resolved_envs = mock_run.call_args.args[2]
    assert resolved_args.project_name == "bili-search-p21001"
    assert resolved_args.container_name == "bili-search-p21001"
    assert resolved_envs["port"] == 21001


def test_bsdk_restart_defaults_to_app_scope_with_sync():
    from cli.bsdk import build_parser

    parser = build_parser()
    args = parser.parse_args(["restart", "-p", "21001"])

    assert args.runtime == "docker"
    assert args.restart_scope == "app"
    assert args.sync_code is True


def test_bsdk_start_parser_accepts_local_runtime_flags():
    from cli.bsdk import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["start", "--runtime", "local", "--foreground", "--reload", "-k", "-p", "21001"]
    )

    assert args.runtime == "local"
    assert args.foreground is True
    assert args.reload is True
    assert args.kill is True


def test_bsdk_build_rejects_local_runtime():
    from cli.bsdk import cmd_build

    try:
        cmd_build(make_runtime_args(runtime="local"))
    except SystemExit as exc:
        assert str(exc) == "build only supports --runtime docker"
    else:
        assert False, "expected SystemExit"


def test_bsdk_restart_app_scope_uses_container_app_action():
    result = MagicMock(stdout="reloaded\n", stderr="", returncode=0)
    with (
        patch(
            "cli.bsdk.find_bili_search_container_by_port",
            return_value={
                "project_name": "bili-search-p21001",
                "service_name": "bili-search",
                "name": "bili-search-p21001",
                "image": "bili-search:p21001",
                "port": 21001,
                "elastic_index": "bili_videos_dev6",
                "elastic_env_name": "elastic_dev",
                "llm_config": "deepseek",
            },
        ),
        patch(
            "cli.bsdk.run_container_app_action", return_value=result
        ) as mock_app_action,
        patch("cli.bsdk.run_compose") as mock_run_compose,
        patch("cli.bsdk.ensure_base_image") as mock_base,
        patch("cli.bsdk._report_compose_result"),
    ):
        from cli.bsdk import cmd_restart

        cmd_restart(make_runtime_args(restart_scope="app", sync_code=True))

    mock_app_action.assert_called_once()
    mock_run_compose.assert_not_called()
    mock_base.assert_not_called()


def test_bsdk_restart_container_scope_without_sync_uses_compose_restart():
    result = MagicMock(stdout="restarted\n", stderr="", returncode=0)
    with (
        patch(
            "cli.bsdk.find_bili_search_container_by_port",
            return_value={
                "project_name": "bili-search-p21001",
                "service_name": "bili-search",
                "name": "bili-search-p21001",
                "image": "bili-search:p21001",
                "port": 21001,
                "elastic_index": "bili_videos_dev6",
                "elastic_env_name": "elastic_dev",
                "llm_config": "deepseek",
            },
        ),
        patch("cli.bsdk.run_container_app_action") as mock_app_action,
        patch("cli.bsdk.run_compose", return_value=result) as mock_run_compose,
        patch("cli.bsdk.ensure_base_image") as mock_base,
        patch("cli.bsdk._report_compose_result"),
    ):
        from cli.bsdk import cmd_restart

        cmd_restart(
            make_runtime_args(
                restart_scope="container",
                sync_code=False,
            )
        )

    mock_app_action.assert_not_called()
    mock_base.assert_not_called()
    assert mock_run_compose.call_args.args[1] == "restart"


def test_bsdk_restart_container_scope_with_sync_recreates_container():
    base_result = MagicMock(stdout="base\n", stderr="", returncode=0)
    result = MagicMock(stdout="recreated\n", stderr="", returncode=0)
    with (
        patch(
            "cli.bsdk.find_bili_search_container_by_port",
            return_value={
                "project_name": "bili-search-p21001",
                "service_name": "bili-search",
                "name": "bili-search-p21001",
                "image": "bili-search:p21001",
                "port": 21001,
                "elastic_index": "bili_videos_dev6",
                "elastic_env_name": "elastic_dev",
                "llm_config": "deepseek",
            },
        ),
        patch("cli.bsdk.run_container_app_action") as mock_app_action,
        patch("cli.bsdk.ensure_base_image", return_value=base_result) as mock_base,
        patch("cli.bsdk.run_compose", return_value=result) as mock_run_compose,
        patch("cli.bsdk._report_compose_result") as mock_report,
    ):
        from cli.bsdk import cmd_restart

        cmd_restart(
            make_runtime_args(
                restart_scope="container",
                sync_code=True,
            )
        )

    mock_app_action.assert_not_called()
    mock_base.assert_called_once()
    assert mock_run_compose.call_args.args[1] == "recreate"
    assert mock_report.call_args_list[0].args[0] is base_result


def test_run_compose_recreate_uses_force_recreate_build():
    app_envs = {
        "port": 21001,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "deepseek",
    }
    with patch("docker.manager.subprocess.run") as mock_run:
        from docker.manager import run_compose

        run_compose(make_runtime_args(), "recreate", app_envs)

    command = mock_run.call_args.args[0]
    assert command[-5:] == ["up", "-d", "--force-recreate", "--build", "bili-search"]


def test_bsdk_ps_prints_container_table():
    with (
        patch(
            "cli.bsdk.list_bili_search_containers",
            return_value=[
                {
                    "port": 21001,
                    "status": "running",
                    "started_at": "2026-03-18 09:02:03",
                    "uptime": "2h 3min 4s",
                    "llm_config": "deepseek",
                    "name": "bili-search-p21001",
                }
            ],
        ) as mock_list,
        patch("builtins.print") as mock_print,
    ):
        from cli.bsdk import cmd_ps

        cmd_ps(make_runtime_args())

    mock_list.assert_called_once_with(
        filters={
            "port": 21001,
            "elastic_index": "bili_videos_dev6",
            "elastic_env_name": "elastic_dev",
            "llm_config": "deepseek",
        },
        include_all=False,
    )
    printed = mock_print.call_args.args[0]
    printed = decolored(printed)
    assert "PORTS" in printed
    assert "UPTIME" in printed
    assert "21001" in printed
    assert "2h 3min 4s" in printed
    assert "bili-search-p21001" in printed


def test_bsdk_status_reports_app_runtime_metadata():
    health = {"status": "ok"}
    with (
        patch(
            "cli.bsdk.find_bili_search_container_by_port",
            return_value={
                "port": 21001,
                "name": "bili-search-p21001",
                "network_mode": "host",
                "ports": "21001",
                "status": "running/healthy",
                "started_at": "2026-03-18 09:00:00",
                "uptime": "2min 3s",
                "project_name": "bili-search-p21001",
                "service_name": "bili-search",
                "image": "bili-search:p21001",
                "elastic_index": "bili_videos_dev6",
                "elastic_env_name": "elastic_dev",
                "llm_config": "deepseek",
            },
        ),
        patch(
            "cli.bsdk.read_container_app_state",
            return_value={
                "started_at": "2026-03-18 09:02:03",
                "uptime": "12s",
                "restart_count": 3,
            },
        ),
        patch("cli.bsdk.fetch_health", return_value=health),
        patch("builtins.print") as mock_print,
    ):
        from cli.bsdk import cmd_status

        cmd_status(make_runtime_args())

    rendered = mock_print.call_args.args[0]
    printed = decolored(rendered)
    lines = printed.splitlines()
    assert len(lines[0]) == len(lines[-1])
    assert len(lines[1]) == len(lines[2])
    assert "\x1b[95m" in rendered
    assert "\x1b[92m" in rendered
    assert "Ports" in printed
    assert "21001" in printed
    assert "App Restart Count" in printed
    assert "2026-03-18 09:02:03" in printed
    assert '{"status": "ok"}' in printed


def test_bsdk_ps_alternates_column_colors():
    with (
        patch(
            "cli.bsdk.list_bili_search_containers",
            return_value=[
                {
                    "port": 21001,
                    "ports": "21001",
                    "status": "running",
                    "started_at": "2026-03-18 09:02:03",
                    "uptime": "2h 3min 4s",
                    "llm_config": "deepseek",
                    "name": "bili-search-p21001",
                }
            ],
        ),
        patch("builtins.print") as mock_print,
    ):
        from cli.bsdk import cmd_ps

        cmd_ps(make_runtime_args())

    printed = mock_print.call_args.args[0]
    decolored_printed = decolored(printed)
    assert printed != decolored_printed
    assert "\x1b[95m" in printed
    assert "\x1b[92m" in printed
    assert "PORTS" in decolored_printed
    assert "STATUS" in decolored_printed
    assert "NAME" in decolored_printed


def test_bsdk_prune_runs_local_and_docker_cleanup_for_runtime_all():
    with (
        patch("cli.bsdk.local_runtime_cli.cmd_prune") as mock_local_prune,
        patch("cli.bsdk._cmd_docker_prune") as mock_docker_prune,
    ):
        from cli.bsdk import cmd_prune

        cmd_prune(make_runtime_args(runtime="all", port=None))

    mock_local_prune.assert_called_once()
    mock_docker_prune.assert_called_once()
