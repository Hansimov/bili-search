"""Tests for cli.bsdk docker CLI helpers."""

import json

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch


def make_runtime_args(**overrides):
    values = {
        "mode": "dev",
        "host": None,
        "port": 21031,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "gpt",
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
        "pip_trusted_host": None,
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
        "mode": "dev",
        "port": 21031,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "gpt",
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
    from docker.manager import default_image_name
    from docker.manager import default_project_name, resolve_compose_settings

    app_envs = {
        "mode": "dev",
        "port": 21031,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "gpt",
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


def test_list_bili_search_containers_reads_runtime_metadata():
    ps_result = MagicMock(returncode=0, stdout="abc123\n")
    inspect_payload = [
        {
            "Id": "abc123def4567890",
            "Name": "/bili-search-dev-p21031",
            "Config": {
                "Image": "bili-search:dev-p21031",
                "Labels": {
                    "com.docker.compose.project": "bili-search-dev-p21031",
                    "com.docker.compose.service": "bili-search",
                },
                "Env": [
                    "BILI_SEARCH_APP_MODE=dev",
                    "BILI_SEARCH_APP_PORT=21031",
                    "BILI_SEARCH_APP_ELASTIC_INDEX=bili_videos_dev6",
                    "BILI_SEARCH_APP_ELASTIC_ENV_NAME=elastic_dev",
                    "BILI_SEARCH_APP_LLM_CONFIG=gpt",
                ],
            },
            "State": {
                "Status": "running",
                "Running": True,
                "StartedAt": "2026-03-18T01:02:03.123456789Z",
            },
            "HostConfig": {"NetworkMode": "host"},
        }
    ]
    inspect_result = MagicMock(returncode=0, stdout=json.dumps(inspect_payload))

    with patch(
        "docker.manager.subprocess.run",
        side_effect=[ps_result, inspect_result],
    ):
        from docker.manager import list_bili_search_containers

        containers = list_bili_search_containers(filters={"mode": "dev"})

    assert len(containers) == 1
    assert containers[0]["port"] == 21031
    assert containers[0]["mode"] == "dev"
    assert containers[0]["started_at"] == "2026-03-18 09:02:03"


def test_bsdk_ps_prints_container_table():
    with (
        patch(
            "cli.bsdk.list_bili_search_containers",
            return_value=[
                {
                    "port": 21031,
                    "status": "running",
                    "started_at": "2026-03-18 01:02:03 UTC",
                    "mode": "dev",
                    "llm_config": "gpt",
                    "name": "bili-search-dev-p21031",
                }
            ],
        ) as mock_list,
        patch("builtins.print") as mock_print,
    ):
        from cli.bsdk import cmd_ps

        cmd_ps(make_runtime_args())

    mock_list.assert_called_once_with(
        filters={
            "mode": "dev",
            "port": 21031,
            "elastic_index": "bili_videos_dev6",
            "elastic_env_name": "elastic_dev",
            "llm_config": "gpt",
        },
        include_all=False,
    )
    printed = mock_print.call_args.args[0]
    assert "PORT" in printed
    assert "21031" in printed
    assert "bili-search-dev-p21031" in printed
