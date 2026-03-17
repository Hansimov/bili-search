"""Tests for apps.search_app_cli service management helpers."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch


def make_runtime_args(**overrides):
    values = {
        "mode": "dev",
        "host": None,
        "port": 21099,
        "elastic_index": "bili_videos_dev6",
        "elastic_env_name": "elastic_dev",
        "llm_config": "deepseek",
        "timeout": 1.0,
        "lines": 20,
        "follow": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_start_passes_runtime_env_to_service():
    manager = MagicMock()
    manager.status.return_value = {"status": "not_running", "pid": None}
    manager.start.return_value = {"status": "started", "pid": 4321}
    manager.log_file = Path("/tmp/server.dev.p21099.log")
    with patch(
        "apps.search_app_cli._build_service_manager",
        return_value=manager,
    ):
        from apps.search_app_cli import cmd_start

        args = make_runtime_args()
        cmd_start(args)

    kwargs = manager.start.call_args.kwargs
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 21099
    assert kwargs["extra_env"]["BILI_SEARCH_APP_MODE"] == "dev"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_PORT"] == "21099"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_ELASTIC_INDEX"] == "bili_videos_dev6"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_ELASTIC_ENV_NAME"] == "elastic_dev"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_LLM_CONFIG"] == "deepseek"


def test_status_cleans_stale_pid():
    manager = MagicMock()
    manager.status.return_value = {"status": "dead", "pid": 7788}
    manager.log_file = Path("/tmp/server.dev.p21099.log")
    with patch(
        "apps.search_app_cli._build_service_manager",
        return_value=manager,
    ), patch("apps.search_app_cli._sanitize_log_file") as mock_sanitize:
        from apps.search_app_cli import cmd_status

        args = make_runtime_args(port=21001, elastic_index=None, elastic_env_name=None)
        cmd_status(args)

    mock_sanitize.assert_called_once_with(manager.log_file)


def test_status_uses_runtime_specific_service_manager():
    manager = MagicMock()
    manager.status.return_value = {"status": "not_running", "pid": None}
    manager.log_file = Path("/tmp/server.dev.p21001.log")

    with patch(
        "apps.search_app_cli._build_service_manager",
        return_value=manager,
    ):
        from apps.search_app_cli import cmd_status

        args = make_runtime_args(port=21001, elastic_index=None, elastic_env_name=None)
        cmd_status(args)

    manager.status.assert_called_once_with()


def test_logs_reads_from_service_manager_and_decolors_output():
    manager = MagicMock()
    manager.log_file = MagicMock()
    manager.log_file.exists.return_value = True
    manager.read_logs.return_value = "\x1b[95mline1\x1b[0m\nline2\n"

    with patch(
        "apps.search_app_cli._build_service_manager",
        return_value=manager,
    ), patch("apps.search_app_cli._sanitize_log_file") as mock_sanitize, patch(
        "builtins.print"
    ) as mock_print:
        from apps.search_app_cli import cmd_logs

        args = make_runtime_args()
        cmd_logs(args)

    mock_sanitize.assert_called_once_with(manager.log_file)
    manager.read_logs.assert_called_once_with(lines=20)
    mock_print.assert_called_once_with("line1\nline2\n", end="")


def test_restart_uses_runtime_specific_service_manager():
    manager = MagicMock()
    manager.stop.return_value = {"status": "stopped", "pid": 7788}
    manager.start.return_value = {"status": "started", "pid": 8899}
    manager.log_file = Path("/tmp/server.dev.p21099.log")

    with patch(
        "apps.search_app_cli._build_service_manager",
        return_value=manager,
    ):
        from apps.search_app_cli import cmd_restart

        cmd_restart(make_runtime_args())

    manager.stop.assert_called_once_with()
    manager.start.assert_called_once()
