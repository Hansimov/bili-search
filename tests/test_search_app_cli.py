"""Tests for apps.search_app_cli service management helpers."""

from argparse import Namespace
from unittest.mock import MagicMock, patch


def test_start_passes_runtime_env_to_service():
    with patch(
        "apps.search_app_cli.SERVICE_MANAGER.status",
        return_value={"status": "not_running", "pid": None},
    ), patch(
        "apps.search_app_cli.SERVICE_MANAGER.start",
        return_value={"status": "started", "pid": 4321},
    ) as mock_start:
        from apps.search_app_cli import cmd_start

        args = Namespace(
            mode="dev",
            host=None,
            port=21099,
            elastic_index="bili_videos_dev6",
            elastic_env_name="elastic_dev",
            llm_config="deepseek",
        )
        cmd_start(args)

    kwargs = mock_start.call_args.kwargs
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 21099
    assert kwargs["extra_env"]["BILI_SEARCH_APP_MODE"] == "dev"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_PORT"] == "21099"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_ELASTIC_INDEX"] == "bili_videos_dev6"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_ELASTIC_ENV_NAME"] == "elastic_dev"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_LLM_CONFIG"] == "deepseek"


def test_status_cleans_stale_pid():
    with patch(
        "apps.search_app_cli.SERVICE_MANAGER.status",
        return_value={"status": "dead", "pid": 7788},
    ):
        from apps.search_app_cli import cmd_status

        args = Namespace(
            mode="dev",
            host=None,
            port=None,
            elastic_index=None,
            elastic_env_name=None,
            llm_config=None,
            timeout=1.0,
        )
        cmd_status(args)


def test_logs_reads_from_service_manager():
    with patch(
        "pathlib.Path.exists",
        return_value=True,
    ), patch(
        "apps.search_app_cli.SERVICE_MANAGER.read_logs",
        return_value="line1\nline2\n",
    ) as mock_read_logs, patch("builtins.print") as mock_print:
        from apps.search_app_cli import cmd_logs

        args = Namespace(lines=20, follow=False)
        cmd_logs(args)

    mock_read_logs.assert_called_once_with(lines=20)
    mock_print.assert_called_once_with("line1\nline2\n", end="")
