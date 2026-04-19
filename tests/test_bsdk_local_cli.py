"""Tests for bsdk local runtime helpers."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from tclogger import decolored


def make_runtime_args(**overrides):
    values = {
        "runtime": "local",
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


def test_local_start_passes_runtime_env_to_service():
    manager = MagicMock()
    manager.status.return_value = {"status": "not_running", "pid": None}
    manager.start.return_value = {"status": "started", "pid": 4321}
    manager.log_file = Path("/tmp/server.dev.p21099.log")
    with (
        patch(
            "cli.local_runtime._build_service_manager",
            return_value=manager,
        ),
        patch("cli.local_runtime.sync_service_file_aliases") as mock_sync,
    ):
        from cli.local_runtime import cmd_start

        args = make_runtime_args()
        args.foreground = False
        args.reload = False
        args.kill = False
        cmd_start(args)

    kwargs = manager.start.call_args.kwargs
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 21099
    assert kwargs["extra_env"]["BILI_SEARCH_APP_PORT"] == "21099"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_ELASTIC_INDEX"] == "bili_videos_dev6"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_ELASTIC_ENV_NAME"] == "elastic_dev"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_LLM_CONFIG"] == "deepseek"
    mock_sync.assert_called_once()
    synced_envs = mock_sync.call_args.args[0]
    assert synced_envs["host"] == "0.0.0.0"
    assert synced_envs["port"] == 21099
    assert synced_envs["elastic_index"] == "bili_videos_dev6"
    assert synced_envs["elastic_env_name"] == "elastic_dev"
    assert synced_envs["llm_config"] == "deepseek"


def test_local_start_inherits_runtime_defaults_from_config():
    manager = MagicMock()
    manager.start.return_value = {"status": "started", "pid": 4321}
    manager.log_file = Path("/tmp/server.dev.p21001.log")
    with (
        patch(
            "cli.local_runtime._build_service_manager",
            return_value=manager,
        ),
        patch(
            "cli.local_runtime.build_local_service_snapshot",
            return_value={
                "runtime_state": "stopped",
                "worker_status": "stopped",
                "pid": None,
            },
        ),
        patch("cli.local_runtime.sync_service_file_aliases") as mock_sync,
    ):
        from cli.local_runtime import cmd_start

        args = make_runtime_args(
            port=None,
            elastic_index=None,
            elastic_env_name=None,
            llm_config=None,
        )
        args.foreground = False
        args.reload = False
        args.kill = False
        cmd_start(args)

    kwargs = manager.start.call_args.kwargs
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 21001
    assert kwargs["extra_env"]["BILI_SEARCH_APP_PORT"] == "21001"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_ELASTIC_INDEX"] == "bili_videos_dev6"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_ELASTIC_ENV_NAME"] == "elastic_dev"
    assert kwargs["extra_env"]["BILI_SEARCH_APP_LLM_CONFIG"] == "deepseek"
    mock_sync.assert_called_once()
    synced_envs = mock_sync.call_args.args[0]
    assert synced_envs["port"] == 21001
    assert synced_envs["elastic_index"] == "bili_videos_dev6"
    assert synced_envs["elastic_env_name"] == "elastic_dev"
    assert synced_envs["llm_config"] == "deepseek"


def test_local_status_cleans_stale_pid():
    manager = MagicMock()
    manager.status.return_value = {"status": "dead", "pid": 7788}
    manager.log_file = Path("/tmp/server.dev.p21099.log")
    with (
        patch(
            "cli.local_runtime._build_service_manager",
            return_value=manager,
        ),
        patch("cli.local_runtime._sanitize_log_file") as mock_sanitize,
    ):
        from cli.local_runtime import cmd_status

        args = make_runtime_args(port=21001, elastic_index=None, elastic_env_name=None)
        cmd_status(args)

    mock_sanitize.assert_called_once_with(manager.log_file)


def test_local_status_uses_runtime_specific_service_manager():
    manager = MagicMock()
    manager.status.return_value = {"status": "not_running", "pid": None}
    manager.log_file = Path("/tmp/server.dev.p21001.log")

    with (
        patch(
            "cli.local_runtime._build_service_manager",
            return_value=manager,
        ),
        patch(
            "cli.local_runtime.build_local_service_snapshot",
            return_value={
                "status": "exited",
                "runtime": "local",
                "source": "workspace",
                "manager_status": "not_running",
                "worker_status": "stopped",
                "manager_pid": None,
                "worker_pid": None,
                "started_at": None,
                "uptime": None,
                "url": "http://127.0.0.1:21001/health",
                "log_file": "/tmp/server.dev.p21001.log",
                "health": None,
                "health_error": None,
                "reason": None,
            },
        ) as mock_build_snapshot,
    ):
        from cli.local_runtime import cmd_status

        args = make_runtime_args(port=21001, elastic_index=None, elastic_env_name=None)
        cmd_status(args)

    mock_build_snapshot.assert_called_once()


def test_local_logs_reads_from_service_manager_and_decolors_output():
    manager = MagicMock()
    manager.log_file = MagicMock()
    manager.log_file.exists.return_value = True
    manager.read_logs.return_value = "\x1b[95mline1\x1b[0m\nline2\n"

    with (
        patch(
            "cli.local_runtime._build_service_manager",
            return_value=manager,
        ),
        patch("cli.local_runtime._sanitize_log_file") as mock_sanitize,
        patch("builtins.print") as mock_print,
    ):
        from cli.local_runtime import cmd_logs

        args = make_runtime_args()
        cmd_logs(args)

    mock_sanitize.assert_called_once_with(manager.log_file)
    manager.read_logs.assert_called_once_with(lines=20)
    mock_print.assert_called_once_with("line1\nline2\n", end="")


def test_local_restart_uses_runtime_specific_service_manager():
    manager = MagicMock()
    manager.stop.return_value = {"status": "stopped", "pid": 7788}
    manager.start.return_value = {"status": "started", "pid": 8899}
    manager.log_file = Path("/tmp/server.dev.p21099.log")

    with (
        patch(
            "cli.local_runtime._build_service_manager",
            return_value=manager,
        ),
        patch(
            "cli.local_runtime.build_local_service_snapshot",
            return_value={
                "runtime_state": "running",
                "worker_status": "running",
            },
        ),
        patch("cli.local_runtime.sync_service_file_aliases") as mock_sync,
    ):
        from cli.local_runtime import cmd_restart

        args = make_runtime_args()
        args.foreground = False
        cmd_restart(args)

    manager.stop.assert_called_once_with()
    manager.start.assert_called_once()
    mock_sync.assert_called_once()
    synced_envs = mock_sync.call_args.args[0]
    assert synced_envs["host"] == "0.0.0.0"
    assert synced_envs["port"] == 21099
    assert synced_envs["elastic_index"] == "bili_videos_dev6"
    assert synced_envs["elastic_env_name"] == "elastic_dev"
    assert synced_envs["llm_config"] == "deepseek"


def test_list_managed_service_instances_reads_pid_files(tmp_path):
    pid_file = (
        tmp_path / "server.p21001.ei-bili-videos-dev6.ev-elastic-dev.lc-deepseek.pid"
    )
    pid_file.write_text("4321\n", encoding="utf-8")

    with (
        patch("service.runtime.DATA_DIR", tmp_path),
        patch("service.runtime.process_is_running", return_value=True),
        patch(
            "service.runtime.process_started_at",
            return_value="2026-03-18 09:02:03",
        ),
    ):
        from service.runtime import list_managed_service_instances

        instances = list_managed_service_instances()

    assert len(instances) == 1
    assert instances[0]["port"] == 21001
    assert instances[0]["pid"] == 4321
    assert instances[0]["started_at"] == "2026-03-18 09:02:03"


def test_local_ps_prints_service_table():
    with (
        patch(
            "cli.local_runtime.list_managed_service_instances",
            return_value=[
                {
                    "port": 21001,
                    "status": "running",
                    "runtime": "local",
                    "source": "workspace",
                    "started_at": "2026-03-18 09:02:03",
                    "uptime": "2h 3min 4s",
                    "llm_config": "deepseek",
                    "pid": 4321,
                    "name": "bili-search-p21001",
                }
            ],
        ) as mock_list,
        patch("builtins.print") as mock_print,
    ):
        from cli.local_runtime import cmd_ps

        args = make_runtime_args(port=21001, llm_config="deepseek")
        args.all = False
        cmd_ps(args)

    mock_list.assert_called_once_with(
        filters={
            "port": 21001,
            "elastic_index": "bili_videos_dev6",
            "elastic_env_name": "elastic_dev",
            "llm_config": "deepseek",
        },
        include_all=False,
    )
    printed = decolored(mock_print.call_args.args[0])
    assert "PORT" in printed
    assert "local" in printed
    assert "workspace" in printed
    assert "2026-03-18 09:02:03" in printed


def test_local_status_prints_key_value_table_for_running_service():
    manager = MagicMock()
    manager.status.return_value = {"status": "running", "pid": 7788}
    manager.log_file = Path("/tmp/server.dev.p21099.log")

    with (
        patch("cli.local_runtime._build_service_manager", return_value=manager),
        patch("cli.local_runtime._sanitize_log_file"),
        patch(
            "cli.local_runtime.build_local_service_snapshot",
            return_value={
                "status": "running",
                "runtime": "local",
                "source": "workspace",
                "manager_status": "running",
                "worker_status": "running",
                "manager_pid": 7788,
                "worker_pid": 7788,
                "started_at": "2026-03-18 09:02:03",
                "uptime": "12s",
                "url": "http://127.0.0.1:21099/health",
                "log_file": "/tmp/server.dev.p21099.log",
                "health": {"status": "ok"},
                "health_error": None,
                "reason": None,
            },
        ),
        patch("builtins.print") as mock_print,
    ):
        from cli.local_runtime import cmd_status

        cmd_status(make_runtime_args())

    printed = decolored(mock_print.call_args.args[0])
    assert "Field" in printed
    assert "Value" in printed
    assert "Status" in printed
    assert "running" in printed
    assert "7788" in printed
    assert '{"status": "ok"}' in printed
