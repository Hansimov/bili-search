from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from service.runtime import build_local_service_snapshot, list_managed_service_instances


class LocalRuntimeStatusTestCase(unittest.TestCase):
    def test_orphaned_worker_is_reported_as_degraded(self) -> None:
        with TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            pid_file = (
                temp_root
                / "server.p21001.ei-bili-videos-dev6.ev-elastic-dev.lc-deepseek.pid"
            )
            pid_file.write_text("3860406\n", encoding="utf-8")

            with (
                patch("service.runtime.DATA_DIR", temp_root),
                patch("service.runtime.process_is_running", return_value=False),
                patch(
                    "service.runtime.match_service_worker_process",
                    return_value={
                        "pid": 1820502,
                        "ppid": 1,
                        "user": "asimov",
                        "args": (
                            "/home/asimov/miniconda3/envs/ai/bin/python -m uvicorn "
                            "service.uvicorn_factory:create_app_from_env --host 0.0.0.0 "
                            "--port 21001 --factory"
                        ),
                        "started_at": "2026-04-01 18:00:00",
                        "uptime": "3h 12min",
                    },
                ),
                patch(
                    "service.runtime.probe_local_health",
                    return_value={
                        "supported": True,
                        "ok": True,
                        "payload": {"status": "ok"},
                    },
                ),
            ):
                snapshot = build_local_service_snapshot(
                    {
                        "port": 21001,
                        "elastic_index": "bili_videos_dev6",
                        "elastic_env_name": "elastic_dev",
                        "llm_config": "deepseek",
                    }
                )

        self.assertEqual(snapshot["status"], "degraded")
        self.assertEqual(snapshot["runtime_state"], "stopped")
        self.assertEqual(snapshot["raw_state"], "dead")
        self.assertEqual(snapshot["worker_status"], "running")
        self.assertEqual(snapshot["runtime"], "local")
        self.assertEqual(snapshot["source"], "workspace")

    def test_list_instances_keeps_degraded_local_rows_without_all(self) -> None:
        with TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            pid_file = (
                temp_root
                / "server.p21001.ei-bili-videos-dev6.ev-elastic-dev.lc-deepseek.pid"
            )
            pid_file.write_text("3860406\n", encoding="utf-8")

            with (
                patch("service.runtime.DATA_DIR", temp_root),
                patch("service.runtime.process_is_running", return_value=False),
                patch(
                    "service.runtime.match_service_worker_process",
                    return_value={
                        "pid": 1820502,
                        "ppid": 1,
                        "user": "asimov",
                        "args": "python -m uvicorn service.uvicorn_factory:create_app_from_env --port 21001 --factory",
                        "started_at": "2026-04-01 18:00:00",
                        "uptime": "3h 12min",
                    },
                ),
                patch(
                    "service.runtime.probe_local_health",
                    return_value={
                        "supported": True,
                        "ok": True,
                        "payload": {"status": "ok"},
                    },
                ),
            ):
                items = list_managed_service_instances(include_all=False)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["status"], "degraded")
        self.assertEqual(items[0]["name"], "bili-search-p21001")

    def test_list_instances_claims_one_worker_per_port(self) -> None:
        with TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            (
                temp_root
                / "server.p21001.ei-bili-videos-dev6.ev-elastic-dev.lc-deepseek.pid"
            ).write_text(
                "603703\n",
                encoding="utf-8",
            )
            (
                temp_root
                / "server.p21001.ei-bili-videos-dev6.ev-elastic-dev.lc-gpt.pid"
            ).write_text(
                "3860406\n",
                encoding="utf-8",
            )

            with (
                patch("service.runtime.DATA_DIR", temp_root),
                patch(
                    "service.runtime.manager_process_snapshot",
                    side_effect=[
                        {
                            "pid_file": "deepseek.pid",
                            "log_file": "deepseek.log",
                            "pid": 603703,
                            "status": "running",
                            "started_at": "2026-04-01 14:22:06",
                            "uptime": "24min",
                        },
                        {
                            "pid_file": "gpt.pid",
                            "log_file": "gpt.log",
                            "pid": 3860406,
                            "status": "dead",
                            "started_at": None,
                            "uptime": None,
                        },
                    ],
                ),
                patch(
                    "service.runtime.match_service_worker_process",
                    return_value={
                        "pid": 603703,
                        "ppid": 1,
                        "user": "asimov",
                        "args": "python -m uvicorn service.uvicorn_factory:create_app_from_env --port 21001 --factory",
                        "started_at": "2026-04-01 14:22:06",
                        "uptime": "24min",
                    },
                ),
                patch(
                    "service.runtime.probe_local_health",
                    return_value={
                        "supported": True,
                        "ok": True,
                        "payload": {"status": "ok"},
                    },
                ),
            ):
                items = list_managed_service_instances(include_all=True)

        self.assertEqual([item["status"] for item in items], ["running", "exited"])
        self.assertEqual(items[0]["source"], "workspace")
        self.assertEqual(items[1]["worker_status"], "stopped")
