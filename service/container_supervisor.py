from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path


CHILD_COMMAND = ["bssv", "start", "--foreground"]
STATE_ROOT = Path("/app/logs/search_app")
STATE_FILENAME_TEMPLATE = "container_app_state.p{port}.json"


class AppSupervisor:
    def __init__(self):
        self.child: subprocess.Popen | None = None
        self.reload_requested = False
        self.shutdown_requested = False
        self.restart_count = 0
        self.state_path = self._resolve_state_path()

    def _resolve_state_path(self) -> Path | None:
        port = str(os.environ.get("BILI_SEARCH_APP_PORT") or "").strip()
        if not port:
            return None
        return STATE_ROOT / STATE_FILENAME_TEMPLATE.format(port=port)

    def _write_state(self):
        if self.state_path is None or self.child is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": self.child.pid,
            "started_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "restart_count": self.restart_count,
        }
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    def install_signal_handlers(self):
        signal.signal(signal.SIGHUP, self._handle_reload)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_reload(self, _signum, _frame):
        self.reload_requested = True
        self._stop_child()

    def _handle_shutdown(self, _signum, _frame):
        self.shutdown_requested = True
        self._stop_child()

    def _start_child(self):
        self.child = subprocess.Popen(CHILD_COMMAND)
        self.restart_count += 1
        self._write_state()

    def _stop_child(self):
        if self.child is None or self.child.poll() is not None:
            return
        self.child.terminate()

    def run(self) -> int:
        self.install_signal_handlers()
        while True:
            self._start_child()
            assert self.child is not None
            while True:
                try:
                    returncode = self.child.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue

            if self.shutdown_requested:
                return int(returncode or 0)
            if self.reload_requested:
                self.reload_requested = False
                time.sleep(0.2)
                continue
            return int(returncode or 0)


def main() -> int:
    return AppSupervisor().run()


if __name__ == "__main__":
    raise SystemExit(main())
