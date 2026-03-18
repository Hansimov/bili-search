from __future__ import annotations

import sys

from tclogger import decolored


class _DecoloredStream:
    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        return self._stream.write(decolored(data))

    def flush(self):
        return self._stream.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = _DecoloredStream(sys.stdout)
sys.stderr = _DecoloredStream(sys.stderr)

from service.app import create_app_from_env as _create_app_from_env


def create_app_from_env():
    return _create_app_from_env()
