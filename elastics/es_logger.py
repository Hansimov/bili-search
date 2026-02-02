"""
Elasticsearch debug logger for capturing request/response errors.
Logs to bili-search/logs/es.log
"""

import json
import os
import traceback
from datetime import datetime
from pathlib import Path


class ESDebugLogger:
    """Logger for Elasticsearch request/response debugging."""

    def __init__(self, log_file: str = None):
        if log_file is None:
            # Default to logs/es.log relative to project root
            project_root = Path(__file__).parent.parent
            log_file = project_root / "logs" / "es.log"
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _format_json(self, data: dict) -> str:
        """Format dict as pretty JSON string."""
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return str(data)

    def log_error(
        self,
        request_body: dict,
        error: Exception,
        index_name: str = None,
        context: str = None,
    ) -> None:
        """
        Log ES request body and error details to log file.

        Args:
            request_body: The JSON body sent to Elasticsearch
            error: The exception that was raised
            index_name: Name of the ES index being queried
            context: Additional context about the operation
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Extract error details
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()

        # Try to extract ES-specific error info
        es_error_info = {}
        if hasattr(error, "info"):
            es_error_info["info"] = error.info
        if hasattr(error, "status_code"):
            es_error_info["status_code"] = error.status_code
        if hasattr(error, "error"):
            es_error_info["error"] = error.error
        if hasattr(error, "body"):
            es_error_info["body"] = error.body
        if hasattr(error, "meta"):
            try:
                es_error_info["meta"] = {
                    "status": getattr(error.meta, "status", None),
                    "http_version": getattr(error.meta, "http_version", None),
                    "headers": dict(getattr(error.meta, "headers", {})),
                }
            except Exception:
                pass

        # Build log entry
        log_lines = [
            "=" * 80,
            f"[ES ERROR] {timestamp}",
            "=" * 80,
            "",
            f"Index: {index_name}",
            f"Context: {context}" if context else "",
            "",
            f"Error Type: {error_type}",
            f"Error Message: {error_message}",
            "",
        ]

        if es_error_info:
            log_lines.extend(
                [
                    "--- Elasticsearch Error Info ---",
                    self._format_json(es_error_info),
                    "",
                ]
            )

        log_lines.extend(
            [
                "--- Request Body ---",
                self._format_json(request_body),
                "",
                "--- Traceback ---",
                error_traceback,
                "",
            ]
        )

        log_content = "\n".join(log_lines)

        # Append to log file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_content)
        except Exception as write_error:
            # Fallback: print to console if file write fails
            print(f"[ESDebugLogger] Failed to write to {self.log_file}: {write_error}")
            print(log_content)

    def log_request(
        self,
        request_body: dict,
        index_name: str = None,
        context: str = None,
    ) -> None:
        """
        Log ES request body (for debugging successful requests if needed).

        Args:
            request_body: The JSON body sent to Elasticsearch
            index_name: Name of the ES index being queried
            context: Additional context about the operation
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        log_lines = [
            "-" * 80,
            f"[ES REQUEST] {timestamp}",
            "-" * 80,
            f"Index: {index_name}",
            f"Context: {context}" if context else "",
            "",
            "--- Request Body ---",
            self._format_json(request_body),
            "",
        ]

        log_content = "\n".join(log_lines)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_content)
        except Exception as write_error:
            print(f"[ESDebugLogger] Failed to write to {self.log_file}: {write_error}")


# Singleton instance for convenience
_es_debug_logger = None


def get_es_debug_logger() -> ESDebugLogger:
    """Get the singleton ES debug logger instance."""
    global _es_debug_logger
    if _es_debug_logger is None:
        _es_debug_logger = ESDebugLogger()
    return _es_debug_logger
