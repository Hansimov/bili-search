from __future__ import annotations

import os

import requests

from tclogger import logger

from configs.envs import SECRETS


DEFAULT_BILI_STORE_TIMEOUT = 30.0
_TRANSCRIPT_CONFIG_ERROR = (
    "Transcript endpoint is not configured. "
    "Set bili_store.endpoint in configs/secrets.json or BILI_STORE_BASE_URL."
)


def _read_secret(section: str, key: str, default=None):
    try:
        section_value = SECRETS[section]
    except Exception:
        return default
    if isinstance(section_value, dict):
        return section_value.get(key, default)
    getter = getattr(section_value, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return section_value[key]
    except Exception:
        return default


def _is_placeholder_value(value: str | None) -> bool:
    normalized = str(value or "").strip()
    lowered = normalized.lower()
    if not normalized:
        return True
    if normalized.startswith("${{") or normalized.startswith("$"):
        return True
    if "<" in normalized and ">" in normalized:
        return True
    if "your_" in lowered or "your-" in lowered:
        return True
    if any(
        token in lowered
        for token in ("placeholder", "example", "replace-me", "replace_", "set-me")
    ):
        return True
    return False


def _resolve_base_url(base_url: str | None = None) -> str | None:
    raw_value = (
        base_url
        or os.getenv("BILI_STORE_BASE_URL")
        or _read_secret("bili_store", "endpoint")
    )
    normalized = str(raw_value or "").strip().rstrip("/")
    if _is_placeholder_value(normalized):
        return None
    return normalized


def _resolve_timeout(timeout: float | None = None) -> float:
    if timeout is not None:
        return float(timeout)
    raw_value = os.getenv("BILI_STORE_TIMEOUT") or _read_secret("bili_store", "timeout")
    if raw_value in (None, ""):
        return DEFAULT_BILI_STORE_TIMEOUT
    return float(raw_value)


class BiliStoreTranscriptClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        verbose: bool = False,
    ):
        self.base_url = _resolve_base_url(base_url)
        self.timeout = _resolve_timeout(timeout)
        self.verbose = verbose

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url)

    def _not_configured_error(self) -> dict:
        return {"error": _TRANSCRIPT_CONFIG_ERROR}

    def _post(self, path: str, payload: dict | None = None) -> dict:
        if not self.base_url:
            return self._not_configured_error()
        url = f"{self.base_url}{path}"
        if self.verbose:
            logger.note(f"> Transcript POST: {path}")
        try:
            response = requests.post(url, json=payload or {}, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            return {
                "error": f"Transcript request failed with HTTP {status_code}",
                "status_code": status_code,
                "path": path,
            }
        except requests.RequestException as exc:
            return {
                "error": f"Transcript request failed: {exc}",
                "path": path,
            }
        except ValueError as exc:
            return {
                "error": f"Transcript response was not valid JSON: {exc}",
                "path": path,
            }

    def _get(self, path: str) -> dict:
        if not self.base_url:
            return self._not_configured_error()
        url = f"{self.base_url}{path}"
        if self.verbose:
            logger.note(f"> Transcript GET: {path}")
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            return {
                "error": f"Transcript request failed with HTTP {status_code}",
                "status_code": status_code,
                "path": path,
            }
        except requests.RequestException as exc:
            return {
                "error": f"Transcript request failed: {exc}",
                "path": path,
            }
        except ValueError as exc:
            return {
                "error": f"Transcript response was not valid JSON: {exc}",
                "path": path,
            }

    def health(self) -> dict:
        return self._get("/health")

    def get_video_transcript(
        self, video_id: str, *, request: dict | None = None
    ) -> dict:
        normalized_video_id = str(video_id or "").strip()
        if not normalized_video_id:
            return {"error": "video_id is required"}
        payload = dict(request or {})
        payload.pop("video_id", None)
        payload.pop("bvid", None)
        return self._post(f"/transcripts/{normalized_video_id}", payload)


__all__ = [
    "BiliStoreTranscriptClient",
    "DEFAULT_BILI_STORE_TIMEOUT",
]
