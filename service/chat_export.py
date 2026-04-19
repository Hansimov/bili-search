from __future__ import annotations

import json
import os
import re

from fastapi import Request, Response
from urllib.parse import parse_qs, quote

DEFAULT_FILE_NAME = "chat-session-export.md"


async def handle_export(request: Request) -> Response:
    """Parse the incoming request and return an attachment download response."""
    payload = await _parse_payload(request)
    mime_type = _resolve_mime_type(payload.get("mimeType", ""))
    file_name = _sanitize_filename(
        payload.get("fileName", DEFAULT_FILE_NAME),
        mime_type,
    )
    content = payload.get("content", "")

    return Response(
        content=content.encode("utf-8"),
        media_type=mime_type,
        headers={
            "Content-Disposition": _build_disposition(file_name, mime_type),
            "Cache-Control": "no-store, no-cache, max-age=0",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _parse_payload(request: Request) -> dict[str, str]:
    content_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if content_type == "application/json":
        payload = await request.json()
        if isinstance(payload, dict):
            content = payload.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            return {
                "fileName": str(payload.get("fileName") or DEFAULT_FILE_NAME),
                "mimeType": str(payload.get("mimeType") or ""),
                "content": content,
            }

    raw_body = await request.body()
    parsed = parse_qs(
        raw_body.decode("utf-8", errors="replace"), keep_blank_values=True
    )
    return {
        "fileName": _first_form_value(parsed, "fileName", DEFAULT_FILE_NAME),
        "mimeType": _first_form_value(parsed, "mimeType", ""),
        "content": _first_form_value(parsed, "content", ""),
    }


def _first_form_value(
    values: dict[str, list[str]],
    key: str,
    default: str = "",
) -> str:
    items = values.get(key)
    if not items:
        return default
    return items[0]


def _resolve_mime_type(mime_type: str) -> str:
    normalized = (mime_type or "").split(";", 1)[0].strip().lower()
    if normalized == "application/json":
        return "application/json; charset=utf-8"
    if normalized == "text/markdown":
        return "text/markdown; charset=utf-8"
    if normalized == "text/plain":
        return "text/plain; charset=utf-8"
    return "text/plain; charset=utf-8"


def _sanitize_filename(file_name: str, mime_type: str) -> str:
    candidate = (file_name or "").replace("\x00", "")
    candidate = candidate.replace("\r", "").replace("\n", "").strip()
    candidate = re.sub(r'[\\/:*?"<>|]+', "-", candidate)
    candidate = re.sub(r"\s+", "-", candidate).strip(" .-_")

    default_extension = ".json" if mime_type.startswith("application/json") else ".md"
    if not candidate:
        return f"chat-session-export{default_extension}"

    stem, suffix = os.path.splitext(candidate)
    if not suffix:
        candidate = f"{stem or 'chat-session-export'}{default_extension}"

    return candidate[:180]


def _ascii_filename(file_name: str, mime_type: str) -> str:
    stem, suffix = os.path.splitext(file_name)
    ascii_stem = stem.encode("ascii", "ignore").decode("ascii")
    ascii_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_stem).strip(" .-_")

    ascii_suffix = suffix.encode("ascii", "ignore").decode("ascii")
    ascii_suffix = re.sub(r"[^A-Za-z0-9.]+", "", ascii_suffix)
    default_extension = ".json" if mime_type.startswith("application/json") else ".md"

    return f"{ascii_stem or 'chat-session-export'}{ascii_suffix or default_extension}"


def _build_disposition(file_name: str, mime_type: str) -> str:
    ascii_name = _ascii_filename(file_name, mime_type)
    return (
        f'attachment; filename="{ascii_name}"; ' f"filename*=UTF-8''{quote(file_name)}"
    )
