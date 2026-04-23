from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path


_ASSET_DIR = Path(__file__).with_name("assets")
_TERM_ALIAS_ASSET_PATH = _ASSET_DIR / "term_aliases.json"


@lru_cache(maxsize=1)
def get_term_alias_rules() -> tuple[tuple[str, str], ...]:
    payload = json.loads(_TERM_ALIAS_ASSET_PATH.read_text(encoding="utf-8"))
    items = payload.get("aliases") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("term alias asset must contain an aliases list")

    rules: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        if not source or not target or source == target:
            continue
        rules.append((source, target))
    return tuple(rules)


__all__ = ["get_term_alias_rules"]
