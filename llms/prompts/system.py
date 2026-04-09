"""System prompt helpers for date context."""

from __future__ import annotations

from datetime import datetime, timedelta


def get_date_prompt(*, now: datetime | None = None) -> str:
    """Build day-level date context without per-request clock jitter."""
    current = now or datetime.now()
    today_str = current.strftime("%Y-%m-%d")
    yesterday_str = (current - timedelta(days=1)).strftime("%Y-%m-%d")
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekday_names[current.weekday()]

    return (
        f"[SYSTEM_TIME]\n"
        f"当前日期：{today_str}（{weekday}）\n"
        f"昨天日期：{yesterday_str}\n"
        f"[/SYSTEM_TIME]"
    )
