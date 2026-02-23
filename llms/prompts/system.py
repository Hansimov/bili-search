"""System prompts for date/time context."""

from datetime import datetime, timedelta


def get_date_prompt() -> str:
    """Build a date/time prompt with current system time.

    Regenerated per-request to always reflect the current time.
    """
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today_str = now.strftime("%Y-%m-%d")
    yesterday = now - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekday_names[now.weekday()]

    return (
        f"[SYSTEM_TIME]\n"
        f"当前系统时间：{now_str}（{weekday}）\n"
        f"今天日期：{today_str}\n"
        f"昨天日期：{yesterday_str}\n"
        f"[/SYSTEM_TIME]"
    )
