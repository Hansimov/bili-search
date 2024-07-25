from datetime import datetime, timedelta
from typing import Union


def get_now_ts_str() -> tuple[int, str]:
    now = datetime.now()
    now_ts = round(now.timestamp())
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return now_ts, now_str


def timestamp_to_datetime_str(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def seconds_to_duration(seconds: Union[int, str]) -> str:
    """Example: `666` -> `00:11:06`"""
    dt = timedelta(seconds=int(seconds))
    hours = dt.seconds // 3600
    minutes = (dt.seconds % 3600) // 60
    seconds = dt.seconds % 60
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return duration_str


def decimal_seconds_to_srt_timestamp(dot_seconds: Union[float, str]) -> str:
    """
    * SubRip - Wikipedia
        * https://en.wikipedia.org/wiki/SubRip#Format

    Example:
        `666.123` -> `00:11:06,123`
    """

    seconds, ms = str(dot_seconds).split(".")
    duration = seconds_to_duration(seconds)
    return f"{duration},{ms}"


if __name__ == "__main__":
    decimal_seconds = "666.123"
    print(decimal_seconds_to_srt_timestamp(decimal_seconds))

    # python -m converters.times
