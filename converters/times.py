import re

from datetime import datetime, timedelta
from tclogger import logger
from typing import Union


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


class DateFormatChecker:
    def __init__(self):
        self.year = None
        self.month = None
        self.day = None

    def is_date_format(self, input_str: str, verbose: bool = False):
        logger.enter_quiet(not verbose)
        input_str = input_str.strip()
        logger.note(f"{input_str}:")
        patterns = [
            ("^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}$", "%Y-%m-%d"),  # yyyy-mm-dd
            ("^[0-9]{4}/[0-9]{1,2}/[0-9]{1,2}$", "%Y/%m/%d"),  # yyyy/mm/dd
            ("^[0-9]{4}-[0-9]{1,2}$", "%Y-%m"),  # yyyy-mm
            ("^[0-9]{4}/[0-9]{1,2}$", "%Y/%m"),  # yyyy/mm
            ("^[0-9]{4}$", "%Y"),  # yyyy
            ("^[0-9]{1,2}-[0-9]{1,2}$", "%m-%d"),  # mm/dd
            ("^[0-9]{1,2}/[0-9]{1,2}$", "%m/%d"),  # mm-dd
        ]
        for pattern, date_format in patterns:
            match = re.match(pattern, input_str)
            if match:
                try:
                    date = datetime.strptime(input_str, date_format)
                except Exception as e:
                    logger.warn(f"× Error: {e}")
                    logger.exit_quiet(not verbose)
                    return False
                if date_format in ["%m-%d", "%m/%d"]:
                    self.year = datetime.now().year
                else:
                    self.year = date.year
                self.month = date.month
                self.day = date.day
                logger.success(
                    f"✓ Parsed date: {self.year:04}-{self.month:02}-{self.day:02}"
                )
                logger.exit_quiet(not verbose)
                return True
        logger.warn("× Invalid date format!")
        logger.exit_quiet(not verbose)
        return False

    def is_in_date_range(
        self,
        input_str: str,
        start: Union[str, datetime] = None,
        end: Union[str, datetime] = None,
        verbose: bool = False,
    ):
        logger.enter_quiet(not verbose)
        input_str = input_str.strip()

        if not start and not end:
            return False
        if not self.is_date_format(input_str, verbose=verbose):
            logger.warn("× Invalid date format!")
            return False

        if start:
            if isinstance(start, str):
                start_date = datetime.fromisoformat(start)
            else:
                start_date = start
        else:
            start_date = datetime.min

        if end:
            if isinstance(end, str):
                end_date = datetime.fromisoformat(end)
            else:
                end_date = end
        else:
            end_date = datetime.max

        if start_date <= datetime(self.year, self.month, self.day) <= end_date:
            logger.success(f"✓ In date range")
            logger.exit_quiet(not verbose)
            return True
        else:
            logger.warn("× Out of date range!")
            logger.exit_quiet(not verbose)
            return False


if __name__ == "__main__":
    decimal_seconds = "666.123"
    print(decimal_seconds_to_srt_timestamp(decimal_seconds))

    input_str_list = [
        "2022-02-28",
        "2022-02",
        "2022-2",
        "2022",
        "02-18",
        "02-28",
        "2008",
        "2009-09-09",
        "2099-02-38",
        "2024/2/18",
        "2024/2",
        "2-18",
        "9/18",
    ]
    checker = DateFormatChecker()
    for input_str in input_str_list:
        # checker.is_date_format(input_str, verbose=True)
        checker.is_in_date_range(
            input_str, start="2009-09-09", end=datetime.now(), verbose=True
        )

    # python -m converters.times
