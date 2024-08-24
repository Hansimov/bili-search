import re

from datetime import datetime, timedelta
from tclogger import logger, ts_to_str


class DateRangeConverter:
    RE_MD_SEP = r"[-/\.]*"
    RE_DH_SEP = r"[\_\.]*"
    RE_YEAR = r"(\d{4})"
    RE_MM = r"(\d{1,2})"
    RE_DD = r"(\d{1,2})"
    RE_HH = r"(\d{1,2})"
    RE_SEP_MM = rf"{RE_MD_SEP}{RE_MM}"
    RE_SEP_DD = rf"{RE_MD_SEP}{RE_DD}"
    RE_SEP_HH = rf"{RE_DH_SEP}{RE_HH}"
    RE_YM = rf"(?P<yyyy_mm>{RE_YEAR}{RE_SEP_MM})"
    RE_YMD = rf"(?P<yyyy_mm_dd>{RE_YEAR}{RE_SEP_MM}{RE_SEP_DD})"
    RE_YMDH = rf"(?P<yyyy_mm_dd_hh>{RE_YEAR}{RE_SEP_MM}{RE_SEP_DD}{RE_SEP_HH})"
    RE_MD = rf"(?P<mm_dd>{RE_MM}{RE_SEP_DD})"
    RE_MDH = rf"(?P<mm_dd_hh>{RE_MM}{RE_SEP_DD}{RE_SEP_HH})"
    RE_RANGE_DATE = rf"(?P<range_date>{RE_YMDH}|{RE_YMD}|{RE_YM}|{RE_MDH}|{RE_MD})"

    RE_CH_SEP = r"[\_\.\s]*"

    RE_THIS_YEAR = rf"(?P<this_year>本年|今年|this{RE_CH_SEP}year)"
    RE_LAST_YEAR = rf"(?P<last_year>上年|去年|last{RE_CH_SEP}year)"
    RE_THIS_MONTH = rf"(?P<this_month>本月|this{RE_CH_SEP}month)"
    RE_LAST_MONTH = rf"(?P<last_month>上个?月|last{RE_CH_SEP}month)"
    RE_THIS_WEEK = rf"(?P<this_week>本周|this{RE_CH_SEP}week)"
    RE_LAST_WEEK = rf"(?P<last_week>上周|last{RE_CH_SEP}week)"
    RE_THIS_DAY = rf"(?P<this_day>本日|今日|今天|this{RE_CH_SEP}day|today)"
    RE_LAST_DAY = rf"(?P<last_day>上日|昨日|昨天|last{RE_CH_SEP}day|yesterday)"
    RE_THIS_HOUR = rf"(?P<this_hour>本时|当前小时|this{RE_CH_SEP}hour)"
    RE_LAST_HOUR = rf"(?P<last_hour>上时|上个小时|last{RE_CH_SEP}hour)"
    RE_RANGE_THIS = rf"(?P<range_this>{RE_THIS_YEAR}|{RE_THIS_MONTH}|{RE_THIS_WEEK}|{RE_THIS_DAY}|{RE_THIS_HOUR})"
    RE_RANGE_LAST = rf"(?P<range_last>{RE_LAST_YEAR}|{RE_LAST_MONTH}|{RE_LAST_WEEK}|{RE_LAST_DAY}|{RE_LAST_HOUR})"

    RE_DU_SEP = r"[\s]*"
    RE_YEAR_UNIT = r"(?P<year_unit>年|years?|yr|y)"
    RE_MONTH_UNIT = r"(?P<month_unit>个?月|months?|mon)"
    RE_WEEK_UNIT = r"(?P<week_unit>周|weeks?|wk|w)"
    RE_DAY_UNIT = r"(?P<day_unit>日|天|days?|d)"
    RE_HOUR_UNIT = r"(?P<hour_unit>个?小时|hours?|hr|h)"
    # RE_MINUTE_UNIT = r"(分钟?|minutes?|min)"
    # RE_SECOND_UNIT = r"(秒钟?|seconds?|sec|s)"
    RE_N_YEAR = rf"(?P<n_years>(?P<year_n>\d+){RE_DU_SEP}{RE_YEAR_UNIT})"
    RE_N_MONTH = rf"(?P<n_months>(?P<month_n>\d+){RE_DU_SEP}{RE_MONTH_UNIT})"
    RE_N_WEEK = rf"(?P<n_weeks>(?P<week_n>\d+){RE_DU_SEP}{RE_WEEK_UNIT})"
    RE_N_DAY = rf"(?P<n_days>(?P<day_n>\d+){RE_DU_SEP}{RE_DAY_UNIT})"
    RE_N_HOUR = rf"(?P<n_hours>(?P<hour_n>\d+){RE_DU_SEP}{RE_HOUR_UNIT})"
    # RE_N_MINUTE = rf"(?P<n_minutes>(?P<minute_n>\d+)\s*{RE_MINUTE_UNIT})"
    # RE_N_SECOND = rf"(?P<n_seconds>(?P<second_n>\d+)\s*{RE_SECOND_UNIT})"
    RE_RANGE_DIST = (
        rf"(?P<range_dist>{RE_N_YEAR}|{RE_N_MONTH}|{RE_N_WEEK}|{RE_N_DAY}|{RE_N_HOUR})"
    )

    def get_date_ts_range(self, date_str: str) -> tuple[int, int]:
        """Get the timestamp range of the date.
        Examples:
        1. <range_date>
            - "2014"
                - from: 2014-01-01 00:00:00
                -   to: 2014-12-31 23:59:59
            - "2014-08"
                - from: 2014-08-01 00:00:00
                -   to: 2014-08-31 23:59:59
            - "2014/08/12"
                - from: 2014-08-12 00:00:00
                -   to: 2014-08-12 23:59:59
            - "08/12"
                - from: [THIS_YEAR]-08-12 00:00:00
                - from: [THIS_YEAR]-08-12 23:59:59
            - "2014-08-12.12"
                - from: 2014-08-12 12:00:00
                -   to: 2014-08-12 12:59:59
        2. <range_this>
            - "this_year"
                - from: [THIS_YEAR]-01-01 00:00:00
                -   to: [NOW]
            - "this_month"
                - from: [THIS_YEAR]-[THIS_MONTH]-01 00:00:00
                -   to: [NOW]
            - "this_week"
                - from: [THIS_WEEK_MONDAY_DATE] 00:00:00
                -   to: [NOW]
            - "this_day"
                - from: [TODAY] 00:00:00
                -   to: [NOW]
            - "this_hour"
                - from: [TODAY] [THIS_HOUR]:00:00
                -   to: [NOW]
        3. <range_last>
            - "last_year"
                - from: [LAST_YEAR]-01-01 00:00:00
                -   to: [LAST_YEAR]-12-31 23:59:59
            - "last_month"
                - from: [LAST_MONTH]-01 00:00:00
                -   to: [LAST_MONTH]-[LAST_DAY] 23:59:59
            - "last_week"
                - from: [LAST_WEEK_MONDAY_DATE] 00:00:00
                -   to: [LAST_WEEK_SUNDAY_DATE] 23:59:59
            - "last_day"
                - from: [YESTERDAY] 00:00:00
                -   to: [YESTERDAY] 23:59:59
            - "last_hour"
                - from: [DATE_OF_LAST_HOUR] [LAST_HOUR]:00:00
                - from: [DATE_OF_LAST_HOUR] [LAST_HOUR]:59:59
        4. <range_dist>
            - "1 year"
                - from: [DATE_OF_ONE_YEAR_AGO]
                -   to: [DATE_OF_ONE_YEAR_AGO]
            - "2 months"
                - from: [DATE_OF_TWO_MONTHS_AGO]
                -   to: [DATE_OF_TWO_MONTHS_AGO]
            - "3 weeks"
                - from: [DATE_OF_THREE_WEEKS_AGO]
                -   to: [DATE_OF_THREE_WEEKS_AGO]
            - "4 days"
                - from: [DATE_OF_FOUR_DAYS_AGO]
                -   to: [DATE_OF_FOUR_DAYS_AGO]
            - "5 hours"
                - from: [DATE_OF_FIVE_HOURS_AGO]
                -   to: [DATE_OF_FIVE_HOURS_AGO]
        """

        if re.match(self.RE_RANGE_DATE, date_str):
            return self.get_date_ts_range_of_date(date_str)
        elif re.match(self.RE_RANGE_THIS, date_str):
            return self.get_date_ts_range_of_this(date_str)
        elif re.match(self.RE_RANGE_LAST, date_str):
            return self.get_date_ts_range_of_last(date_str)
        elif re.match(self.RE_RANGE_DIST, date_str):
            return self.get_date_ts_range_of_dist(date_str)
        else:
            logger.warn(f"× No matched range for date_str: {date_str}")
            return None

    def get_date_ts_range_of_date(self, date_str: str) -> tuple[int, int]:
        logger.mesg(f"> <range_date>: {date_str}")
        return 0, 0
        now = datetime.now()
        this_year = now.year

        def to_timestamp(dt: datetime) -> int:
            return int(dt.timestamp())

        match = re.match(self.RE_RANGE_DATE, date_str)
        if match:
            if match.group("yyyy_mm_dd_hh"):
                dt = datetime.strptime(match.group("yyyy_mm_dd_hh"), "%Y-%m-%d-%H")
                start = dt
                end = dt + timedelta(hours=1) - timedelta(seconds=1)
            elif match.group("yyyy_mm_dd"):
                dt = datetime.strptime(match.group("yyyy_mm_dd"), "%Y-%m-%d")
                start = dt
                end = dt + timedelta(days=1) - timedelta(seconds=1)
            elif match.group("yyyy_mm"):
                dt = datetime.strptime(match.group("yyyy_mm"), "%Y-%m")
                start = dt
                end = (dt + timedelta(days=31)).replace(day=1) - timedelta(seconds=1)
            elif match.group("mm_dd_hh"):
                dt = datetime.strptime(
                    f"{this_year}-{match.group('mm_dd_hh')}", "%Y-%m-%d-%H"
                )
                start = dt
                end = dt + timedelta(hours=1) - timedelta(seconds=1)
            elif match.group("mm_dd"):
                dt = datetime.strptime(
                    f"{this_year}-{match.group('mm_dd')}", "%Y-%m-%d"
                )
                start = dt
                end = dt + timedelta(days=1) - timedelta(seconds=1)
            else:
                raise ValueError("Invalid date string format")
            return to_timestamp(start), to_timestamp(end)
        else:
            raise ValueError("Invalid date string format")

    def get_date_ts_range_of_this(self, date_str: str) -> tuple[int, int]:
        logger.mesg(f"> <range_this>: {date_str}")
        return 0, 0
        now = datetime.now()
        this_year = now.year
        this_month = now.month
        this_day = now.day
        this_hour = now.hour

        def to_timestamp(dt: datetime) -> int:
            return int(dt.timestamp())

        if re.match(self.RE_THIS_YEAR, date_str):
            start = datetime(this_year, 1, 1)
            end = now
        elif re.match(self.RE_THIS_MONTH, date_str):
            start = datetime(this_year, this_month, 1)
            end = now
        elif re.match(self.RE_THIS_WEEK, date_str):
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif re.match(self.RE_THIS_DAY, date_str):
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif re.match(self.RE_THIS_HOUR, date_str):
            start = now.replace(minute=0, second=0, microsecond=0)
            end = now
        else:
            raise ValueError("Invalid date string format")

        return to_timestamp(start), to_timestamp(end)

    def get_date_ts_range_of_last(self, date_str: str) -> tuple[int, int]:
        logger.mesg(f"> <range_last>: {date_str}")
        return 0, 0
        now = datetime.now()
        this_year = now.year
        this_month = now.month

        def to_timestamp(dt: datetime) -> int:
            return int(dt.timestamp())

        if re.match(self.RE_LAST_YEAR, date_str):
            start = datetime(this_year - 1, 1, 1)
            end = datetime(this_year - 1, 12, 31, 23, 59, 59)
        elif re.match(self.RE_LAST_MONTH, date_str):
            last_month = this_month - 1 if this_month > 1 else 12
            last_month_year = this_year if this_month > 1 else this_year - 1
            start = datetime(last_month_year, last_month, 1)
            end = (start + timedelta(days=31)).replace(day=1) - timedelta(seconds=1)
        elif re.match(self.RE_LAST_WEEK, date_str):
            start = now - timedelta(days=now.weekday() + 7)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        elif re.match(self.RE_LAST_DAY, date_str):
            start = now - timedelta(days=1)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=23, minutes=59, seconds=59)
        elif re.match(self.RE_LAST_HOUR, date_str):
            start = now - timedelta(hours=1)
            start = start.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(minutes=59, seconds=59)
        else:
            raise ValueError("Invalid date string format")

        return to_timestamp(start), to_timestamp(end)

    def get_date_ts_range_of_dist(self, date_str: str) -> tuple[int, int]:
        logger.mesg(f"> <range_dist>: {date_str}")
        return 0, 0
        now = datetime.now()

        def to_timestamp(dt: datetime) -> int:
            return int(dt.timestamp())

        match = re.match(self.RE_RANGE_DIST, date_str)
        if match:
            if match.group("n_years"):
                n = int(match.group("year_n"))
                start = now - timedelta(days=365 * n)
                end = now
            elif match.group("n_months"):
                n = int(match.group("month_n"))
                start = now - timedelta(days=30 * n)
                end = now
            elif match.group("n_weeks"):
                n = int(match.group("week_n"))
                start = now - timedelta(weeks=n)
                end = now
            elif match.group("n_days"):
                n = int(match.group("day_n"))
                start = now - timedelta(days=n)
                end = now
            elif match.group("n_hours"):
                n = int(match.group("hour_n"))
                start = now - timedelta(hours=n)
                end = now
            else:
                raise ValueError("Invalid date string format")
            return to_timestamp(start), to_timestamp(end)
        else:
            raise ValueError("Invalid date string format")


if __name__ == "__main__":
    date_strs = [
        "2014",
        "2014-08",
        "2014/08/12",
        "08/12",
        "2014-08-12.12",
        "this_year",
        "this_month",
        "this_week",
        "this_day",
        "this_hour",
        "last_year",
        "last_month",
        "last_week",
        "last_day",
        "last_hour",
        "1 year",
        "2 months",
        "3 weeks",
        "4 days",
        "5 hours",
    ]
    converter = DateRangeConverter()
    for date_str in date_strs:
        logger.note(f"{date_str}")
        start_ts, end_ts = converter.get_date_ts_range(date_str)
        start_str = ts_to_str(start_ts)
        end_str = ts_to_str(end_ts)
        # logger.success(f"> start: {start_ts}")
        # logger.success(f">   end: {end_ts}")

    # python -m converters.date_ranger
