import re

from calendar import monthrange
from datetime import timedelta
from tclogger import logger, ts_to_str, get_now_ts, get_now, tcdatetime
from converters.field.operators import OP_MAP, BRACKET_MAP


class DateFieldConverter:
    RE_DATE_FIELD = rf"(?i:(日期|时间|rq|date|dt|d))"
    REP_DATE_FIELD = rf"(?P<date_field>{RE_DATE_FIELD})"

    # DATE: YMDH, YMD, YM, YYYY, MDH, MD
    RE_MM_SEP = r"[-/\.]+"
    RE_DD_SEP = r"[-/\.]+"
    RE_HH_SEP = r"[\_\.]+"
    RE_YEAR = r"\d{4}"
    RE_MM = r"\d{1,2}"
    RE_DD = r"\d{1,2}"
    RE_HH = r"\d{1,2}"
    RE_SEP_MM = rf"{RE_MM_SEP}{RE_MM}"
    RE_SEP_DD = rf"{RE_DD_SEP}{RE_DD}"
    RE_SEP_HH = rf"{RE_HH_SEP}{RE_HH}"

    RE_YMDH = rf"{RE_YEAR}{RE_MM_SEP}{RE_MM}{RE_DD_SEP}{RE_DD}{RE_HH_SEP}{RE_HH}"
    RE_YMD = rf"{RE_YEAR}{RE_MM_SEP}{RE_MM}{RE_DD_SEP}{RE_DD}"
    RE_YM = rf"{RE_YEAR}{RE_MM_SEP}{RE_MM}"
    RE_YYYY = rf"{RE_YEAR}"
    RE_MDH = rf"{RE_MM}{RE_DD_SEP}{RE_DD}{RE_HH_SEP}{RE_HH}"
    RE_MD = rf"{RE_MM}{RE_DD_SEP}{RE_DD}"
    RE_RANGE_DATE = rf"({RE_YMDH}|{RE_YMD}|{RE_YM}|{RE_YYYY}|{RE_MDH}|{RE_MD})"

    REP_YMDH = rf"(?P<yyyy_mm_dd_hh>(?P<ymdh_year>{RE_YEAR}){RE_MM_SEP}(?P<ymdh_mm>{RE_MM}){RE_DD_SEP}(?P<ymdh_dd>{RE_DD}){RE_HH_SEP}(?P<ymdh_hh>{RE_HH}))"
    REP_YMD = rf"(?P<yyyy_mm_dd>(?P<ymd_year>{RE_YEAR}){RE_MM_SEP}(?P<ymd_mm>{RE_MM}){RE_DD_SEP}(?P<ymd_dd>{RE_DD}))"
    REP_YM = rf"(?P<yyyy_mm>(?P<ym_year>{RE_YEAR}){RE_MM_SEP}(?P<ym_mm>{RE_MM}))"
    REP_YYYY = rf"(?P<yyyy>{RE_YEAR})"
    REP_MDH = rf"(?P<mm_dd_hh>(?P<mdh_mm>{RE_MM}){RE_DD_SEP}(?P<mdh_dd>{RE_DD}){RE_HH_SEP}(?P<mdh_hh>{RE_HH}))"
    REP_MD = rf"(?P<mm_dd>(?P<md_mm>{RE_MM}){RE_DD_SEP}(?P<md_dd>{RE_DD}))"
    REP_RANGE_DATE = (
        rf"(?P<range_date>{REP_YMDH}|{REP_YMD}|{REP_YM}|{REP_YYYY}|{REP_MDH}|{REP_MD})"
    )

    # THIS/PAST/LAST + YEAR/MONTH/WEEK/DAY/HOUR
    RE_CH_SEP = r"[\_\.\s\-]*"
    RE_THIS_YEAR = rf"(本年|今年|this{RE_CH_SEP}year)"
    RE_LAST_YEAR = rf"(上年|去年|last{RE_CH_SEP}year)"
    RE_PAST_YEAR = rf"(过去一年|past{RE_CH_SEP}year)"
    RE_THIS_MONTH = rf"(本月|this{RE_CH_SEP}month)"
    RE_LAST_MONTH = rf"(上个?月|last{RE_CH_SEP}month)"
    RE_PAST_MONTH = rf"(过去一个?月|past{RE_CH_SEP}month)"
    RE_THIS_WEEK = rf"(本周|this{RE_CH_SEP}week)"
    RE_LAST_WEEK = rf"(上周|last{RE_CH_SEP}week)"
    RE_PAST_WEEK = rf"(过去一周|past{RE_CH_SEP}week)"
    RE_THIS_DAY = rf"(本日|今日|今天|this{RE_CH_SEP}day|today)"
    RE_LAST_DAY = rf"(上日|昨日|昨天|last{RE_CH_SEP}day|yesterday)"
    RE_PAST_DAY = rf"(过去一天|past{RE_CH_SEP}day)"
    RE_THIS_HOUR = rf"(本时|当前小时|this{RE_CH_SEP}hour)"
    RE_LAST_HOUR = rf"(上时|上个小时|last{RE_CH_SEP}hour)"
    RE_PAST_HOUR = rf"(过去一个小时|past{RE_CH_SEP}hour)"
    RE_RANGE_THIS = (
        rf"({RE_THIS_YEAR}|{RE_THIS_MONTH}|{RE_THIS_WEEK}|{RE_THIS_DAY}|{RE_THIS_HOUR})"
    )
    RE_RANGE_LAST = (
        rf"({RE_LAST_YEAR}|{RE_LAST_MONTH}|{RE_LAST_WEEK}|{RE_LAST_DAY}|{RE_LAST_HOUR})"
    )
    RE_RANGE_PAST = (
        rf"({RE_PAST_YEAR}|{RE_PAST_MONTH}|{RE_PAST_WEEK}|{RE_PAST_DAY}|{RE_PAST_HOUR})"
    )
    RE_RANGE_RECENT = rf"({RE_RANGE_THIS}|{RE_RANGE_LAST}|{RE_RANGE_PAST})"

    REP_THIS_YEAR = rf"(?P<this_year>{RE_THIS_YEAR})"
    REP_LAST_YEAR = rf"(?P<last_year>{RE_LAST_YEAR})"
    REP_PAST_YEAR = rf"(?P<past_year>{RE_PAST_YEAR})"
    REP_THIS_MONTH = rf"(?P<this_month>{RE_THIS_MONTH})"
    REP_LAST_MONTH = rf"(?P<last_month>{RE_LAST_MONTH})"
    REP_PAST_MONTH = rf"(?P<past_month>{RE_PAST_MONTH})"
    REP_THIS_WEEK = rf"(?P<this_week>{RE_THIS_WEEK})"
    REP_LAST_WEEK = rf"(?P<last_week>{RE_LAST_WEEK})"
    REP_PAST_WEEK = rf"(?P<past_week>{RE_PAST_WEEK})"
    REP_THIS_DAY = rf"(?P<this_day>{RE_THIS_DAY})"
    REP_LAST_DAY = rf"(?P<last_day>{RE_LAST_DAY})"
    REP_PAST_DAY = rf"(?P<past_day>{RE_PAST_DAY})"
    REP_THIS_HOUR = rf"(?P<this_hour>{RE_THIS_HOUR})"
    REP_LAST_HOUR = rf"(?P<last_hour>{RE_LAST_HOUR})"
    REP_PAST_HOUR = rf"(?P<past_hour>{RE_PAST_HOUR})"
    REP_RANGE_THIS = rf"(?P<range_this>{REP_THIS_YEAR}|{REP_THIS_MONTH}|{REP_THIS_WEEK}|{REP_THIS_DAY}|{REP_THIS_HOUR})"
    REP_RANGE_LAST = rf"(?P<range_last>{REP_LAST_YEAR}|{REP_LAST_MONTH}|{REP_LAST_WEEK}|{REP_LAST_DAY}|{REP_LAST_HOUR})"
    REP_RANGE_PAST = rf"(?P<range_past>{REP_PAST_YEAR}|{REP_PAST_MONTH}|{REP_PAST_WEEK}|{REP_PAST_DAY}|{REP_PAST_HOUR})"
    REP_RANGE_RECENT = (
        rf"(?P<range_recent>{REP_RANGE_THIS}|{REP_RANGE_LAST}|{REP_RANGE_PAST})"
    )

    # N + YEAR/MONTH/WEEK/DAY/HOUR
    RE_DU_SEP = r"[\s]*"
    RE_YEAR_UNIT = r"(年|years?|yr|y)"
    RE_MONTH_UNIT = r"(个?月|months?|mon|m)"
    RE_WEEK_UNIT = r"(周|weeks?|wk|w)"
    RE_DAY_UNIT = r"(日|天|days?|d)"
    RE_HOUR_UNIT = r"(个?小时|hours?|hr|h)"
    RE_N_YEAR = rf"\d+{RE_DU_SEP}{RE_YEAR_UNIT}"
    RE_N_MONTH = rf"\d+{RE_DU_SEP}{RE_MONTH_UNIT}"
    RE_N_WEEK = rf"\d+{RE_DU_SEP}{RE_WEEK_UNIT}"
    RE_N_DAY = rf"\d+{RE_DU_SEP}{RE_DAY_UNIT}"
    RE_N_HOUR = rf"\d+{RE_DU_SEP}{RE_HOUR_UNIT}"
    RE_RANGE_DIST = rf"({RE_N_YEAR}|{RE_N_MONTH}|{RE_N_WEEK}|{RE_N_DAY}|{RE_N_HOUR})"

    REP_N_YEAR = rf"(?P<n_years>(?P<year_n>\d+){RE_DU_SEP}{RE_YEAR_UNIT})"
    REP_N_MONTH = rf"(?P<n_months>(?P<month_n>\d+){RE_DU_SEP}{RE_MONTH_UNIT})"
    REP_N_WEEK = rf"(?P<n_weeks>(?P<week_n>\d+){RE_DU_SEP}{RE_WEEK_UNIT})"
    REP_N_DAY = rf"(?P<n_days>(?P<day_n>\d+){RE_DU_SEP}{RE_DAY_UNIT})"
    REP_N_HOUR = rf"(?P<n_hours>(?P<hour_n>\d+){RE_DU_SEP}{RE_HOUR_UNIT})"
    REP_RANGE_DIST = rf"(?P<range_dist>{REP_N_YEAR}|{REP_N_MONTH}|{REP_N_WEEK}|{REP_N_DAY}|{REP_N_HOUR})"

    RE_DATE_VAL = rf"({RE_RANGE_DATE}|{RE_RANGE_RECENT}|{RE_RANGE_DIST})"

    def get_date_ts_range(self, date_str: str) -> tuple[str, int, int]:
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
        4. <range_dist> (<range_past> is the special case of <range_dist>)
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

        if re.match(self.REP_RANGE_DATE, date_str):
            return "date", *self.get_date_ts_range_of_date(date_str)
        elif re.match(self.REP_RANGE_THIS, date_str):
            return "this", *self.get_date_ts_range_of_this(date_str)
        elif re.match(self.REP_RANGE_LAST, date_str):
            return "last", *self.get_date_ts_range_of_last(date_str)
        elif re.match(self.REP_RANGE_DIST, date_str):
            return "dist", *self.get_date_ts_range_of_dist(date_str)
        elif re.match(self.REP_RANGE_PAST, date_str):
            return "past", *self.get_date_ts_range_of_past(date_str)
        else:
            logger.warn(f"× No matched date_ts_range for date_str: {date_str}")
            return "", 0, 0

    def get_date_ts_range_of_date(self, date_str: str) -> tuple[int, int]:
        now = get_now()
        match = re.match(self.REP_RANGE_DATE, date_str)
        if match:
            if match.group("yyyy_mm_dd_hh"):
                year = int(match.group("ymdh_year"))
                month = int(match.group("ymdh_mm"))
                day = int(match.group("ymdh_dd"))
                hour = int(match.group("ymdh_hh"))
                start_dt = tcdatetime(year, month, day, hour)
                end_dt = start_dt + timedelta(hours=1) - timedelta(milliseconds=1)
            elif match.group("yyyy_mm_dd"):
                year = int(match.group("ymd_year"))
                month = int(match.group("ymd_mm"))
                day = int(match.group("ymd_dd"))
                start_dt = tcdatetime(year, month, day)
                end_dt = start_dt + timedelta(days=1) - timedelta(milliseconds=1)
            elif match.group("yyyy_mm"):
                year = int(match.group("ym_year"))
                month = int(match.group("ym_mm"))
                start_dt = tcdatetime(year, month, 1)
                month_days = monthrange(year, month)[1]
                end_dt = (
                    start_dt + timedelta(days=month_days) - timedelta(milliseconds=1)
                )
            elif match.group("yyyy"):
                year = int(match.group("yyyy"))
                start_dt = tcdatetime(year, 1, 1)
                end_dt = tcdatetime(year, 12, 31, 23, 59, 59)
            elif match.group("mm_dd_hh"):
                month = int(match.group("mdh_mm"))
                day = int(match.group("mdh_dd"))
                hour = int(match.group("mdh_hh"))
                start_dt = tcdatetime(now.year, month, day, hour)
                end_dt = start_dt + timedelta(hours=1) - timedelta(milliseconds=1)
            elif match.group("mm_dd"):
                month = int(match.group("md_mm"))
                day = int(match.group("md_dd"))
                start_dt = tcdatetime(now.year, month, day)
                if start_dt.timestamp() > now.timestamp():
                    start_dt = tcdatetime(now.year - 1, month, day)
                end_dt = start_dt + timedelta(days=1) - timedelta(milliseconds=1)
            else:
                logger.warn(f"× No match for type <range_date>: {date_str}")
                start_dt = None
                end_dt = None

            if start_dt and end_dt:
                return int(start_dt.timestamp()), int(end_dt.timestamp())
            else:
                return 0, 0
        else:
            logger.warn(f"× No match for type <range_date>: {date_str}")

        return 0, 0

    def get_date_ts_range_of_this(self, date_str: str) -> tuple[int, int]:
        now = get_now()
        match = re.match(self.REP_RANGE_THIS, date_str)
        if match:
            if match.group("this_year"):
                start_dt = tcdatetime(now.year, 1, 1)
            elif match.group("this_month"):
                start_dt = tcdatetime(now.year, now.month, 1)
            elif match.group("this_week"):
                start_dt = now - timedelta(days=now.weekday())
                start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            elif match.group("this_day"):
                start_dt = tcdatetime(now.year, now.month, now.day)
            elif match.group("this_hour"):
                start_dt = tcdatetime(now.year, now.month, now.day, now.hour)
            else:
                logger.warn(f"× No match for type <range_this>: {date_str}")
                start_dt = None

            if start_dt:
                return int(start_dt.timestamp()), int(now.timestamp())
            else:
                return 0, 0
        else:
            logger.warn(f"× No match for type <range_this>: {date_str}")

        return 0, 0

    def get_date_ts_range_of_last(self, date_str: str) -> tuple[int, int]:
        now = get_now()
        match = re.match(self.REP_RANGE_LAST, date_str)
        if match:
            if match.group("last_year"):
                start_dt = tcdatetime(now.year - 1, 1, 1)
                end_dt = tcdatetime(now.year - 1, 12, 31, 23, 59, 59)
            elif match.group("last_month"):
                if now.month == 1:
                    start_dt = tcdatetime(now.year - 1, 12, 1)
                else:
                    start_dt = tcdatetime(now.year, now.month - 1, 1)
                month_days = monthrange(start_dt.year, start_dt.month)[1]
                end_dt = (
                    start_dt + timedelta(days=month_days) - timedelta(milliseconds=1)
                )
            elif match.group("last_week"):
                start_dt = now - timedelta(days=now.weekday() + 7)
                start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                end_dt = start_dt + timedelta(days=7) - timedelta(milliseconds=1)
            elif match.group("last_day"):
                start_dt = now - timedelta(days=1)
                start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                end_dt = start_dt + timedelta(days=1) - timedelta(milliseconds=1)
            elif match.group("last_hour"):
                start_dt = now - timedelta(hours=1)
                start_dt = start_dt.replace(minute=0, second=0, microsecond=0)
                end_dt = start_dt + timedelta(hours=1) - timedelta(milliseconds=1)
            else:
                logger.warn(f"× No match for type <range_last>: {date_str}")
                start_dt = None

            if start_dt:
                return int(start_dt.timestamp()), int(end_dt.timestamp())
            else:
                return 0, 0
        else:
            logger.warn(f"× No match for type <range_last>: {date_str}")

        return 0, 0

    def get_date_ts_range_of_dist(self, date_str: str) -> tuple[int, int]:
        now = get_now()

        match = re.match(self.REP_RANGE_DIST, date_str)
        if match:
            if match.group("n_years"):
                n = int(match.group("year_n"))
                start_dt = tcdatetime(
                    now.year - n, now.month, now.day, now.hour, now.minute, now.second
                )
            elif match.group("n_months"):
                n = int(match.group("month_n"))
                start_year = now.year - (n + 12 - now.month) // 12
                start_month = ((now.month - n) % 12) or 12
                start_dt = tcdatetime(
                    start_year, start_month, now.day, now.hour, now.minute, now.second
                )
            elif match.group("n_weeks"):
                n = int(match.group("week_n"))
                start_dt = now - timedelta(weeks=n)
            elif match.group("n_days"):
                n = int(match.group("day_n"))
                start_dt = now - timedelta(days=n)
            elif match.group("n_hours"):
                n = int(match.group("hour_n"))
                start_dt = now - timedelta(hours=n)
            else:
                logger.warn(f"× No match for type <range_dist>: {date_str}")
                start_dt = None

            if start_dt:
                return int(start_dt.timestamp()), int(start_dt.timestamp())
            else:
                return 0, 0
        else:
            logger.warn(f"× No match for type <range_dist>: {date_str}")

        return 0, 0

    def get_date_ts_range_of_past(self, date_str: str) -> tuple[int, int]:
        match = re.match(self.REP_RANGE_PAST, date_str)
        if match:
            if match.group("past_year"):
                return self.get_date_ts_range_of_dist("1 year")
            elif match.group("past_month"):
                return self.get_date_ts_range_of_dist("1 month")
            elif match.group("past_week"):
                return self.get_date_ts_range_of_dist("1 week")
            elif match.group("past_day"):
                return self.get_date_ts_range_of_dist("1 day")
            elif match.group("past_hour"):
                return self.get_date_ts_range_of_dist("1 hour")
            else:
                logger.warn(f"× No match for type <range_past>: {date_str}")
                return 0, 0

        return 0, 0

    def op_val_to_es_dict(self, op: str, val: str, use_date_str: bool = False) -> dict:
        """
        Examples:
        - {'field':'date', 'field_type':'date', 'op':'>', 'val':'2024','val_type':'value'}
            -> {"pubdate": {"gt": ts of (end of 2024)}}
        - {'field':'date', 'field_type':'date', 'op':'<=', 'val':'2024-11','val_type':'value'}
            -> {"pubdate": {"gte": ts of (start of 2024-11)}}
        - {'field':'date', 'field_type':'date', 'op':'=', 'val':'2024-11','val_type':'value'}
            -> {"pubdate": {"gte": ts of (start of 2024-11), "lte": ts of (end of 2024-11)}}
        """
        range_type, start_ts, end_ts = self.get_date_ts_range(val)
        now_ts = get_now_ts()
        if op in OP_MAP.keys():
            op_str = OP_MAP[op]
        res = {}
        res_val = {}
        if range_type == "dist":
            if op not in OP_MAP.keys():
                res_val = {"gte": start_ts, "lte": now_ts}
            else:
                op_range = {
                    "gt": {"lt": start_ts},
                    "lt": {"gt": start_ts},
                    "gte": {"lte": start_ts},
                    "lte": {"gte": start_ts},
                }
                res_val = op_range[op_str]
        elif range_type in ["date", "this", "last", "past"]:
            if op not in OP_MAP.keys():
                res_val = {"gte": start_ts, "lte": end_ts}
            else:
                op_range = {
                    "gt": {"gt": end_ts},
                    "lt": {"lt": start_ts},
                    "gte": {"gte": start_ts},
                    "lte": {"lte": end_ts},
                }
                res_val = op_range[op_str]
        else:
            logger.warn(f"× No matching date range type for: {val}")
            return {}

        res = {"pubdate": res_val}
        if use_date_str:
            res = {"pubdate_str": {k: ts_to_str(v) for k, v in res_val.items()}}
        return res

    def range_val_to_es_dict(
        self, lb: str, lval: str, rval: str, rb: str, use_date_str: bool = False
    ) -> dict:
        """
        Rules:
            * [: >= lval_start
            * (: >  lval_end
            * ]: <= rval_end
            * ): <  rval_start
        """
        res = {}
        res_val = {}

        if lval:
            lval_type, lval_start, lval_end = self.get_date_ts_range(lval)
            lb_str = BRACKET_MAP[lb]
        else:
            lval_type, lval_start, lval_end = None, None, None

        if rval:
            rval_type, rval_start, rval_end = self.get_date_ts_range(rval)
            rb_str = BRACKET_MAP[rb]
        else:
            rval_type, rval_start, rval_end = None, None, None

        # ignore order of lval and rval for "dist" val_type
        if (
            lval_type == "dist"
            and rval_type == "dist"
            and lval_start is not None
            and rval_start is not None
        ):
            if lval_start < rval_start:
                lval_start, rval_start = rval_start, lval_start
                lval_end, rval_end = rval_end, lval_end

        if lval_start is not None and lb_str:
            if lb_str == "gt":
                if lval_type == "dist":
                    res_val["lt"] = lval_end
                else:
                    res_val["gt"] = lval_end
            elif lb_str == "gte":
                if lval_type == "dist":
                    res_val["lte"] = lval_start
                else:
                    res_val["gte"] = lval_start
            else:
                logger.warn(f"× No matching lb: {lb_str}")

        if rval_start is not None and rb_str:
            if rb_str == "lt":
                if rval_type == "dist":
                    res_val["gt"] = rval_start
                else:
                    res_val["lt"] = rval_start
            elif rb_str == "lte":
                if rval_type == "dist":
                    res_val["gte"] = rval_end
                else:
                    res_val["lte"] = rval_end
            else:
                logger.warn(f"× No matching rb: {rb_str}")

        res = {"pubdate": res_val}
        if use_date_str:
            res = {"pubdate_str": {k: ts_to_str(v) for k, v in res_val.items()}}

        return res

    def filter_dict_to_es_dict(
        self, filter_dict: dict, use_date_str: bool = False
    ) -> dict:
        res = {}
        if filter_dict["val_type"] == "value":
            res = self.op_val_to_es_dict(
                filter_dict["op"], filter_dict["val"], use_date_str=use_date_str
            )
        elif filter_dict["val_type"] == "range":
            res = self.range_val_to_es_dict(
                filter_dict["lb"],
                filter_dict["lval"],
                filter_dict["rval"],
                filter_dict["rb"],
                use_date_str=use_date_str,
            )
        elif filter_dict["val_type"] == "list":
            logger.warn(f"× Not implemented val type: {filter_dict['val_type']}")
        else:
            logger.warn(f"× No matching val type: {filter_dict['val_type']}")
        return res


if __name__ == "__main__":
    from converters.field.test import test_date_strs

    converter = DateFieldConverter()
    for date_str in test_date_strs:
        logger.note(f"{date_str}")
        range_type, start_ts, end_ts = converter.get_date_ts_range(date_str)
        start_str = ts_to_str(start_ts)
        end_str = ts_to_str(end_ts)
        logger.mesg(f"> type : {range_type}")
        logger.success(f"> start: {start_str}")
        logger.success(f"> end  : {end_str}")

    # python -m converters.field.date
