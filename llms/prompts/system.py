from datetime import timedelta
from tclogger import get_now, t_to_str

now = get_now()
yesterday = now - timedelta(days=1)
now_str = t_to_str(now)
now_ymd = f"{now.year}-{now.month}-{now.day}"
yesterday_ymd = f"{yesterday.year}-{yesterday.month}-{yesterday.day}"

# It is better to place this at end of combined prompts to utilize cache hit feature
NOW_PROMPT = f"""[SYSTEM_TIME] 现在的系统时间是：{now_str}。[/SYSTEM_TIME]"""

TODAY_PROMPT = f"""[SYSTEM_DATE] 现在的系统日期是：{now_ymd}。[/SYSTEM_DATE]"""
