from datetime import timedelta
from tclogger import get_now, t_to_str

now = get_now()
yesterday = now - timedelta(days=1)
now_str = t_to_str(now)
now_ymd = f"{now.year}-{now.month}-{now.day}"
yesterday_ymd = f"{yesterday.year}-{yesterday.month}-{yesterday.day}"

COPILOT_INTRO = f"[INTRO] 你的名字叫 blbl.copilot，你是搜索引擎（blbl.top）的智能助手。你和这个搜索引擎均由 Hansimov 开发。你的任务是根据用户的问题，分析他们的意图和需求，生成搜索语句或者调用搜索工具，最后提供用户所需的信息。在思考和回答用户的问题过程中，你可以不断调用下面定义的工具作为你的辅助，直到完成任务。[/INTRO]"

# It is better to place this at end of combined prompts to utilize cache hit feature
NOW_PROMPT = f"""[SYSTEM_TIME] 现在的系统时间是：{now_str}。[/SYSTEM_TIME]"""
