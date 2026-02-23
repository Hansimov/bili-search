"""Tool definitions in OpenAI function calling format.

These schemas are sent to the LLM as the `tools` parameter so it can
natively call functions to search videos and check authors.
"""

SEARCH_VIDEOS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_videos",
        "description": (
            "搜索B站视频。根据搜索语句返回视频结果列表。"
            "搜索语句支持关键词和DSL过滤器语法（参见 SEARCH_SYNTAX）。"
            "示例查询：'黑神话'、'黑神话 :view>=1w :date<=30d'、':user=影视飓风 :date<=7d'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "搜索语句，可包含关键词和/或DSL过滤器。"
                        "关键词用空格分隔，过滤器以冒号':'起始。"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

CHECK_AUTHOR_TOOL = {
    "type": "function",
    "function": {
        "name": "check_author",
        "description": (
            "检查用户输入是否匹配B站视频作者（UP主）的昵称。"
            "返回匹配的关键词高亮信息和相关作者列表（含UID和占比）。"
            "用于在搜索前判断用户意图：是关键词搜索还是特定UP主搜索。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要检查的名称或关键词文本",
                },
            },
            "required": ["name"],
        },
    },
}

# All tool definitions for the chat handler
TOOL_DEFINITIONS = [SEARCH_VIDEOS_TOOL, CHECK_AUTHOR_TOOL]
