"""Tool definitions in OpenAI function calling format.

These schemas are sent to the LLM as the `tools` parameter so it can
natively call functions to search videos and check authors.
"""

SEARCH_VIDEOS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_videos",
        "description": (
            "搜索B站视频。支持一次传入多个搜索语句，并行搜索并合并结果。"
            "搜索语句支持关键词和DSL过滤器。"
            "过滤器以冒号':'开头，格式为 :<字段><操作符><值>。"
            "常用过滤器：:view>=1w(播放量) :date<=7d(日期) :user=名字(UP主) :t>5m(时长)。"
            "示例queries：['黑神话 :view>=1w', ':user=影视飓风 :date<=7d']。"
            "当需要搜索不同关键词或不同UP主时，应传入多个query以提高效率。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "搜索语句列表。每个语句可包含关键词和/或DSL过滤器。"
                        "关键词用空格分隔，过滤器以冒号':'起始。"
                        "示例: ['黑神话 :view>=1w', ':user=影视飓风 :date<=7d']"
                    ),
                },
            },
            "required": ["queries"],
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

READ_SPEC_TOOL = {
    "type": "function",
    "function": {
        "name": "read_spec",
        "description": (
            "读取搜索引擎的完整规格文档。"
            "系统提示中已包含常用DSL语法速查，大部分查询不需要调用此工具。"
            "仅在需要查阅完整语法细节（如范围过滤器、搜索模式等高级用法）时使用。"
            "可用文档: search_syntax"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "文档名称。可选值: search_syntax",
                    "enum": ["search_syntax"],
                },
            },
            "required": ["name"],
        },
    },
}

# Default tool definitions for the chat handler.
# read_spec is available but not in the default list since DSL syntax
# is already inline in the system prompt.
TOOL_DEFINITIONS = [SEARCH_VIDEOS_TOOL, CHECK_AUTHOR_TOOL]
