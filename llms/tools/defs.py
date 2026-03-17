"""Tool definitions in OpenAI function calling format."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_SEARCH_CAPABILITIES = {
    "service_type": "unknown",
    "default_query_mode": "wv",
    "rerank_query_mode": "vwr",
    "supports_multi_query": True,
    "supports_author_check": True,
    "docs": ["search_syntax"],
}


def _merge_capabilities(capabilities: dict | None = None) -> dict:
    merged = dict(DEFAULT_SEARCH_CAPABILITIES)
    merged.update(capabilities or {})
    return merged


def build_search_videos_tool(capabilities: dict | None = None) -> dict:
    caps = _merge_capabilities(capabilities)
    default_mode = caps.get("default_query_mode", "wv")
    rerank_mode = caps.get("rerank_query_mode", "vwr")
    multi_query_text = (
        "支持一次传入多个搜索语句，并行搜索并合并结果。"
        if caps.get("supports_multi_query", True)
        else "当前服务按单语句执行，但仍接受 queries 数组格式。"
    )
    return {
        "type": "function",
        "function": {
            "name": "search_videos",
            "description": (
                "搜索B站视频。"
                f"{multi_query_text}"
                "搜索语句支持关键词和DSL过滤器。"
                "过滤器以冒号':'开头，格式为 :<字段><操作符><值>。"
                "常用过滤器：:view>=1w(播放量) :date<=7d(日期) :user=名字(UP主) :t>5m(时长)。"
                f"搜索模式：默认q={default_mode}，需要更高相关性时添加q={rerank_mode}。"
                f"示例queries：['黑神话 :view>=1w q={rerank_mode}', ':user=影视飓风 :date<=7d']。"
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
                            f"需要高相关性时在末尾添加 q={rerank_mode}。"
                        ),
                    },
                },
                "required": ["queries"],
            },
        },
    }


def build_check_author_tool(capabilities: dict | None = None) -> dict:
    caps = _merge_capabilities(capabilities)
    description = (
        "检查用户输入是否匹配B站视频作者（UP主）的昵称。"
        "返回匹配的关键词高亮信息和相关作者列表（含UID和占比）。"
        "用于在搜索时判断用户意图：是关键词搜索还是特定UP主搜索。"
        "建议与 search_videos 在同一轮并行调用，避免单独占用一轮。"
    )
    if not caps.get("supports_author_check", True):
        description += " 当前服务不支持作者检查时，应降级为普通视频搜索。"
    return {
        "type": "function",
        "function": {
            "name": "check_author",
            "description": description,
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


def build_read_spec_tool(capabilities: dict | None = None) -> dict:
    caps = _merge_capabilities(capabilities)
    docs = list(caps.get("docs") or ["search_syntax"])
    return {
        "type": "function",
        "function": {
            "name": "read_spec",
            "description": (
                "读取搜索引擎的完整规格文档。"
                "系统提示中已包含常用DSL语法速查，大部分查询不需要调用此工具。"
                "仅在需要查阅完整语法细节（如范围过滤器、搜索模式等高级用法）时使用。"
                f"可用文档: {', '.join(docs)}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "文档名称。",
                        "enum": docs,
                    },
                },
                "required": ["name"],
            },
        },
    }


def build_tool_definitions(
    capabilities: dict | None = None,
    *,
    include_read_spec: bool = False,
) -> list[dict]:
    tools = [
        build_search_videos_tool(capabilities),
        build_check_author_tool(capabilities),
    ]
    if include_read_spec:
        tools.append(build_read_spec_tool(capabilities))
    return tools


TOOL_DEFINITIONS = build_tool_definitions()
SEARCH_VIDEOS_TOOL = deepcopy(TOOL_DEFINITIONS[0])
CHECK_AUTHOR_TOOL = deepcopy(TOOL_DEFINITIONS[1])
READ_SPEC_TOOL = build_read_spec_tool()
