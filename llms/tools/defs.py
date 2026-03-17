"""Tool definitions in OpenAI function calling format."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_SEARCH_CAPABILITIES = {
    "service_type": "unknown",
    "default_query_mode": "wv",
    "rerank_query_mode": "vwr",
    "supports_multi_query": True,
    "supports_author_check": False,
    "supports_google_search": False,
    "relation_endpoints": [],
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


def build_search_google_tool(capabilities: dict | None = None) -> dict:
    caps = _merge_capabilities(capabilities)
    description = (
        "搜索站外网页信息，用于补充B站内搜索无法直接回答的背景知识、新闻、产品信息或跨站事实。"
        "优先在需要外部事实核对时使用，而不是替代 B 站视频搜索。"
    )
    return {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的网页查询语句",
                    },
                },
                "required": ["query"],
            },
        },
    }


def build_relation_tool(
    name: str,
    description: str,
    properties: dict,
    required: list[str],
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
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
    ]
    if _merge_capabilities(capabilities).get("supports_google_search", False):
        tools.append(build_search_google_tool(capabilities))
    relation_endpoints = set(
        _merge_capabilities(capabilities).get("relation_endpoints") or []
    )
    if "related_tokens_by_tokens" in relation_endpoints:
        tools.append(
            build_relation_tool(
                "related_tokens_by_tokens",
                "基于给定文本寻找相关联的 token 补全、主题词或纠错候选。适合补全关键词、发现相关话题词。",
                {
                    "text": {"type": "string", "description": "输入文本"},
                    "mode": {
                        "type": "string",
                        "enum": [
                            "auto",
                            "prefix",
                            "associate",
                            "next_token",
                            "correction",
                        ],
                        "description": "关系模式，默认 auto",
                    },
                    "size": {
                        "type": "integer",
                        "description": "返回候选数量",
                        "default": 8,
                    },
                },
                ["text"],
            )
        )
    if "related_owners_by_tokens" in relation_endpoints:
        tools.append(
            build_relation_tool(
                "related_owners_by_tokens",
                "根据话题文本寻找相关 UP 主。适合找某个领域、事件或作品下活跃的创作者。",
                {
                    "text": {"type": "string", "description": "输入话题文本"},
                    "size": {
                        "type": "integer",
                        "description": "返回作者数量",
                        "default": 8,
                    },
                },
                ["text"],
            )
        )
    for endpoint, description, key_name, key_desc in [
        (
            "related_videos_by_videos",
            "根据种子视频找相似或相关视频。",
            "bvids",
            "种子视频 BV 号数组",
        ),
        (
            "related_owners_by_videos",
            "根据种子视频找相关作者。",
            "bvids",
            "种子视频 BV 号数组",
        ),
        (
            "related_videos_by_owners",
            "根据种子作者找相关视频。",
            "mids",
            "种子作者 mid 数组",
        ),
        (
            "related_owners_by_owners",
            "根据种子作者找相关作者。",
            "mids",
            "种子作者 mid 数组",
        ),
    ]:
        if endpoint in relation_endpoints:
            tools.append(
                build_relation_tool(
                    endpoint,
                    description,
                    {
                        key_name: {
                            "type": "array",
                            "items": {
                                "type": "string" if key_name == "bvids" else "integer"
                            },
                            "description": key_desc,
                        },
                        "size": {
                            "type": "integer",
                            "description": "返回候选数量",
                            "default": 10,
                        },
                    },
                    [key_name],
                )
            )
    if include_read_spec:
        tools.append(build_read_spec_tool(capabilities))
    return tools


TOOL_DEFINITIONS = build_tool_definitions()
SEARCH_VIDEOS_TOOL = deepcopy(TOOL_DEFINITIONS[0])
READ_SPEC_TOOL = build_read_spec_tool()
