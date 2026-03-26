"""Tool definitions in OpenAI function calling format."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_SEARCH_CAPABILITIES = {
    "service_type": "unknown",
    "default_query_mode": "wv",
    "rerank_query_mode": "vwr",
    "supports_multi_query": True,
    "supports_author_check": False,
    "supports_owner_search": False,
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
                "搜索B站视频。这是大多数 B 站问题的主力工具和默认首选。"
                f"{multi_query_text}"
                "搜索语句必须是规范 DSL 搜索语句，而不是用户原话整句。"
                "queries 中优先保留关键实体和检索条件，例如作品名、人物名、作者名、产品名、主题词、时间窗、热度条件。"
                "不要把寒暄、能力提问、纯口语追问、助词、功能词、账号关系问题直接塞进 queries。"
                "搜索语句支持关键词和DSL过滤器。"
                "过滤器以冒号':'开头，格式为 :<字段><操作符><值>。"
                "常用过滤器：:view>=1w(播放量) :date<=7d(日期) :user=名字(UP主) :t>5m(时长)。"
                "如果用户最终要的是具体视频清单、时间线、代表作、热视频、教程/攻略/解读，优先使用本工具。"
                "如果用户给的是抽象偏好、口语标签、黑话或隐含主题，应先把它翻译成更具体、更可检索的主题词，再并行构造多条 queries。"
                "如果请求很短、很抽象、缺稳定实体，不应只保留一条 literal query 直接搜索；优先先做 related_tokens_by_tokens 语义展开，再回到本工具。"
                "如果用户问的是作者资料/关联账号/矩阵号，这通常不是本工具的首选。"
                "如果用户给出的作者词像简称、别名片段、混合中英数字昵称，或者你不确定它到底是不是作者名，"
                "不要直接把原词写成 :user=xxx，先调用 search_owners 再回到本工具。"
                f"搜索模式：默认q={default_mode}（泛搜热门），精确主题匹配用q={rerank_mode}。"
                f"示例queries：['黑神话 :view>=1w :date<=30d', 'Stable Diffusion 教程 q={rerank_mode}']。"
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
                            "每个 query 都应是整理后的检索语句，尽量只保留关键实体和检索条件，而不是用户对话原句。"
                            "如果原始需求比较抽象，可以把它拆成多条并行 query，分别覆盖不同的具体搜索假设。"
                            f"精确主题搜索时在末尾添加 q={rerank_mode}。"
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
        "搜索 Google 网页结果。它有三类主要用途："
        "1) 官网、公告、release notes、跨站事实核对；"
        "2) 当用户需求很模糊、属于深度意图、黑话、口语标签，或者 B 站内暂时缺稳定关键词时，"
        "先做关键词启发，"
        "先用它摸到更像样的主题词、产品名、作者名、标题写法或搜索短语；"
        "3) 当目标仍然是 B 站内容时，可直接在 query 中使用 Google `site:` 语法做辅助站内搜索。"
        "最重要的 site 范围包括：`site:bilibili.com`(全 B 站)、`site:space.bilibili.com`(用户页)、"
        "`site:bilibili.com/video`(视频)、`site:bilibili.com/read`(文章/专栏)。"
        "如果最终目标仍是 B 站视频、作者或 B 站文章，search_google 通常只是侦察/启发层；"
        "拿到线索后通常还应继续调用 search_videos 或 search_owners，而不是停在 Google 结果层。"
        "query 应整理成紧凑搜索短语，可直接包含 site 过滤，不要把整句口语原样塞进去。"
        "如果用户同时要官方更新和 B 站解读，通常应与 search_videos 同轮配合使用。"
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
                        "description": (
                            "要搜索的 Google 查询语句。可以是官网/更新日志查询，也可以是关键词启发查询，"
                            "还可以直接带 site 过滤，例如 `site:bilibili.com/video Gemini CLI MCP`、"
                            "`site:space.bilibili.com ComfyUI 教程`、`site:bilibili.com/read AI coding agent`。"
                        ),
                    },
                    "num": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5",
                        "default": 5,
                    },
                    "lang": {
                        "type": "string",
                        "description": "可选语言代码，例如 zh-CN、en",
                    },
                },
                "required": ["query"],
            },
        },
    }


def build_search_owners_tool(capabilities: dict | None = None) -> dict:
    _merge_capabilities(capabilities)
    return {
        "type": "function",
        "function": {
            "name": "search_owners",
            "description": (
                "搜索作者/UP主。用于作者名查找、别名补全、作者候选发现、关联账号/矩阵号/相近作者扩展。"
                "当用户目标是找作者本身，而不是直接列视频时，优先使用本工具。"
                "若用户明确提到作者名、简称、缩写、混合中英数字昵称，优先用它。"
                "当用户想看‘某作者最近发了什么视频’但作者词本身不够稳定时，也应先用它确认作者，再继续 search_videos。"
                "若用户在问谁在做某个主题内容，也优先用它，而不是把作者问题硬转成视频搜索。"
                "当用户给的是抽象主题、风格偏好或模糊圈内标签时，也可以用 mode=topic 先摸到相关创作者集合。"
                "mode=relation 适合关联账号、矩阵号、主副号、类似作者；mode=topic 适合主题找作者；mode=name 适合名字查作者。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "作者名、别名、主题词或作者线索文本",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "name", "topic", "relation"],
                        "description": "搜索模式，默认 auto",
                        "default": "auto",
                    },
                    "size": {
                        "type": "integer",
                        "description": "返回作者数量",
                        "default": 8,
                    },
                },
                "required": ["text"],
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
    if _merge_capabilities(capabilities).get("supports_owner_search", False):
        tools.append(build_search_owners_tool(capabilities))
    relation_endpoints = set(
        _merge_capabilities(capabilities).get("relation_endpoints") or []
    )
    if "related_tokens_by_tokens" in relation_endpoints:
        tools.append(
            build_relation_tool(
                "related_tokens_by_tokens",
                "辅助工具。基于给定文本寻找相关 token 补全、主题词、语义联想或纠错候选。适用于别名、错写、简称，也适用于口语黑话、抽象标签、隐含主题的展开。对于很短、抽象、缺稳定实体的请求，通常应先调用它做语义展开，而不是直接发起 literal 视频搜索。它不是最终结果来源；拿到候选后通常还应继续调用 search_videos 或 search_owners。",
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
    if include_read_spec:
        tools.append(build_read_spec_tool(capabilities))
    return tools


TOOL_DEFINITIONS = build_tool_definitions()
SEARCH_VIDEOS_TOOL = deepcopy(TOOL_DEFINITIONS[0])
READ_SPEC_TOOL = build_read_spec_tool()
