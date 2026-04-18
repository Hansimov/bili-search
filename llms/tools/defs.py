"""Tool definitions in OpenAI function calling format."""

from __future__ import annotations

from copy import deepcopy


_PROMPT_TOOL_EXAMPLES = {
    "search_videos": "<search_videos queries='[\"黑神话 :view>=1w q=vwr\"]'/>",
    "get_video_transcript": "<get_video_transcript video_id='BV1YXZPB1Erc' head_chars='6000' include_segments='true'/>",
    "search_google": "<search_google query='Gemini 2.5 更新 site:bilibili.com/video' num='5'/>",
    "search_owners": "<search_owners text='黑神话悟空' size='8'/>",
    "expand_query": "<expand_query text='康夫UI' mode='correction' size='8'/>",
    "read_spec": "<read_spec name='search_syntax'/>",
    "read_prompt_assets": "<read_prompt_assets tool_names='[\"search_videos\"]' levels='[\"examples\"]'/>",
    "inspect_tool_result": "<inspect_tool_result result_ids='[\"R1\"]' focus='只看最相关 BV' max_items='5'/>",
    "run_small_llm_task": "<run_small_llm_task task='把候选结果压成 3 条要点' result_ids='[\"R1\",\"R2\"]' output_format='短要点'/>",
}


_OWNER_RELATION_ENDPOINTS = {
    "related_owners_by_tokens",
    "related_owners_by_videos",
    "related_owners_by_owners",
}


DEFAULT_SEARCH_CAPABILITIES = {
    "service_type": "unknown",
    "default_query_mode": "wv",
    "rerank_query_mode": "vwr",
    "supports_multi_query": True,
    "supports_author_check": False,
    "supports_owner_search": False,
    "supports_google_search": False,
    "supports_transcript_lookup": False,
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
                "搜索 B 站视频。这是默认终局工具，适合视频、代表作、时间线、热门、教程和解读。"
                f"{multi_query_text}"
                "普通检索时用 queries；若已拿到作者 mids 或种子视频 bvids，也可直接基于种子继续发掘相关视频。"
                "bvids 只用于 discover 模式下从种子继续找相关视频，不是单个 BV 的详情或字幕读取接口。"
                "若用户已给出具体 BV 并要求总结视频内容、字幕或转写，应优先使用 get_video_transcript；"
                "若当前没有该工具，就应直接说明无法读取转写，而不是把 bvids 当成转写接口。"
                "queries 必须是整理后的 DSL 搜索语句，而不是用户原话整句。"
                "优先保留关键实体、主题词、作者、时间窗、热度和时长条件。"
                "作者关系问题通常不该直接用它；作者名不稳时先 search_owners。"
                "抽象偏好、口语标签、黑话或 vibe 请求，通常先 expand_query 再回到本工具。"
                f"搜索模式：默认q={default_mode}（泛搜热门），精确主题匹配用q={rerank_mode}。"
                f"示例queries：['黑神话 :view>=1w :date<=30d', 'Stable Diffusion 教程 q={rerank_mode}']。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "search", "discover"],
                        "description": "普通检索或基于 mids/bvids 的继续发掘，默认 auto",
                        "default": "auto",
                    },
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
                    "bvids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "已知种子视频 BV 号。仅用于从现有视频继续发掘相关视频，不用于直接读取该 BV 的详情或转写。",
                    },
                    "mids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "已知作者 mid 列表。用于从作者继续发掘代表作或相关视频。",
                    },
                    "size": {
                        "type": "integer",
                        "description": "discover 模式下返回候选数量",
                        "default": 10,
                    },
                },
                "required": [],
            },
        },
    }


def build_search_google_tool(capabilities: dict | None = None) -> dict:
    caps = _merge_capabilities(capabilities)
    description = (
        "搜索 Google 网页结果。适合官网、公告、release notes、跨站事实核对，"
        "也适合作为 B 站内容检索前的关键词侦察层。"
        "最重要的 site 范围包括：`site:bilibili.com`(全 B 站)、`site:space.bilibili.com`(用户页)、"
        "`site:bilibili.com/video`(视频)、`site:bilibili.com/read`(文章/专栏)。"
        "作者/UP 主发现优先用 search_owners；它已经内置聚合 `site:space.bilibili.com` 的侦察结果。"
        "使用 `site:` 时，默认把关键词写前面、`site:` 放在最后。"
        "若最终目标仍是 B 站视频、作者或专栏，search_google 通常只是侦察/启发层，拿到线索后应继续调用终局工具。"
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
                            "还可以直接带 site 过滤，例如 `Gemini CLI MCP site:bilibili.com/video`、"
                            "`ComfyUI 教程 site:space.bilibili.com`、`AI coding agent site:bilibili.com/read`。"
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


def build_get_video_transcript_tool(capabilities: dict | None = None) -> dict:
    _merge_capabilities(capabilities)
    return {
        "type": "function",
        "function": {
            "name": "get_video_transcript",
            "description": (
                "获取某个具体 B 站视频的音频转写文本。"
                "适合已知 BV 号/具体视频后，回答“这个视频讲了什么”“帮我总结重点”“视频里说了啥”“给我字幕/转写”之类的问题。"
                "长转写通常先取 head_chars/head_segments，再配合 run_small_llm_task 做归纳。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "视频标识，通常传 BV 号。也可传 bvid 的别名字段。",
                    },
                    "page_index": {
                        "type": "integer",
                        "description": "分 P 页码，默认 1",
                        "default": 1,
                    },
                    "head_chars": {
                        "type": "integer",
                        "description": "只截取前 N 个字符，适合先做快速阅读或摘要。",
                    },
                    "head_segments": {
                        "type": "integer",
                        "description": "只截取前 N 个分段。",
                    },
                    "max_segments": {
                        "type": "integer",
                        "description": "最多返回多少个分段。",
                    },
                    "include_segments": {
                        "type": "boolean",
                        "description": "是否返回分段信息。",
                        "default": False,
                    },
                    "ranges": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选时间范围列表，例如 ['00:30-01:20', '120-180']。",
                    },
                    "force_refresh": {
                        "type": "boolean",
                        "description": "是否强制刷新转写缓存。",
                        "default": False,
                    },
                    "force_audio_refresh": {
                        "type": "boolean",
                        "description": "是否强制刷新音频缓存。",
                        "default": False,
                    },
                    "long_audio": {
                        "type": "string",
                        "enum": ["auto", "always", "never"],
                        "description": "长音频处理策略，默认 auto。",
                    },
                },
                "required": ["video_id"],
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
                "搜索作者/UP主。适合作者名查找、别名补全、作者候选发现、关联账号、矩阵号和相近作者扩展。"
                "作者问题优先用它，不要机械转成视频搜索。"
                "作者最近视频这类问题，如果作者词不稳，应先用它确认作者，再继续 search_videos。"
                "按主题或作者线索找作者时传 text；若已经拿到 bvids 或 mids，也可直接继续做作者关系发掘。"
                "对 text 查询会自动并行聚合名字匹配、主题发现、关系发现，以及 `site:space.bilibili.com` 的 Google 侦察结果；"
                "不需要再猜 mode，也不需要单独补一次 `search_google site:space.bilibili.com`。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "作者名、别名、主题词或作者线索文本。主题找作者时也写在 text 里，不要传 queries。",
                    },
                    "bvids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "已知种子视频 BV 号。用于从视频继续查作者或关联作者。",
                    },
                    "mids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "已知作者 mid 列表。用于扩展关联账号、矩阵号或相近作者。",
                    },
                    "size": {
                        "type": "integer",
                        "description": "返回作者数量。text 聚合模式下表示每个来源的候选上限，最终会做合并去重。",
                        "default": 8,
                    },
                },
                "required": [],
            },
        },
    }


def build_expand_query_tool(capabilities: dict | None = None) -> dict:
    _merge_capabilities(capabilities)
    return {
        "type": "function",
        "function": {
            "name": "expand_query",
            "description": (
                "抽象 query 的语义展开工具。基于给定文本寻找相关 token 补全、主题词、语义联想或纠错候选。"
                "适用于别名、错写、简称，也适用于口语黑话、抽象标签、隐含主题的展开。"
                "对于很短、抽象、缺稳定实体的请求，通常应先调用它做语义展开，而不是直接发起 literal 视频搜索。"
                "它不是最终结果来源；拿到候选后通常还应继续调用 search_videos 或 search_owners。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
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
                        "description": "展开模式，默认 auto",
                    },
                    "size": {
                        "type": "integer",
                        "description": "返回候选数量",
                        "default": 8,
                    },
                },
                "required": ["text"],
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
                "读取搜索引擎的完整规格文档。默认提示只给出精简 DSL 速查；"
                "只有在需要查阅完整语法细节时才调用。"
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
    include_internal: bool = False,
) -> list[dict]:
    caps = _merge_capabilities(capabilities)
    relation_endpoints = set(caps.get("relation_endpoints") or [])
    tools = [build_search_videos_tool(caps)]
    if caps.get("supports_transcript_lookup", False):
        tools.append(build_get_video_transcript_tool(caps))
    if caps.get("supports_google_search", False):
        tools.append(build_search_google_tool(caps))
    if caps.get("supports_owner_search", False) or (
        relation_endpoints & _OWNER_RELATION_ENDPOINTS
    ):
        tools.append(build_search_owners_tool(caps))
    if "related_tokens_by_tokens" in relation_endpoints:
        tools.append(build_expand_query_tool(caps))
    if include_read_spec:
        tools.append(build_read_spec_tool(caps))
    if include_internal:
        tools.extend(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_prompt_assets",
                        "description": "按 tool_name / levels 读取分级提示资产。只有 brief guidance 不够时才调用。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "tool_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "levels": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": ["brief", "detailed", "examples"],
                                    },
                                },
                                "asset_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "inspect_tool_result",
                        "description": "根据 result_ids 读取更细的工具结果摘要，不直接把整批原始结果塞进上下文。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "result_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "focus": {"type": "string"},
                                "max_items": {"type": "integer", "default": 5},
                            },
                            "required": ["result_ids"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "run_small_llm_task",
                        "description": "把窄任务委托给小模型，可并行执行，适合关键词整理、结果压缩、候选对比和证据归纳。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},
                                "context": {"type": "string"},
                                "result_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "output_format": {"type": "string"},
                            },
                            "required": ["task"],
                        },
                    },
                },
            ]
        )
    return tools


def _compact_description(text: str, limit: int = 180) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _parameter_summary(parameters: dict | None) -> str:
    schema = parameters or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    parts = []
    for name, prop in properties.items():
        type_name = str(prop.get("type") or "any")
        marker = "*" if name in required else ""
        parts.append(f"{name}{marker}:{type_name}")
    return ", ".join(parts) if parts else "无参数"


def build_tool_prompt_overview(
    capabilities: dict | None = None,
    *,
    include_read_spec: bool = False,
    include_internal: bool = True,
) -> str:
    tools = build_tool_definitions(
        capabilities,
        include_read_spec=include_read_spec,
        include_internal=include_internal,
    )
    lines = [
        "[TOOL_OVERVIEW]",
        "统一使用 XML 工具协议，不依赖 provider function calling。",
        "如果需要工具，只输出 XML 自闭合标签；一行一个；不要放进 Markdown 代码块。",
        "如果当前消息里输出了工具标签，就不要同时输出最终答案。",
        "数组或对象参数必须写成 JSON，再整体放进单引号属性值里。",
        "多个工具标签可以同轮并列输出，系统会并行执行并把摘要结果回灌给你。",
        "参数说明中带 * 的字段是必填。",
        "可用工具：",
    ]
    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name") or "unknown_tool"
        description = _compact_description(function.get("description") or "")
        params = _parameter_summary(function.get("parameters"))
        example = _PROMPT_TOOL_EXAMPLES.get(name, f"<{name}/>")
        lines.extend(
            [
                f"{name}",
                f"- 用途: {description}",
                f"- 参数: {params}",
                f"- XML 示例: {example}",
            ]
        )
    lines.append("[/TOOL_OVERVIEW]")
    return "\n".join(lines)


TOOL_DEFINITIONS = build_tool_definitions()
SEARCH_VIDEOS_TOOL = deepcopy(TOOL_DEFINITIONS[0])
READ_SPEC_TOOL = build_read_spec_tool()
