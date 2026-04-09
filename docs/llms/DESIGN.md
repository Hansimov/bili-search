# LLMs 设计说明

## 目标

- 用“意图路由 + 多模型编排”替换旧的单体大提示词 + 单大模型循环。
- 固定 `deepseek` 为大模型，固定 `qwen3.5-4b` 为小模型，并为后续模型扩展保留统一入口。
- 默认只加载当前任务最需要的提示资产，避免把整套长提示无差别塞进上下文。
- 避免把大块原始工具结果直接回灌给大模型，只提供必要摘要、链接标识和 `result_id`。
- 保持现有后端 OpenAI 兼容接口和前端 `tool_events` 契约基本不变。
- 明确禁止回到“写几个关键词 / 正则就决定路由”的实现方式。意图与 planning 的决策必须优先落在 taxonomy、examples、intent signals 和 execution signals 上。

## 结构重排

- `llms/contracts/`
  - 统一承载 `IntentProfile`、`ToolCallRequest`、`OrchestrationResult` 等共享数据契约。
- `llms/models/`
  - `client.py`：LLM API client compatibility layer。
  - `registry.py`：模型注册表和 large/small client 初始化。
- `llms/runtime/`
  - `cli.py`：本地调试和单次查询入口。
- `llms/intent/`
  - `taxonomy.py`：定义 final target、task mode 和 facet taxonomy，以及 example-driven matcher。
  - `signals.py`：集中管理 topic/entity 抽取、ambiguity/complexity、term normalization 等 signal 规则。
  - `prompt_selection.py`：按 intent signals 选择 prompt assets。
  - `classifier.py`：只负责拼装 taxonomy ranking + intent signals，构建 `IntentProfile`。
- `llms/planning/`
  - `pipeline.py`：根据 intent 和工具执行信号选择 planning plugins。
  - `mixin.py`：保留具体的 rewrite / bootstrap 实现。
  - `owner_resolution.py`：承载作者名消歧和 owner resolution mixin，不再挂在 `chat/` 下。
- `llms/orchestration/`
  - `policies.py`：coverage、nudge、final answer 收口策略。
  - `engine.py`：`ChatOrchestrator` 的真实执行核心。
  - `tool_markup.py`：统一 XML / DSML 工具命令解析与内容清洗。
  - `result_store.py`：统一 result store、摘要和 inspection 视图。
- `llms/chat/`
  - `handler.py`：保留对外 OpenAI 兼容 API 和 SSE 封装。
  - 其余旧模块逐步退化为兼容壳，不再承载核心实现。
- 旧路径 `llms/config.py`、`llms/llm_client.py`、`llms/protocol.py`、`llms/cli.py`、`llms/routing.py`、`llms/routing_rules.py`、`llms/chat/policies.py`、`llms/chat/tool_planning.py`、`llms/chat/orchestrator.py`、`llms/chat/owner_resolution.py` 只保留兼容壳，不再承载核心逻辑。

## 核心组件

### 模型注册表

- `llms/models/registry.py` 中的 `ModelRegistry` 负责声明和公开当前可用模型。
- 当前固定默认值：
  - 大模型：`deepseek`
  - 小模型：`qwen3.5-4b`
- `llms/models/client.py` 提供 `LLMClient` / `ChatResponse` / `ToolCall` compatibility layer。
- `create_model_clients(...)` 会一次性构造大小模型 client，并把公开能力暴露给运行时和测试脚本。

### 意图路由

- `llms/intent/classifier.py` 会基于最新一轮用户输入和近两轮用户上下文构建 `IntentProfile`。
- `llms/intent/taxonomy.py` 用 label description + examples 做 similarity matching，而不是靠 route-specific regex 命中。
- `llms/intent/signals.py` 把 topic/entity 提取、ambiguity/complexity 估计、term normalization 和 route flags 集中成结构化 signal 规则，避免这些阈值散落在 classifier 主流程里。
- `llms/intent/prompt_selection.py` 会根据 `needs_keyword_expansion`、`needs_term_normalization`、`needs_owner_resolution` 等 signals 装配 prompt assets。
- 当前会识别和估计的核心字段包括：
  - 最终目标：`videos / owners / relations / external / mixed`
  - 任务模式：`exploration / lookup_entity / collect_compare / repeat / known_item`
  - 歧义度与复杂度
  - 路由标记，例如：关键词扩展、term normalization、作者名消歧、站外检索
- 同一个 `IntentProfile` 同时驱动：
  - 提示资产选择
  - planner / response / delegate 模型选择
  - 调试输出与 live 回归观察

### 分级提示资产

- `llms/prompts/assets.py` 把提示拆成 `brief / detailed / examples` 三层。
- `llms/prompts/copilot.py` 会按当前意图拼装系统提示词，主要来源包括：
  - 基础资产
  - route 级资产
  - 可选的工具级资产
- 默认只装入最小 `brief` 集合。
- 当模型明确需要更细规则或示例时，才通过内部工具 `read_prompt_assets` 增量读取更高层级内容。

### 编排器

- `llms/orchestration/engine.py` 是新的执行核心，旧的 `llms/chat/orchestrator.py` 仅保留 re-export 兼容层。
- `llms/orchestration/policies.py` 提供 coverage / nudge / fallback policy，避免把这类逻辑继续堆回 orchestrator 的 if/else 中。
- `llms/orchestration/tool_markup.py` 负责 function-calling / XML fallback 共用的命令解析与清洗，避免 handler 和 engine 双份实现。
- 它负责：
  - 选择 planner / response / delegate 模型
  - 优先走 OpenAI 风格 function calling
  - 在必要时回退到 XML 工具命令兼容路径
  - 执行内部编排工具：
    - `read_prompt_assets`
    - `inspect_tool_result`
    - `run_small_llm_task`
- 当前模型选择策略额外收紧为：
  - 只要 planner / response 阶段需要真实工具编排，就优先使用大模型
  - 小模型主要保留给 `run_small_llm_task` 这类窄任务委托
- 对 mixed 场景增加了额外保护：
  - 当 `search_google + search_videos` 这一组终局工具已经完整执行过一轮后，不允许再重复同组扩搜
  - 若还缺细节，只能走 `inspect_tool_result`
  - 否则直接进入最终回答阶段
- 对 live 里常见的 0-hit 场景，编排器还会做两类兜底：
  - mixed 任务里，若站内视频连续 0 hit，会明确引导改用 `search_google + site:bilibili.com/video`
  - 显式视频任务里，若站内查询 0 hit 且用户实体明确，也会引导走同样的 B 站站点搜索兜底
- 对纯官方查询（`final_target=external`）增加了保护：
  - 禁止继续扩展到 `search_videos`
  - 一旦拿到一轮足够的官方 Google 结果，就直接收口回答

### 结果隔离

- 每次工具执行都会保存成一个 `ToolExecutionRecord`，并分配稳定的 `result_id`。
- `llms/orchestration/result_store.py` 统一维护 result store、摘要拼装和 `inspect_tool_result` 视图，避免这些逻辑继续堆在 orchestrator 大文件里。
- 默认不会把完整原始结果再塞回模型上下文，而是只提供：
  - 摘要
  - 链接标识，例如 `BV`、`space` 链接、官方来源链接
  - `result_id`
- 只有模型明确调用 `inspect_tool_result` 时，才会读取更细结果。
- 这样可以同时满足两件事：
  - 降低大模型上下文膨胀
  - 保留生成最终链接和引用所需的关键标识

## 端到端流程

1. `ChatHandler` 接收 OpenAI 兼容 `messages`。
2. `llms/intent/classifier.py` 中的 `build_intent_profile(...)` 基于 taxonomy 识别当前请求意图。
3. `build_prompt_selection(...)` 选择最合适的提示资产集合。
4. `ChatOrchestrator` 按阶段选择模型：
   - planner
   - response
   - delegate
5. 外部工具和内部工具执行后，结果进入 result store。
6. 模型默认只看到摘要，而不是整批原始 JSON。
7. `ChatHandler` 再把最终结果包装成现有 completion / SSE 输出格式。

## 前端契约

- `tool_events` 结构保持兼容。
- 每个 tool call 现在可能额外带上：
  - `visibility`
  - `result_id`
  - `summary`
- 内部编排步骤会标记为 `visibility=internal`，前端默认可以隐藏它们，只展示用户真正关心的外部工具调用。

## 调试脚本

- `conda run -n ai python debugs/inspect_llm_routing.py "帮我找最近的 Gemini 2.5 解读视频"`
  - 输出：意图画像、选中的提示资产、planner/response/delegate 模型选择
- `conda run -n ai python debugs/inspect_llm_prompt_selection.py "来点让我开心的视频" --full-prompt`
  - 输出：选中的提示资产、各 section 长度、可选的完整系统提示词
- `conda run -n ai python debugs/inspect_mixed_ascii_owner_query.py`
  - 用 live dev 索引检查混合中英文作者名查询的召回稳定性

## Live 验证

### 直接跑 handler 级 live 回归

推荐命令：

```bash
conda run -n ai python -m tests.llm.test_live_chat \
  --elastic-env-name elastic_dev \
  --test 3 \
  --test 5 \
  --test 8 \
  --test 14 \
  --test 16 \
  --test 17
```

这组 case 覆盖了几类真实任务：

- `3`：官方更新 + B 站解读的 mixed 任务
- `5`：直接找视频
- `8`：多作者对比
- `14`：作者关系追问代表作
- `16`：别名/错写纠正后再搜视频
- `17`：抽象情绪型视频需求

### 当前 live 回归关注点

- mixed 场景下不要反复重跑 `search_google + search_videos`
- 视频回答要尽量给出可点击 BV 链接，而不只是标题
- 作者回答要尽量给出真实 `space.bilibili.com/{mid}` 链接
- 抽象 query 要么先语义展开，要么直接落成可检索的视频 query，避免误走 Google
- 回答中不应泄漏 DSML / function-calling 中间标记
- 若 live 后端出现 relation / auto-constraint 的 `429 circuit_breaking_exception`，优先通过更早收口、少轮次补搜和 Google 站点兜底减压，而不是继续堆更多 planner 迭代

## 验证命令

- llms 单测：
  - `pytest tests/llm -q`
- 定向 live 回归：
  - `conda run -n ai python -m tests.llm.test_live_chat --elastic-env-name elastic_dev --test 3 --test 5 --test 14`
- 后端全量：
  - `pytest -q`

## 当前状态

- llms 核心架构已经切换到多模型编排。
- intent / planning / orchestration 已拆成独立子包，`chat/` 正在收缩为对外 API / compat 层。
- 路由主干已经改成 taxonomy + similarity matcher，避免继续新增 route-specific regex。
- planning plugin 不再默认全量串行执行，而是先由 `llms/planning/pipeline.py` 根据 intent 和结果信号做 selector。
- 提示资产、结果隔离、前端 tool event 兼容层已落地。
- live 回归入口已经改成真正使用大小模型 client，而不是旧的单模型入口。
- 已验证通过的 live 类型至少包括：
  - mixed 官方更新 + B 站解读
  - 显式视频搜索
  - official-only 查询
  - official-only follow-up 查询
- 当前仍受 live 后端 `429 circuit_breaking_exception` 影响较明显的链路，主要集中在 relation 扩展和抽象语义展开类任务。
- 设计文档现在以中文为主，便于后续直接作为团队维护入口。