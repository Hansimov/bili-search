# 语义改写与意图识别架构说明

## 2026-04-26 当前状态

`bili-search-algo/models/semantics` 和 `es-tok` 的 compact semantic bundle 目前默认禁用。代码、资源格式和接口可以保留，方便后续重新评估或 A/B，但默认 local-dev 和生产链路不应消费该 bundle。

当前约束：

- `bili-search` 不进入 search semantic rewrite 分支；搜索响应只保留 `semantic_rewrite_info.disabled=true` 作为观测字段。
- `es-tok` 未显式设置 `ES_TOK_SEMANTICS_ENABLED=true` 或 `-Des.tok.semantics.enabled=true` 时不加载 semantic bundle。
- 即使请求传入 `mode=semantic`，`es-tok` 也应回退到 `auto`。
- 不在 orchestration 代码中追加自然语言词表、例子或正则来替代 semantic 功能；query 质量由大模型规划阶段负责，执行层只做稳定协议门禁。

## 2026-04-25 历史架构更新

当时主线已经停止使用 `es-tok` 内部的索引期 semantic snapshot。

当时的职责边界是：

- `bili-search-algo/models/semantics` 负责离线生成 compact semantic bundle，并用 group 级 SQLite cursor 记录已处理文档以支持增量更新。
- bundle 中按文件拆分承载 `rewrite / synonym / near_synonym / doc_cooccurrence`，构建期落在 `bili-search-algo/data/semantics/<version>/merged`，插件重载时复制到 `es_tok/semantics/v1/merged`。
- `es-tok` 只负责加载 compact bundle 并在查询期消费，不再在 `postIndex`/`postDelete` 中维护语义状态。加载优先级为 JVM 参数、环境变量、插件目录、开发态相邻仓库、内置兜底资源。
- 配置类资源继续使用 JSON；批量语义数据使用 TSV。`es-tok` 的内置 semantic TSV 只保留空兜底，避免把人工 case rule 混进数据集生成产物。
- 需要 A/B 或回滚时，可以通过 `BILI_SEARCH_SEMANTIC_REWRITE_ENABLED=false`、`BILI_SEARCH_ALIAS_REWRITE_ENABLED=false`、`BILI_SEARCH_RELATION_REWRITE_ENABLED=false`、`BILI_SEARCH_EXACT_RELAX_RETRY_ENABLED=false` 控制 bili-search 侧行为；通过 `ES_TOK_SEMANTICS_ENABLED=false` 或 `-Des.tok.semantics.enabled=false` 让 es-tok 的 `semantic` 模式回退到 `auto`。

因此，下文中所有关于 `SemanticSnapshotManager`、`SemanticSnapshotIndexListener`、`IndexSemanticExpansionSnapshot` 的描述，都属于上一轮方案的历史记录，不再是当前实现。2026-04-26 之后，compact semantic bundle 本身也进入默认禁用状态。

## 范围

本文说明本轮重构后 bili-search 与 es-tok 共享的语义改写、意图识别与语义资产架构。

核心目标不是继续在代码里堆局部常量和特判，而是把确定性知识迁移到可版本化的资产文件，把真实样本文档归纳出的语义证据迁移到离线生成的 bundle 中。

## 哪些规则已经移出代码

### 1. bili-search 的术语别名归一化

- 术语别名不再以内嵌 `_TERM_ALIAS_MAP` 的形式写死在 Python 逻辑里。
- 别名规则统一放在 `llms/intent/assets/term_aliases.json`。
- `llms/intent/semantic_assets.py` 是稳定的加载边界。
- `llms.intent.focus.rewrite_known_term_aliases()` 只负责按 token 感知地匹配和替换，不再持有别名数据本体。

### 2. bili-search 的作者意图策略

- 作者意图相关的阈值、屏蔽词、来源标签规则、模型名模式等，不再散落在 searcher 私有常量中。
- 这些规则现在统一放在 `elastics/videos/assets/owner_intent_policy.json`。
- `elastics/videos/owner_intent_policy.py` 对外提供缓存后的 `OwnerIntentPolicy`，由 `VideoSearcherV2` 消费。
- 这样一来，检索流程仍然保留在 searcher 中，但策略值已经变成了可独立审阅、可单测、可扩展的数据资产。

## es-tok 的语义存储层

### 1. 查询期抽象层

es-tok 的 semantic 扩展不再直接依赖单一的静态调优加载器，而是统一走抽象存储接口：

- `SemanticExpansionStore`
- `SemanticExpansionRule`
- `SemanticQueryExpansionSuggester`

默认实现由 `SemanticArtifactStore` 提供，直接加载 `src/main/resources/tuning/semantic/*.tsv` 下的 compact bundle，并与查询期建议器共享同一套存储契约。

### 2. 历史方案说明

以下类和概念只保留为历史记录，便于对照旧日志、旧报告与旧代码：

- `SemanticSnapshotManager`
- `SemanticSnapshotIndexListener`
- `IndexSemanticExpansionSnapshot`

它们代表的是上一轮“索引期增量 snapshot”方案，当前主线已经不再依赖这条链路。

### 3. 当前离线生成流程

当前语义资产的生成和消费流程是：

1. `bili-search-algo/models/semantics` 从真实文档统计中生成 compact semantic bundle，不在模型代码中注入人工挑选的 query/product seed rule。
2. bundle 按关系类型拆分为四个固定文件：
  - `rewrite.tsv`
  - `synonym.tsv`
  - `near_synonym.tsv`
  - `doc_cooccurrence.tsv`
3. 每个文件都使用紧凑的单行格式：一行对应一个 source term，后续列按 `target weight target weight ...` 展开，不再使用嵌套 JSON。
4. `es-tok` 启动时由 `SemanticArtifactStore` 优先加载插件目录下的 merged bundle，找不到时回退到 `src/main/resources/tuning/semantic/*.tsv`。
5. 查询期仍由 `SemanticQueryExpansionSuggester` 消费统一的 `SemanticExpansionStore` 接口，但底层数据源已经变成离线 bundle，而不是运行中 snapshot。

当前 bundle 中同时承载四类关系：

- `rewrite`：可安全改写的确定性表述
- `synonym`：强同义项
- `near_synonym`：同一词或实体的表面变体，只允许包含关系、数字签名一致，或 ASCII 型号/英文词轻微拼写差异
- `doc_cooccurrence`：来自真实文档共现的动态语义证据

## 为什么这样拆分

在本轮重构前，语义知识主要分散在三块互不对齐的面：

- es-tok 的静态 query expansion 调优
- bili-search 本地的 alias rewrite
- searcher 内部的作者意图阈值与正则

这会导致一个面改了，另外两个面不跟进，搜索质量就开始漂移。

现在边界被明确拆开了：

- 确定性规则属于资产文件
- 真实文档归纳出的语义证据属于离线 semantic bundle
- 检索和编排代码只消费策略对象和存储对象，不再内嵌规则表

## 本轮验证结论

当前这轮验证不再围绕索引期 snapshot，而是围绕“离线 bundle -> es-tok compact artifact store -> bili-search relation client contract”这条链路。

已经完成的关键验证如下：

1. `models.semantics` 的窄测试通过，确认 compact 格式写出、增量去重和 near_synonym 质量门槛稳定。
2. `es-tok` 的 `SemanticArtifactStore` 窄测试、`SemanticQueryExpansionSuggester` 相关测试、`compileJava` 和 `assemble` 通过，确认 Java 侧已经切到 compact bundle，且内置兜底 TSV 会进入插件 jar。
3. `bili-search` 的语义关系调用仍通过 `RelationsClient.related_tokens_by_tokens(mode="semantic")`，响应契约不变，因此 `elastics.videos` 和 `llms` 不需要直接读取离线产物。
4. `data/semantics/v1/merged` 已由 1M Mongo live run 生成，并通过 `SEMANTICS_BUNDLE_PATH=... ./load.sh -a` 复制到开发 ES 插件目录。
5. 开发 ES 重启后，后端和前端 local-dev 链路均已重新验证。
6. live endpoint 验证 `b200 价格` 从严格 `+b200 +价格` 的 1 条结果恢复为普通 `b200 价格` 的 20 条页面结果，证明低召回 retry 修复了型号词+属性词过窄召回问题。

## 本轮二次验证

围绕当前 offline bundle 主线，本轮已经完成小、中、大三档 Mongo live build + merge，并继续保留真实 probe 资产：

1. 小规模 `live_small_stats`：10k docs，active processing 10266.94 docs/s，merge 后 10863 allowed terms、7959 `doc_cooccurrence` rows。
2. 中规模 `live_medium_stats`：200k docs，端到端 10743.04 docs/s，active processing 19011.41 docs/s，merge 后 53081 allowed terms、42568 `doc_cooccurrence` rows。
3. 大规模 `live_large_stats`：1M docs，端到端 18352.53 docs/s，active processing 21551.72 docs/s，merge 后 92453 allowed terms、76632 `doc_cooccurrence` rows，未出现 OOM 或 exit 137。
4. reload 后的 live semantic probe：`debugs/live_case_reports/semantic_probe_live_large_bundle_20260425_after_composable_fix.json`，统计为 `28 docs -> 67 probe terms -> 57 non-empty cases`，返回中可见 `doc_cooccurrence / rewrite / prefix / correction` 等关系类型，同时短 token 的 `doc_cooccurrence` 局部拼接噪声已收敛。

历史调试中还保留了三类真实验证资产，可继续作为下一轮 live probe 输入：

1. 新一轮 `10 x 10` 真实样本集：
  - 输出文件：`debugs/live_case_reports/live_case_corpus_round2_20260424.json`
  - 规模：100 个 case，10 类，每类 10 个，均来自当前 dev index 的真实热门文档。
2. 当前主线下的真实文档 bundle 输入：
  - 输出文件：`debugs/live_case_reports/semantic_docs_merged_real.jsonl`
  - 汇总文件：`debugs/live_case_reports/semantic_docs_merged_real.summary.json`
  - 当前版本由两部分合并而成：
    - 历史 live case corpus 中抽出的种子文档
    - recent-window live probe 中实际回放过的真实文档
3. reload 后的上一轮 live semantic probe 历史输出：
  - 输出文件：`debugs/live_case_reports/semantic_probe_merged_real_post_reload.json`
  - 当前统计：`28 docs -> 47 probe terms -> 44 non-empty cases`

这些资产对下一轮验证有三个用途：

1. 用真实文档继续扩大 `doc_cooccurrence` 覆盖面。
2. 在插件 reload 后复用 probe 输入确认 live ES 返回的 relation type 是否包含 compact bundle 中的关系。
3. 把 residual noise 从“链路是否生效”转移到“哪些高频主题词仍需要继续收敛”这一层面。

## 当前残留问题

虽然当前主链已经打通，但 live probe 仍暴露出两类残留问题：

1. 一批更常见、也更难直接一刀切的高频主题词仍然容易成为 expansion，例如 `生活`、`日常`、`娱乐`、`综艺`、`音乐`。
2. 部分短 ASCII 或标题片段触发的局部替换容易把 `doc_cooccurrence` 噪声拼回长标题；当前 es-tok 已收口为只允许 `rewrite / synonym / near_synonym` 参与局部替换，`doc_cooccurrence` 保留为整词扩展证据。
3. 当前 recent-window Mongo 导出路径在更大窗口上仍然偏慢，因此“更多领域、更多数量”的真实 corpus 构建还需要继续优化导出方式。
4. 下一轮应优先把这些泛化标签沉到策略文件里，区分：
  - 禁止进入 expansion 的 broad tags
  - 只允许作为 anchor 的高频主题
  - 需要基于 doc frequency 或 field role 下调权重的弱语义词

## 下一步

当前实现已经可运行，但仍然只是第一阶段，不是最终形态。后续重点包括：

1. 优化 recent-window 真实文档导出路径，避免更大时间窗下 Mongo 聚合过慢。
2. 扩展文档语义抽取范围，不再局限于 `title/tags/rtags`，尤其要提升长中文标题、多段主题和弱结构文本的建模能力。
3. 把 live probe 中反复出现的高频泛化词继续收敛到策略资产，而不是让它们在代码里零散特判。
4. 继续评估作者意图中剩余的启发式规则，哪些可以从策略资产进一步演进为语料驱动或学习型先验。
5. 把当前 bundle 生成结果和 live probe 统计进一步反馈给 query planning，而不只是 token 扩展。
