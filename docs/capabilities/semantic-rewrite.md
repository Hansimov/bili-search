# 语义改写与意图识别架构说明

## 范围

本文说明本轮重构后 bili-search 与 es-tok 共享的语义改写、意图识别与语义状态架构。

核心目标不是继续在代码里堆局部常量和特判，而是把确定性知识迁移到可版本化的资产文件，把动态语义证据迁移到索引期增量维护的语义快照中。

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

默认的资源文件规则仍由 `QueryExpansionTuning` 提供，但它现在和未来的快照型、混合型语义提供者共享同一套存储契约。

### 2. 索引期增量 semantic snapshot

es-tok 目前已经接入第一版索引期增量语义快照链路：

- `SemanticSnapshotManager`
- `SemanticSnapshotIndexListener`
- `IndexSemanticExpansionSnapshot`

当前行为如下：

1. 成功的 `postIndex` 事件会读取本次写入文档的 `_source`。
2. 第一阶段只使用 `title`、`tags`、`rtags` 三类字段构造语义档案。
3. 每个文档都会生成一个基于归一化 `title + tags + rtags` 的轻量指纹。
4. 若同一文档再次写入且指纹未变，则跳过快照更新。
5. 若文档内容变更，会先移除旧贡献，再写入新贡献。
6. 成功的删除事件会把该文档对快照的贡献移除。
7. 查询期的 `semantic` 模式会把静态规则扩展和运行中快照产生的 `doc_cooccurrence` 扩展合并返回。

### 3. 第一版噪声控制策略

第一版 snapshot 有意把标题中抽出的词更多视为“语义锚点”，而不是无条件的主扩展目标。

这样可以保留有价值的关系，例如：

- `ComfyUI -> 教程`
- `教程 -> 讲解`

同时尽量避免低价值扩展，例如：

- `教程 -> ComfyUI 教程`
- `教程 -> ComfyUI`

## 为什么这样拆分

在本轮重构前，语义知识主要分散在三块互不对齐的面：

- es-tok 的静态 query expansion 调优
- bili-search 本地的 alias rewrite
- searcher 内部的作者意图阈值与正则

这会导致一个面改了，另外两个面不跟进，搜索质量就开始漂移。

现在边界被明确拆开了：

- 确定性规则属于资产文件
- 动态语义证据属于 semantic snapshot store
- 检索和编排代码只消费策略对象和存储对象，不再内嵌规则表

## 本轮验证结论

围绕索引期 semantic snapshot，本轮已经完成一次真实写入验证，结论如下：

1. 直接用 `workers.elastic_videos.commander` 做目标时间窗的大批量回灌时，ES bulk 阶段容易超时，不适合作为窄窗口语义链路的首选验证入口。
2. 更稳定的做法是使用小量真实文档回放：先从 Mongo 中取目标时间窗文档，再少量写入 `bili_videos_dev6`，然后立刻对同一时间窗做 semantic probe。
3. 该验证方式已经确认 `postIndex -> semantic snapshot -> semantic query` 这条链路可以真实生效。
4. 在目标窗口 `2026-04-22 12:00:00` 到 `2026-04-22 18:00:00` 的小样本真实回放后，已经观测到新的 `doc_cooccurrence` 扩展，例如：
  - `鬼畜 -> 搞笑`
  - `汽车 -> 汽车知识`
5. semantic relation 探测需要给足超时预算。5 秒会产生假阴性，30 秒可以稳定观察到结果，因此 live probe 不应使用过短超时。

## 下一步

当前实现已经可运行，但仍然只是第一阶段，不是最终形态。后续重点包括：

1. 让 semantic snapshot 支持跨节点重启持久化，而不只依赖重启后的新写流量重新积累。
2. 在 shard 启动时重建已有文档的 snapshot 状态，而不是只依赖新增写入。
3. 扩展文档语义抽取范围，不再局限于 `title/tags/rtags`，尤其要提升长中文标题、多段主题和弱结构文本的建模能力。
4. 把 snapshot 产生的语义证据进一步反馈给作者意图识别和 query planning，而不只是 token 扩展。
5. 继续评估作者意图中剩余的启发式规则，哪些可以从策略资产进一步演进为语料驱动或学习型先验。