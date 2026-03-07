# Owner Semantic Upgrade, CPU-First

## 目标

当前 owner 检索已经补上了 route-aware query handling，但语义抽象仍然偏弱，尤其是以下几类 query：

1. name 和 domain 共享部分词片时，容易互相污染。
2. 长尾 phrase 的字面约束变强了，但泛化仍然主要依赖 BM25。
3. 新领域、新梗、新说法出现时，纯词面特征更新慢。

这里的目标不是直接上一个 GPU 依赖很重的 dense 检索系统，而是设计一条 CPU 更擅长、可以渐进接入生产的语义升级路径。

## 方法论

建议把 owner 语义建模拆成三层：

1. 全库预训练的 sparse semantic encoder
2. 面向 name/domain/phrase 的对比学习微调
3. 在线 CPU 友好的 hybrid retrieval + rerank

### 1. 全库预训练

训练对象不是视频级 doc，而是 owner 级语义单元。每个 owner 形成一个聚合语义包：

1. `name`
2. `top_tags`
3. `topic_phrases`
4. `domain_text`
5. `mentioned_names`

预训练输入可以采用字符 `n-gram` + subword hashing，目标不是生成 dense embedding，而是学习一个稀疏语义权重空间：

1. 输入编码：char bigram/trigram + 子词 hash bucket
2. 主干模型：`FTRL / SGDClassifier / linear ranking / SPLADE-like sparse objective`
3. 训练任务：遮盖恢复、邻域区分、owner-profile 对齐

这样做的原因：

1. CPU 上训练和推理都便宜。
2. 稀疏权重天然适合倒排和 ES 体系。
3. 可以和当前 `domain_text`、`topic_phrases` 直接拼接，不需要先大改线上架构。

### 2. 对比学习微调

正样本不只来自人工标注，还来自全库结构信号：

1. 同 owner 的 `name <-> domain_text`
2. 同 owner 的 `topic_phrases <-> sample title phrase clusters`
3. 同领域高重合 owner 对
4. query log 中点击或停留好的 query-owner 对

负样本分三类：

1. 头部 hard negative：高影响力但领域无关的 owner
2. 近邻 hard negative：top_tags 有交集但 phrase 不同的 owner
3. 同名/近名 negative：名字相近但领域不同的 owner

训练目标建议采用 margin-based pairwise ranking 或 in-batch contrastive loss，但输出仍约束为 sparse lexical expansion 权重，而不是纯 dense 向量。

## 线上结构

线上不要直接切到单路语义召回，而是四段式：

1. Route detection
2. BM25 / exact / phrase strict first-stage retrieval
3. Sparse semantic expansion rerank
4. 现有 influence / quality / activity 融合

建议新增一个离线字段，例如 `semantic_terms`，只保留 top weighted sparse terms：

1. 每个 owner 保留 64 到 128 个语义展开词
2. 每个 term 带离散化权重或 rank bucket
3. 继续存到 ES 倒排字段，避免在线 dense ANN 依赖

这样 CPU 推理时只需要：

1. 把 query 编码成少量 sparse terms
2. 和 `semantic_terms` 做检索或 rerank
3. 再与五大维度分数融合

## 训练与评测

训练集不应该只看热门 query。至少拆成三组：

1. `head-name`: 明确作者名，目标是强 precision
2. `head-domain`: 热门领域，目标是 recall + quality
3. `tail-phrase`: 长尾整句，目标是 phrase robustness

另外建议单独维护一组 `hard-negative` panel：

1. query 是明确的 creator/domain intent
2. top-k 中禁止出现一组已知高影响力但无关的 owner
3. 单独统计 `forbidden_name` 通过率，而不只看普通 Hit@k

和当前新增的 query panel 对齐，离线指标建议至少看：

1. Hit@1 / Hit@3
2. MRR
3. 头部与长尾分桶 pass rate
4. 高影响力误召回率

## 生产接入顺序

建议按下面顺序集成，避免一次性替换：

1. 先离线生成 `semantic_terms`，不改线上排序。
2. 再在 `phrase` route 上加入 semantic rerank，对长尾 query 先试点。
3. 稳定后把 semantic rerank 受控扩到 tail-like 的 `domain + quality/activity` 查询。
4. 最后才考虑是否需要额外的 ANN 或 dense 双塔。

## 为什么这条路更适合 CPU

1. 稀疏编码 + 线性模型对 CPU 更友好。
2. 训练可分片、可增量，不要求大 batch GPU。
3. 在线查询仍基于倒排，运维成本低。
4. 可以和现有 ES schema 渐进兼容，而不是重做整套检索栈。