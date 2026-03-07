# Owner Profile Retrieval Redesign

## Why

当前 owner-domain coarse 分类实验已经说明两件事：

1. 这条线仍然有价值，但更适合做低成本 sanity check。
2. 它的上限已经越来越接近 pseudo-label 噪声上限，而不是 owner retrieval 的真实上限。

因此，后续主线不再是“继续把 coarse 分类 accuracy 往上拱”，而是转到 retrieval-first：直接围绕真实 query、真实 owner profile、真实 ES index 做优化。

## New Objective

新的核心目标分成三层：

1. Mongo snapshot 更轻，但保留足够 retrieval 信号。
2. ES owner doc 更短，但对 `name / domain / phrase` 三类 query 仍然稳定。
3. 回归不只看总体通过率，而是按维度拆开看退化点。

## Data Path

新的统一链路是：

1. `videos -> owner_profile_snapshots_*`
2. `owner_profile_snapshots_* -> ES owners slim index`
3. `query panels -> live regression -> experiment record`

这里 Mongo snapshot 是训练、调试、索引的统一上游；ES index 只保存 retrieval-facing slices；实验记录则是版本切换的唯一对比基准。

## Validation Axes

回归至少拆成三组：

1. balanced: 看 head/tail 基本覆盖。
2. hard-negative: 看误召回抑制。
3. broad-dimension: 看不同内容维度是否一起稳定。

当前 broad-dimension 已覆盖：

1. `3c`
2. `photography`
3. `game`
4. `mobile_game`
5. `moba_game`
6. `sandbox_game`
7. `movie_analysis`

## Optimization Priorities

下一阶段优先做三类优化，而不是继续围绕 coarse label classifier 打转：

1. snapshot slimming:
   - 收紧 `top_tags / sample_titles / topic_phrases`
   - 去掉旧的 `profile_text` 和不再消费的重字段
2. retrieval evaluation:
   - 让 regression record 输出 `by_dimension`
   - 在新 slim index 上和旧 index 做同口径对比
3. retrieval-oriented training:
   - 从 owner profile 构造 query-profile pair / triplet 数据
   - 未来以 pairwise / contrastive 目标替代 coarse multi-class accuracy 作为主优化指标

## Exit Criteria

新的 profile-first retrieval 路线，至少要满足下面三条，才算比旧路线更值得继续投入：

1. slim snapshot / slim ES index 的体积明显下降。
2. balanced + hard-negative + broad-dimension 三组 live regression 不退化。
3. 后续 retrieval-oriented 训练能在真实 panel 上带来可解释的收益，而不是只在 pseudo-label accuracy 上涨分。

## 2026-03-08 Status

这一轮已经完成了第一阶段验收：

1. `v2slim1` snapshot 和 `sem64` slim index 已完成真实重建。
2. Mongo / ES sample doc size 已分别下降约 `27.1%` 和 `16.7%`。
3. balanced + hard-negative + broad-dimension 三套 panel 在真实 slim index 上是 `26/26` 全通过。

同时也暴露了下一阶段的真实优化点：

1. head-domain precision 仍然是最值得继续打磨的方向。
2. 轻量 strict gate 已经证明可以收窄 `黑神话悟空` 这类短实体 query 的泛词噪声。
3. 下一步不应再继续围绕 coarse 分类 accuracy 做主优化，而应开始构造 retrieval-oriented pair / triplet 数据，在真实 panel 上比较收益。