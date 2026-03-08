# Owner Search Transition Plan 2

## Current state

旧的 owner semantic 文本链路已经退场，线上和索引侧现在都只保留 token contract：

1. `core_tag_token_ids`
2. `core_tag_token_weights`
3. `core_text_token_ids`
4. `core_text_token_weights`
5. `core_tokenizer_version`
6. `profile_domain_ready`

`OwnerSearcher` 也已经支持两种 domain 查询入口：

1. 显式传入 query token ids
2. 通过 coretok bundle 在查询时自动编码

## What changed

这一轮的重点已经从“删旧字段”切到“训练新模型”：

1. `bili-search-algo/models/coretok` 已有 bundle 序列化能力
2. `bili-search-algo/models/owners/profile.py` 已输出 token snapshot
3. `bili-scraper` v3 profile-source mapping 已切到 token 字段
4. `bili-search` 运行时已可加载 bundle 做 query encoding

## Active training strategy

训练不再依赖固定低信息词表，而是采用数据驱动流程：

1. 从 tag/title/desc 学习 `CoreCorpusStats`
2. 自动识别高覆盖 stop candidates
3. 在 holdout owner retrieval 上评估 tokenizer 和 importance 统计
4. 按 `tiny -> small -> medium` 分规模扩张

## Acceptance criteria for each scale

每个规模都必须先满足以下条件，才继续放大：

1. `query_coverage` 稳定，不出现大面积空编码
2. `Recall@5` 和 `MRR` 在多组阈值上有清晰最优点
3. 指标标准差足够低，说明该规模配置稳定
4. bundle 词表增长合理，没有失控膨胀

## Operational output

训练和调优结果统一写入：

1. `data/coretok/runs/<run>/events.jsonl`
2. `data/coretok/runs/<run>/<scale>/summary.json`
3. `data/coretok/runs/<run>/<scale>/best_bundle.json`
