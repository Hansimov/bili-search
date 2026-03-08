# Owner Search Transition Plan 2

## Goal

这一轮不再继续修补旧的 owner 文本特征链路，而是先把线上检索 contract 收紧，为新的 CoreTagTokenizer / CoreTexTokenizer 训练链路让路。

## Why the old profile text path is being removed

当前这批字段已经确认不再适合作为主语义来源：

1. `top_tags`
2. `topic_phrases`
3. `domain_text`
4. `semantic_terms`
5. `vector_bucket_ids`
6. `vector_bucket_weights`
7. `primary_tid`
8. `primary_ptid`
9. `latest_pic`
10. `profile_version`

问题不是某个字段调权不够，而是整条思路本身噪声太大：

1. 词粒度太粗，混入大量无意义 token。
2. 各字段语义边界不清，重复表达严重。
3. bucket 和主分区字段并不对应真实 owner 语义。
4. 这些字段继续在线消费，只会把错误检索结果固化下来。

## Current service contract

从这一版开始，`OwnerSearcher` 只保留四类稳定能力：

1. `name` route
2. `relation` route
3. `suggest`
4. `top_owners`

domain / phrase 检索不再消费旧文本字段，而是只保留 token 接口占位：

1. `core_tag_token_ids`
2. `core_tag_token_weights`
3. `core_text_token_ids`
4. `core_text_token_weights`
5. `core_tokenizer_version`
6. `profile_domain_ready`

如果调用方没有显式提供 query token ids，则 domain route 直接返回空结果，并附带：

1. `query_route`
2. `domain_status=query_tokens_missing`

这样做是刻意的保守策略，用来阻止旧噪声字段继续在线误召回。

## Integration impact

受影响的上层 contract 已同步收口：

1. LLM tool 输出不再暴露 `top_tags` / `latest_pic`
2. author group enrichment 不再回灌旧 profile 文本字段
3. owner search 单元测试改成验证 token-placeholder 路径

## Next implementation steps

下一阶段按下面顺序推进：

1. 在 `bili-search-algo/models/coretok` 训练 `CoreTagTokenizer`
2. 在 title / desc 上微调 `CoreTexTokenizer`
3. 训练 `CoreImpEvaluator`，输出 token importance / information weights
4. 产出 owner profile token ids 后，再把 domain route 接回线上
