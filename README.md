# Bili-Search

Backend of Bili Search Engine (https://blbl.top) developped by Hansimov.

Bili-Search 是 B 站搜索服务后端，提供视频搜索、探索、关系查询以及 LLM 工具调用能力。

## Documentation

- 运行与部署：[docs/run/bssv.md](/home/asimov/repos/bili-search/docs/run/bssv.md), [docs/run/bsdk.md](/home/asimov/repos/bili-search/docs/run/bsdk.md)
- 测试与联调：[docs/tests/TEST.md](/home/asimov/repos/bili-search/docs/tests/TEST.md)
- 能力说明：[docs/capabilities/relations.md](/home/asimov/repos/bili-search/docs/capabilities/relations.md)
- DSL 文档：[docs/dsl/SYNTAX.md](/home/asimov/repos/bili-search/docs/dsl/SYNTAX.md), [docs/dsl/USAGE.md](/home/asimov/repos/bili-search/docs/dsl/USAGE.md), [docs/dsl/DESIGN.md](/home/asimov/repos/bili-search/docs/dsl/DESIGN.md)
- 视频探索：[docs/videos_explore/USAGE.md](/home/asimov/repos/bili-search/docs/videos_explore/USAGE.md), [docs/videos_explore/DESIGN.md](/home/asimov/repos/bili-search/docs/videos_explore/DESIGN.md)

## Breaking Change

上游 `es-tok` 的 `es_tok_query_string` 已经从“近似兼容 Lucene `query_string`”重写为“项目自定义的最小文本 DSL”。bili-search 已经同步迁移：

- `+片段` / `-片段` / `"片段"` 会保留在 `es_tok_query_string` 文本里，不再提前改写成 `es_tok_constraints.have_token`。
- `constraint_filter` 也改为走同一套 `es_tok_query_string` exact 语义，避免 analyzer-split 中文片段在 KNN / filter-only 路径上退化。
- DSL 侧不会再给 `es_tok_query_string` 发送 `type` 这类已下线参数。

如果你在外部调用 bili-search 相关模块，请优先阅读 [docs/dsl/USAGE.md](/home/asimov/repos/bili-search/docs/dsl/USAGE.md) 里的迁移说明，以及 [docs/tests/TEST.md](/home/asimov/repos/bili-search/docs/tests/TEST.md) 里的 smoke / benchmark / profiling 入口。