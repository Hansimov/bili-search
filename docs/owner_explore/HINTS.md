# Owner Search Hints

## 已确认的实现细节

1. `OwnerSearcher.search()` 之前虽然导入了 `detect_owner_query_type` 和 `compute_owner_rank_score`，但主流程并没有真正使用它们。
2. `ToolExecutor` 之前已经实现了 `search_owners`，但 `ChatHandler` 不识别该 XML 命令，导致工具链上层断开。
3. `AuthorGrouper` 之前只聚合视频命中里的 owner 字段，无法补上 owners 索引中的画像统计。
4. `bili-scraper` 之前虽然已有 `workers/elastic_owners` 主链，但缺少基于时间窗口的增量入口和 action。

## 调试建议

1. 如果 `authors` 里没有 owner 画像字段，优先检查 `VideoExplorer.owner_searcher` 是否已挂载。
2. 如果 `search_owners` 没有被 LLM 触发，先看 prompt 是否给了明确示例，其次再看 handler 是否正确解析了 XML。
3. 如果 relevance 排序看起来异常，先打印 `query_type`、`_name_score`、`_domain_score` 三个值定位问题。
4. 如果 `bili-scraper` 的增量更新没有产出 owner 文档，先检查时间窗口内 `videos.insert_at` 是否真的有新增/更新 bvid。

## 风险提醒

1. 现在的 relevance 融合是在应用层完成的，不是 ES 原生排序，因此总数是近似值，不是全局精确 distinct owner 数。
2. owner 画像补全依赖 owners 索引命中文档的 `mid` 与视频结果里的 `owner.mid` 一致。
3. 真实线上效果最终仍取决于 owners 索引的数据质量，而不是这层查询封装本身。
4. `bili-scraper` 的范围增量目前是通过 `videos` 集合的时间字段找受影响 bvid，再映射到 owner mids；如果未来需要更精细的变更检测，再考虑单独扩展 flags 机制。
