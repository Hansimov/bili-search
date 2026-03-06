# Owner Search Hints

## 已确认的实现细节

1. `OwnerSearcher.search()` 之前虽然导入了 `detect_owner_query_type` 和 `compute_owner_rank_score`，但主流程并没有真正使用它们。
2. `ToolExecutor` 之前已经实现了 `search_owners`，但 `ChatHandler` 不识别该 XML 命令，导致工具链上层断开。
3. `AuthorGrouper` 之前只聚合视频命中里的 owner 字段，无法补上 owners 索引中的画像统计。
4. `bili-scraper` 之前虽然已有 `workers/elastic_owners` 主链，但缺少基于时间窗口的增量入口和 action。
5. `bili-search-algo` 之前没有 owner 画像的小实验入口，现在补了一条 pseudo-label owner domain baseline，用来快速验证 owner 文本聚合和领域标签稳定性。
6. `bili-scraper` 的范围增量已经改成按 owner 聚合，不再依赖大窗口下会撞 16MB 限制的 `distinct("bvid")`。
7. `bili-scraper` commander 现在支持 `--ensure-index`，适合在 `elastic_dev` 上非破坏式建 owner 索引。
8. owner-domain baseline 现在支持 `centroid`、`naive_bayes` 和 `compare` 三种模式；在当前受控样本上，`naive_bayes` 比 `centroid` 略强。

## 调试建议

1. 如果 `authors` 里没有 owner 画像字段，优先检查 `VideoExplorer.owner_searcher` 是否已挂载。
2. 如果 `search_owners` 没有被 LLM 触发，先看 prompt 是否给了明确示例，其次再看 handler 是否正确解析了 XML。
3. 如果 relevance 排序看起来异常，先打印 `query_type`、`_name_score`、`_domain_score` 三个值定位问题。
4. 如果 `bili-scraper` 的增量更新没有产出 owner 文档，先检查时间窗口内 `videos.insert_at` 是否真的有新增/更新 bvid。
5. 如果 owner 增量 dry-run 看不出覆盖面，直接看 commander 新输出的 summary，其中会列出 sample `bvids`、sample `mids` 和空 owner mids。
6. 如果 `elastic_dev` 上 owner 索引还不存在，先跑 `--ensure-index --count`，不要直接用 `-r` 做交互式重建。
7. 如果 owner-domain baseline 跑得过慢，先缩小 `pubdate` 时间窗，再用 `-m` 和 `--max-scanned-videos` 双重约束预算。
8. 如果要做 DEV 联调抽检，优先先跑 `-m 1000` 这种量级，再查 count、exact-name、domain-search 三类结果，不要一开始就扩大到几万 owner。

## 风险提醒

1. 现在的 relevance 融合是在应用层完成的，不是 ES 原生排序，因此总数是近似值，不是全局精确 distinct owner 数。
2. owner 画像补全依赖 owners 索引命中文档的 `mid` 与视频结果里的 `owner.mid` 一致。
3. 真实线上效果最终仍取决于 owners 索引的数据质量，而不是这层查询封装本身。
4. `bili-scraper` 的范围增量目前是通过 `videos` 集合的时间字段找受影响 bvid，再映射到 owner mids；如果未来需要更精细的变更检测，再考虑单独扩展 flags 机制。
5. `bili-search-algo` 里的 owner domain baseline 是轻量 centroid 分类器，适合做特征验证，不适合作为最终线上模型结论。
6. 这次真实 DEV 联调里，`2026-03-07 00:00:00 ~ 02:00:00` 的 `insert_at` 窗口命中约 `565` 万更新视频，因此所有线上调试都应该先对受影响 owner 数做上限控制，而不是直接全量回灌。
7. 当前 `1000` owner 的 DEV 样本已经足够做 owner search 行为联调，但还不足以拿来判断最终线上召回质量。
