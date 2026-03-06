# Owner Search Runbook

## 本仓库内推荐验证顺序

1. 运行 owner 专项测试：

```bash
pytest tests/owner_search -q
```

2. 运行 owner 相关旧测试，确认没有直接回归：

```bash
pytest recalls/tests/test_recall_optimization.py -q
pytest tests/llm/test_tools.py -q
```

3. 如果本地配置了 owners 索引，启动搜索服务后做手工验证：

```bash
python -m apps.search_app
```

手工检查以下场景：

1. `check_author("影视飓风")` 优先走 owners 索引。
2. `search_owners(query="黑神话悟空", sort_by="influence")` 可以返回 owner 列表。
3. 视频搜索结果中的 `authors` 列表包含 `influence_score`、`top_tags`、`latest_pic` 等 owners 字段。

## `bili-scraper` 侧 owners 索引命令

1. 创建或重建索引：

```bash
cd /home/asimov/repos/bili-scraper
python -m workers.elastic_owners.commander -ei bili_owners_dev1 -ev elastic_dev -r
```

2. 全量构建：

```bash
cd /home/asimov/repos/bili-scraper
python -m workers.elastic_owners.commander -ei bili_owners_dev1 -ev elastic_dev -n
```

3. 小样本全量构建验证：

```bash
cd /home/asimov/repos/bili-scraper
python -m workers.elastic_owners.commander -ei bili_owners_dev1 -ev elastic_dev -n -m 10000
```

4. 基于时间窗口的增量更新：

```bash
cd /home/asimov/repos/bili-scraper
python -m workers.elastic_owners.commander -ei bili_owners_dev1 -ev elastic_dev -s "2026-03-07 00:00:00"
```

5. 通过 stdin 指定 bvid 的增量更新：

```bash
cd /home/asimov/repos/bili-scraper
echo -e "BV1xx\nBV2yy" | python -m workers.elastic_owners.commander -ei bili_owners_dev1 -ev elastic_dev --incremental
```

6. 定时 action：

```bash
cd /home/asimov/repos/bili-scraper
python -m actions.elastic_owners_indexer
```

## 真实环境联调建议

1. 先在开发索引验证 owners mapping 和少量样本导入。
2. 确认 `bili-search` 的 `elastic_owners_index` 指向新索引后，再启用搜索服务。
3. 最后再做 LLM 工具链回归，验证 prompt 输出的 `<search_owners/>` 能被正确解析与执行。
4. 如果要长期跑增量更新，优先用 `actions.elastic_owners_indexer`，默认按 65 分钟窗口扫描 `videos.insert_at`。

## 当前这次改动的测试边界

本次提交主要验证的是：

1. owner 双路召回后的融合排序逻辑。
2. owner 画像注入 author grouping 的链路。
3. `search_owners` XML 指令在 chat handler 中的解析能力。
4. `bili-scraper` 里 owners 索引的全量/增量命令入口。

不包含：

1. 真实 ES 查询结果质量评估。
2. MongoDB owners 构建管线吞吐验证。
3. embedding / relation extraction 的离线任务运行。
