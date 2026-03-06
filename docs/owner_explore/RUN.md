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

先推荐非破坏式确保索引存在，只在 `elastic_dev` 上做：

```bash
cd /home/asimov/repos/bili-scraper
python -m workers.elastic_owners.commander -ei bili_owners_dev1 -ev elastic_dev --ensure-index --count
```

只有在确认要清空 DEV 索引时，才使用重建：

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

这条命令现在会先打印 owner generator summary，再打印 owner indexer summary，至少能看到：时间窗口、命中的 `bvid` 数、影响的 `mid` 数、采样 `bvid/mid`、空 owner mids 采样，以及 dry-run / 实际批次数。

在真实大窗口验证时，优先给 `-m`，先把受影响 owner 数限制在几十到几百：

```bash
cd /home/asimov/repos/bili-scraper
python -m workers.elastic_owners.commander -ei bili_owners_dev1 -ev elastic_dev -s "2026-03-07 00:00:00" -e "2026-03-07 02:00:00" -m 50 -d
```

当前一次真实 DEV 联调结果：`2026-03-07 00:00:00 ~ 02:00:00` 的 `insert_at` 窗口里约有 `5,652,812` 个更新 `bvid`，先限制为 `50` 个受影响 owner 后，实际写入 DEV 索引成功 `50` 条。

如果要把 DEV 样本扩到更有代表性的规模，可以先提到 `1000`：

```bash
cd /home/asimov/repos/bili-scraper
python -m workers.elastic_owners.commander -ei bili_owners_dev1 -ev elastic_dev -s "2026-03-07 00:00:00" -e "2026-03-07 02:00:00" -m 1000
```

当前一次真实扩容结果：同一窗口下成功写入 `1000` 个 owner 到 `bili_owners_dev1`，随后 DEV 查询 count 为 `1000`，`不祥之赵` 的 name search 正常命中，`黑神话悟空` 的 domain search 返回 `31` 个 owner。

如果要做 `bili-search` 服务级联调，可以直接把 DEV owner 索引挂进 `search_app`：

```bash
cd /home/asimov/repos/bili-search
python -m apps.search_app -m dev -ei bili_videos_dev6 -ev elastic_dev -eoi bili_owners_dev1 -p 21011
```

当前一次真实 API 联调结果：

1. `/health` 返回 `status=ok`，并显示 `llm_model=gpt-5.2`
2. `/explore` 查询 `黑神话悟空` 时，live 响应里返回 `129644` total hits、`25` 个 author groups
3. `/chat/completions` 对“推荐几个做黑神话悟空内容的UP主”这类请求，tool events 已真实触发 `search_owners`

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

## `bili-search-algo` 侧 owner 实验命令

1. 构建 owner 领域样本：

```bash
cd /home/asimov/repos/bili-search-algo
python -m models.owners.domain --build-only --min-videos 8 --dominant-ratio 0.6 -m 5000
```

2. 构建样本并跑轻量 baseline：

```bash
cd /home/asimov/repos/bili-search-algo
python -m models.owners.domain --min-videos 8 --dominant-ratio 0.6 --sample-per-owner 12 -m 5000
```

在 9 亿 docs / 6000 万 owner 的量级下，不要默认全扫。快速验证时至少给一个预算约束：`-m`、`--max-scanned-videos` 或时间窗口。

推荐先跑这种受控实验：

```bash
cd /home/asimov/repos/bili-search-algo
python -m models.owners.domain -m 300 --max-scanned-videos 200000 -s "2026-02-01 00:00:00" -e "2026-03-07 00:00:00" --min-videos 5 --sample-per-owner 10
```

当前一次真实实验结果：扫描 `7,637` 条 recent videos，构建 `300` 个 owner 样本，覆盖 `10` 个 label，baseline accuracy 为 `0.5179`。

如果要比较两种轻量基线，直接加 `--model compare`：

```bash
cd /home/asimov/repos/bili-search-algo
python -m models.owners.domain --model compare -m 300 --max-scanned-videos 200000 -s "2026-02-01 00:00:00" -e "2026-03-07 00:00:00" --min-videos 5 --sample-per-owner 10
```

当前一次真实对比结果：在同一批 `300` owner 样本、同一随机切分上，`centroid=0.5179`，`naive_bayes=0.5536`。

继续加入线性基线后的最新结果：`linear=0.4643`，低于 `naive_bayes`，因此当前最值得保留的低成本 baseline 仍然是 `naive_bayes`。

默认会输出到：

1. `data/owners/owner_domain_samples.jsonl`
2. `data/owners/owner_domain_metrics.json`
3. `data/owners/owner_domain_predictions.jsonl`

## 真实环境联调建议

1. 先在开发索引验证 owners mapping 和少量样本导入。
2. 关于 Elasticsearch 的 owner 测试，当前阶段只在 `elastic_dev` 进行，不要直接落到 `elastic_pro`。
3. 确认 `bili-search` 的 `elastic_owners_index` 指向新索引后，再启用搜索服务。
4. 最后再做 LLM 工具链回归，验证 prompt 输出的 `<search_owners/>` 能被正确解析与执行。
5. 如果要长期跑增量更新，优先用 `actions.elastic_owners_indexer`，默认按 65 分钟窗口扫描 `videos.insert_at`。

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
