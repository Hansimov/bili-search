# Owner Search Runbook

## 本仓库内推荐验证顺序

1. 运行 owner 专项测试：

```bash
pytest tests/owner_search -q
```

如果只想先验证新的 head + tail query panel 结构，也可以单独跑：

```bash
pytest tests/owner_search/test_owner_query_panel.py -q
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

如果本地已经有 owner DEV 索引，推荐先跑一次面板化验证，而不是只测单个 query：

```bash
cd /home/asimov/repos/bili-search
python -m debugs.owners_search.eval_owner_panel -i bili_owners_dev_poc3_100k4_v2 -ev elastic_dev
```

如果要专门看“高影响力但领域错误”的误召回，可以跑 hard-negative panel：

```bash
cd /home/asimov/repos/bili-search
python -m debugs.owners_search.eval_owner_panel -i bili_owners_dev_poc3_100k4_v2lean_sem1 -ev elastic_dev -p debugs/owners_search/owner_query_panel_hardneg.json
```

当前 panel 位于：

```bash
debugs/owners_search/owner_query_panel.json
```

面板的设计约束是：

1. 同时覆盖头部明确作者名、头部热门领域、长尾 phrase/domain。
2. 对明确作者名校验 `expected_route=name`，避免被 influence 排序误污染。
3. 对长尾 phrase 校验 `expected_route=phrase`，确保 strict phrase path 持续生效。

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

4. 先对大分片做 `plan-only`，确认 1/10 owner shard 的真实规模和吞吐：

```bash
cd /home/asimov/repos/bili-scraper
PYTHONPATH=. python -m workers.elastic_owners.commander --plan-only -n --owner-partition-count 10 --owner-partition-index 0
```

这条命令现在不会先卡在静默的 `count_documents()`。对于带 owner 分片的全量计划，它会直接进入流式扫描，并且每 `20` 万视频打印一次进度，例如：

1. `200,000 videos, owners=5,757, rate=4824/s`
2. `1,000,000 videos, owners=31,733, rate=3274/s`
3. `2,400,000 videos, owners=80,944, rate=3328/s`

这至少可以区分两类问题：

1. 真正的索引/查询瓶颈
2. 任务本身规模很大但仍在稳定推进

5. 确认 plan 规模后，按 owner 分片直接写入 DEV：

```bash
cd /home/asimov/repos/bili-scraper
PYTHONPATH=. python -m workers.elastic_owners.commander -ei bili_owners_dev1 -ev elastic_dev --ensure-index -n --owner-partition-count 10 --owner-partition-index 0
```

当前一次真实验证里，我没有继续往 `bili_owners_dev1` 追加，而是单独写到了 `bili_owners_dev10p0_raw1` 以避免污染已有 DEV 索引。raw 1/10 shard 在写入前几个 batch 后，经手动 refresh 验证：

1. 已成功写入 `45,000` 个 owner
2. `黑神话悟空` 的 domain search 返回 `445` 个 owner
3. 对返回的 top owner 再做 exact-name search 可以正常命中同一 `mid`

这说明 owner 分片全量写入链路、ES 落盘和独立 owner searcher 查询都已经打通。

如果想先做更聚焦的 DEV 样本，可以把分片和过滤器叠加起来，例如只写最近窗口或高热视频对应的 owner：

```bash
cd /home/asimov/repos/bili-scraper
PYTHONPATH=. python -m workers.elastic_owners.commander -ei bili_owners_dev1 -ev elastic_dev --ensure-index -n -x "u:stat.view>=1w" --owner-partition-count 10 --owner-partition-index 0
```

现在带 `-x` 且带 owner 分片的 full build 也不会再卡在静默的 upfront exact count。它会直接进入流式构建。当前一次真实高热视频 shard 验证中，我把结果写到 `bili_owners_dev10p0_hot1w1`，首个 batch 已成功写入 `5,000` 个 owner；手动 refresh 后：

1. `黑神话悟空` 的 domain search 返回 `48` 个 owner
2. 结果集明显比 raw shard 更窄，但仍然存在泛二创/泛娱乐噪声

这说明高热视频过滤对“缩小候选集”是有效的，但还不能单独解决 owner domain 的标签噪声问题。

6. 基于时间窗口的增量更新：

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

如果要批量验证 creator intent 是否稳定走 owner 搜索，可以直接跑内置 chat suite：

```bash
cd /home/asimov/repos/bili-search
python -m elastics.tests.diag_search_app_dev -u http://127.0.0.1:21011 -m chat_suite --max-iterations 3 --request-timeout 180 --fail-on-miss
```

当前一次真实 suite 结果：4 个 creator-intent query 全部在第 1 轮触发了 `search_owners`，没有 miss，覆盖了泛推荐、攻略向、剧情解析向、整活搞笑向四类表达。

如果要验证“明确视频搜索 / 明确作者时间线”不会误走 owner 搜索，可以再跑对照 suite：

```bash
cd /home/asimov/repos/bili-search
python -m elastics.tests.diag_search_app_dev -u http://127.0.0.1:21011 -m chat_contrast_suite --max-iterations 3 --request-timeout 180 --fail-on-miss
```

当前一次真实对照结果：3 个非 owner 场景全部通过。

1. `推荐几条高播放的黑神话悟空视频` → 第 1 轮触发 `search_videos`
2. `影视飓风最近有什么新视频` → 触发 `check_author + search_videos`，后续只继续视频搜索，不再误触发 `search_owners`
3. `找几条黑神话悟空剧情解析视频` → 第 1 轮触发 `search_videos`

为稳定这组 live 路由，当前服务端除了 prompt 约束，还加了两层轻量 guard：

1. 明确作者时间线请求会在 handler 中过滤 `search_owners`
2. 如果模型只输出“我来搜/没收到结果”的兜底文本而没有真正发命令，handler 会为明确的视频搜索或作者时间线请求注入最小必要的 fallback 命令

7. 通过 stdin 指定 bvid 的增量更新：

```bash
cd /home/asimov/repos/bili-scraper
echo -e "BV1xx\nBV2yy" | python -m workers.elastic_owners.commander -ei bili_owners_dev1 -ev elastic_dev --incremental
```

8. 定时 action：

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

当前一次真实对比结果：在同一批 `300` owner 样本、同一随机切分上，`centroid=0.5179`，`naive_bayes=0.5536`，`naive_bayes_weighted=0.6071`，`linear=0.4643`。

这次新增的 `naive_bayes_weighted` 会对 `owner_name`、`top_tags`、`sample_titles`、`desc_samples` 做轻量字段加权。按当前这组预算和切分，它已经明显超过未加权 `naive_bayes`，因此当前最值得继续保留和迭代的低成本 baseline 已变成 `naive_bayes_weighted`。

如果把窗口放大到用户这次已经在 DEV 侧使用的更大时间范围，也可以先跑一个中等预算的对比：

```bash
cd /home/asimov/repos/bili-search-algo
python -m models.owners.domain --model compare -m 5000 --max-scanned-videos 2000000 -s "2025-12-15 00:00:00" -e "2026-03-07 16:00:00" --min-videos 5 --sample-per-owner 10
```

当前一次真实大窗口结果：`5,000` 个 owner 样本来自约 `186,789` 条视频，同一切分上 `centroid=0.5793`，`naive_bayes=0.6034`，`naive_bayes_weighted=0.6185`，`linear=0.5301`。结论仍然是 `naive_bayes_weighted` 最优，但优势已经明显缩小，说明小样本上调出来的字段权重还需要在更大窗口上重新调优。

如果要继续在同一预算下自动调一轮权重和 `alpha`，可以直接跑：

```bash
cd /home/asimov/repos/bili-search-algo
python -m models.owners.domain --model tune_naive_bayes_weighted -m 300 --max-scanned-videos 200000 -s "2026-02-01 00:00:00" -e "2026-03-07 00:00:00" --min-videos 5 --sample-per-owner 10
```

当前一次真实 tuning 结果：best accuracy 提升到 `0.7143`，最佳配置为：

1. `alpha=1.5`
2. `owner_name=3.0`
3. `top_tags=3.0`
4. `sample_titles=1.0`
5. `desc_samples=0.5`

`owner_domain_metrics.json` 里现在还会额外输出 `error_summary`，方便直接看每个真值 label 最常错到哪里，以及对应的 owner 误判样例。

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
