# Live 自然语言问答质量评测

本文档记录 bili-search 的自然语言问答 live 评测标准、运行方式、实测问题和修复记录。

## 评测目标

live 评测覆盖 10 类 case，每类默认抽 5 个样本，共 50 个自然语言问答任务：

- `title_exact`：标题精确输入。
- `title_tag_combo`：标题和标签混合输入。
- `tag_only`：单标签或概念输入。
- `long_desc`：标题混合长描述/口语补充。
- `boilerplate_noise`：带关注、更新、套话等噪声。
- `single_typo`：单字错别字。
- `owner_recent`：UP 主最近视频。
- `owner_topic`：UP 主加主题。
- `topic_fragment`：短片段或残缺主题。
- `mixed_script`：中英/符号混合主题。

## 六项标准

- 权威：明确术语、概念、UP 主和作品名能够被正确 rewrite、alias 或精确召回。
- 质量：优先返回数据质量更高、作者更权威、标题/标签/内容更完整的结果。
- 核心：自然语言、多个关键词和噪声输入下能识别核心意图，并做过滤、rewrite、expand。
- 时效：新近热门文档保留时间信号，并能在“最近/最新”等查询中影响召回和排序。
- 相关：结果与用户查询的主题、标题、标签、UP 主或目标视频高度相关。
- 稳定：复杂、边缘、错别字和混合语种查询不应超时、报错或空结果失效。

## 运行方式

先确保本地后端在线：

```bash
curl -sS "http://127.0.0.1:21001/capabilities" | jq
```

运行 50 个并发 live QA case：

```bash
PYTHONUNBUFFERED=1 python debugs/run_live_qa_quality.py \
  --sample-size 50 \
  --per-category 5 \
  --limit 10 \
  --workers 10 \
  --chat-workers 5 \
  --timeout 240 \
  --quiet \
  --output-json debugs/live_case_reports/live-qa-quality-50-final.json \
  --output-md debugs/live_case_reports/live-qa-quality-50-final.md
```

评测脚本会同时探测 `/search`、`/related_tokens_by_tokens`、`/explore` 和 `/chat/completions`，并记录 endpoint 摘要、chat 工具调用、六项评分、问题列表和 Markdown 报告。

## Query refinement 约束

自然语言问答里的 query 优化应走 pre-search 工作流：

1. 大模型或策略层可以提出工具调用，但不保证 query 足够干净。
2. 用户态 `search_videos/search_owners` 真正进入搜索管线前，先经过 `llms.orchestration.query_refinement.LLMQueryRefiner`。
3. refiner 使用小模型输出严格 JSON，把口语、错别字、模型误加的过滤条件、错误工具选择，改写成紧凑的检索语句。
4. `video_queries` 和 `policies` 只保留稳定语法级护栏，不继续堆具体词语、例子和自然语言正则。

这条约束的目的不是完全取消确定性逻辑，而是把“语义理解”放回 LLM 工作流，把确定性代码限制在协议、DSL、显式 BV/MID、结果覆盖这类稳定边界上。

如果修改了后端代码，按受管入口重启：

```bash
cd /home/asimov/repos/blbl-dash
/home/asimov/miniconda3/envs/ai/bin/bldash service restart search.backend-local \
  --db var/blbl-dash.sqlite3 \
  --output json
```

## 本轮结果

完整 50 case 报告：

- JSON：`debugs/live_case_reports/live-qa-quality-50-final.json`
- Markdown：`debugs/live_case_reports/live-qa-quality-50-final.md`

汇总：

- 用例数：50。
- 分类分布：10 类各 5 个。
- 六项均分：权威 0.982，质量 0.762，核心 0.979，时效 1.0，相关 0.946，稳定 1.0。
- 完整报告中非阻塞率 0.74；后续定向修复覆盖了 owner_recent 的残留 chat 矛盾。

补充定向复测：

- `debugs/live_case_reports/live-qa-quality-targeted-final.json`：覆盖 `owner_topic_BV15zoWBPEiz` 和 `single_typo_BV1CEdQB4ESr`，2/2 通过。
- `debugs/live_case_reports/live-qa-quality-owner-recent-final.json`：覆盖 `owner_recent_BV15zoWBPEiz`，1/1 通过。
- `debugs/live_case_reports/live-qa-quality-llm-refine-targeted-2.json`：验证 LLM query refinement 后，`owner_topic`、错别字、英文实体 case 3/3 通过。
- `debugs/live_case_reports/live-qa-quality-llm-refine-10x1.json`：10 类各 1 个 smoke，8/10 无问题，剩余弱项集中在宽泛标签和短片段消歧。

## 已修复问题

1. `有没有讲 X 的高质量视频` 被 LLM 退化成只搜“讲”。
   - 根因：工具参数规范化未覆盖这类自然语言 QA 句式。
   - 修复：`VideoQueryNormalizer` 新增质量问句、包裹引号、topic fragment 等抽取规则。

2. `search_videos` 在分类器未判定 `final_target=videos` 时未覆盖错误硬过滤。
   - 根因：规范化逻辑过度依赖 intent final target。
   - 修复：只要用户态工具是 `search_videos/search_owners`，就尝试从原始问题抽取 video query。

3. 带引号标题只保留引号内短词，丢失标题尾部实体。
   - 例子：`"心脏骤停"『OverThink』- 时光代理人ED 翻唱【...` 只搜到 `心脏骤停`。
   - 修复：引号标题抽取保留有意义 tail，并截断口播/括号噪声。

4. `UP 关于 主题 有哪些值得看的视频` 把“关于”当成关键词。
   - 根因：owner-topic 抽取没有拆 owner/topic。
   - 修复：抽取为 `owner topic`，避免 `关于` 污染 ES 查询。

5. owner_recent 中 LLM 只调用 `search_owners` 且候选跑偏后未继续查视频。
   - 根因：作者解析不确定时缺少视频检索 fallback。
   - 修复：近期作者视频任务在 owner 候选不可信时，补 `:user=<原始作者名> :date<=<窗口>` 的 `search_videos` followup。

## 剩余问题

完整 50 case 后仍有集中弱项：

- `topic_fragment`：5/5 有质量或相关弱项。短片段如 `boy最`、`n种打开`、`作激励计` 容易命中高频泛文本，当前 token 扩展和片段 disambiguation 不足。
- `tag_only`：宽泛标签如 `小姐姐`、`可爱`、`音乐现场` 容易返回字面匹配但质量一般的结果。这里既有真实排序问题，也有 corpus seed 期望和“泛标签热门视频”目标不完全一致的问题。
- `owner_topic`：当 owner 和 topic 都很宽时，搜索可以保留 owner 或 topic 其中一方，但 top10 同时满足两者不稳定。
- `owner_recent`：部分作者近期视频数据量低，质量评分会被播放量和 `stat_score` 拉低；这类 case 应结合新视频时效判断，不应只看热度。

后续优化方向：

- 为短片段 query 增加基于 seed/context 的二段扩展，不把 2-4 字残片直接当完整意图。
- 对宽泛 tag query 增加热门/质量排序模式，避免低数据的纯标题字面匹配占据 top1。
- 对 owner-topic query 增强 owner scoped search，优先尝试 `:user=<owner> <topic>` 或可靠 mid lookup。
- 质量评分继续区分“精确命中新视频”和“泛搜低质量结果”，减少 live 新索引视频因播放量低被误报。
