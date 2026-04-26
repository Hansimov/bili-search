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

## Query 构造约束

自然语言问答里的 query 优化不应走隐藏的 pre-search 小模型链路：

1. 大模型在输出 XML 工具命令前，必须先把用户意图转成紧凑、可检索的 DSL。
2. `search_videos` query 只保留实体、主题和必要过滤，不带“是谁”“最近发了哪些视频”“投稿视频”等问句套话。
3. 作者身份和近期投稿混合问题必须按工作流拆开：先 `search_owners` 或显式 lookup 解析作者，再用返回的 `mid` / `:uid` 查近期视频。
4. 执行层只做稳定协议门禁：如果同一轮已经在解析作者，就延后 `expand_query` 和未定向的 `search_videos` 宽搜，避免错误 query 进入 ES。
5. `video_queries` 和 `policies` 只保留稳定语法级护栏，不继续堆具体词语、例子和自然语言正则。
6. `bili-search` 当前禁用 search semantic rewrite；`expand_query` 默认使用 `auto`，即使模型传入 `semantic` 也按 `auto` 执行。
7. 当模型只输出“我来搜索/下一步查询”等计划文本、但目标尚无工具结果覆盖时，运行时必须继续推动真实工具调用，不能把规划文字当最终回答。
8. 作者搜索默认优先站内来源；外部空间页侦察只在本地候选不足或显式要求时补充，避免外部搜索延迟拖慢常规路径。

这条约束的目的不是完全取消确定性逻辑，而是把“语义理解”放回大模型规划，把确定性代码限制在协议、DSL、显式 BV/MID、结果覆盖和工具执行顺序这类稳定边界上。

2026-04-26 更新：

- 已清理 `deterministic.py`、`policies.py`、`runtime.py` 中针对具体自然语言词语的特殊分支。
- 已将 `video_queries.py` 收敛为语法级处理，只保留显式 DSL、括号标题、引号标题、标点和空白清理。
- 已关闭 `bili-search` search semantic rewrite；禁用时搜索准备流程不进入 semantic rewrite 分支，只保留 `semantic_rewrite_info.disabled=true` 作为观测字段。
- 已从 `es-tok` 清理 semantic store、suggester、内置 TSV 资产和 bundle 复制逻辑；`mode=semantic` 仅作为兼容输入映射到 `auto`。
- 已废弃上一轮通过固定词表修复对战、近期、采访等场景的做法；这些场景应由大模型规划先产出高质量 query，再进入搜索管线。
- 真实 local-dev 验证：`/search` 返回 `semantic_rewrite_info.disabled=true`；`/related_tokens_by_tokens` 传入 `mode=semantic` 时返回 `mode=auto`；前端“直接查找”和“快速问答”均可正常完成。
- 快速问答的作者近期视频答案优先使用结构化 `search_owners` 请求文本作为展示主体，避免 intent 中混入问句片段后污染回答。
- 已移除 `search_owners` 结果到 `search_videos` 的 deterministic 自动 follow-up；作者候选必须回到大模型规划上下文，由模型判断最高分作者是否可信、是否需要多作者查询或追加作者搜索。
- 作者过滤之外没有内容匹配文本的视频请求统一走结构化 lookup：`mid`/`uid`、`:uid=... :date<=...` 和数字型 `mid` 参数都会被归一化为 Mongo 优先的 `lookup_videos`。
- transcript 远端 404/网络错误会返回结构化工具错误，不再击穿 `/chat/completions` 导致 500。
- 定向 live 复测“月亮3最近3期视频内容”：工具链分为 `search_owners`、候选检查/分析、`search_videos mode=lookup mid=674510452 limit=3`，视频结果 `source_counts` 显示 Mongo 命中；前端“快速问答”和“直接查找”均完成。
- `response-5.log` 暴露的作者查找场景中，模型多轮只输出准备搜索文本、没有 XML 工具调用，并在后续“继续”中复述或幻觉作者结果。修复方向：提示词明确禁止无工具结果时停在计划；运行时在目标未覆盖且无工具调用时触发确定性恢复调用；引号中的作者名优先作为 focus。
- 同一日志显示 `search_owners` 感知变慢的主要风险来自默认并入外部空间页搜索。修复方向：站内 name/topic/relation/related_tokens 先并发完成并融合，只有本地无候选或显式 `include_google` 时再补 Google 空间页。

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

- `debugs/live_case_reports/live-owner-recent-red-alert-08-after-fix.json`：验证“红色警戒08是谁？最近发了哪些视频？”修复后只执行 `search_owners` 和基于 `mid=1629347259` 的 `search_videos lookup`，未再执行 `expand_query` 或未解析作者的 `:user=红色警戒08` 宽搜；服务端 elapsed 约 5.26 秒。
- `debugs/live_case_reports/live-response2-round2-after-fix.json`：复测“给出他们对战的视频”，不再把街头霸王或普通红警视频包装成明确对战命中；回答只保留标签含 `红警月亮3/HBK08` 的直播回放作为低到中置信入口，并明确提示缺少标题级直接命中。
- `debugs/live_case_reports/live-response2-round3-after-fix.json`：复测“红警月亮3 和 红警HBK08 有一场决赛”，首轮 0 命中后使用干净 fallback `月亮3 红警HBK08 决赛对局`，站内命中 `BV1jmdvBYEPr`，不再生成含“给出他们/他和08什么关系”的污染查询。
- `debugs/live_case_reports/live-response2-round4-after-fix.json`：复测 `BV1jmdvBYEPr` 内容总结，工具摘要保留分 P 标题 `第一部分：08 红警阿V vs 月亮3 国米 2V2 抢7`，最终回答不再误称无法确认月亮3相关内容。
- `debugs/live_case_reports/live-response2-round5-after-fix.json`：复测“月亮3最近3期视频内容”，执行层把被历史污染的 `search_owners text=这期` 纠正为 `月亮3`，用 `mid=674510452` 精确 lookup，`limit=3` 且不强加 30 天窗口。
- `debugs/live_case_reports/live-qa-quality-targeted-final.json`：覆盖 `owner_topic_BV15zoWBPEiz` 和 `single_typo_BV1CEdQB4ESr`，2/2 通过。
- `debugs/live_case_reports/live-qa-quality-owner-recent-final.json`：覆盖 `owner_recent_BV15zoWBPEiz`，1/1 通过。
- `debugs/live_case_reports/live-qa-quality-llm-refine-targeted-2.json`：历史报告，曾用于验证 pre-search LLM refinement；该方案后续因隐藏延迟和错误改写被移除。
- `debugs/live_case_reports/live-qa-quality-llm-refine-10x1.json`：10 类各 1 个 smoke，8/10 无问题，剩余弱项集中在宽泛标签和短片段消歧。

## 历史修复记录

以下记录保留用于追踪问题来源。2026-04-26 之后，其中依赖自然语言词表、例子或正则的实现已被移除；相同问题应通过大模型规划 query 和结构化执行门禁处理。

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
