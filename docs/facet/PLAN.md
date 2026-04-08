# 视频搜索引擎多层 Facet / Taxonomy / 多模态检索方案（贴合当前资源的工程设计稿）

## 目录

- [1. 文档目标](#1-文档目标)
- [2. 当前资源与约束](#2-当前资源与约束)
- [3. 核心问题重述：为什么标题+tags 不够](#3-核心问题重述为什么标题tags-不够)
- [4. 总体设计原则](#4-总体设计原则)
- [5. 顶层系统架构：Need → Promise → Evidence → Payoff](#5-顶层系统架构need--promise--evidence--payoff)
- [6. 为什么视频搜索的 Facet 不能照搬文本搜索](#6-为什么视频搜索的-facet-不能照搬文本搜索)
- [7. Facet / Taxonomy 系统工程规格](#7-facet--taxonomy-系统工程规格)
  - [7.1 Layer A：Need Facets（用户为什么搜）](#71-layer-aneed-facets用户为什么搜)
  - [7.2 Layer B：Promise Facets（点击前用户以为会看到什么）](#72-layer-bpromise-facets点击前用户以为会看到什么)
  - [7.3 Layer C：Evidence Facets（视频里实际上如何展开）](#73-layer-cevidence-facets视频里实际上如何展开)
  - [7.4 Layer D：Payoff Facets（用户看完后得到了什么）](#74-layer-dpayoff-facets用户看完后得到了什么)
  - [7.5 横切层：Safety / Policy / Trust](#75-横切层safety--policy--trust)
- [8. Query Facet 与 Doc Facet 的映射关系](#8-query-facet-与-doc-facet-的映射关系)
- [9. Doc 数据模型与字段设计](#9-doc-数据模型与字段设计)
- [10. Query 数据模型与字段设计](#10-query-数据模型与字段设计)
- [11. 索引与存储架构设计](#11-索引与存储架构设计)
- [12. 离线生产链路（DAG）](#12-离线生产链路dag)
- [13. 在线检索与排序链路](#13-在线检索与排序链路)
- [14. 模型与组件选择建议](#14-模型与组件选择建议)
- [15. 冷启动：没有真实 query 时如何构建 baseline](#15-冷启动没有真实-query-时如何构建-baseline)
- [16. 评估体系](#16-评估体系)
- [17. 按你当前资源给出的阶段性实施方案](#17-按你当前资源给出的阶段性实施方案)
- [18. 关键工程任务拆解](#18-关键工程任务拆解)
- [19. 风险、坑点与边界](#19-风险坑点与边界)
- [20. 推荐的第一版交付物](#20-推荐的第一版交付物)
- [21. 后续可扩展方向](#21-后续可扩展方向)
- [22. 一页式结论](#22-一页式结论)

---

## 1. 文档目标

本文档的目标，是把前面关于视频搜索、query taxonomy、facet 设计、多模态检索、冷启动和工程落地的讨论，整理成一份**可以直接指导实施的系统方案**。

本文档重点解决的问题：

1. 用户的真实查询往往是“想获得某种体验/效果”，而不是“视频客观上属于什么题材”。
2. 标题、tags、简介无法完整表达视频会不会被用户点击。
3. 封面是当前最可用、最贴近点击决策的信号。
4. 视频又不同于普通图文：它还有时间结构、表现形式、音频、节奏、情绪轨迹与观看方式。
5. 在缺少真实 query 与用户行为日志的阶段，需要先构建一个**高质量、可扩展的 baseline**。

本文档不是泛泛而谈，而是刻意贴合你当前的资源与约束：

- 10 亿视频元数据已在 MongoDB 中
- 已有写入 Elasticsearch 的链路
- 封面链接齐全，但未全量下载
- 视频快照链接齐全，但文件更大
- 有一台大内存高核数服务器
- 有 6 张 3080 20GB 显卡
- 可以使用小参数开源模型和部分付费 API

---

## 2. 当前资源与约束

### 2.1 你已经具备的资源

#### 数据侧

- 约 10 亿视频 doc
- 每个 doc 典型字段包括：
  - 标题
  - tags
  - 简介
  - 播放量 / 点赞 / 收藏 / 评论 / 分享等统计
  - 作者昵称 / 作者 ID
  - 视频封面图片链接
  - 视频快照链接

#### 基础设施侧

- MongoDB：已有全量元数据
- Elasticsearch：已有写入链路
- 单机服务器：
  - 128 核 256 线程
  - 1TB 内存
  - 16TB 存储
  - 2 × 4TB SSD（其中一块用于主系统+MongoDB，另一块用于 Elasticsearch 和 RocksDB）

#### 算力侧

- 6 × RTX 3080 20GB
- 可部署小中型视觉语言模型、文本 embedding 模型、reranker
- 可以调用部分外部 API 作为补充

### 2.2 你当前的关键约束

#### 约束 1：不能一开始就做“全量视频深理解”

- 10 亿视频 × 封面下载 × embedding × caption × taxonomy 打标，本身就是非常大的工程
- 如果再叠加视频快照、ASR、音频处理，成本会指数上升

#### 约束 2：当前最强信号是“封面 + 标题/tags + 统计”

- 这决定了第一版方案必须把封面用到极致
- 快照与正文理解应属于第二阶段增强，而非第一阶段基础

#### 约束 3：真实 query 与行为日志不足

- 不能直接依赖点击日志做学习排序
- 需要先用规则、合成 query、人工 benchmark 构建 baseline

#### 约束 4：目标不是学术论文，而是工程上能跑、能扩、能持续迭代

- 顶层 taxonomy 必须稳定
- 中层字段必须可回溯、可重算、可版本化
- 在线链路必须能在现有 ES / Mongo / 本地计算资源下闭环

---

## 3. 核心问题重述：为什么标题+tags 不够

你遇到的问题，本质上不是“文本召回模型还不够强”，而是：

> 用户使用的是“体验语言 / 需求语言 / 动机语言”，而视频 doc 大多只显式提供“题材语言 / 作者语言 / 表层描述语言”。

例如：

- “我想看让我开心的视频”
- “我想看性感的小姐姐”
- “睡前适合刷的视频”
- “想看有氛围感的”
- “想看讲得特别清楚的教程”

这些 query 在很多情况下：

- 不是一个明确 topic
- 不是一个明确实体
- 甚至不一定会在标题与 tags 中出现

但是：

- 封面往往能强烈传递“风格、人物、吸引点、氛围、点击冲动”
- 视频的表现形式、节奏、情绪轨迹，决定了它是否真的满足用户需求

因此，系统不能只把 query 与 doc 当成“文本相似度匹配”问题，而要升级成：

> **用户需求理解 + 点击前承诺建模 + 视频证据建模 + 满足结果建模**

这就是后文四层体系 `Need → Promise → Evidence → Payoff` 的核心出发点。

---

## 4. 总体设计原则

### 4.1 不做扁平大标签表，做多层、多 facet、软标签系统

不建议：

- 试图维护一个“几十万类别”的平铺 taxonomy
- 让 query 落到单个 hard label
- 让每个 doc 只属于一个类别

建议：

- 顶层少量稳定维度
- 每个维度内部多值、概率化、可扩展
- query 和 doc 都表示成 facet 分布，而不是单点类别

### 4.2 Query Facet 与 Doc Facet 不强行同构

这是视频搜索最重要的系统设计原则之一。

用户 query 往往表达的是：

- 想获得什么
- 想怎么消费
- 想看到什么感觉
- 想满足什么情绪 / 任务

而 doc 当前可观测到的，往往是：

- 封面里是谁 / 什么东西
- 标题 tags 在讲什么
- 视频大概是什么形式
- 是否像某种内容

所以系统必须有一层显式桥接：

- `Need Facet`（query 侧）
- 通过映射层
- 对齐到 `Promise / Evidence / Payoff`（doc 侧）

### 4.3 先做点击前信号最强的一层，再扩展到正文层

第一版优先级应当是：

1. 封面
2. 标题 / tags / 简介
3. 统计先验
4. 热门热视频 / 热门作者
5. 少量快照增强
6. 之后才是音频、ASR、长视频结构

### 4.4 先做热集，再做全量

不建议第一天做全量 10 亿封面的下载、embedding 和 caption。

建议：

- 先定义 hot set
- 再对 hot set 做多模态理解
- 对 warm/cold 集做懒计算和按需升级

### 4.5 第一版目标是“Exploratory Query 明显变好”，而不是“一步到位做最强通用搜索”

最值得体现差异化价值的，恰恰是：

- 情绪类 query
- 审美类 query
- 模糊风格类 query
- 观看场景类 query
- “我也说不清，但我就想看这种”类 query

---

## 5. 顶层系统架构：Need → Promise → Evidence → Payoff

整个系统建议围绕这四层来设计。

### 5.1 Need

用户为什么搜。

例如：

- 我想开心
- 我想放松
- 我想学会某件事
- 我想看养眼的内容
- 我想找适合摸鱼刷的
- 我想找一个值得发给朋友的

### 5.2 Promise

用户点击前，以为这个视频会是什么样。

它主要由以下信号构成：

- 封面主体
- 封面风格
- 封面吸引点
- 标题口吻
- tags 与简介里暴露出的题材 / 形式
- 播放量、点赞率、收藏率等社交证明

### 5.3 Evidence

视频里实际上发生了什么、如何展开。

典型包括：

- 主体 / 场景 / 动作 / 事件
- 表现形式
- 节奏与时间结构
- 音频结构
- 情绪轨迹
- 是否完整演示过程

### 5.4 Payoff

用户看完后到底获得了什么。

例如：

- 被逗笑
- 被治愈
- 获得知识
- 做出决策
- 得到养眼满足
- 得到社交谈资
- 感到陪伴

### 5.5 为什么这四层对视频尤其重要

对于视频来说：

- 点击由 Promise 强烈驱动
- 满意度由 Evidence 和 Payoff 驱动
- Query 多数表达 Need 与 Payoff
- 而你当前可直接观测到的，恰恰主要是 Promise

所以第一版系统应该做的是：

- 强建模 Promise
- 粗建模 Need
- 用有限的文本和封面推断部分 Evidence / Payoff
- 后续再靠快照、日志逐渐补强

---

## 6. 为什么视频搜索的 Facet 不能照搬文本搜索

普通文本搜索中，常见维度是：

- topic
- entity
- intent
- freshness
- authority

但视频搜索额外多出几个非常关键的维度：

1. **点击前封面承诺**
2. **观看方式**（快刷 / 深看 / 背景播放 / 循环看）
3. **表现形式**（口播 / 混剪 / vlog / 直播切片 / reaction / 教程）
4. **时间结构**（开头即高潮、慢热、单峰、多高光、完整过程）
5. **音频结构**（人声主导、音乐主导、环境声主导、ASMR）
6. **情绪轨迹**（治愈 / 紧张 / 爽感 / 混乱搞笑 / 陪伴感）
7. **观看 payoff**（开心、解压、学会、可转发、养眼、陪伴）

也就是说，视频搜索的 facet 不能仅仅围绕“视频讲什么”，而必须同时围绕：

- 用户此刻想怎么消费
- 视频在点击前如何诱发点击
- 视频内部如何兑现承诺
- 用户看完后是否得到满足

---

## 7. Facet / Taxonomy 系统工程规格

下面给出推荐的顶层 facet schema。

设计原则：

- 顶层维度固定
- 维度之间近似独立
- 维度内部允许层级和多值
- 所有标签都用 soft label（概率）表示

---

### 7.1 Layer A：Need Facets（用户为什么搜）

这是 query 侧主层。

#### A1. Task Mode

表示用户当前的搜索任务模式。

建议枚举：

- `known_item`
- `lookup_entity`
- `exploration`
- `repeat`
- `collect_compare`

说明：

- `known_item`：想找某个具体视频或系列
- `lookup_entity`：想找某个作者、人物、作品、游戏、主题
- `exploration`：并不明确，想逛、想发现
- `repeat`：想重复消费熟悉内容
- `collect_compare`：想找多个候选来对比或收藏

#### A2. Motivation

表示用户此刻的核心动机。

建议枚举：

- `entertainment`
- `information`
- `emotion_regulation`
- `aesthetic`
- `social_shareability`
- `utility_task`
- `companionship`

#### A3. Consumption Mode

表示用户想怎么消费。

建议枚举：

- `quick_scroll`
- `deep_watch`
- `background_play`
- `loopable`
- `share_first`
- `learn_first`

#### A4. Expected Payoff

表示用户预期获得什么结果。

建议枚举：

- `funny`
- `relaxing`
- `healing`
- `stimulating`
- `eye_candy`
- `clear_explanation`
- `decision_help`
- `topic_for_chat`
- `cozy_companionship`

#### A5. Constraints

表示用户隐式或显式约束。

建议枚举：

- `duration_short`
- `duration_medium`
- `duration_long`
- `safe_only`
- `authoritative_only`
- `recent_only`
- `popular_only`
- `no_jump_scare`
- `low_cognitive_load`

---

### 7.2 Layer B：Promise Facets（点击前用户以为会看到什么）

这是 doc 侧 v1 最重要的一层。

#### B1. Cover Subject

封面主体是什么。

建议枚举示例：

- `young_female`
- `young_male`
- `group_people`
- `pet_cat`
- `pet_dog`
- `food`
- `car`
- `game_ui`
- `landscape`
- `room_scene`
- `object_closeup`
- `cosplay_character`
- `livestream_room`

#### B2. Cover Hook

封面最直接的点击诱因是什么。

建议枚举示例：

- `pretty_face`
- `funny_expression`
- `cute_animal`
- `strong_motion_pose`
- `visual_contrast`
- `curiosity_gap`
- `suspense`
- `professional_setup`
- `dramatic_text_overlay`
- `before_after_tease`
- `high_energy_pose`

#### B3. Surface Style

点击前能感觉到的审美或风格。

建议枚举示例：

- `cute`
- `sexy`
- `cool`
- `clean`
- `cinematic`
- `chaotic`
- `meme_like`
- `cozy`
- `premium`
- `raw_authentic`
- `pure_desire_style`
- `healing_style`
- `high_energy_style`

#### B4. Social Proof

点击前可见的社交证明。

建议枚举：

- `very_popular`
- `high_like_rate`
- `high_favorite_rate`
- `trusted_author`
- `series_content`
- `viral_signal`

#### B5. Promise Format

用户在没点开前，觉得这是什么形式。

建议枚举：

- `looks_like_tutorial`
- `looks_like_vlog`
- `looks_like_dance`
- `looks_like_clip`
- `looks_like_livestream_cut`
- `looks_like_reaction`
- `looks_like_review`
- `looks_like_amv`
- `looks_like_pet_daily`

---

### 7.3 Layer C：Evidence Facets（视频里实际上如何展开）

这一层是视频相比图文的关键新增部分。

#### C1. Semantic Content

视频实际内容讲什么。

建议枚举：

- `person`
- `pet`
- `food`
- `vehicle`
- `gameplay`
- `dance`
- `makeup`
- `fashion`
- `travel`
- `comedy`
- `science`
- `sports`
- `music_performance`
- `screen_tutorial`
- `storytelling`
- `review`
- `daily_life`

#### C2. Presentation Form

视频是通过什么形式表达的。

建议枚举：

- `talking_head`
- `step_by_step_tutorial`
- `montage`
- `reaction`
- `livestream_cut`
- `street_shot`
- `dance_performance`
- `daily_vlog`
- `commentary`
- `screen_recording`
- `compilation`
- `before_after`
- `interview`
- `asmr`

#### C3. Temporal Structure

视频如何随时间展开。

建议枚举：

- `hook_first`
- `slow_build`
- `single_peak`
- `multiple_highlights`
- `process_complete`
- `story_arc`
- `loop_friendly`
- `clip_based_fragmented`
- `steady_low_intensity`

#### C4. Pace / Density

视频节奏和信息密度。

建议枚举：

- `fast_cut`
- `medium_pace`
- `slow_pace`
- `high_info_density`
- `low_info_density`
- `requires_context`
- `easy_to_drop_in`

#### C5. Audio Structure

音频信息结构。

建议枚举：

- `speech_dominant`
- `music_dominant`
- `ambient_dominant`
- `asmr_like`
- `laugh_reaction_audio`
- `clear_narration`
- `subtitle_dependent`
- `noise_heavy`

#### C6. Emotion Trajectory

情绪轨迹或主导情感。

建议枚举：

- `positive_high_arousal`
- `positive_low_arousal`
- `negative_high_arousal`
- `tense`
- `healing`
- `fear_inducing`
- `nostalgic`
- `chaotic_fun`
- `soft_companion`

#### C7. Trust / Authenticity

内容是否像真实、专业、可相信的表达。

建议枚举：

- `personal_experience`
- `professional_explanation`
- `obvious_editing`
- `scripted_performance`
- `livestream_authentic`
- `reupload_or_clip`
- `uncertain_source`

---

### 7.4 Layer D：Payoff Facets（用户看完后得到了什么）

这一层在 v1 阶段可以先做弱监督预测，在有真实日志后逐步校正。

#### D1. Emotional Payoff

建议枚举：

- `funny`
- `relaxing`
- `healing`
- `exciting`
- `stress_relief`
- `eye_candy`
- `companionship`
- `adrenaline`

#### D2. Informational Payoff

建议枚举：

- `learned_method`
- `understood_principle`
- `got_answer`
- `got_comparison`
- `decision_reduced`
- `found_reference`

#### D3. Social Payoff

建议枚举：

- `shareable`
- `talk_worthy`
- `meme_material`
- `community_signal`
- `identity_signal`

#### D4. Habit Payoff

建议枚举：

- `can_binge`
- `follow_series`
- `sleep_routine`
- `background_companion`
- `rewatchable`

---

### 7.5 横切层：Safety / Policy / Trust

这一层不应该简单当成加分项，而是更适合作为：

- 过滤条件
- 降权条件
- 需要人工复核的标记

建议枚举：

- `sexual_suggestive_risk`
- `minor_uncertain_risk`
- `violence_risk`
- `medical_misinfo_risk`
- `copyright_clip_risk`
- `hate_or_harassment_risk`
- `unsafe_behavior_risk`
- `gambling_or_fraud_risk`

---

## 8. Query Facet 与 Doc Facet 的映射关系

这是整个系统最关键的桥接层。

### 8.1 为什么必须显式建映射层

因为：

- Query 常常表达 Need / Payoff
- Doc 在 v1 阶段主要可观测的是 Promise / 部分 Evidence

所以不能假设“query facet = doc facet”。

### 8.2 映射层的职责

把 query 中的 Need / Payoff，映射成 doc 侧可检索的 Promise / Evidence 信号。

例如：

#### Query：`我想看让我开心的视频`

Need：

- `motivation = emotion_regulation`
- `expected_payoff = funny`
- `consumption_mode = quick_scroll`

映射到 Doc 侧偏好的信号：

- Promise：
  - `cover_hook = funny_expression`
  - `cover_hook = cute_animal`
  - `surface_style = chaotic`
  - `surface_style = meme_like`
- Evidence：
  - `temporal_structure = hook_first`
  - `temporal_structure = multiple_highlights`
  - `presentation_form = compilation`
  - `presentation_form = reaction`
- Prior：
  - 高互动、高分享、高复看

#### Query：`我想看性感的小姐姐`

Need：

- `motivation = aesthetic`
- `expected_payoff = eye_candy`
- `consumption_mode = quick_scroll`

映射到 Doc 侧偏好的信号：

- Promise：
  - `cover_subject = young_female`
  - `cover_hook = pretty_face`
  - `surface_style = sexy`
  - `surface_style = pure_desire_style`
- Evidence：
  - `presentation_form = dance_performance`
  - `presentation_form = street_shot`
  - `presentation_form = livestream_cut`
  - `presentation_form = fashion`
- Safety：
  - 需要做边界控制与降权策略

### 8.3 映射层的表示方式

推荐使用结构化配置，而不是散落在代码中的隐式规则。

示例：

```json
{
  "need_label": "funny",
  "mapped_doc_signals": {
    "promise.cover_hook": ["funny_expression", "cute_animal", "visual_contrast"],
    "promise.surface_style": ["chaotic", "meme_like", "cute"],
    "evidence.temporal_structure": ["hook_first", "multiple_highlights"],
    "evidence.presentation_form": ["compilation", "reaction", "short_clip"]
  }
}
```

### 8.4 映射层的构建来源

建议分三步：

1. 人工规则初始化
2. LLM / VLM 扩展候选映射
3. 上线后使用真实行为数据校正映射权重

---

## 9. Doc 数据模型与字段设计

每个视频 doc 建议拆成以下几类字段。

### 9.1 原始字段

```json
{
  "bvid": "BV...",
  "title": "...",
  "tags": ["..."],
  "desc": "...",
  "owner_name": "...",
  "owner_id": 123,
  "pubdate": 1710000000,
  "insert_at": 1710000000,
  "stats": {
    "view": 0,
    "like": 0,
    "favorite": 0,
    "coin": 0,
    "share": 0,
    "danmaku": 0
  },
  "cover_url": "https://...",
  "videoshot_urls": ["..."]
}
```

### 9.2 视觉衍生字段

```json
{
  "cover_embedding_clip_v1": "vector_ref",
  "cover_caption_short": "...",
  "cover_caption_long": "...",
  "cover_visual_tags": ["..."],
  "cover_ocr_text": ["..."],
  "cover_quality_score": 0.0,
  "cover_face_count": 0,
  "cover_person_count": 0
}
```

### 9.3 Facet 字段

```json
{
  "promise_facets": {
    "cover_subject": [
      {"label": "young_female", "score": 0.89, "source": "vlm_cover"}
    ],
    "surface_style": [
      {"label": "sexy", "score": 0.74, "source": "vlm_cover"}
    ]
  },
  "evidence_facets": {},
  "payoff_facets": {},
  "safety_facets": {}
}
```

### 9.4 Enriched Text 字段

这是为 BM25、dense embedding、reranker 准备的增强文本表示。

推荐模板：

```text
{title}
[SEP]
{tags}
[SEP]
{desc}
[SEP]
作者: {owner_name}
[SEP]
封面描述: {cover_caption_short}
[SEP]
视觉标签: {cover_visual_tags}
[SEP]
主体: {subject labels}
题材: {semantic labels}
形式: {presentation labels}
风格: {surface style labels}
效果: {payoff labels}
```

### 9.5 质量与先验字段

```json
{
  "prior_popularity_score": 0.0,
  "prior_ctr_proxy_score": 0.0,
  "prior_quality_score": 0.0,
  "freshness_score": 0.0,
  "author_trust_score": 0.0
}
```

### 9.6 版本与回溯字段

```json
{
  "facet_version": "facet_v1.2",
  "caption_model_version": "qwen2.5-vl-7b-2026-04",
  "embedding_model_version": "jina-clip-v2-2026-04",
  "updated_at": 1710000000
}
```

### 9.7 设计要求

- 每个字段都要能重算
- 每个模型输出都要带版本号
- 不要把所有结果压成单个不透明 blob
- 要允许局部回滚、局部重算和 AB 对比

---

## 10. Query 数据模型与字段设计

Query 不应该只变成一个 embedding，而应该先解析成结构化对象。

推荐 schema：

```json
{
  "raw_query": "我想看让我开心的视频",
  "language": "zh",
  "normalized_query": "我想看让我开心的视频",
  "query_type": {
    "known_item": 0.02,
    "exploration": 0.91,
    "repeat": 0.07
  },
  "need_facets": {
    "motivation": [
      {"label": "emotion_regulation", "score": 0.93},
      {"label": "entertainment", "score": 0.71}
    ],
    "consumption_mode": [
      {"label": "quick_scroll", "score": 0.68}
    ],
    "expected_payoff": [
      {"label": "funny", "score": 0.84},
      {"label": "relaxing", "score": 0.55}
    ],
    "constraints": [
      {"label": "safe_only", "score": 0.88}
    ]
  },
  "explicit_entities": [],
  "explicit_topics": [],
  "visual_intent_hints": [
    {"label": "cute_animal", "score": 0.44},
    {"label": "funny_expression", "score": 0.52}
  ],
  "facet_weights": {
    "promise": 0.45,
    "evidence": 0.20,
    "payoff": 0.35
  },
  "ambiguity": 0.76,
  "safety_requirement": "strict"
}
```

### 10.1 Query parser 输出内容建议

至少包括：

- query 类型分布
- need facet 分布
- 显式主题 / 实体
- 视觉意图 hints
- ambiguity
- safety requirement
- facet 权重

### 10.2 高歧义 query 的多假设机制

对于高歧义 query，建议不要强行输出一个 interpretation，而是允许生成 2~4 个 hypotheses。

例如：

Query：`想看性感的小姐姐`

可拆成：

- H1：舞蹈 + 女生出镜 + 性感风
- H2：穿搭 / 街拍 + 高颜值女性 + 氛围感
- H3：直播切片 + 吸睛风格
- H4：更安全、更收敛的女性出镜内容

这会显著降低因为单一错误解释导致的召回失败。

---

## 11. 索引与存储架构设计

### 11.1 推荐分工

#### MongoDB

适合存：

- 原始元数据
- 原始链路输入
- 部分任务状态
- 补充调试信息

#### Elasticsearch

适合存：

- 原始文本字段
- enriched_text
- facet labels
- prior scores
- 结构化过滤字段
- 可选：hot set 向量索引

#### RocksDB

适合存：

- 封面处理状态
- embedding / caption 中间产物缓存
- URL → 本地文件路径映射
- prompt / model version 缓存
- 失败重试状态

#### 本地 blob / 对象存储

适合存：

- hot set 封面原图
- 标准化缩略图
- 后续热视频快照

### 11.2 关于 ES 与向量存储的建议

第一版可以：

- ES 负责混合检索编排
- 热集封面向量可以先放 ES 或独立 ANN 层
- 不建议一开始就全量 10 亿向量全部硬上生产检索

### 11.3 热温冷分层

#### Hot 集

- 最近 180 天
- 高频召回、高互动
- 重点垂类
- 预计 1000 万到 5000 万

#### Warm 集

- 暂不下载封面，但保留 URL
- 当进入热门候选或被频繁点击时，再升级为 hot

#### Cold 集

- 只保留元数据
- 先不做封面 embedding
- 通过文本路先参与粗召回

---

## 12. 离线生产链路（DAG）

下面给出建议的离线生产流程。

### 12.1 Step 0：选择 hot set

输入：

- 最近时间窗口
- 播放 / 点赞 / 收藏 / 分享等统计
- 高频 query 候选结果
- 重点垂类白名单

输出：

- `doc_hot_candidates`

### 12.2 Step 1：下载封面与缓存

产物：

- 原图缓存
- 标准化缩略图
- 下载失败队列
- URL 指纹与去重结果

建议：

- 原图按内容哈希去重
- 标准化后统一尺寸
- 下载链路有断点续传 / 重试 / 超时控制

### 12.3 Step 2：封面理解

分两条子链路。

#### Step 2A：Embedding

输入：封面图

输出：

- `cover_embedding_clip_v1`

#### Step 2B：Caption / Attributes / Promise Facet

输入：封面图

输出：

- `cover_caption_short`
- `cover_caption_long`
- `cover_visual_tags`
- `promise_facets`
- 粗粒度 `evidence_facets`
- `safety_facets`

### 12.4 Step 3：文本理解

输入：

- title
- tags
- desc
- owner_name

输出：

- 文本 topic labels
- 表现形式 labels
- 实体 labels
- 同义 rewrite 候选
- 文本侧补充 facet

### 12.5 Step 4：Facet 融合

把封面理解与文本理解结果融合。

示例逻辑：

```python
final_score(label) = (
    0.55 * score_from_cover_vlm
  + 0.25 * score_from_title_tags
  + 0.20 * score_from_stats_prior
)
```

融合原则：

- 同一 facet 内做 calibrated merge
- source reliability 可配置
- 遇到冲突，优先保留多值
- safety 风险取更保守结果

### 12.6 Step 5：构造 enriched_text

把：

- 原始文本
- 封面 caption
- 视觉 tags
- facet labels

融合成一个统一可检索文本表示。

### 12.7 Step 6：写入索引与缓存

写入：

- ES：文本 + facet + prior + 部分向量引用
- RocksDB：任务状态与中间产物
- blob/object：封面与后续资源

### 12.8 Step 7：增量更新

新视频入库时：

1. 写 Mongo
2. 写 ES 原始字段
3. 根据是否进入 hot/warm 决定是否立刻抓封面
4. 异步补做多模态字段
5. 重新写回 ES enriched 索引

---

## 13. 在线检索与排序链路

### 13.1 Query 解析阶段

输入 query 后，先做：

1. normalize
2. query type 判定
3. need facets 预测
4. ambiguity 估计
5. safety requirement 判定
6. 生成 1~4 个 query hypotheses（仅在高歧义时）

### 13.2 多路召回设计

推荐第一版至少 5 路。

#### R1. Lexical Retrieval

字段：

- `title`
- `tags`
- `desc`
- `owner_name`
- `enriched_text_v1`

适合：

- 显式 topic / entity query
- known-item
- 精确关键词匹配

#### R2. Dense Text Retrieval

使用 enriched_text 的 embedding。

适合：

- 表达更长、概念更模糊的语义 query
- 标题 tags 中未直接出现的近义需求

#### R3. Cross-modal Cover Retrieval

让 query 文本直接去匹配封面 embedding。

适合：

- 养眼、氛围感、可爱、性感、搞怪、电影感等强视觉需求
- 标题不写，但封面很对味的内容

#### R4. Facet Bucket Retrieval

根据 Need → Promise/Evidence 映射，直接按 facet inverted buckets 取候选。

例如：

- `need = funny`
- boost：
  - `cover_hook = funny_expression`
  - `surface_style = chaotic`
  - `temporal_structure = multiple_highlights`

#### R5. Prior Retrieval

基于：

- 热度
- 作者质量
- 新鲜度
- 系列内容补齐

适合作为兜底和召回稳定器。

### 13.3 候选融合

建议：

- 每路取 top 300~1000
- 合并成 1000~3000 候选
- 使用 RRF 或等价融合策略
- 去重
- Safety 预过滤

### 13.4 两级重排

#### Stage 1：Text / Facet Rerank

输入：

- query
- enriched_text
- doc facet summary

输出：

- 文本和结构化信息综合相关性分

#### Stage 2：Visual Rerank

仅对 top 50~200 做：

- query + cover image + cover caption + promise facets

目的：

- 把封面真正对味的提上来
- 把文本像、封面不像的压下去

### 13.5 最终打分公式建议

```text
S(q,d) =
  w1 * S_rrf
+ w2 * S_text_rerank
+ w3 * S_visual_rerank
+ w4 * S_facet_alignment
+ w5 * S_prior
- w6 * S_risk
```

其中：

- `S_rrf`：多路召回融合分
- `S_text_rerank`：文本+facet 精排分
- `S_visual_rerank`：封面对齐分
- `S_facet_alignment`：Need 与 Promise/Evidence/Payoff 的匹配分
- `S_prior`：热度、作者质量、新鲜度
- `S_risk`：风险分

---

## 14. 模型与组件选择建议

### 14.1 第一版推荐模型角色分工

#### 封面 embedding

推荐优先：

- `jina-clip-v2`

用途：

- text → image 检索
- cover embedding 存储
- 与 query 的跨模态匹配

适合原因：

- 多语言
- 多模态
- 适合作为封面检索第一套基线

#### 封面 caption / facet 生成

推荐：

- `Qwen2.5-VL-7B`

用途：

- 封面描述
- 视觉 tags
- Promise facet 抽取
- 粗粒度 Evidence / Safety 抽取

#### 文本 embedding

推荐：

- `BGE-M3`

用途：

- enriched_text embedding
- hybrid retrieval 兼容

#### 文本 / 结构化 rerank

推荐：

- `bge-reranker-v2-m3` 或等价 cross-encoder

用途：

- 小候选集精排

### 14.2 第二版增强

- 快照理解：继续使用视觉语言模型
- ASR：语音转文本
- 音频风格模型：识别音频节奏、说话清晰度、ASMR 风格
- 行为学习模型：用真实曝光点击日志学习映射与排序

### 14.3 为什么不建议第一版就追求“一个大模型统一解决所有问题”

因为当前阶段更实际的系统策略是：

- 每个环节分治
- 输出显式中间结果
- 易于观察、易于调试、易于回滚

统一大模型在实验上也许很优雅，但在你这个阶段，代价是：

- 解释性差
- 迭代慢
- 难以定位失败点
- 难以渐进式利用现有 ES 链路

---

## 15. 冷启动：没有真实 query 时如何构建 baseline

这是当前阶段成败的关键之一。

### 15.1 不要直接“让 LLM 随便编 query”

这样做的问题：

- 容易只生成表面词
- 难覆盖真实 query 空间
- 容易模板化、同质化
- 容易忽略用户动机和观看场景

### 15.2 推荐的冷启动方法：先造 `intent × state × expression`

建议把 synthetic query 的生成目标，从“造句子”改成生成三元组：

- `need / intent`
- `user state`
- `surface expression`

例如：

- need：`funny`
- state：`下班后快速刷一会`
- expression：
  - “下班后想刷点开心的”
  - “来点轻松搞笑的”
  - “我想看能让我笑出来的”

### 15.3 种子构造方法

建议先选 8 个重点垂类：

- 搞笑 / 整活
- 萌宠
- 舞蹈
- 穿搭 / 街拍
- 美妆
- vlog
- 美食
- 游戏 / 鬼畜

每个垂类先做：

- 30 个核心 intent
- 每个 intent 人工写 20 条 query

这样：

- 8 × 30 × 20 = 4800 条高质量人工种子

然后再用 LLM 扩到 5~10 万条。

### 15.4 扩展方式

每条种子 query 建议生成以下变体：

- 口语型
- 简短型
- 长句型
- 委婉型
- 场景型
- 效果型
- 模糊表达型
- 更明确表达型

### 15.5 覆盖而不是堆数量

关键不是“生成多少 query”，而是：

- 覆盖多少 Need facets
- 覆盖多少 Promise/Evidence 映射组合
- 覆盖多少用户状态
- 覆盖多少垂类
- 覆盖多少风险边界

建议维护一个 coverage 统计表。

---

## 16. 评估体系

评估必须分成三层。

### 16.1 Facet 标注质量评估

抽样 2000~5000 条视频人工评审：

重点评估：

- Promise facet top-1/top-3 准确率
- 表现形式判断质量
- 风格标签质量
- Safety 风险识别质量

### 16.2 Offline Retrieval 评估

构建一个 query benchmark，按 query 类型分 strata：

- known-item
- explicit topic
- affective query
- aesthetic query
- consumption-context query
- tutorial-quality query
- high-ambiguity query

指标：

- Recall@50
- NDCG@10
- MRR@10
- facet satisfaction rate
- unsafe rate@20

### 16.3 Online 行为评估

上线后记录：

- impression
- click
- dwell time
- long click
- back rate
- favorite
- share
- rewatch
- reformulation

并按 query facet 分桶观测：

- affective query CTR
- exploratory query reformulation drop
- tutorial query long-click rate
- unsafe suppression success rate

---

## 17. 按你当前资源给出的阶段性实施方案

### Phase 0：冻结规格（优先做）

先冻结以下 6 件事：

1. 顶层 facet schema
2. doc/query JSON schema
3. enriched text 模板
4. Need → Promise/Evidence 映射格式
5. offline benchmark strata
6. safety 标签口径

这是最关键的“架构稳定层”。

### Phase 1：两到四周

目标：做出第一版可运行 baseline。

建议交付：

- hot set 定义
- 封面抓取链路
- cover embedding
- cover caption / promise facets
- enriched text 生成
- query parser v1
- 5 路召回原型
- text rerank

### Phase 2：一个到两个月

目标：明显提升 exploratory / 体验类 query 的质量。

建议交付：

- visual rerank
- synthetic query benchmark
- facet 映射规则库 v1
- 风险打标更稳定
- query parser 从纯 prompt 化过渡到半结构小模型

### Phase 3：两到三个月

目标：进入视频内部结构理解的初级阶段。

建议交付：

- 热门候选的快照抽样理解
- temporal structure 初版
- ASR / 音频字段接入
- Need → Promise 映射的行为学习
- 个性化 cohort prior

### Phase 4：中长期

目标：真正建立多模态视频搜索壁垒。

建议交付：

- query decomposition + multi-hypothesis retrieval
- 热门 query 的 segment-level rerank
- 强化 personalized payoff modeling
- Promise 与真实满意度之间的 gap 建模
- 交互式澄清检索

---

## 18. 关键工程任务拆解

下面给出更贴近实施的任务列表。

### 18.1 数据与缓存层

- [ ] hot set 策略定义
- [ ] cover 下载器
- [ ] cover 缓存目录结构与 hash 去重
- [ ] 下载失败重试队列
- [ ] RocksDB 状态表设计

### 18.2 视觉处理层

- [ ] cover embedding 服务
- [ ] cover caption / facet 抽取服务
- [ ] OCR 提取服务（可选）
- [ ] promise facet 融合器
- [ ] safety facet 抽取器

### 18.3 文本处理层

- [ ] title/tags/desc 规范化
- [ ] 文本侧 facet 抽取
- [ ] enriched text 构造器
- [ ] enriched_text embedding 服务

### 18.4 Query 理解层

- [ ] query normalize
- [ ] query parser prompt/schema
- [ ] ambiguity 估计器
- [ ] multi-hypothesis generator
- [ ] Need → Promise/Evidence 映射配置服务

### 18.5 检索层

- [ ] lexical retrieval
- [ ] dense text retrieval
- [ ] cross-modal cover retrieval
- [ ] facet bucket retrieval
- [ ] prior retrieval
- [ ] RRF 融合器

### 18.6 排序层

- [ ] stage-1 text/facet rerank
- [ ] stage-2 visual rerank
- [ ] safety suppressor
- [ ] final score combiner

### 18.7 评估层

- [ ] synthetic query benchmark
- [ ] 人工评测工具
- [ ] 离线指标计算脚本
- [ ] 在线曝光点击日志结构
- [ ] query 分桶报表

---

## 19. 风险、坑点与边界

### 19.1 坑 1：taxonomy 设计得过于扁平、过于大

结果会变成：

- 难维护
- 难解释
- 难覆盖
- 冷启动极差

正确做法：

- 顶层 facet 少而稳
- 叶子值持续增长

### 19.2 坑 2：query 与 doc 强行同构

这样会导致：

- 大量体验类 query 找不到合适 doc
- 只能处理显式 topic query
- exploratory 表现很差

### 19.3 坑 3：一开始就全量做快照/视频深理解

问题在于：

- 代价大
- 周期长
- 很多 query 用不到
- 延迟看到价值

正确做法：

- 先封面
- 后快照
- 只对热集 / 高价值候选做正文增强

### 19.4 坑 4：直接拿点击当真标签

上线后容易出现：

- 位置偏差
- 曝光偏差
- 标题党/封面党被过拟合
- 旧系统偏见被复制

正确做法：

- 先做 bias-aware 数据使用
- 保留 exploration bucket
- 明确区分“被看到但没点”与“根本没看到”

### 19.5 坑 5：安全边界不独立建模

如果把：

- 吸引力
- 点击率
- 性感 / 擦边
- 风险边界

混在一起学，后续会很难收敛，且工程上会带来很大隐患。

正确做法：

- Safety 作为横切层单独建模
- 参与过滤、降权、复核
- 不要简单和 relevance 混为一个分

---

## 20. 推荐的第一版交付物

第一版不应追求“全量最强视频多模态搜索”。

更合理的交付定义是：

> 一个基于封面 + enriched text + query need parsing 的视频搜索 baseline。

它至少应具备以下能力：

1. 显式 topic / entity query 不弱于 title/tags-only 系统
2. exploratory / affective / aesthetic query 明显更好
3. query 可以被解析为 Need facets
4. doc 可以被解析为 Promise facets 和部分 Evidence facets
5. 有可回溯、可版本化、可扩展的 facet schema
6. 有基础的安全约束机制
7. 有离线 benchmark 与上线后日志采集闭环

---

## 21. 后续可扩展方向

### 21.1 快照增强

只对：

- top-k 候选
- 热门 query
- 高价值垂类

做快照抽样理解和二次重排。

### 21.2 音频与 ASR

增强：

- 说话清晰度
- 音乐主导 / 人声主导
- ASMR / 环境音
- 教程可听性

### 21.3 Personalized Cohort Priors

不是一开始就做强个性化，而是先做：

- 二次元偏好 cohort
- 摄影党 cohort
- 车迷 cohort
- 宠物党 cohort
- 穿搭 / 舞蹈 / 美妆偏好 cohort

### 21.4 Segment-level Retrieval

对高价值 query，进一步升级为：

- 视频级召回
- 片段级定位
- 关键高光重排

### 21.5 Promise / Payoff Gap Modeling

长期非常有价值的一个方向：

- 看起来很吸引，但看完不满足
- 标题党 / 封面党
- 点击高但长停留低

这是你后续从“点击导向搜索”走向“满意度导向搜索”的关键。

---

## 22. 一页式结论

### 核心判断

你当前的根问题不是“文本召回器不够强”，而是：

- 用户 query 主要表达的是 Need / Payoff
- 你的 doc 当前显式提供的是文本题材与部分 Promise
- 视频搜索又天然多出时间结构、表现形式、音频和观看方式这些维度

### 正确方向

把系统升级成：

- **Need → Promise → Evidence → Payoff** 四层 facet system
- 其中：
  - query 主建模 Need
  - doc v1 主建模 Promise
  - 用封面 + enriched text 粗推 Evidence / Payoff
  - 后续再用快照、ASR、日志补强

### 第一版最值得做的事

1. 冻结 facet schema / doc schema / query schema
2. 做 hot set 封面下载
3. 做封面 embedding + cover caption + promise facet
4. 做 enriched text
5. 做 query parser + 5 路召回 + 两级重排 baseline
6. 做 synthetic query benchmark
7. 做 safety 横切层

### 你此刻最不该做的事

- 一开始就全量下载 10TB+ 封面并做全量重资产处理
- 一开始就全量做快照 / 正文深理解
- 一开始就依赖点击日志直接训练排序模型
- 设计一个庞大扁平的“万能标签表”

### 最现实的第一阶段目标

先让以下 query 明显比原来更强：

- 让我开心的
- 下饭刷的
- 养眼的 / 有氛围的
- 讲得清楚的教程
- 某种感觉 / 某种风格 / 某种观看场景下的视频

只要这一步跑通，你就不再只是一个“视频元数据文本搜索器”，而是在向“多模态视频意图搜索系统”演进。

---

如果要继续往下拆，最合适的下一份文档有四种：

1. `facet label 枚举表 v1`（把各一级 facet 的二级值表详细列出来）
2. `Elasticsearch mapping 设计文档`
3. `离线 DAG 任务拆分与调度设计`
4. `query parser / cover facet tagger 的 prompt 与输出协议`

