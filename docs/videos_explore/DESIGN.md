# 视频搜索与探索系统 — 设计文档

## 目录

- [1. 系统概述](#1-系统概述)
- [2. 架构设计](#2-架构设计)
  - [2.1 模块结构](#21-模块结构)
  - [2.2 类继承与数据流](#22-类继承与数据流)
  - [2.3 依赖组件](#23-依赖组件)
- [3. 搜索流程](#3-搜索流程)
  - [3.1 查询模式 (qmod)](#31-查询模式-qmod)
  - [3.2 统一入口 unified_explore](#32-统一入口-unified_explore)
  - [3.3 词语搜索流程 (explore_v2)](#33-词语搜索流程-explore_v2--多车道召回)
  - [3.4 向量搜索流程 (knn_explore_v2)](#34-向量搜索流程-knn_explore_v2)
  - [3.5 混合搜索流程 (hybrid_explore_v2)](#35-混合搜索流程-hybrid_explore_v2)
- [4. 核心机制](#4-核心机制)
  - [4.1 DSL 查询解析](#41-dsl-查询解析)
  - [4.2 向量嵌入与 LSH](#42-向量嵌入与-lsh)
  - [4.3 多车道召回 (Multi-Lane Recall)](#43-多车道召回-multi-lane-recall)
  - [4.4 多轮召回与 UP 主意图检测](#44-多轮召回与-up-主意图检测)
  - [4.5 召回池优化器 (RecallPoolOptimizer)](#45-召回池优化器-recallpooloptimizer)
  - [4.6 召回噪声过滤 (Noise Filtering)](#46-召回噪声过滤-noise-filtering)
  - [4.7 精排重排 (Reranking)](#47-精排重排-reranking)
  - [4.8 排序策略](#48-排序策略)
  - [4.9 多样化排序 (Diversified Ranking)](#49-多样化排序-diversified-ranking)
  - [4.10 UP 主意图感知排序](#410-up-主意图感知排序)
  - [4.11 UP 主分组](#411-up-主分组)
  - [4.12 窄过滤器处理](#412-窄过滤器处理)
- [5. ES 索引设计](#5-es-索引设计)
  - [5.1 索引结构](#51-索引结构)
  - [5.2 文本字段与分词](#52-文本字段与分词)
  - [5.3 向量字段](#53-向量字段)
- [6. 返回数据结构](#6-返回数据结构)
  - [6.1 步骤结果格式](#61-步骤结果格式)
  - [6.2 单条命中结果](#62-单条命中结果)
- [7. 性能考量](#7-性能考量)

---

## 1. 系统概述

本系统是基于 Elasticsearch 的 B 站视频搜索引擎，核心目标是**识别和命中用户的搜索意图**。支持三种搜索模式：

| 模式 | 代号 | 说明 |
|------|------|------|
| 词语搜索 | `q=w` | 基于 BM25 的关键词匹配搜索 |
| 向量搜索 | `q=v` | 基于 LSH bit vector 的 KNN 语义搜索 |
| 混合搜索 | `q=wv` | 词语 + 向量并行搜索，RRF 融合排序 |

每种模式均可附加精排重排（`r`），如 `q=vr`、`q=wr`、`q=wvr`。

系统索引使用 **8 个分片**，支持丰富的 DSL 过滤表达式（日期、播放量、UP 主、BV 号等）。每条文档包含预计算的 `stat_score`（由 `blux.doc_score.DocScorer` 在入库时计算），用于衡量文档质量。

**核心pipeline**：
```
查询 → DSL 解析 → 多车道召回 (6 lanes) → 多轮补充 (3 rounds)
     → 召回池优化 (PoolHints) → 噪声过滤 → 多样化排序 (3 phases)
     → UP 主分组 → 返回结果
```

---

## 2. 架构设计

### 2.1 模块结构

```
elastics/
├── structure.py          # ES 查询构造工具函数
├── es_logger.py          # ES 调试日志
├── videos/
│   ├── constants.py      # 搜索/KNN/排序等全部常量
│   ├── searcher_v2.py    # VideoSearcherV2 — 底层搜索引擎
│   ├── explorer.py       # VideoExplorer — 多步骤探索引擎
│   └── hits.py           # VideoHitsParser — ES 响应解析
├── tests/
│   ├── test_videos.py         # 综合测试
│   ├── test_optimization_v2.py # 7 查询 recall+rank 质量评估
│   ├── diag_deep.py           # 向量质量诊断
│   ├── diag_float_vs_lsh.py   # Float vs LSH 对比诊断
│   ├── diag_knn.py            # KNN 召回诊断
│   ├── diag_owner_recall.py   # UP主召回分数诊断
│   ├── diag_single.py         # 单查询诊断 (top-20 详情)
│   └── compare_official.py    # 与官方 Bilibili 结果对比

recalls/                       # 多车道召回模块
├── __init__.py           # 模块文档
├── base.py               # RecallResult / RecallPool / NoiseFilter
├── word.py               # MultiLaneWordRecall — 6 车道词语召回
├── vector.py             # VectorRecall — 向量+词语补充召回
├── manager.py            # RecallManager — 多轮召回策略编排
├── optimizer.py          # RecallPoolOptimizer — 召回池特征分析 ← NEW
└── tests/
    ├── test_base.py           # RecallPool 合并测试
    ├── test_diversified.py    # 多样化排序测试
    ├── test_noise.py          # 噪声过滤测试
    └── test_recall_optimization.py  # 召回优化测试

converters/
├── embed/
│   └── embed_client.py   # TextEmbedClient — 文本嵌入客户端
├── dsl/
│   ├── elastic.py        # DSL 表达式 → ES 查询转换
│   ├── rewrite.py        # DSL 重写
│   ├── filter.py         # 过滤器合并
│   └── fields/
│       └── qmod.py       # 查询模式解析 (q=w/v/wv/r)

ranks/
├── constants.py          # 排序相关所有常量
├── ranker.py             # VideoHitsRanker — 排序引擎
├── diversified.py        # DiversifiedRanker — 多样化排序 + UP主意图感知
├── reranker.py           # EmbeddingReranker — 精排
├── grouper.py            # AuthorGrouper — UP主分组
├── scorers.py            # 评分器：Stats/Pubdate/Relate
└── fusion.py             # ScoreFuser — 分数融合
```

### 2.2 类继承与数据流

```
VideoSearcherV2
    ├── search()              # 词语搜索
    ├── knn_search()          # KNN 搜索
    ├── hybrid_search()       # 混合搜索
    ├── filter_only_search()  # 纯过滤
    ├── random() / latest() / doc()
    └── ...
        │
        ▼
VideoExplorer (继承 VideoSearcherV2)
    ├── _run_explore_pipeline()     # 共享探索管线
    ├── explore_v2()           # 词语探索 → _run_explore_pipeline(mode="word")
    ├── knn_explore_v2()       # 向量探索 → _run_explore_pipeline(mode="vector")
    ├── hybrid_explore_v2()    # 混合探索 → _run_explore_pipeline(mode="hybrid")
    ├── unified_explore()      # 统一分发入口 → 路由到 V2 方法
    └── _filter_only_explore() # 无关键词回退 (纯过滤 + stats/recency 排序)

RecallManager (多轮召回策略编排)
    ├── MultiLaneWordRecall   # 6 车道并行词语召回
    ├── VectorRecall          # 向量召回 + 词语补充
    ├── RecallPoolOptimizer   # 召回池特征分析 → PoolHints
    ├── _detect_owner_intent()    # UP主意图检测
    ├── _owner_focused_recall()   # UP主定向召回 (Round 3)
    └── pool.filter_noise()       # 合并后噪声过滤 (3 信号)

NoiseFilter (噪声过滤)
    ├── filter_by_score_ratio()         # BM25 车道噪声过滤
    ├── filter_knn_by_score_ratio()     # KNN 向量噪声过滤
    └── apply_content_quality_penalty() # 短文本/低互动惩罚

DiversifiedRanker (三阶段排序 + UP主意图感知)
    ├── _score_all_dimensions()   # 5 维评分 (接收 pool_hints)
    ├── _select_headline_top_n()  # Phase 1: 头部质量选择 (top-3)
    ├── _allocate_slots()         # Phase 2: 相关性门控槽位分配 (位置 4-10)
    └── fused scoring             # Phase 3: 融合评分 (>10)

RecallPoolOptimizer (召回池优化器 — NEW)
    ├── analyze()                 # 主入口 → PoolHints
    ├── _analyze_owners()         # UP主集中度/意图分析
    ├── _extract_title_keywords() # 标题关键词提取
    └── _analyze_tags()           # 标签频率分析
```

**数据流**：
```
RecallManager
  ├── Round 1/2/3 → RecallPool (hits + lane_tags + _owner_matched flags)
  ├── RecallPoolOptimizer.analyze() → PoolHints
  └── pool.pool_hints = PoolHints
         │
         ▼
VideoExplorer._fetch_and_rank()
  ├── pool_hints → VideoHitsRanker.rank()
  └── pool_hints → DiversifiedRanker._score_all_dimensions()
         │
         ▼
使用 PoolHints.owner_analysis.intent_strength 代替重新计算
  → 更准确的 UP主 意图评估 → 更好的排序
```

### 2.3 依赖组件

| 组件 | 类 | 说明 |
|------|----|------|
| Elasticsearch | `ElasticOperator` | 搜索、KNN、聚合 |
| MongoDB | `MongoOperator` | UP 主头像等补充信息 |
| TEI 嵌入服务 | `TextEmbedClient` | 文本 → 向量（float 1024 维 + LSH 2048 bit） |
| DSL 解析器 | `DslExprRewriter` + `DslExprToElasticConverter` | 查询重写与 ES 查询构建 |
| 召回管理器 | `RecallManager` | 多轮多车道召回策略编排 + 噪声过滤 + 优化 |
| 召回池优化器 | `RecallPoolOptimizer` | 分析召回池特征 → 生成 PoolHints |
| 排序引擎 | `VideoHitsRanker` | 多策略排序（diversified/stats/relevance/tiered/rrf） |
| 多样化排序器 | `DiversifiedRanker` | 三阶段排序 + UP主意图感知（默认） |
| 精排器 | `EmbeddingReranker` | 基于 float 向量余弦相似度的精排 |
| 分组器 | `AuthorGrouper` | 按 UP 主聚合结果 |

---

## 3. 搜索流程

### 3.1 查询模式 (qmod)

用户通过 DSL 表达式 `q=<mode>` 指定搜索模式：

| 表达式 | 模式列表 | 分发方法 | 排序策略 | 精排 |
|--------|----------|----------|----------|------|
| `q=w` | `["word"]` | `explore_v2()` | diversified | 否 |
| `q=wr` | `["word", "rerank"]` | `explore_v2()` | diversified | **是** |
| `q=v` | `["vector"]` | `knn_explore_v2()` | diversified | **始终开启** |
| `q=vr` | `["vector", "rerank"]` | `knn_explore_v2()` | diversified | **始终开启** |
| `q=wv` | `["word", "vector"]` | `hybrid_explore_v2()` | diversified | 否 |
| `q=wvr` | `["word", "vector", "rerank"]` | `hybrid_explore_v2()` | diversified | **是** |
| 无指定 | `["word", "vector"]` | `hybrid_explore_v2()` | diversified | 否 |

> **重要**：`q=v` 模式始终开启精排，因为原始 KNN hamming 分数精度不足以产生有意义的排序。

### 3.2 统一入口 unified_explore

`unified_explore()` 是所有搜索请求的统一入口：

```
用户查询 → 提取 qmod → 判断搜索模式组合 → 分发到对应 V2 explore 方法
```

处理逻辑：
1. 解析 DSL 表达式，提取 `qmod` 字段
2. 如无 `qmod`，使用默认配置 `QMOD = ["word", "vector"]`（混合搜索）
3. 判断是否包含 `word`、`vector`、`rerank`
4. 分发到 `explore_v2()` / `knn_explore_v2()` / `hybrid_explore_v2()`
5. 在结果中附加 `qmod` 信息

### 3.3 词语搜索流程 (explore_v2 — 多车道召回)

**6 车道并行召回 + 3 轮渐进式补充 + 多样化排序**

```
┌─ Step 0: construct_query_dsl_dict ─────────────────────────┐
│  解析 DSL → 构建 boosted fields → 生成 ES 查询字典         │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ Step 1: multi_lane_recall (3 轮) ─────────────────────────┐
│  ┌───────────────────────────────────────────────────────────────┐│
│  │           ThreadPoolExecutor (6 并行)                         ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        ││
│  │  │ 相关性   │ │ 标题匹配 │ │ UP主名称 │ │ 热度     │        ││
│  │  │ BM25     │ │ title+   │ │ owner+   │ │ stat.view│        ││
│  │  │ _score   │ │ tags     │ │ title    │ │ desc     │        ││
│  │  │ 600 条   │ │ 400 条   │ │ 200 条   │ │ 300 条   │        ││
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘        ││
│  │       └────────┬────┴────────┬───┴──────────┘                ││
│  │                └─────────────┘                                ││
│  │            + ┌──────────┐ ┌──────────┐                        ││
│  │              │ 时效性   │ │ 质量     │                        ││
│  │              │ pubdate  │ │stat_score│                        ││
│  │              │ desc     │ │ desc     │                        ││
│  │              │ 250 条   │ │ 300 条   │                        ││
│  │              └────┬─────┘ └────┬─────┘                        ││
│  └───────────────────┴────────────┴──────────────────────────────┘│
│                         │                                         │
│  Round 1: 合并 + 去重 (by bvid), 标记 _title_matched/_owner_matched│
│  Round 2: (如不足) 扩大搜索范围补充召回                            │
│  Round 3: (如检测到UP主意图) UP主定向召回 → 合成分数注入           │
│                         │                                         │
│  RecallPoolOptimizer.analyze() → 生成 PoolHints                   │
│  约 500-1200 候选文档, 每条文档带 lane_tags                       │
└────────────────────────────────────────────────────────────────────┘
                         │
                    (可选: rerank)
                         │
                         ▼
┌─ fetch_and_rank (pool_hints 传递) ─────────────────────────┐
│  ① fetch_docs_by_bvids: 获取完整文档                        │
│  ② diversified_rank: 三阶段排序 (使用 pool_hints)            │
│     Phase 1: 头部质量选择 (top-3 by headline_score)          │
│     Phase 2: 相关性门控槽位分配 (位置 4-10)                  │
│     Phase 3: 融合评分 (剩余文档)                              │
│  ③ UP主意图感知: owner_intent_strength 影响排序逻辑          │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ Step 2: group_hits_by_owner ──────────────────────────────┐
│  按 UP 主分组，添加 MongoDB 头像，按首次出现顺序排列       │
└────────────────────────────────────────────────────────────┘
```

### 3.4 向量搜索流程 (knn_explore_v2)

```
┌─ Step 0: construct_knn_query ──────────────────────────────┐
│  ① 解析 DSL → 提取过滤条件                                 │
│  ② 判断窄过滤器 (用户/BV 号过滤)                            │
│  ③ VectorRecall 协调嵌入和检索                               │
└────────────────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
    ┌─ 窄过滤器 ─┐          ┌─ 常规查询 ──────────┐
    │ dual_sort   │          │ KNN search           │
    │ filter      │          │ + 词语补充召回        │
    │ search      │          │ 并行执行              │
    └─────────────┘          └──────────────────────┘
                         │
                         ▼
┌─ Step 1.5: word_recall_supplement ─────────────────────────┐
│  合并词语召回结果到 KNN 池中 (去重 by bvid)                │
│  扩大精排候选池，补充 KNN 语义漏召的关键词匹配结果          │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ Step 2: rerank ───────────────────────────────────────────┐
│  ① 对合并后的候选池 (KNN + 词语召回) 进行 float 向量精排   │
│  ② cosine similarity × keyword_boost                       │
│  ③ 按精排分数重排                                           │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ fetch_and_rank (pool_hints 传递) ─────────────────────────┐
│  获取完整文档 → diversified_rank (意图感知) → 字符级高亮   │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ Step 3: group_hits_by_owner ──────────────────────────────┐
│  按 UP 主分组，按首次出现顺序排列                          │
└────────────────────────────────────────────────────────────┘
```

### 3.5 混合搜索流程 (hybrid_explore_v2)

```
┌─ Step 0: construct_query_dsl_dict ─────────────────────────┐
│  解析查询，提取关键词，设置 qmod=["word","vector"]          │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ Step 1: hybrid_recall (3 轮) ────────────────────────────┐
│  RecallManager 协调两种召回:                                │
│  ┌─────────────┐  ┌──────────────┐                        │
│  │  6 车道词语  │  │  KNN 向量    │   ← 并行执行           │
│  │  recall      │  │  recall      │                        │
│  └──────┬──────┘  └──────┬───────┘                        │
│         └───────┬────────┘                                │
│                 ▼                                          │
│   RecallPool.merge → 合并去重                               │
│   Round 2: 补充召回 (如不足)                                │
│   Round 3: UP主定向召回 (如有意图)                          │
│   RecallPoolOptimizer → PoolHints                          │
│   保留 lane_tags, _owner_matched, _title_matched           │
└────────────────────────────────────────────────────────────┘
                         │
                    (可选: rerank)
                         │
                         ▼
┌─ fetch_and_rank (pool_hints 传递) ────────────────────────┐
│  fetch_docs → diversified_rank (意图感知) → group_by_owner│
└───────────────────────────────────────────────────────────┘
```

---

## 4. 核心机制

### 4.1 DSL 查询解析

用户查询支持丰富的 DSL 表达式语法：

```
影视飓风 d>2024-01-01 v>1w u="某UP主" q=vr
```

解析流程：
1. **DslExprRewriter** 将查询文本解析为 DSL 表达式树
2. **DslExprToElasticConverter** 将表达式树转换为 ES 查询字典
3. **QueryDslDictFilterMerger** 合并额外的过滤条件

支持的 DSL 字段类型：

| 字段 | 示例 | 说明 |
|------|------|------|
| `d` (date) | `d>2024`, `d=7d`, `d=[1d,3d]` | 日期/时间过滤 |
| `v` (view) | `v>1w`, `v=[1k,10000]` | 播放量过滤 |
| `u` (user) | `u="影视飓风"`, `u=红警HBK08` | UP 主过滤 |
| `q` (qmod) | `q=w`, `q=vr`, `q=wvr` | 搜索模式 |
| `bv` (bvid) | `bv=BV1xx...` | BV 号过滤 |
| `coin/like/...` | `:coin>1000` | 各类统计量过滤 |

### 4.2 向量嵌入与 LSH

**嵌入管线：**

```
查询文本 → TEI 服务 → 1024 维 float 向量 → LSH → 2048 bit 向量 → hex 字符串 → byte 数组
```

**文档嵌入文本格式：**
```
【UP主名】 标题 (标签) 简介
```
由 `blux.text_doc.build_sentence()` 构建。

**索引存储：**
- ES 字段类型：`dense_vector`，`element_type="bit"`，`dims=2048`
- 相似度：`l2_norm`（对 bit vector 即 Hamming 距离）
- 索引方法：HNSW（m=16, ef_construction=100）

**ES 评分公式：**
```
score = (2048 - hamming_distance) / 2048
```

**已验证结论：**
LSH 2048-bit 向量与 float 1024 维向量的排序结果几乎一致（经 `diag_float_vs_lsh.py` 验证）。KNN 召回问题的根本原因是嵌入模型对实体名的语义误解（如将"影视飓风"理解为"影视+飓风"），而非 LSH 精度损失。

### 4.3 多车道召回 (Multi-Lane Recall)

**问题**：旧的单次 ES 搜索（limit=10000）召回维度单一、速度慢。只按 BM25 相关性取 top-k，无法保证召回高热度、高质量、高时效性的文档。

**解决方案**：`recalls/` 模块实现 6 车道并行召回策略。

#### 6 车道词语召回 (MultiLaneWordRecall)

同一个查询，6 个车道并行执行不同排序/字段组合的 ES 查询：

| 车道 | 排序依据 | 取回量 | 搜索字段 | 目标 |
|------|----------|--------|----------|------|
| `relevance` | BM25 `_score` | 600 | 全字段 (title, tags, owner.name, desc) | 相关性最高的文档 |
| `title_match` | BM25 `_score` | 400 | title.words (5.0) + tags.words (3.0) | 标题/标签精准匹配 |
| `owner_name` | BM25 `_score` | 200 | owner.name.words (8.0) + title.words (3.0) + tags.words (2.0) | UP主名称匹配 |
| `popularity` | `stat.view` desc | 300 | 全字段 (match filter) | 热度最高的文档 |
| `recency` | `pubdate` desc | 250 | 全字段 (match filter) | 时间最近的文档 |
| `quality` | `stat_score` desc | 300 | 全字段 (match filter) | 质量最高的文档 |

> **6 车道设计说明**：
> - `title_match` 搜索 `title.words` (boost=5.0) 和 `tags.words` (boost=3.0)，确保标题或标签匹配查询的文档一定被召回。标签包含实体名称、电影标题、主题关键词等强信号。
> - `owner_name` 专门针对 UP主名称匹配，使用 `owner.name.words` (boost=8.0) 高权重，确保创作者相关文档被召回。这是 UP主意图检测的基础。
> - `popularity` — 按原始曝光量排序 — 与 `quality`（stat_score = 综合满意度）定位不同。

**召回后标记：**
- `_tag_title_matches()` — 为标题**或标签**包含查询关键词的文档标记 `_title_matched=True`
- `_tag_owner_matches()` — 使用三重策略为 UP主名匹配的文档标记 `_owner_matched=True` 和 `_matched_owner_name`

**UP主名匹配三重策略：**
1. **Token-set 匹配**：查询所有 token 出现在 UP主名 token 中（如 `红警08` → `红警HBK08`）
2. **CJK 子串匹配**：纯中文查询出现在 UP主名中（如 `米娜` → `大聪明罗米娜`）
3. **通用子串匹配**：所有查询 token 出现在 UP主名中

**合并逻辑** (`RecallPool.merge`)：
- 按 `bvid` 去重，同一文档可能出现在多个车道
- 每条文档记录 `lane_tags`（如 `{"relevance", "owner_name", "popularity"}`）
- 传播元数据标记：`_title_matched`、`_owner_matched`、`_owner_lane`、`_matched_owner_name`
- 保留各车道最高分数
- 合并后的候选池约 500-1200 条

**性能**：6 个并行 ES 查询 vs 1 个扫描 10000 条的查询。每个车道的 `size` 远小于 10000，且并行执行，总延迟约等于单个最慢车道的延迟。

#### 向量召回 (VectorRecall)

- **广泛查询**：KNN搜索 + 并行词语补充召回 → 合并去重
- **窄过滤器查询**：`dual_sort_filter_search()` 按时间和热度双排序

#### RecallManager

根据搜索模式自动选择召回策略：

| 模式 | 召回策略 |
|------|----------|
| `word` | `MultiLaneWordRecall` |
| `vector` | `VectorRecall` |
| `hybrid` | 两者并行 → `RecallPool.merge` |

合并后自动调用 `pool.filter_noise()` 移除噪声文档（见 [4.6 召回噪声过滤](#46-召回噪声过滤-noise-filtering)）。

### 4.4 多轮召回与 UP 主意图检测

RecallManager 实现三轮渐进式召回，逐步完善候选集：

| 轮次 | 名称 | 触发条件 | 说明 |
|------|------|----------|------|
| Round 1 | 标准召回 | 始终执行 | 6 车道词语/向量/混合召回 + 噪声过滤 |
| Round 2 | 补充召回 | `len(pool) < target_count` | 更大搜索范围填充缺口 (relevance_broad 1500+, popularity_broad 1000+, quality_broad 1000+) |
| Round 3 | UP主定向召回 | 检测到 UP主 意图 | 从检测到的 UP主 补充内容 (top-50 per owner) |

#### UP主 意图检测 (`_detect_owner_intent`)

分析 Round 1 结果中的 `owner.name` 字段，使用与 `_tag_owner_matches` 相同的三重策略判断查询是否包含 UP主 意图：

1. **Token-set 匹配**：查询所有 token 出现在 UP主 名称 token 中（如 `红警08` → `红警HBK08`）
2. **CJK 子串匹配**：纯中文查询出现在 UP主 名称中（如 `米娜` → `大聪明罗米娜`）
3. **通用子串匹配**：所有查询 token 出现在 UP主 名称中

**UP主 集中度分析**（区分"UP主查询"和"话题查询"）：
- **查询-名称长度比**：`len(query) / len(owner_name)` < 0.40 且匹配 UP主 ≥ 2 → 抑制（如 `米娜` / `大聪明罗米娜` = 0.33）
- **分散度检查**：匹配 UP主 ≥ 6 且无主导 UP主 → 抑制
- **主导性检查**：最多 doc 的 UP主 < 5 个 doc 且匹配 UP主 > 2 → 抑制

UP主 定向召回使用 `owner.name.keyword` 精确匹配从 ES 过滤查询该 UP主 的 top-50 文档（按 `stat_score` 降序），确保被检测到的创作者有足够的候选文档参与排序。

**合成分数注入**：UP主定向召回使用 ES filter 查询，不产生 BM25 `_score`。为使这些文档能在排序中与其他文档竞争，系统从当前池的 BM25 分数分布中取 **p25 分位数** 作为合成分数注入这些文档。

### 4.5 召回池优化器 (RecallPoolOptimizer)

**模块**：`recalls/optimizer.py`

RecallPoolOptimizer 在所有召回轮次完成后，对完整召回池进行特征分析，生成 `PoolHints` 供排序阶段使用。这是**召回信息到排序信息**的桥梁。

#### PoolHints 数据结构

```python
@dataclass
class PoolHints:
    owner_analysis: OwnerAnalysis      # UP主集中度、意图强度
    query_type: str                    # "owner" / "topic" / "mixed"
    content_types: ContentTypeDistribution  # 标签频率分布
    title_keywords: list[tuple[str, int]]  # 高频标题关键词
    score_dist: ScoreDistribution      # BM25 分数分布 (max, min, p25, p75, std...)
    view_dist: ScoreDistribution       # 播放量分布
    suggested_owner_bonus: float       # 建议的 UP主 加成 (0.05-0.35)
    suggested_title_bonus: float       # 建议的标题加成 (default 0.20)
    noise_gate_ratio: float            # 建议的噪声门限 (基于 CV)
    summary: str                       # 人类可读摘要
```

#### OwnerAnalysis

```python
@dataclass
class OwnerAnalysis:
    total_owners: int              # 池中不同 UP主 总数
    total_owner_matched: int       # 匹配查询的 UP主 文档数
    dominant_owner: str            # 最多文档的匹配 UP主
    dominant_owner_count: int      # 该 UP主 的文档数
    concentration: float           # 集中度 = dominant_count / total_matched
    diversity: float               # 多样性 = num_unique_matched / 6.0 (cap 1.0)
    intent_strength: float         # 意图强度 [0, 1] — 核心输出
    matched_owners: list[tuple[str, int]]  # 所有匹配 UP主 及其文档数
```

**意图强度计算**：
```
intent_strength = concentration × (1 - diversity × 0.7)
```
- 集中度高 (少数 UP主 占多数文档) + 多样性低 (匹配 UP主 少) → 强 UP主 意图
- ≤2 匹配 UP主 且主导 UP主 ≥3 文档 → 额外 +0.3
- ≥5 匹配 UP主 且集中度 <0.3 → ×0.2 (话题查询特征)

**查询类型分类**：
- `intent_strength ≥ 0.6` → `"owner"`（UP主查询，如 `红警08`）
- `0.3 ≤ intent_strength < 0.6` → `"mixed"`（混合查询）
- `intent_strength < 0.3` → `"topic"`（话题查询，如 `蝴蝶刀`、`chatgpt`）

#### 数据流

```
RecallManager.recall()
    ↓ (Round 1/2/3 完成)
RecallPoolOptimizer.analyze(pool.hits, query, pool.lane_tags)
    ↓
pool.pool_hints = PoolHints(...)
    ↓
VideoExplorer._fetch_and_rank(pool_hints=pool.pool_hints)
    ↓
VideoHitsRanker.rank(pool_hints=pool_hints)
    ↓
DiversifiedRanker._score_all_dimensions(pool_hints=pool_hints)
    ↓
if pool_hints.owner_analysis available:
    owner_intent_strength = pool_hints.owner_analysis.intent_strength
else:
    owner_intent_strength = _analyze_owner_intent_strength(hits, ...)
```

### 4.6 召回噪声过滤 (Noise Filtering)

**问题**：召回阶段会引入大量噪声文档：
- **BM25 稀有关键词噪声**：IDF 高的稀有词产生高 BM25 分数，即使文档与查询语义无关
- **KNN bit vector 噪声**：LSH 量化后 hamming 距离区分度有限

**解决方案**：三层 + 三信号噪声过滤架构

```
                    ES 层                    车道层                    池层 (三信号)
                ┌──────────┐          ┌──────────────┐         ┌────────────────────┐
                │min_score │          │ NoiseFilter.  │         │ RecallPool         │
   Query ──→   │ = 3.0    │ ──→      │ filter_by_    │  ──→    │ .filter_noise()    │ ──→ 排序
                │(ES body) │          │ score_ratio() │         │ ① score-ratio      │
                └──────────┘          │ + content     │         │ ② short-text pen.  │
                    │                 │   quality     │         │ ③ low-engage pen.  │
              不返回 BM25             │   penalty     │         └────────────────────┘
              分数 < 3.0 的           └──────────────┘
              文档
```

#### 第一层：ES 级过滤 (`min_score`)
在 ES 搜索请求 body 中设置 `"min_score": 3.0`（`MIN_BM25_SCORE`），阻止 BM25 分数极低的文档被返回。仅适用于非相关性车道（popularity/recency/quality 车道按其他字段排序，BM25 分数可能偏低但文档仍有价值）。

#### 第二层：车道级过滤 (`NoiseFilter`)
每个车道返回的 hits 经过 `filter_by_score_ratio()` 过滤：
- **BM25 车道**：移除 `score < max_score × NOISE_SCORE_RATIO_GATE (0.10)` 的命中
- **KNN 车道**：使用更严格的比率 `score < max_score × NOISE_KNN_SCORE_RATIO (0.50)`
- **最小数量保护**：若总命中 ≤ 30 条，跳过过滤以避免结果过少
- **内容质量预处理** (`apply_content_quality_penalty`)：在 score-ratio 过滤之前，对短文本和低互动文档的分数施加惩罚

#### 第三层：池级过滤 (`RecallPool.filter_noise()` — 三信号)
合并后的召回池使用三信号进行最终过滤：

1. **Score-ratio 门控**：全局 gate = `max_score × ratio`
2. **短文本惩罚**：`title + desc` 长度 < 15 字符的文档，有效分数 × 0.3
3. **低互动惩罚**：播放量 < 100 的文档，有效分数 × 0.15

- **多车道感知**：出现在 2+ 个车道的文档使用更低阈值（`gate × 0.5`），因为跨车道出现本身就是相关性信号

### 4.7 精排重排 (Reranking)

精排使用 float 向量的余弦相似度替代 LSH hamming 相似度进行精细排序：

**公式：**
```
final_score = cosine_similarity(query_vec, doc_vec) × keyword_boost_multiplier
```

**关键词加权规则：**

| 匹配位置 | 加权公式 | 默认值 |
|----------|----------|--------|
| 标题/UP 主名 | `1 + title_keyword_boost × match_count` | boost=2.0 |
| 标签 | `1 + keyword_boost × count × 0.8` | boost=1.5 |
| 简介 | `1 + keyword_boost × count × 0.3` | boost=1.5 |

**性能参数：**
- `RERANK_MAX_HITS = 2000`（最大精排候选数）
- `RERANK_TIMEOUT = 30s`
- `RERANK_MAX_PASSAGE_LENGTH = 4096` 字符

### 4.8 排序策略

系统支持 5 种排序策略，由 `VideoHitsRanker` 实现：

#### heads — 首部截取
直接截取前 top_k 条结果，不做任何评分。用于快速返回。

#### stats — 统计排序
```
rank_score = (STATS_BASE + stats_score) × pubdate_score × (relate_score³)
```
- **stats_score**：由 `blux.doc_score.DocScorer` 计算的文档质量分，∈ [0, 1)
- **pubdate_score**：分段线性衰减（今天=4.0, 7 天=1.0, 30 天=0.6, 1 年=0.3）
- **relate_score**：搜索相关度分数，门控 + 幂变换

#### relevance — 纯相关度排序
```
rank_score = transform(score, power=2.0, min=0.4, high_threshold=0.85, high_boost=2.0)
```

#### tiered — 分层排序
- **高相关区**（≥ max_score × 0.7）：按 `0.7 × stats_score + 0.3 × recency` 排序
- **低相关区**（< max_score × 0.7）：按纯相关度排序

#### rrf — 倒数排序融合
```
rrf_score = Σ weight[i] / (k + rank[i])
```

### 4.9 多样化排序 (Diversified Ranking)

**问题**：连续加权融合 (`w × quality + w × relevance + w × recency`) 导致 Top 10 同质化 — 所有结果都是各维度中等偏上的"均衡型"文档。

**解决方案**：`DiversifiedRanker` 使用**三阶段排序**算法，保证 Top-3 质量，Top-10 多样性，并通过**相关性门控**防止不相关文档占据 top-10。

#### 5 维评分体系

每条文档计算 5 维分数：

| 维度 | 来源 | 归一化方式 | 说明 |
|------|------|-----------|------|
| `relevance_score` | BM25 `_score` | 除以 `max_score` | 经过**内容深度惩罚** + **标题关键词覆盖** + **TM/OM 加成** + **相关性底线** |
| `quality_score` | `stat_score` | DocScorer 已归一化 [0,1) | 时长 < 30s 额外 ×0.8 |
| `recency_score` | `pubdate` | 线性衰减 | 时间越近越高 |
| `popularity_score` | `stat.view` | `log1p(view) / max_log_view` | **对数归一化**抑制幂律分布 |
| `headline_score` | 复合 | 加权融合 4 维 | 0.50×rel + 0.30×qual + 0.10×rec + 0.10×pop |

**Popularity 对数归一化**：播放量呈幂律分布（如 100M vs 50K），线性归一化会导致一个超高播放视频压制所有其他文档。使用 `log1p(view) / max_log_view` 将比率从 2000:1 压缩到约 1.7:1。

**健壮性处理**：每个 hit 的评分逻辑使用 try/except 保护，单条文档的异常数据（如负播放量）不会导致整个排序流程崩溃。`math.log1p` 对负值进行安全处理 (`max(0, view)`)。

#### relevance_score 修正流程

```
ES BM25 _score → 归一化 (÷max_score) → 内容深度惩罚 → 标题关键词覆盖检查
→ TM 加成 (+0.20) → OM 加成 (+0.30, 按 intent_strength 缩放)
→ 相关性底线 (OWNER_RELEVANCE_FLOOR × intent_strength)
→ cap at 1.0
```

**内容深度惩罚**（抑制超短标题 BM25 膨胀）：

标题 `gta` 对查询 `gta` 会获得最高 BM25 分，但实际提供零额外信息：
```
remaining_chars = len(title - query_keywords)
depth_factor = max(0.30, remaining_chars / 20)
relevance_score *= depth_factor
```

**标题关键词覆盖检查**（CJK 连续子串匹配）：

针对纯中文复合查询（如 `小红书推荐系统`），如果查询的 CJK 部分 ≥ 4 字符，检查标题中最长连续匹配子串是否覆盖查询的 50% 以上：
- `小红书推荐系统` in `小红书推荐用户冷启动实践` → 匹配 `小红书推荐` (5/6=83%) ✓
- `小红书推荐系统` in `台湾省小红书反向广告` → 匹配 `小红书` (3/6=50%) ✗

未通过此检查的**非 UP主意图文档**受到 `RANK_NO_TITLE_KEYWORD_PENALTY=0.50` 的 relevance 惩罚。

> **关键改进**：UP主意图文档（`_owner_matched=True` 且 `owner_intent_strength ≥ 0.6`）**豁免**此惩罚。这解决了 `红警08` 查询中 UP主 `红警HBK08` 的视频标题不含 "红警08" 但仍为高度相关内容的问题。

**标题匹配信号 (Title-Match Bonus)**：带 `_title_matched=True` 的文档获得 `TITLE_MATCH_BONUS=0.20` 加分。

**UP主匹配信号 (Owner-Match Bonus)**：带 `_owner_matched=True` 的文档获得 `OWNER_MATCH_BONUS=0.30` 加成，**按 `owner_intent_strength` 缩放**。详见 [4.10 UP主意图感知排序](#410-up-主意图感知排序)。

#### 伪相关反馈 — 标签亲和度 (Tag Affinity)

Phase 3 融合评分阶段引入基于伪相关反馈的标签亲和度加分。从 relevance_score 最高的 top-20 文档中提取高频标签（出现 ≥ 3 次），为 Phase 3 中与这些标签重叠的文档添加小幅加分（上限 0.10），促进主题一致性。

#### 三阶段排序算法

```
Phase 1 — 头部质量选择 (Top-3)
  ┌──────────────────────────────────────────┐
  │ 计算 headline_score (复合分数):          │
  │   = 0.50×relevance + 0.30×quality       │
  │     + 0.10×recency  + 0.10×popularity   │
  │                                          │
  │ 前置处理：                               │
  │   - 内容深度惩罚 (短标题 BM25 修正)     │
  │   - 标题关键词覆盖检查 (CJK 连续匹配)  │
  │   - TM/OM 加成 (OM 按 intent 缩放)     │
  │   - 相关性底线 (OM 文档保底 relevance)  │
  │                                          │
  │ 从 relevance ≥ HEADLINE_MIN_RELEVANCE   │
  │ 且时长 ≥ RANK_HEADLINE_MIN_DURATION 的  │
  │ 候选中选取 headline_score 最高的 3 个    │
  │ (同分时以 relevance_score 做 tiebreak)   │
  └──────────────────────────────────────────┘
                 │
                 ▼
Phase 2 — 相关性门控槽位分配 (位置 4-10)
  ┌──────────────────────────────────────────┐
  │ 从剩余候选中按维度分配 5 个槽位:        │
  │   balanced: {rel:2, quality:1,           │
  │              recency:1, popularity:1}    │
  │                                          │
  │ ⚠ 相关性门控 (Relevance Gating):        │
  │ 每个维度的候选分数 =                     │
  │   dim_score × relevance_factor           │
  │ 其中 relevance_factor:                   │
  │   = 1.0  如果 rel ≥ 0.45                 │
  │   = (rel/0.45)^2  否则                   │
  │                                          │
  │ ⚠ 渐进式阈值放松:                       │
  │   ① relevance ≥ 0.30 的候选              │
  │   ② 不足时放松到 ≥ 0.15                  │
  │   ③ 仍不足时允许所有候选                 │
  └──────────────────────────────────────────┘
                 │
                 ▼
Phase 3 — 融合评分 (位置 11+)
  ┌──────────────────────────────────────────┐
  │ 剩余文档按加权融合分数排序:             │
  │   FUSED_WEIGHTS = {rel:0.50, qual:0.20, │
  │                    rec:0.18, pop:0.12}  │
  │                                          │
  │ + 标签亲和度加分 (tag_bonus, max 0.10)  │
  │   从 top-20 高频标签中获取权重,          │
  │   tag_bonus = sum(affinity) × 0.05      │
  │                                          │
  │ 保证总返回 min(top_k, pool_size) 条:    │
  │   fused_quota = target - headline - slot │
  └──────────────────────────────────────────┘
```

#### 与旧排序策略的对比

| 特性 | stats (旧默认) | 新三阶段 diversified |
|------|---------------|---------------------|
| Top 3 质量 | 低 (仅看分数) | **高 (复合质量分 + TM/OM + 底线)** |
| Top 10 多样性 | 低 (都是均衡型) | **高 (相关性门控各维度代表)** |
| 不相关热门文档 | 可能进入 top-10 | **被相关性门控阻止** |
| 标题匹配文档 | 无特殊处理 | **+0.20 relevance 加成** |
| UP主匹配文档 | 无特殊处理 | **+0.30 (按 intent 缩放 + 底线保障)** |
| 短标题 BM25 膨胀 | 无处理 | **内容深度惩罚** |
| CJK 复合查询 | 无处理 | **最长连续子串匹配检查** |
| 主题一致性 | 无处理 | **标签亲和度 (Phase 3)** |
| UP主意图适应 | 无处理 | **动态 intent_strength 全流程影响** |

### 4.10 UP 主意图感知排序

**核心问题**：不同查询的 UP主 匹配语义差异极大：
- `红警08` → 用户想找 UP主 `红警HBK08`，应大幅提升其内容排名
- `米娜` → 用户想找 "米娜" 相关话题，`大聪明罗米娜` 不应被过度提升
- `chatgpt` → 纯话题查询，UP主匹配几乎没有意义

**解决方案**：使用 `owner_intent_strength` ∈ [0, 1] 动态调控排序中所有 UP主 相关的信号强度。

#### owner_intent_strength 的来源

优先使用 `pool_hints.owner_analysis.intent_strength`（由 RecallPoolOptimizer 从完整召回池计算，使用 `_owner_matched` 标记），如果不可用则在排序阶段通过 `_analyze_owner_intent_strength()` 重新计算。

**关键设计决策**：排序阶段不再重新匹配 UP主 名称（旧方法使用 CJK 子串匹配会将 `红警` 匹配到 `红警V神`、`红警魔鬼蓝天` 等无关 UP主），而是直接复用召回阶段精确标记的 `_owner_matched` 标记。这使得 `红警08` 的 intent_strength 从 0.16 正确提升到 1.0。

#### intent_strength 影响的排序行为

| 行为 | 条件 | 效果 |
|------|------|------|
| OM 加成 | `_owner_matched` | `+OWNER_MATCH_BONUS(0.30) × intent_strength` |
| 相关性底线 | `_owner_matched` 且 `intent ≥ 0.6` | `rel_norm = max(rel_norm, OWNER_RELEVANCE_FLOOR(0.75) × intent_strength)` |
| 标题关键词惩罚豁免 | `_owner_matched` 且 `intent ≥ 0.6` | 不受 `RANK_NO_TITLE_KEYWORD_PENALTY` 影响 |
| OM 加成条件 | 强意图 (`intent ≥ 0.6`) | 无需标题包含查询关键词 |
| OM 加成条件 | 弱意图 (`intent < 0.6`) | 需标题包含查询关键词才给加成 |

**实际效果示例**：

| 查询 | intent_strength | OM docs in top-20 | 说明 |
|------|----------------|-------------------|------|
| `红警08` | 1.00 | 16/20 | UP主 `红警HBK08` 的视频占据主导 |
| `吴恩达大模型` | ~0.80 | 4/20 | 相关 UP主 视频适度提升 |
| `米娜` | ~0.30 | 7/20 | 话题为主，UP主 匹配适度 |
| `chatgpt` | ~0.10 | 3/20 | 几乎纯话题查询 |

### 4.11 UP 主分组

`AuthorGrouper` 将搜索结果按 UP 主聚合，返回有序列表：

**输出字段：** `mid`, `name`, `face`, `latest_pubdate`, `sum_view`, `sum_count`, `top_rank_score`, `first_appear_order`, `hits`

**排序方式（默认 `first_appear_order`）：** 保持 UP 主在搜索结果中首次出现的顺序，确保 UP 主顺序与视频列表一致。

**返回类型为 list（而非 dict）**，以确保 JSON 序列化/反序列化时顺序不丢失。

### 4.12 窄过滤器处理

当查询包含用户过滤器（`u=xxx`）或 BV 号过滤器（`bv=xxx`）时，结果集通常很小。此时 KNN 搜索效率低下。

**解决方案：使用 filter-first 方式**

`dual_sort_filter_search()` 运行两次过滤搜索：
1. 按发布时间倒序，取最近 N 条
2. 按播放量倒序，取最热 N 条
3. 合并去重

然后可选地进行精排（如 `q=vr` 模式）。

---

## 5. ES 索引设计

### 5.1 索引结构

- 索引名：`bili_videos_dev6`（开发）/ `bili_videos_pro1`（生产）
- 分片数：**8 个主分片**（每个分片约 625 万文档）
- 副本数：0（单节点部署）
- KNN 总候选量：8 × num_candidates = 8 × 10,000 = **80,000**

### 5.2 文本字段与分词

文本字段采用 **父字段 + 子字段** 的设计：

| 父字段 | `index` | 子字段 | 分词器 | 用途 |
|--------|---------|--------|--------|------|
| `title` | `false` | `title.words` | `chinese_analyzer` (es_tok) | 搜索匹配 |
| `desc` | `false` | `desc.words` | `chinese_analyzer` | 搜索匹配 |
| `tags` | `false` | `tags.words` | `chinese_analyzer` | 搜索匹配 |
| `owner.name` | `false` | `owner.name.words` | `chinese_analyzer` | 搜索匹配 |
| — | — | `owner.name.keyword` | `keyword` | 精确匹配（UP主定向召回） |

> **设计说明**：父字段 `index=false` 是有意为之。所有搜索查询都使用 `.words` 子字段（由 `es_tok` 中文分词器索引），父字段不直接参与搜索，节省索引空间。

### 5.3 向量字段

```json
{
  "text_emb": {
    "type": "dense_vector",
    "element_type": "bit",
    "dims": 2048,
    "index": true,
    "similarity": "l2_norm",
    "index_options": {
      "type": "hnsw",
      "m": 16,
      "ef_construction": 100
    }
  }
}
```

- 99.98% 的文档有 `text_emb` 字段
- 存储格式：signed int8 数组（256 个元素 = 2048 bits）
- 配置 `index.mapping.exclude_source_vectors: true` 以节省存储

---

## 6. 返回数据结构

### 6.1 步骤结果格式

所有 explore 方法返回统一格式：

```python
{
    "query": "影视飓风",
    "status": "finished",     # "finished" | "timedout" | "error"
    "data": [                 # 步骤结果列表
        {
            "step": 0,
            "name": "construct_knn_query",
            "name_zh": "构建向量查询",
            "status": "finished",
            "input": {...},
            "output_type": "info",    # "hits" | "info"
            "output": {...},
            "comment": "",
        },
        ...
    ],
    "perf": {                 # 性能追踪
        "query_parsing_ms": 12.5,
        "lsh_embedding_ms": 45.2,
        "word_recall_ms": 320.1,
        "knn_search_ms": 180.3,
        "rerank_ms": 1200.0,
        "fetch_docs_ms": 150.0,
        "total_ms": 2100.5,
    },
}
```

**步骤名称与中文对照：**

| name | name_zh | output_type |
|------|---------|-------------|
| `construct_query_dsl_dict` | 解析查询 | info |
| `construct_knn_query` | 构建向量查询 | info |
| `multi_lane_recall` | 多车道召回 | hits |
| `knn_search` | 向量搜索 | hits |
| `rerank` | 精排重排 | info |
| `hybrid_search` | 混合搜索 | hits |
| `group_hits_by_owner` | UP 主聚合 | info |

### 6.2 单条命中结果

```python
{
    "bvid": "BV1xx...",
    "title": "视频标题",
    "desc": "视频简介",
    "tags": "标签1,标签2",
    "owner": {"mid": 12345, "name": "UP主名"},
    "pic": "https://...",
    "duration": 300,
    "stat": {"view": 10000, "coin": 500, ...},
    "pubdate": 1704067200,
    "insert_at": 1704067210,
    "tid": 17,
    "ptid": 4,
    "tname": "单机游戏",
    "rtags": "游戏,单机",
    "score": 28.5,                   # ES BM25 分数
    "rank_score": 15.2,              # 排序分数
    "stat_score": 0.72,              # 预计算文档质量分 (DocScorer)
    "rerank_score": 0.8523,          # 精排分数 (如有)
    "cosine_similarity": 0.8523,     # float 余弦相似度 (如有)
    "hybrid_score": 0.0156,          # 混合融合分数 (如有)
    # 排序阶段附加的多维分数 (diversified 模式):
    "relevance_score": 0.85,         # 归一化相关性 (含 TM/OM 加成)
    "quality_score": 0.72,           # 质量分
    "recency_score": 0.35,           # 时效分
    "popularity_score": 0.68,        # 热度分 (对数归一化)
    "headline_score": 0.71,          # 复合头部分数
    # 召回阶段标记:
    "_title_matched": true,          # 标题/标签匹配查询
    "_owner_matched": true,          # UP主名匹配查询
    "_matched_owner_name": "红警HBK08",  # 匹配的 UP主 名
    "highlights": {...},
    "region_info": {...},
}
```

---

## 7. 性能考量

### KNN 搜索延迟分解

| 阶段 | 典型耗时 | 说明 |
|------|----------|------|
| 查询解析 | 5-15ms | DSL 解析 + 过滤提取 |
| LSH 嵌入 | 30-60ms | TEI 服务调用 |
| KNN 搜索 | 100-300ms | ES HNSW 搜索 (8 分片) |
| 词语召回 | 200-500ms | 并行于 KNN，通常先完成 |
| 精排 | 800-1500ms | float 向量余弦相似度计算 |
| 文档获取 | 100-200ms | 按 bvid 批量获取 |
| **总计** | **1.5-3.0s** | 典型 `q=v` 端到端延迟 |

### 词语搜索 (6 车道) 延迟分解

| 阶段 | 典型耗时 | 说明 |
|------|----------|------|
| 查询解析 | 5-15ms | DSL 解析 |
| 6 车道并行召回 | 200-600ms | 最慢车道决定总时间 |
| 噪声过滤 | 5-10ms | 三层过滤 |
| RecallPoolOptimizer | 10-30ms | 特征分析 |
| 文档获取 | 100-200ms | 批量获取完整文档 |
| 多样化排序 | 20-50ms | 三阶段排序 + 意图感知 |
| **总计** | **0.5-1.0s** | 典型 `q=w` 端到端延迟 |

### 优化策略

1. **并行化**：6 车道词语召回并行执行；词语召回与 KNN 搜索并行
2. **两阶段搜索**：先用最小字段搜索获取 bvid，再批量获取完整文档
3. **窄过滤器检测**：自动切换到 filter-first 策略避免 KNN 低效遍历
4. **精排候选池控制**：`RERANK_MAX_HITS=2000` 控制精排候选数量
5. **嵌入缓存**：`TextEmbedClient` 支持查询嵌入缓存
6. **去重合并**：词语召回与 KNN 结果按 bvid 去重，避免重复精排
7. **健壮性保护**：per-hit try/except 防止单条异常数据导致整体失败

---

## 附录 A: 常量配置速查

> 配置文件：`elastics/videos/constants.py` 和 `ranks/constants.py`

### 多车道召回配置

| 车道 | 限制 | 排序 | 搜索字段 Boost |
|------|------|------|---------------|
| `relevance` | 600 | `_score` | 全字段默认权重 |
| `title_match` | 400 | `_score` | title.words=5.0, tags.words=3.0 |
| `owner_name` | 200 | `_score` | owner.name.words=8.0, title.words=3.0, tags.words=2.0 |
| `popularity` | 300 | `stat.view desc` | 全字段 match filter |
| `recency` | 250 | `pubdate desc` | 全字段 match filter |
| `quality` | 300 | `stat_score desc` | 全字段 match filter |

### 噪声过滤常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `MIN_BM25_SCORE` | 3.0 | ES 级最低分数阈值 |
| `NOISE_SCORE_RATIO_GATE` | 0.10 | BM25 分数比率门限 |
| `NOISE_KNN_SCORE_RATIO` | 0.50 | KNN 分数比率门限 |
| `NOISE_MIN_HITS_FOR_FILTER` | 30 | 最少命中数 (低于此不过滤) |
| `NOISE_MULTI_LANE_GATE_FACTOR` | 0.5 | 多车道文档阈值缩放因子 |
| `NOISE_SHORT_TEXT_MIN_LENGTH` | 15 | 短文本判定 (字符数) |
| `NOISE_SHORT_TEXT_PENALTY` | 0.3 | 短文本分数惩罚 |
| `NOISE_MIN_ENGAGEMENT_VIEWS` | 100 | 低互动判定 (播放量) |
| `NOISE_LOW_ENGAGEMENT_PENALTY` | 0.15 | 低互动分数惩罚 |

### 多样化排序配置

| 常量 | 值 | 说明 |
|------|-----|------|
| `TITLE_MATCH_BONUS` | 0.20 | 标题匹配 relevance 加成 |
| `OWNER_MATCH_BONUS` | 0.30 | UP主匹配 relevance 加成 (按 intent 缩放) |
| `OWNER_RELEVANCE_FLOOR` | 0.75 | UP主意图文档的 relevance 底线 (× intent) |
| `RANK_NO_TITLE_KEYWORD_PENALTY` | 0.50 | 标题无关键词时 relevance 乘数 |
| `RANK_CONTENT_DEPTH_MIN_FACTOR` | 0.30 | 深度因子下限 |
| `RANK_CONTENT_DEPTH_NORM_LENGTH` | 20 | 标准化长度 |
| `RANK_HEADLINE_MIN_DURATION` | 30 | Phase 1 最低时长 (秒) |
| `RANK_SLOT_MIN_DURATION` | 15 | Phase 2 最低时长 (秒) |
| `HEADLINE_MIN_RELEVANCE` | 0.50 | Phase 1 最低 relevance |
| `SLOT_MIN_RELEVANCE` | 0.55 | Phase 2 最低 relevance |
| `SLOT_RELEVANCE_DECAY_THRESHOLD` | 0.45 | Phase 2 相关性衰减阈值 |
| `SLOT_RELEVANCE_DECAY_POWER` | 2.0 | Phase 2 衰减指数 |
| `HEADLINE_TOP_N` | 3 | Phase 1 头部数量 |
| `EXPLORE_RANK_TOP_K` | 400 | 最终返回 Top-K |

### UP主意图检测配置

| 常量 | 值 | 说明 |
|------|-----|------|
| `OWNER_DOMINANT_MIN_DOCS` | 5 | 单一 UP主 最少文档数 |
| `OWNER_DOMINANT_RATIO` | 0.15 | 单一 UP主 最低占比 |
| `OWNER_DISPERSE_MAX_OWNERS` | 6 | 分散判断最大匹配 UP主 数 |

### 排序权重

```python
# Phase 1: 头部复合分
HEADLINE_WEIGHTS = {"relevance": 0.50, "quality": 0.30, "recency": 0.10, "popularity": 0.10}

# Phase 2: 槽位预设
SLOT_PRESETS = {
    "balanced":         {"relevance": 2, "quality": 1, "recency": 1, "popularity": 1},
    "prefer_relevance": {"relevance": 3, "quality": 1, "recency": 1, "popularity": 1},
    "prefer_quality":   {"relevance": 1, "quality": 2, "recency": 1, "popularity": 1},
    "prefer_recency":   {"relevance": 1, "quality": 1, "recency": 3, "popularity": 1},
}

# Phase 3: 融合权重
DIVERSIFIED_FUSED_WEIGHTS = {"relevance": 0.50, "quality": 0.20, "recency": 0.18, "popularity": 0.12}
```

### KNN 设置

| 常量 | 值 | 说明 |
|------|-----|------|
| `KNN_TEXT_EMB_FIELD` | `"text_emb"` | 向量字段 |
| `KNN_K` | 1000 | 每分片近邻数 |
| `KNN_NUM_CANDIDATES` | 10000 | 每分片候选数 |
| `KNN_TIMEOUT` | 8 | 搜索超时 (秒) |

### 精排配置

| 常量 | 值 | 说明 |
|------|-----|------|
| `RERANK_MAX_HITS` | 2000 | 最大精排候选 |
| `RERANK_KEYWORD_BOOST` | 1.5 | 一般关键词加权 |
| `RERANK_TITLE_KEYWORD_BOOST` | 2.0 | 标题关键词加权 |
| `RERANK_TIMEOUT` | 30 | 精排超时 (秒) |
