# 视频搜索与探索系统 — 设计文档

## 目录

- [1. 系统概述](#1-系统概述)
- [2. 架构设计](#2-架构设计)
  - [2.1 模块结构](#21-模块结构)
  - [2.2 类继承关系](#22-类继承关系)
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
  - [4.4 召回噪声过滤 (Noise Filtering)](#44-召回噪声过滤-noise-filtering)
  - [4.5 精排重排 (Reranking)](#45-精排重排-reranking)
  - [4.6 排序策略](#46-排序策略)
  - [4.7 多样化排序 (Diversified Ranking)](#47-多样化排序-diversified-ranking)
  - [4.8 UP 主分组](#48-up-主分组)
  - [4.9 窄过滤器处理](#49-窄过滤器处理)
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

本系统是基于 Elasticsearch 的 B 站视频搜索引擎，支持三种搜索模式：

| 模式 | 代号 | 说明 |
|------|------|------|
| 词语搜索 | `q=w` | 基于 BM25 的关键词匹配搜索 |
| 向量搜索 | `q=v` | 基于 LSH bit vector 的 KNN 语义搜索 |
| 混合搜索 | `q=wv` | 词语 + 向量并行搜索，RRF 融合排序 |

每种模式均可附加精排重排（`r`），如 `q=vr`、`q=wr`、`q=wvr`。

系统索引约 **5000 万** 条视频文档，使用 **8 个分片**，支持丰富的 DSL 过滤表达式（日期、播放量、UP 主、BV 号等）。每条文档包含预计算的 `stat_score`（由 `blux.doc_score.DocScorer` 在入库时计算），用于衡量文档质量。

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
│   ├── test_videos.py    # 综合测试
│   ├── diag_deep.py      # 向量质量诊断
│   ├── diag_float_vs_lsh.py  # Float vs LSH 对比诊断
│   └── diag_knn.py       # KNN 召回诊断

recalls/                      # 多车道召回模块
├── __init__.py           # 模块文档
├── base.py               # RecallResult / RecallPool / NoiseFilter
├── word.py               # MultiLaneWordRecall — 多车道词语召回
├── vector.py             # VectorRecall — 向量+词语补充召回
├── manager.py            # RecallManager — 召回策略编排
└── tests/
    ├── test_base.py      # RecallPool 合并测试 (7 tests)
    ├── test_diversified.py # 多样化排序测试 (21 tests)
    └── test_noise.py     # 噪声过滤测试 (27 tests)

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
├── diversified.py        # DiversifiedRanker — 多样化槽位排序 ← NEW
├── reranker.py           # EmbeddingReranker — 精排
├── grouper.py            # AuthorGrouper — UP主分组
├── scorers.py            # 评分器：Stats/Pubdate/Relate
└── fusion.py             # ScoreFuser — 分数融合
```

### 2.2 类继承关系

```
VideoSearcherV2
    ├── search()           # 词语搜索
    ├── knn_search()       # KNN 搜索
    ├── hybrid_search()    # 混合搜索
    ├── filter_only_search()  # 纯过滤
    ├── random() / latest() / doc()
    └── ...
        │
        ▼
VideoExplorer (继承 VideoSearcherV2)
    ├── _run_explore_pipeline()     # 共享探索管线 (所有 V2 方法委托至此)
    ├── explore_v2()           # 词语探索 → _run_explore_pipeline(mode="word")
    ├── knn_explore_v2()       # 向量探索 → _run_explore_pipeline(mode="vector")
    ├── hybrid_explore_v2()    # 混合探索 → _run_explore_pipeline(mode="hybrid")
    ├── unified_explore()      # 统一分发入口 → 路由到 V2 方法
    └── _filter_only_explore() # 无关键词回退 (纯过滤 + stats/recency 排序)

RecallManager (召回策略编排)
    ├── MultiLaneWordRecall   # 5 车道并行词语召回
    ├── VectorRecall          # 向量召回 + 词语补充
    └── pool.filter_noise()   # 合并后噪声过滤 (3 信号)

NoiseFilter (噪声过滤)
    ├── filter_by_score_ratio()         # BM25 车道噪声过滤
    ├── filter_knn_by_score_ratio()     # KNN 向量噪声过滤
    └── apply_content_quality_penalty() # 短文本/低互动惩罚

DiversifiedRanker (三阶段排序)
    ├── _select_headline_top_n()  # Phase 1: 头部质量选择 (top-3)
    ├── _allocate_slots()         # Phase 2: 相关性门控槽位分配 (位置 4-10)
    └── fused scoring              # Phase 3: 融合评分 (>10)
```

`VideoSearcherV2` 提供单次搜索能力，返回命中列表。  
`VideoExplorer` 在此基础上编排多步骤流程（搜索 → 重排 → 取全量文档 → UP 主分组），返回步骤结果列表。

### 2.3 依赖组件

| 组件 | 类 | 说明 |
|------|----|------|
| Elasticsearch | `ElasticOperator` | 搜索、KNN、聚合 |
| MongoDB | `MongoOperator` | UP 主头像等补充信息 |
| TEI 嵌入服务 | `TextEmbedClient` | 文本 → 向量（float 1024 维 + LSH 2048 bit） |
| DSL 解析器 | `DslExprRewriter` + `DslExprToElasticConverter` | 查询重写与 ES 查询构建 |
| 召回管理器 | `RecallManager` | 多车道召回策略编排 + 噪声过滤 |
| 排序引擎 | `VideoHitsRanker` | 多策略排序（diversified/stats/relevance/tiered/rrf） |
| 多样化排序器 | `DiversifiedRanker` | 槽位分配多样化排序（默认） |
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
| 无指定 | 由默认设置决定 | 根据 `QMOD` 常量 | 随模式 | 随模式 |

> **重要**：`q=v` 模式始终开启精排，因为原始 KNN hamming 分数精度不足以产生有意义的排序。

### 3.2 统一入口 unified_explore

`unified_explore()` 是所有搜索请求的统一入口：

```
用户查询 → 提取 qmod → 判断搜索模式组合 → 分发到对应 V2 explore 方法
```

处理逻辑：
1. 解析 DSL 表达式，提取 `qmod` 字段
2. 如无 `qmod`，使用默认配置 `QMOD = ["word", "vector"]`
3. 判断是否包含 `word`、`vector`、`rerank`
4. 分发到 `explore_v2()` / `knn_explore_v2()` / `hybrid_explore_v2()`
5. 在结果中附加 `qmod` 信息

### 3.3 词语搜索流程 (explore_v2 — 多车道召回)

**旧流程问题**：单次搜索扫描 10000 条文档（`terminate_after=2M`），速度慢且召回维度单一（仅 BM25 相关性）。

**新流程**：4 车道并行召回 + 多样化排序

```
┌─ Step 0: construct_query_dsl_dict ─────────────────────────┐
│  解析 DSL → 构建 boosted fields → 生成 ES 查询字典         │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ Step 1: multi_lane_recall ────────────────────────────────┐
│  ┌───────────────────────────────────────────────────────────┐ │
│  │           ThreadPoolExecutor (5 并行)                     │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐     │ │
│  │  │ 相关性   │ │ 标题匹配 │ │ 热度     │ │ 时效性  │     │ │
│  │  │ BM25     │ │ title    │ │ stat.view│ │ pubdate │     │ │
│  │  │ _score   │ │ only     │ │ desc     │ │ desc    │     │ │
│  │  │ 500 条   │ │ 300 条   │ │ 200 条   │ │ 200 条  │     │ │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘     │ │
│  │       └────────┬────┴────────┬───┴──────────┘            │ │
│  │                └─────────────┘  + ┌──────────┐           │ │
│  │                                   │ 质量     │           │ │
│  │                                   │stat_score│           │ │
│  │                                   │ desc     │           │ │
│  │                                   │ 200 条   │           │ │
│  │                                   └────┬─────┘           │ │
│  └────────────────────────────────────────┼─────────────────┘ │
│                         │                                  │
│              合并 + 去重 (by bvid)                          │
│              标记 _title_matched                            │
│              约 500-1000 候选文档                            │
│              每条文档带 lane_tags                            │
└────────────────────────────────────────────────────────────┘
                         │
                    (可选: rerank)
                         │
                         ▼
┌─ fetch_and_rank ───────────────────────────────────────────┐
│  ① fetch_docs_by_bvids: 获取完整文档                        │
│  ② diversified_rank: 三阶段排序                              │
│     Phase 1: 头部质量选择 (top-3 by headline_score)          │
│     Phase 2: 相关性门控槽位分配 (位置 4-10)                  │
│     Phase 3: 融合评分 (剩余文档)                              │
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
┌─ fetch_and_rank ───────────────────────────────────────────┐
│  获取完整文档 → diversified_rank → 字符级高亮              │
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
┌─ Step 1: hybrid_recall ───────────────────────────────────┐
│  RecallManager 协调两种召回:                                │
│  ┌─────────────┐  ┌──────────────┐                        │
│  │  多车道词语  │  │  KNN 向量    │   ← 并行执行           │
│  │  recall      │  │  recall      │                        │
│  └──────┬──────┘  └──────┬───────┘                        │
│         └───────┬────────┘                                │
│                 ▼                                          │
│   RecallPool.merge → 合并去重                               │
│   保留 lane_tags, rank info                                │
└────────────────────────────────────────────────────────────┘
                         │
                    (可选: rerank)
                         │
                         ▼
┌─ fetch_and_rank ──────────────────────────────────────────┐
│  fetch_docs → diversified_rank → group_by_owner           │
└───────────────────────────────────────────────────────────┘
│  fetch_docs_by_bvids → tiered_rank → group_hits_by_owner  │
└────────────────────────────────────────────────────────────┘
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

**解决方案**：`recalls/` 模块实现多车道并行召回策略。

#### 多车道词语召回 (MultiLaneWordRecall)

同一个查询，5 个车道并行执行不同排序的 ES 查询：

| 车道 | 排序依据 | 取回量 | 目标 |
|------|----------|--------|------|
| `relevance` | BM25 `_score` | 500 | 相关性最高的文档 |
| `title_match` | BM25 `_score` (title.words + tags.words) | 300 | 标题/标签匹配的文档 |
| `popularity` | `stat.view` desc | 200 | 热度最高的文档（原始曝光量） |
| `recency` | `pubdate` desc | 200 | 时间最近的文档 |
| `quality` | `stat_score` desc | 200 | 质量最高的文档（综合满意度） |

> **设计说明**：`title_match` 车道搜索 `title.words` (boost=5.0) 和 `tags.words` (boost=3.0) 字段，确保标题或标签匹配查询的文档一定被召回。标签包含了实体名称、电影标题、主题关键词等强信号，与标题互补。召回后通过 `_tag_title_matches()` 为所有标题**或标签**包含查询关键词的文档添加 `_title_matched=True` 标记，供排序阶段使用。
>
> **重要修复**：`_title_matched` 标记通过 `merge_scores_into_hits()` 从召回结果传递到完整文档中。此标记在排序阶段的 `_score_all_dimensions()` 中被使用，为标题/标签匹配的文档添加 `TITLE_MATCH_BONUS=0.15` 的额外相关性加分。

> 旧版 5 车道包含 `engagement`（按 `stat.coin` 排序），已被替换为 `title_match`。因为 `quality` 车道使用的 `stat_score`（由 DocScorer 计算）已综合了 coin/favorite/like 等互动指标，`engagement` 与 `quality` 定位重合。`popularity`（stat.view = 原始曝光量）与 `quality`（stat_score = 内容满意度）的语义边界清晰，不再重叠。`title_match` 车道搜索 `title.words` (boost=5.0) 和 `tags.words` (boost=3.0)，全面覆盖实体名称和主题关键词。

车道的 `source_fields` 包含 `title`、`tags`、`desc`，用于后续标题匹配标记和内容质量过滤。

**合并逻辑** (`RecallPool.merge`)：
- 按 `bvid` 去重，同一文档可能出现在多个车道
- 每条文档记录 `lane_tags`（如 `{"relevance", "popularity"}`）
- 合并后的候选池约 500-1000 条

**性能**：5 个并行 ES 查询 vs 1 个扫描 10000 条的查询。每个车道的 `size` 远小于 10000，且并行执行，总延迟约等于单个最慢车道的延迟。

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

合并后自动调用 `pool.filter_noise()` 移除噪声文档（见 [4.4 召回噪声过滤](#44-召回噪声过滤-noise-filtering)）。

### 4.4 召回噪声过滤 (Noise Filtering)

**问题**：召回阶段会引入大量噪声文档：
- **BM25 稀有关键词噪声**：IDF 高的稀有词（如"飓风"）产生高 BM25 分数，即使文档与查询语义无关
- **KNN bit vector 噪声**：LSH 量化后 hamming 距离区分度有限，大量语义无关文档获得相近分数

**解决方案**：三层 + 三信号噪声过滤架构

```
                    ES 层                    车道层                    池层 (三信号)
                ┌──────────┐          ┌──────────────┐         ┌────────────────────┐
                │min_score │          │ NoiseFilter.  │         │ RecallPool         │
   Query ──→   │ = 2.0    │ ──→      │ filter_by_    │  ──→    │ .filter_noise()    │ ──→ 排序
                │(ES body) │          │ score_ratio() │         │ ① score-ratio      │
                └──────────┘          │ + content     │         │ ② short-text pen.  │
                    │                 │   quality     │         │ ③ low-engage pen.  │
              不返回 BM25             │   penalty     │         └────────────────────┘
              分数 < 2.0 的           └──────────────┘
              文档
```

#### 第一层：ES 级过滤 (`min_score`)
在 ES 搜索请求 body 中设置 `"min_score": 2.0`，阻止 BM25 分数极低的文档被返回。仅适用于非相关性车道（popularity/recency/quality 车道按其他字段排序，BM25 分数可能偏低但文档仍有价值）。

#### 第二层：车道级过滤 (`NoiseFilter`)
每个车道返回的 hits 经过 `filter_by_score_ratio()` 过滤：
- **BM25 车道**：移除 `score < max_score × 0.12` 的命中
- **KNN 车道**：使用更严格的比率 `score < max_score × 0.5`（KNN 分数范围窄）
- **最小数量保护**：若总命中 ≤ 30 条，跳过过滤以避免结果过少
- **内容质量预处理** (`apply_content_quality_penalty`)：在 score-ratio 过滤之前，对短文本和低互动文档的分数施加惩罚，使它们更容易被 score-ratio 过滤移除

#### 第三层：池级过滤 (`RecallPool.filter_noise()` — 三信号)
合并后的召回池使用三信号进行最终过滤：

1. **Score-ratio 门控**：全局 gate = `max_score × ratio`
2. **短文本惩罚**：`title + desc` 长度 < 15 字符的文档，有效分数 × 0.3（BM25 field-length normalization 对短文本过度加分）
3. **低互动惩罚**：播放量 < 100 的文档，有效分数 × 0.15（极低互动通常为垃圾/测试内容）

- **多车道感知**：出现在 2+ 个车道的文档使用更低阈值（`gate × 0.5`），因为跨车道出现本身就是相关性信号

#### 噪声过滤常量

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `MIN_BM25_SCORE` | 3.0 | ES 级最低分数阈值 |
| `NOISE_SCORE_RATIO_GATE` | 0.18 | BM25 分数比率门限 |
| `NOISE_KNN_SCORE_RATIO` | 0.60 | KNN 分数比率门限（更严格） |
| `NOISE_MIN_HITS_FOR_FILTER` | 30 | 最少命中数，低于此值不过滤 |
| `NOISE_MULTI_LANE_GATE_FACTOR` | 0.5 | 多车道文档的阈值缩放因子 |
| `NOISE_SHORT_TEXT_MIN_LENGTH` | 15 | 短文本判定阈值（title+desc 字符数） |
| `NOISE_SHORT_TEXT_PENALTY` | 0.3 | 短文本分数惩罚因子 |
| `NOISE_MIN_ENGAGEMENT_VIEWS` | 100 | 低互动判定阈值（播放量） |
| `NOISE_LOW_ENGAGEMENT_PENALTY` | 0.15 | 低互动分数惩罚因子 |

### 4.5 精排重排 (Reranking)

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

**文档段落构建：**
```python
passage = build_sentence(title, tags, desc, owner_name)
# 格式: 【UP主名】 标题 (标签) 简介
```

**性能参数：**
- `RERANK_MAX_HITS = 2000`（最大精排候选数）
- `RERANK_TIMEOUT = 30s`
- `RERANK_MAX_PASSAGE_LENGTH = 4096` 字符

### 4.6 排序策略

系统支持 5 种排序策略，由 `VideoHitsRanker` 实现：

#### heads — 首部截取
直接截取前 top_k 条结果，不做任何评分。用于快速返回。

#### stats — 统计排序（默认，用于词语搜索）
```
rank_score = (STATS_BASE + stats_score) × pubdate_score × (relate_score³)
```
- **stats_score**：由 `blux.doc_score.DocScorer` 计算的文档质量分，∈ [0, 1)，综合饱和评分 + 异常检测
  - 饱和评分：对每个统计字段 (view, like, coin, favorite, danmaku, reply) 应用 `1 - 1/(1 + field/10^α)` 加权平均
  - 异常检测：高播放但低互动时惩罚因子 ∈ [0.3, 1.0]
  - 最终：`stat_quality = saturated_score × anomaly_factor`
- **STATS_BASE**：0.1 偏移量，防止零统计量文档被完全压制
- **pubdate_score**：分段线性衰减（今天=4.0, 7 天=1.0, 30 天=0.6, 1 年=0.3）
- **relate_score**：搜索相关度分数，门控 + 幂变换

#### relevance — 纯相关度排序（用于向量搜索）
```
rank_score = transform(score, power=2.0, min=0.4, high_threshold=0.85, high_boost=2.0)
```
仅按语义相似度排序，不考虑统计量和时效性。

#### tiered — 分层排序（用于混合搜索）
将结果分为两个区域：
- **高相关区**（≥ max_score × 0.7）：按 `0.7 × stats_score + 0.3 × recency` 排序（stats_score 为 DocScorer 的有界质量分 [0,1)，直接参与加权无需额外归一化）
- **低相关区**（< max_score × 0.7）：按纯相关度排序

#### rrf — 倒数排序融合
```
rrf_score = Σ weight[i] / (k + rank[i])
```
融合多个维度的排序（分数、统计量、发布时间）。

### 4.7 多样化排序 (Diversified Ranking)

**问题**：连续加权融合 (`w × quality + w × relevance + w × recency`) 导致 Top 10 同质化 — 所有结果都是各维度中等偏上的"均衡型"文档，缺少某一维度特别突出的文档。

**解决方案**：`DiversifiedRanker` 使用**三阶段排序**算法，保证 Top-3 质量，Top-10 多样性，并通过**相关性门控**防止不相关文档占据 top-10。

#### 标题匹配信号 (Title-Match Bonus)

召回阶段的 `title_match` 车道为标题或标签包含查询关键词的文档添加 `_title_matched=True` 标记。
在排序阶段，带此标记的文档获得 `TITLE_MATCH_BONUS=0.15` 的额外相关性加分（加到归一化后的 relevance_score 上，上限 1.0）。
这确保标题匹配查询的文档在所有排序阶段都被强烈偏好，有效解决 `通义实验室`、`飓风营救`、`红警08` 等实体查询中 top 6-10 被不相关文档占据的问题。

#### 三阶段排序算法

```
Phase 1 — 头部质量选择 (Top-3)
  ┌──────────────────────────────────────────┐
  │ 计算 headline_score (复合分数):          │
  │   = 0.55×relevance + 0.20×quality       │
  │     + 0.15×recency  + 0.10×popularity   │
  │                                          │
  │ 从 relevance_score ≥ 0.35 的候选中      │
  │ 选取 headline_score 最高的 3 个          │
  │ (同分时以 relevance_score 做 tiebreak)   │
  │                                          │
  │ _title_matched 文档额外 +0.15 relevance  │
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
  │   = 1.0  如果 rel ≥ 0.40                 │
  │   = (rel/0.40)^2  否则                   │
  │                                          │
  │ 这确保不相关但热门/新鲜的文档不能占据   │
  │ 维度槽位 — 它们的维度分数会被相关性      │
  │ 因子大幅衰减。                           │
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
  │                    rec:0.15, pop:0.15}  │
  │                                          │
  │ 保证总返回 min(top_k, pool_size) 条:    │
  │   fused_quota = target - headline - slot │
  └──────────────────────────────────────────┘
```

每条文档的 5 维分数:
- **relevance_score**: ES BM25 分数归一化 + 标题/标签匹配加成 (TITLE_MATCH_BONUS=0.15)
- **quality_score**: stat_score (DocScorer 计算的质量分)，短时长视频 (<30s) 额外惩罚 ×0.8
- **recency_score**: pubdate 线性衰减
- **popularity_score**: log1p(stat.view) 对数归一化
- **headline_score**: 复合分数，兼顾相关性 + 质量 + 时效 + 热度

**Popularity 对数归一化**：播放量呈幂律分布（如 100M vs 50K），线性归一化会导致一个超高播放视频压制所有其他文档。使用 `log1p(view) / max_log_view` 将比率从 2000:1 压缩到约 1.7:1，使不同量级的播放量都能公平参与评分。

#### Phase 1: 头部质量选择

**问题**：Top-3 纯按 BM25 相关性选取时，容易选中短标题/低质量的文档（BM25 对短文本过度加分）。

**方案**：使用 `headline_score` 复合分数选取 Top-3，确保最显眼位置的结果同时具备高相关性、高质量和合理时效性：

| 维度 | 权重 | 说明 |
|------|------|------|
| relevance | 0.55 | 搜索匹配度是主导因素（含标题匹配加成） |
| quality | 0.20 | 质量是用户满意度关键 |
| recency | 0.15 | 时效性作为加分项 |
| popularity | 0.10 | 热度作为辅助参考 |

#### Phase 2: 槽位预设 (Slot Presets)

槽位预设决定 Top-3 后的位置 4-10 如何分配（按预设总数补满）：

| 预设 | 相关 | 质量 | 时效 | 热度 | 总槽 | 适用场景 |
|------|------|------|------|------|------|----------|
| `balanced` (默认) | 2 | 1 | 1 | 1 | 5 | 通用搜索 |
| `prefer_relevance` | 3 | 1 | 1 | 1 | 6 | 精确查找 |
| `prefer_quality` | 1 | 2 | 1 | 1 | 5 | 质量优先 |
| `prefer_recency` | 1 | 1 | 3 | 1 | 6 | 时效优先 |

#### 与旧排序策略的对比

| 特性 | stats (旧默认) | 旧 diversified | 新三阶段 diversified |
|------|---------------|---------------|---------------------|
| Top 3 质量 | 低 (仅看分数) | 中 (纯相关性) | **高 (复合质量分 + 标题匹配加成)** |
| Top 10 多样性 | 低 (都是均衡型) | **高** | **高 (相关性门控各维度代表)** |
| 不相关热门文档 | 可能进入 top-10 | 可能进入 top-10 | **被相关性门控阻止** |
| 标题匹配文档 | 无特殊处理 | 无特殊处理 | **+0.15 relevance 加成** |
| 超高播放量文档 | 可能被稀释 | **保证出现** | **保证出现 (如果相关)** |
| 最新发布文档 | 可能排名偏后 | **保证出现** | **保证出现 (如果相关)** |
| 排序可控性 | 仅通过调权重 | 调槽位数 | **调头部权重 + 调槽位数 + 门控阈值** |
| 短文本噪声 | 可能排前列 | 可能排前列 | **被惩罚，不会排前列** |

### 4.8 UP 主分组

`AuthorGrouper` 将搜索结果按 UP 主聚合，返回有序列表：

**输出字段：** `mid`, `name`, `face`, `latest_pubdate`, `sum_view`, `sum_count`, `top_rank_score`, `first_appear_order`, `hits`

**排序方式（默认 `first_appear_order`）：** 保持 UP 主在搜索结果中首次出现的顺序，确保 UP 主顺序与视频列表一致。

**返回类型为 list（而非 dict）** ，以确保 JSON 序列化/反序列化时顺序不丢失。

### 4.9 窄过滤器处理

当查询包含用户过滤器（`u=xxx`）或 BV 号过滤器（`bv=xxx`）时，结果集通常很小（如某 UP 主的几十到几百个视频）。此时 KNN 搜索效率低下（HNSW 图中只有极少数节点满足过滤条件）。

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
| — | — | `owner.name.keyword` | `keyword` | 精确匹配 |

> **设计说明**：父字段 `index=false` 是有意为之。所有搜索查询都使用 `.words` 子字段（由 `es_tok` 中文分词器索引），父字段不直接参与搜索，节省索引空间。

> **注意**：v6 索引不包含 `.pinyin` 子字段（已从 v5 移除）。`constants.py` 中的拼音字段配置已被注释。

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
    "query": "影视飓风",                 # 原始查询
    "status": "finished",               # "finished" | "timedout" | "error"
    "data": [                           # 步骤结果列表
        {
            "step": 0,                  # 步骤序号
            "name": "construct_knn_query",  # 步骤名称
            "name_zh": "构建向量查询",   # 中文名称
            "status": "finished",       # 步骤状态
            "input": {...},             # 输入参数
            "output_type": "info",      # "hits" | "info"
            "output": {...},            # 输出数据
            "comment": "",              # 人类可读注释
        },
        ...
    ],
    "perf": {                           # 性能追踪 (knn_explore 专有)
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
    "rerank_score": 0.8523,          # 精排分数 (如有)
    "cosine_similarity": 0.8523,     # float 余弦相似度 (如有)
    "hybrid_score": 0.0156,          # 混合融合分数 (如有)
    "word_rank": 5,                  # 词语排名 (混合搜索)
    "knn_rank": 12,                  # KNN 排名 (混合搜索)
    "highlights": {...},             # 高亮信息
    "region_info": {...},            # 分区信息
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

### 优化策略

1. **并行化**：词语召回与 LSH 嵌入/KNN 搜索并行执行
2. **两阶段搜索**：先用最小字段搜索获取 bvid，再批量获取完整文档
3. **窄过滤器检测**：自动切换到 filter-first 策略避免 KNN 低效遍历
4. **精排候选池控制**：`RERANK_MAX_HITS=2000` 控制精排候选数量
5. **嵌入缓存**：`TextEmbedClient` 支持查询嵌入缓存
6. **去重合并**：词语召回与 KNN 结果按 bvid 去重，避免重复精排
