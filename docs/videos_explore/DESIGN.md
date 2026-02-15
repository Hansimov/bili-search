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
  - [4.4 精排重排 (Reranking)](#44-精排重排-reranking)
  - [4.5 排序策略](#45-排序策略)
  - [4.6 多样化排序 (Diversified Ranking)](#46-多样化排序-diversified-ranking)
  - [4.7 UP 主分组](#47-up-主分组)
  - [4.8 窄过滤器处理](#48-窄过滤器处理)
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

recalls/                      # ← NEW: 多车道召回模块
├── __init__.py           # 模块文档
├── base.py               # RecallResult / RecallPool 数据类
├── word.py               # MultiLaneWordRecall — 多车道词语召回
├── vector.py             # VectorRecall — 向量+词语补充召回
└── manager.py            # RecallManager — 召回策略编排

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
    ├── explore_v2()       # 词语探索 V2 (多车道召回 + 多样化排序)
    ├── knn_explore_v2()   # 向量探索 V2 (向量召回 + 多样化排序)
    ├── hybrid_explore_v2()# 混合探索 V2 (混合召回 + 多样化排序)
    ├── unified_explore()  # 统一分发入口 → 路由到 V2 方法
    ├── explore()          # [Legacy] 词语探索
    ├── knn_explore()      # [Legacy] 向量探索
    └── hybrid_explore()   # [Legacy] 混合探索

RecallManager (召回策略编排)
    ├── MultiLaneWordRecall   # 4 车道并行词语召回
    └── VectorRecall          # 向量召回 + 词语补充

DiversifiedRanker (多样化排序)
    └── diversified_rank()    # 槽位分配多样化排序
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
| 召回管理器 | `RecallManager` | 多车道召回策略编排 |
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
│  ┌───────────────────────────────────────────────────────┐ │
│  │           ThreadPoolExecutor (4 并行)                 │ │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐ │ │
│  │  │ 相关性   │ │ 热度     │ │ 时效性  │ │ 质量     │ │ │
│  │  │ BM25     │ │ stat.view│ │ pubdate │ │stat_score│ │ │
│  │  │ _score   │ │ desc     │ │ desc    │ │ desc     │ │ │
│  │  │ 200 条   │ │ 100 条   │ │ 100 条  │ │ 100 条   │ │ │
│  │  └────┬─────┘ └────┬─────┘ └────┬────┘ └────┬─────┘ │ │
│  │       └────────┬────┴────────┬───┘           │       │ │
│  │                └─────────────┴───────────────┘       │ │
│  └───────────────────────────────────────────────────────┘ │
│                         │                                  │
│              合并 + 去重 (by bvid)                          │
│              约 300-500 候选文档                             │
│              每条文档带 lane_tags                            │
└────────────────────────────────────────────────────────────┘
                         │
                    (可选: rerank)
                         │
                         ▼
┌─ fetch_and_rank ───────────────────────────────────────────┐
│  ① fetch_docs_by_bvids: 获取完整文档                        │
│  ② diversified_rank: 槽位分配排序                            │
│     Top 10 = 3×相关 + 2×质量 + 3×时效 + 2×热度              │
│  ③ 剩余文档按融合分数排序                                    │
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

同一个查询，4 个车道并行执行不同排序的 ES 查询：

| 车道 | 排序依据 | 取回量 | 目标 |
|------|----------|--------|------|
| `relevance` | BM25 `_score` | 200 | 相关性最高的文档 |
| `popularity` | `stat.view` desc | 100 | 热度最高的文档 |
| `recency` | `pubdate` desc | 100 | 时间最近的文档 |
| `quality` | `stat_score` desc | 100 | 质量最高的文档 |

**合并逻辑** (`RecallPool.merge`)：
- 按 `bvid` 去重，同一文档可能出现在多个车道
- 每条文档记录 `lane_tags`（如 `{"relevance", "popularity"}`）
- 合并后的候选池约 300-500 条

**性能**：4 个并行 ES 查询 vs 1 个扫描 10000 条的查询。每个车道的 `size` 远小于 10000，且并行执行，总延迟约等于单个最慢车道的延迟。

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

### 4.4 精排重排 (Reranking)

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

### 4.5 排序策略

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

### 4.6 多样化排序 (Diversified Ranking)

**问题**：连续加权融合 (`w × quality + w × relevance + w × recency`) 导致 Top 10 同质化 — 所有结果都是各维度中等偏上的"均衡型"文档，缺少某一维度特别突出的文档。

**解决方案**：`DiversifiedRanker` 使用**槽位分配**算法，保证 Top-K 中各维度都有代表。

#### 槽位分配算法

```
1. 计算每条文档的 4 维分数:
   - relevance_score: ES BM25 分数归一化
   - quality_score: stat_score (DocScorer 计算的质量分)
   - recency_score: pubdate 线性衰减
   - popularity_score: stat.view 排名分

2. 按维度优先级依次分配槽位:
   slot_preset = {
     "relevance": 3,   # 3 个相关性最高的
     "quality": 2,     # 2 个质量最高的
     "recency": 3,     # 3 个最新的
     "popularity": 2,  # 2 个最热门的
   }   # 总计 10 个槽位

3. 每个维度取该维度分数最高且未被选中的文档

4. 若有剩余文档，按加权融合分数排序作为 fallback

5. 被选中的文档按其槽位排名赋予 rank_score
```

#### 槽位预设 (Slot Presets)

| 预设 | 相关 | 质量 | 时效 | 热度 | 适用场景 |
|------|------|------|------|------|----------|
| `balanced` (默认) | 3 | 2 | 3 | 2 | 通用搜索 |
| `prefer_relevance` | 5 | 2 | 2 | 1 | 精确查找 |
| `prefer_quality` | 2 | 4 | 2 | 2 | 质量优先 |
| `prefer_recency` | 2 | 1 | 5 | 2 | 时效优先 |

#### 与旧排序策略的对比

| 特性 | stats (旧默认) | diversified (新默认) |
|------|---------------|---------------------|
| Top 10 多样性 | 低 (都是均衡型) | **高 (各维度代表)** |
| 超高播放量文档 | 可能被平均化稀释 | **保证出现** |
| 最新发布文档 | 可能排名偏后 | **保证出现** |
| 排序可控性 | 仅通过调权重 | **通过调槽位数** |
| 融合分数连续性 | 连续 | Top-K 后连续 |

### 4.7 UP 主分组

`AuthorGrouper` 将搜索结果按 UP 主聚合，返回有序列表：

**输出字段：** `mid`, `name`, `face`, `latest_pubdate`, `sum_view`, `sum_count`, `top_rank_score`, `first_appear_order`, `hits`

**排序方式（默认 `first_appear_order`）：** 保持 UP 主在搜索结果中首次出现的顺序，确保 UP 主顺序与视频列表一致。

**返回类型为 list（而非 dict）** ，以确保 JSON 序列化/反序列化时顺序不丢失。

### 4.8 窄过滤器处理

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
| `most_relevant_search` | 搜索相关 | hits |
| `knn_search` | 向量搜索 | hits |
| `word_recall_supplement` | 词语补充召回 | info |
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
