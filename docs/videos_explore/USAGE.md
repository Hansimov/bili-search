# 视频搜索与探索系统 — 使用文档

## 目录

- [1. 快速开始](#1-快速开始)
- [2. 基础搜索](#2-基础搜索)
  - [2.1 词语搜索 (search)](#21-词语搜索-search)
  - [2.2 KNN 向量搜索 (knn_search)](#22-knn-向量搜索-knn_search)
  - [2.3 混合搜索 (hybrid_search)](#23-混合搜索-hybrid_search)
  - [2.4 辅助搜索](#24-辅助搜索)
- [3. 探索模式](#3-探索模式)
  - [3.1 统一入口 (unified_explore)](#31-统一入口-unified_explore)
  - [3.2 词语探索 (explore_v2)](#32-词语探索-explore_v2)
  - [3.3 向量探索 (knn_explore_v2)](#33-向量探索-knn_explore_v2)
  - [3.4 混合探索 (hybrid_explore_v2)](#34-混合探索-hybrid_explore_v2)
- [4. 查询语法 (DSL)](#4-查询语法-dsl)
  - [4.1 基本语法](#41-基本语法)
  - [4.2 搜索模式 (qmod)](#42-搜索模式-qmod)
  - [4.3 过滤表达式](#43-过滤表达式)
- [5. 解读搜索结果](#5-解读搜索结果)
  - [5.1 步骤结果](#51-步骤结果)
  - [5.2 命中结果字段](#52-命中结果字段)
  - [5.3 UP 主分组](#53-up-主分组)
- [6. 运行测试](#6-运行测试)
- [7. 注意事项](#7-注意事项)
- [附录 A: VideoSearcherV2 完整参数](#附录-a-videosearcherv2-完整参数)
- [附录 B: VideoExplorer 完整参数](#附录-b-videoexplorer-完整参数)

---

## 1. 快速开始

### 环境依赖

- Elasticsearch（已创建索引 `bili_videos_dev6` 或 `bili_videos_pro1`）
- MongoDB（UP 主头像等补充数据）
- TEI 嵌入服务（向量搜索需要）
- Python 包：`elasticsearch`, `sedb`, `tclogger`, `btok`, `tfmx`, `numpy`

### 最简示例

```python
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.explorer import VideoExplorer

# 初始化
explorer = VideoExplorer(
    index_name=ELASTIC_VIDEOS_DEV_INDEX,   # "bili_videos_dev6"
    elastic_env_name=ELASTIC_DEV,          # "elastic_dev"
)

# 搜索（默认混合模式 q=wv）
result = explorer.unified_explore(query="影视飓风")

# 遍历步骤结果
for step in result["data"]:
    print(f"[{step['name_zh']}] {step['status']}")
    if step["output_type"] == "hits":
        hits = step["output"].get("hits", [])
        for hit in hits[:3]:
            print(f"  {hit['title']}")
```

---

## 2. 基础搜索

`VideoSearcherV2` 提供单次搜索能力，适合直接获取命中列表。

### 2.1 词语搜索 (search)

```python
from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV

searcher = VideoSearcherV2(
    index_name=ELASTIC_VIDEOS_DEV_INDEX,
    elastic_env_name=ELASTIC_DEV,
)

# 基本词语搜索
res = searcher.search(
    query="deepseek v3",
    limit=50,
    rank_top_k=50,
    verbose=True,
)

print(f"总命中: {res['total_hits']}")
for hit in res["hits"][:5]:
    print(f"  [{hit['score']:.2f}] {hit['title']}")
```

**带过滤的搜索：**

```python
res = searcher.search(
    query="黑神话 悟空 d>2024-06-01 v>1w",
    limit=100,
    rank_top_k=100,
    rank_method="stats",   # 按热度排序
    verbose=True,
)
```

### 2.2 KNN 向量搜索 (knn_search)

```python
res = searcher.knn_search(
    query="红警08 小块地",
    k=400,                   # 每分片返回 400 个近邻
    num_candidates=10000,    # 每分片搜索 10000 个候选
    limit=50,
    rank_top_k=50,
    verbose=True,
)
```

**带 DSL 过滤的 KNN 搜索：**

```python
res = searcher.knn_search(
    query="donk 高光集锦 d>2024-01-01 v>1w",
    limit=10,
    verbose=True,
)
```

### 2.3 混合搜索 (hybrid_search)

```python
res = searcher.hybrid_search(
    query="深度学习 教程",
    fusion_method="rrf",     # RRF 融合 (默认)
    limit=400,
    rank_top_k=400,
    verbose=True,
)

print(f"词语命中: {res.get('word_hits_count', 0)}")
print(f"KNN 命中: {res.get('knn_hits_count', 0)}")
print(f"融合结果: {res['return_hits']}")
```

### 2.4 辅助搜索

```python
# 随机推荐
res = searcher.random(limit=3)

# 最新视频
res = searcher.latest(limit=10)

# 按 BV 号获取文档
doc = searcher.doc(bvid="BV1xx411c7cH")

# 批量获取文档
res = searcher.fetch_docs_by_bvids(
    bvids=["BV1xx...", "BV1yy..."],
)

# 纯过滤搜索（无关键词）
res = searcher.filter_only_search(
    query='u="红警HBK08"',
    limit=100,
)
```

---

## 3. 探索模式

`VideoExplorer` 继承 `VideoSearcherV2`，提供多步骤的搜索流程，返回完整的步骤结果（含搜索、排序、分组等信息）。

### 3.1 统一入口 (unified_explore)

**推荐使用此方法**——根据查询中的 `q=<mode>` 自动分发到对应的探索方法。

```python
from elastics.videos.explorer import VideoExplorer
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV

explorer = VideoExplorer(
    index_name=ELASTIC_VIDEOS_DEV_INDEX,
    elastic_env_name=ELASTIC_DEV,
)

# 词语搜索
result = explorer.unified_explore("影视飓风 q=w")

# 向量搜索（自动精排 + 词语补充召回）
result = explorer.unified_explore("影视飓风 q=v")

# 混合搜索
result = explorer.unified_explore("影视飓风 q=wv")

# 不指定模式 → 使用默认配置 QMOD=["word","vector"]
result = explorer.unified_explore("影视飓风")

# 带过滤
result = explorer.unified_explore("影视飓风 q=v d>2024 v>1w")

# 指定参数
result = explorer.unified_explore(
    query="deepseek",
    qmod=["vector"],          # 也可以直接指定
    rank_top_k=400,
    group_owner_limit=25,
    verbose=True,
)
```

### 3.2 词语探索 (explore_v2)

适用于 `q=w` 和 `q=wr` 模式。使用 6 车道并行召回 + 3 轮渐进补充 + UP主意图感知排序。

```python
result = explorer.explore_v2(
    query="黑神话 悟空",
    rank_top_k=400,              # 返回 Top-400
    group_owner_limit=25,        # UP 主分组上限
    enable_rerank=False,         # 是否精排
    verbose=True,
)
```

### 3.3 向量探索 (knn_explore_v2)

适用于 `q=v` 和 `q=vr` 模式。自动包含词语补充召回和精排。

```python
result = explorer.knn_explore_v2(
    query="原神 角色",
    knn_k=400,                    # 每分片近邻数
    knn_num_candidates=10000,     # 每分片候选数
    enable_rerank=True,           # 精排（默认开启）
    word_recall_enabled=True,     # 词语补充召回（默认开启）
    word_recall_limit=1000,       # 词语搜索上限
    rank_top_k=400,
    group_owner_limit=25,
    verbose=True,
)
```

### 3.4 混合探索 (hybrid_explore_v2)

适用于 `q=wv` 和 `q=wvr` 模式。6 车道词语召回 + KNN 向量召回并行。

```python
result = explorer.hybrid_explore_v2(
    query="深度学习 教程 d>2024",
    rrf_k=60,                    # RRF 常数
    fusion_method="rrf",         # "rrf" 或 "weighted"
    rank_top_k=400,
    group_owner_limit=25,
    enable_rerank=False,
    verbose=True,
)
```

---

## 4. 查询语法 (DSL)

### 4.1 基本语法

查询字符串由**搜索关键词**和**过滤表达式**组成，二者可以混合使用：

```
关键词1 关键词2 过滤1 过滤2 ...
```

示例：
```
影视飓风              → 纯关键词搜索
影视飓风 d>2024       → 关键词 + 日期过滤
d>2024-01-01 v>1w    → 纯过滤（无关键词）
"影视飓风" "罗永浩"   → 引号内完整匹配
```

### 4.2 搜索模式 (qmod)

通过 `q=<chars>` 指定搜索模式，每个字母代表一种模式：

| 字母 | 含义 | 说明 |
|------|------|------|
| `w` | word | 词语（关键词匹配）搜索 |
| `v` | vector | 向量（语义相似）搜索 |
| `r` | rerank | 精排重排 |

可以组合使用，顺序无关：

```
影视飓风 q=w      → 仅词语搜索
影视飓风 q=v      → 向量搜索（自动精排 + 词语召回）
影视飓风 q=vr     → 同上（r 在 v 模式下自动开启）
影视飓风 q=wv     → 混合搜索
影视飓风 q=wvr    → 混合搜索 + 精排
```

等价语法：`qm=w`, `qmod=v` 也可使用。

### 4.3 过滤表达式

#### 日期过滤 (d/date)

```
d>2024              → 2024 年之后
d>2024-06-01        → 2024-06-01 之后
d=7d                → 最近 7 天
d=3d                → 最近 3 天
d=[1d,3d]           → 1 到 3 天前
:date<=7d           → 同 d=7d（冒号语法）
```

#### 统计量过滤 (v/view, coin, like, ...)

```
v>1w                → 播放量 > 1 万
v>1k                → 播放量 > 1000
v=[1k,10000]        → 播放量在 1000-10000 之间
:coin>1000          → 投币数 > 1000
:view>=100          → 播放量 ≥ 100
```

支持的后缀：`k`（千）, `w`（万）

#### UP 主过滤 (u/user)

```
u="红警HBK08"       → 精确匹配 UP 主名（推荐加引号）
u=影视飓风          → 匹配 UP 主名
```

#### BV 号过滤 (bv)

```
bv=BV1LP4y1H7TJ     → 精确匹配 BV 号
bv=[BV1xx,BV1yy]    → 多个 BV 号
```

#### 组合示例

```
影视飓风 d>2024 v>1w q=v
  → 搜索"影视飓风"，限制 2024 年后、播放量 > 1 万，使用向量搜索

deepseek v3 :coin>100 q=wv
  → 搜索"deepseek v3"，投币 > 100，混合搜索

u="红警HBK08" d>2024 q=v
  → 查看 UP 主"红警HBK08" 2024 年后的视频，按语义排序
```

---

## 5. 解读搜索结果

### 5.1 步骤结果

explore 方法返回的 `data` 字段是步骤列表：

```python
result = explorer.unified_explore("影视飓风 q=v")

for step in result["data"]:
    print(f"步骤: {step['name_zh']} ({step['name']})")
    print(f"状态: {step['status']}")

    if step["name"] == "knn_search":
        output = step["output"]
        hits = output.get("hits", [])
        print(f"返回 {output.get('return_hits', 0)} 条命中 "
              f"(总计 {output.get('total_hits', 0)})")

    if step["name"] == "word_recall_supplement":
        info = step["output"]
        print(f"词语补充: +{info.get('supplement_count', 0)} 条")

    if step["name"] == "rerank":
        info = step["output"]
        print(f"精排: {info.get('reranked_count', 0)} 条候选")

    if step["name"] == "group_hits_by_owner":
        authors = step["output"].get("authors", [])
        for author in authors[:5]:
            print(f"  UP主: {author['name']} ({author['sum_count']} 个视频)")
```

### 5.2 命中结果字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `bvid` | str | BV 号 |
| `title` | str | 视频标题 |
| `desc` | str | 视频简介 |
| `tags` | str | 标签（逗号分隔） |
| `owner` | dict | `{"mid": int, "name": str}` |
| `pic` | str | 封面 URL |
| `duration` | int | 时长（秒） |
| `stat` | dict | 统计量 `{view, coin, favorite, like, reply, share, danmaku}` |
| `pubdate` | int | 发布时间戳 |
| `tid` / `ptid` | int | 分区 ID / 父分区 ID |
| `tname` | str | 分区名称 |
| `score` | float | ES BM25 原始分数 |
| `rank_score` | float | 排序综合分数 |
| `stat_score` | float | 预计算文档质量分（来自 ES 索引，由 DocScorer 生成，∈ [0,1)） |
| `relevance_score` | float | 归一化相关性分数（含 TM/OM 加成、深度惩罚等） |
| `quality_score` | float | 质量维度分数 |
| `recency_score` | float | 时效维度分数 |
| `popularity_score` | float | 热度维度分数（对数归一化） |
| `headline_score` | float | 复合头部分数（Phase 1 使用） |
| `rerank_score` | float | 精排分数（如有） |
| `cosine_similarity` | float | 余弦相似度（如有） |
| `hybrid_score` | float | 混合融合分数（如有） |
| `_title_matched` | bool | 标题/标签是否匹配查询关键词 |
| `_owner_matched` | bool | UP主名是否匹配查询 |
| `_matched_owner_name` | str | 匹配的 UP主 名称（如有） |
| `highlights` | dict | 高亮信息 |
| `region_info` | dict | 分区详细信息 |

### 5.3 UP 主分组

`group_hits_by_owner` 步骤返回 UP 主列表：

```python
authors = step["output"]["authors"]
for author in authors:
    print(f"UP主: {author['name']}")
    print(f"  mid: {author['mid']}")
    print(f"  头像: {author.get('face', '')}")
    print(f"  视频数: {author['sum_count']}")
    print(f"  总播放: {author['sum_view']}")
    for video in author["hits"][:3]:
        print(f"    - {video['title']}")
```

---

## 6. 运行测试

### 测试文件

测试文件位于 `elastics/tests/` 和 `recalls/tests/` 目录：

```bash
# 综合功能测试
python -m elastics.tests.test_videos

# 7 查询 recall+rank 质量评估（核心测试）
python -m elastics.tests.test_optimization_v2

# 单查询诊断（详细 top-20 输出）
python -m elastics.tests.diag_single "红警08"

# UP主召回诊断
python -m elastics.tests.diag_owner_recall

# 与官方 Bilibili 结果对比
python -m elastics.tests.compare_official

# 向量质量诊断
python -m elastics.tests.diag_deep
python -m elastics.tests.diag_float_vs_lsh

# 召回模块单元测试
python -m pytest recalls/tests/ -v
```

### 核心测试查询

`test_optimization_v2` 使用 7 个代表性查询评估系统质量：

| 查询 | 类型 | 评估重点 |
|------|------|----------|
| `红警08` | UP主查询 | UP主 `红警HBK08` 内容应占主导 (om_top20≈16) |
| `小红书推荐系统` | 技术话题 | CJK 连续子串匹配准确性 |
| `吴恩达大模型` | 混合 (UP主+话题) | UP主内容与话题内容平衡 |
| `chatgpt` | 纯话题 | 无过度 UP主 提升 |
| `gta` | 纯话题 | 内容深度惩罚（短标题处理） |
| `米娜` | 歧义查询 | UP主 `大聪明罗米娜` 不被过度提升 |
| `蝴蝶刀` | 纯话题 | 数据健壮性（负值/异常数据处理） |

评估指标：
- **tm_top20**：Top-20 中标题匹配查询的文档数
- **om_top20**：Top-20 中 UP主 匹配查询的文档数
- **short_top20**：Top-20 中短标题（≤5 字符）文档数

### 测试覆盖

| 测试类别 | 文件 | 说明 |
|----------|------|------|
| 基础搜索 | `test_videos.py` | 词语/KNN/混合搜索基本功能 |
| 质量评估 | `test_optimization_v2.py` | 7 查询 recall+rank 质量评估 |
| 单查询诊断 | `diag_single.py` | 详细 top-20 分析 |
| UP主召回诊断 | `diag_owner_recall.py` | UP主 分数分布、位置分析 |
| 官方对比 | `compare_official.py` | 与 Bilibili 官方结果对比 |
| 向量诊断 | `diag_deep.py`, `diag_float_vs_lsh.py` | 向量质量和 LSH 精度 |
| 召回单元 | `recalls/tests/test_base.py` | RecallPool 合并逻辑 |
| 排序单元 | `recalls/tests/test_diversified.py` | 多样化排序逻辑 |
| 噪声过滤 | `recalls/tests/test_noise.py` | 三层噪声过滤 |
| 召回优化 | `recalls/tests/test_recall_optimization.py` | 召回优化集成 |

---

## 7. 注意事项

### 默认搜索模式

未指定 `q=` 时，默认使用 `QMOD = ["word", "vector"]`（混合搜索），分发到 `hybrid_explore_v2()`。

### 向量搜索 (q=v) 始终开启精排

`q=v` 模式下精排（rerank）始终开启，即使未显式指定 `r`。这是因为 LSH hamming 距离的分数精度有限（分数集中在 0.01-0.02 的范围内），直接使用 hamming 分数排序效果不佳。精排使用 float 向量的余弦相似度，提供更精细的排序。

### 多样化排序 (diversified) 为默认排序策略

所有 V2 explore 方法默认使用 `diversified` 排序策略（三阶段排序 + UP主意图感知），而不是旧版的 `stats` 或 `tiered` 排序。

### UP主意图自适应

系统会自动检测查询中的 UP主 意图并动态调整排序行为：
- **强 UP主 意图**（如 `红警08`）：owner_intent_strength ≈ 1.0，UP主 视频获得显著提升
- **弱 UP主 意图**（如 `米娜`）：intent ≈ 0.3，UP主 匹配适度影响
- **纯话题查询**（如 `chatgpt`）：intent ≈ 0.1，UP主 匹配几乎不影响排序

这一行为由 RecallPoolOptimizer 分析召回池自动决定，无需用户干预。

### 实体查询的语义漏召

搜索"影视飓风"（一个 UP 主的名字）时，嵌入模型会将其理解为"影视 + 飓风"，返回与飓风/气象相关的视频。6 车道词语召回中的 `owner_name` 和 `title_match` 车道以及补充词语召回机制通过关键词搜索来弥补这一不足。

### 窄过滤器自动切换

包含 UP 主过滤器（`u=xxx`）或 BV 号过滤器的查询会自动切换到 filter-first 模式（`dual_sort_filter_search`），避免 KNN 搜索在小结果集上的低效问题。

### 健壮性处理

排序阶段对每条文档的评分使用 try/except 保护，单条文档的异常数据（如负播放量、缺失字段）不会导致整体流程崩溃。`math.log1p` 对负值进行安全处理 (`max(0, value)`)。

### 环境配置

- 开发环境：`ELASTIC_DEV` → `bili_videos_dev6`（约 5000 万文档）
- 生产环境：`ELASTIC_PRO` → `bili_videos_pro1`
- TEI 嵌入服务需在线运行（`http://ai122:28800`）

### JSON 序列化

explore 方法返回的结果可直接用 FastAPI 的 `jsonable_encoder` 序列化。UP 主分组返回 **list**（而非 dict），以确保 JSON 传输后顺序不变。

---

## 附录 A: VideoSearcherV2 完整参数

### search()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串（可含 DSL 表达式） |
| `match_fields` | list[str] | `SEARCH_MATCH_FIELDS` | 搜索匹配字段 |
| `source_fields` | list[str] | `SOURCE_FIELDS` | 返回的文档字段 |
| `match_type` | str | `"cross_fields"` | 匹配类型 |
| `match_bool` | str | `"must"` | 布尔逻辑 |
| `match_operator` | str | `"or"` | 词语间操作符 |
| `extra_filters` | list[dict] | `[]` | 额外过滤条件 |
| `boost` | bool | `True` | 是否使用字段加权 |
| `boosted_fields` | dict | `SEARCH_BOOSTED_FIELDS` | 字段权重配置 |
| `rank_method` | str | `"stats"` | 排序方法 |
| `limit` | int | `50` | 返回结果数上限 |
| `rank_top_k` | int | `50` | 排序后取 Top-K |
| `terminate_after` | int | `2000000` | ES terminate_after |
| `timeout` | float | `2` | 搜索超时（秒） |
| `verbose` | bool | `False` | 详细日志 |

### knn_search()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串 |
| `knn_field` | str | `"text_emb"` | 向量字段名 |
| `k` | int | `400` | 每分片返回近邻数 |
| `num_candidates` | int | `10000` | 每分片候选数 |
| `rank_method` | str | `"stats"` | 排序方法 |
| `limit` | int | `50` | 结果上限 |
| `rank_top_k` | int | `50` | Top-K |
| `timeout` | float | `8` | 超时 |
| `verbose` | bool | `False` | 详细日志 |

### hybrid_search()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串 |
| `fusion_method` | str | `"rrf"` | 融合方法 (`"rrf"` / `"weighted"`) |
| `rrf_k` | int | `60` | RRF 常数 k |
| `knn_k` | int | `400` | KNN 近邻数 |
| `knn_num_candidates` | int | `10000` | KNN 候选数 |
| 其余参数 | — | — | 同 search() |

---

## 附录 B: VideoExplorer 完整参数

### unified_explore()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串（含 DSL 表达式） |
| `qmod` | str/list | `None` | 查询模式（覆盖 DSL 中的 q=） |
| `extra_filters` | list[dict] | `[]` | 额外过滤 |
| `suggest_info` | dict | `{}` | 建议信息 |
| `verbose` | bool | `False` | 详细日志 |
| `rank_top_k` | int | `400` | 返回 Top-K |
| `group_owner_limit` | int | `25` | UP 主分组上限 |
| `knn_field` | str | `"text_emb"` | KNN 向量字段 |
| `knn_k` | int | `400` | KNN 近邻数 |
| `knn_num_candidates` | int | `10000` | KNN 候选数 |

### explore_v2()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串 |
| `extra_filters` | list[dict] | `[]` | 额外过滤 |
| `boost` | bool | `True` | 字段加权 |
| `verbose` | bool | `False` | 详细日志 |
| `rank_top_k` | int | `400` | Top-K |
| `group_owner_limit` | int | `25` | UP 主分组上限 |
| `enable_rerank` | bool | `False` | 是否精排 |

### knn_explore_v2()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串 |
| `extra_filters` | list[dict] | `[]` | 额外过滤 |
| `verbose` | bool | `False` | 详细日志 |
| `knn_field` | str | `"text_emb"` | 向量字段 |
| `knn_k` | int | `400` | 每分片近邻数 |
| `knn_num_candidates` | int | `10000` | 每分片候选数 |
| `enable_rerank` | bool | `True` | 是否精排 |
| `word_recall_enabled` | bool | `True` | 启用词语补充召回 |
| `word_recall_limit` | int | `1000` | 词语搜索上限 |
| `rank_top_k` | int | `400` | Top-K |
| `group_owner_limit` | int | `25` | UP 主分组上限 |

### hybrid_explore_v2()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串 |
| `extra_filters` | list[dict] | `[]` | 额外过滤 |
| `verbose` | bool | `False` | 详细日志 |
| `knn_field` | str | `"text_emb"` | 向量字段 |
| `knn_k` | int | `400` | KNN 近邻数 |
| `knn_num_candidates` | int | `10000` | KNN 候选数 |
| `rrf_k` | int | `60` | RRF 常数 |
| `fusion_method` | str | `"rrf"` | 融合方法 |
| `rank_top_k` | int | `400` | Top-K |
| `group_owner_limit` | int | `25` | UP 主分组上限 |
| `enable_rerank` | bool | `False` | 是否精排 |
