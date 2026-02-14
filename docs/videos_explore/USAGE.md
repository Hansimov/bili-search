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
  - [3.2 词语探索 (explore)](#32-词语探索-explore)
  - [3.3 向量探索 (knn_explore)](#33-向量探索-knn_explore)
  - [3.4 混合探索 (hybrid_explore)](#34-混合探索-hybrid_explore)
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
- [附录 C: 常量配置速查](#附录-c-常量配置速查)
- [附录 D: 排序策略配置](#附录-d-排序策略配置)

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

# 搜索（默认混合模式）
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
    num_candidates=4000,     # 每分片搜索 4000 个候选
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

### 3.2 词语探索 (explore)

适用于 `q=w` 和 `q=wr` 模式。

```python
result = explorer.explore(
    query="黑神话 悟空",
    most_relevant_limit=10000,   # 初始搜索范围
    rank_method="stats",         # 按热度排序
    rank_top_k=400,              # 返回 Top-400
    group_owner_limit=25,        # UP 主分组上限
    enable_rerank=False,         # 是否精排
    verbose=True,
)
```

### 3.3 向量探索 (knn_explore)

适用于 `q=v` 和 `q=vr` 模式。自动包含词语补充召回和精排。

```python
result = explorer.knn_explore(
    query="原神 角色",
    knn_k=400,                    # 每分片近邻数
    knn_num_candidates=4000,      # 每分片候选数
    enable_rerank=True,           # 精排（默认开启）
    word_recall_enabled=True,     # 词语补充召回（默认开启）
    word_recall_limit=1000,       # 词语搜索上限
    rank_method="relevance",      # 纯相关度排序
    rank_top_k=400,
    group_owner_limit=25,
    verbose=True,
)
```

**禁用词语补充召回：**

```python
result = explorer.knn_explore(
    query="原神",
    word_recall_enabled=False,   # 仅使用 KNN 结果
    verbose=True,
)
```

### 3.4 混合探索 (hybrid_explore)

适用于 `q=wv` 和 `q=wvr` 模式。

```python
result = explorer.hybrid_explore(
    query="深度学习 教程 d>2024",
    rrf_k=60,                    # RRF 常数
    fusion_method="rrf",         # "rrf" 或 "weighted"
    rank_method="tiered",        # 分层排序
    rank_top_k=400,
    group_owner_limit=25,
    enable_rerank=False,         # 混合搜索通常不需要精排
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
| `rerank_score` | float | 精排分数（如有） |
| `cosine_similarity` | float | 余弦相似度（如有） |
| `hybrid_score` | float | 混合融合分数（如有） |
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

测试文件位于 `elastics/tests/` 目录：

```bash
# 运行全部测试
python -m elastics.tests.test_videos

# 诊断脚本
python -m elastics.tests.diag_deep          # 向量质量诊断
python -m elastics.tests.diag_float_vs_lsh  # Float vs LSH 对比
python -m elastics.tests.diag_knn           # KNN 召回诊断
```

### 测试覆盖

测试文件包含以下类别的测试：

| 测试类别 | 测试函数 | 说明 |
|----------|----------|------|
| 基础搜索 | `test_search`, `test_filter`, `test_suggest` | 词语搜索基本功能 |
| KNN 搜索 | `test_knn_search`, `test_knn_search_with_filters` | 向量搜索 |
| 混合搜索 | `test_hybrid_search`, `test_rrf_fusion_fill` | 混合搜索和融合 |
| 探索模式 | `test_explore`, `test_knn_explore`, `test_unified_explore` | 多步骤探索 |
| 过滤搜索 | `test_filter_only_search`, `test_filter_only_explore` | 纯过滤 |
| 窄过滤器 | `test_narrow_filter_detection`, `test_knn_filter_bug` | 用户/BV 过滤 |
| 精排 | `test_rerank_step_by_step`, `test_owner_name_keyword_boost` | 精排机制 |
| 词语召回 | `test_word_recall_supplement`, `test_word_recall_overlap_improvement` | 补充召回 |
| DSL 解析 | `test_qmod_parser`, `test_dsl_query_construction` | 查询解析 |
| 分组 | `test_author_grouper_unit`, `test_author_ordering` | UP 主分组 |

---

## 7. 注意事项

### 向量搜索 (q=v) 始终开启精排

`q=v` 模式下精排（rerank）始终开启，即使未显式指定 `r`。这是因为 LSH hamming 距离的分数精度有限（分数集中在 0.01-0.02 的范围内），直接使用 hamming 分数排序效果不佳。精排使用 float 向量的余弦相似度，提供更精细的排序。

### 实体查询的语义漏召

搜索"影视飓风"（一个 UP 主的名字）时，嵌入模型会将其理解为"影视 + 飓风"，返回与飓风/气象相关的视频。补充词语召回机制通过并行运行关键词搜索来弥补这一不足。

### 窄过滤器自动切换

包含 UP 主过滤器（`u=xxx`）或 BV 号过滤器的查询会自动切换到 filter-first 模式（`dual_sort_filter_search`），避免 KNN 搜索在小结果集上的低效问题。

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
| `match_fields` | list[str] | `["title.words", "tags.words", "owner.name.words", "desc.words"]` | 搜索匹配字段 |
| `source_fields` | list[str] | `SOURCE_FIELDS` | 返回的文档字段 |
| `match_type` | str | `"cross_fields"` | 匹配类型 |
| `match_bool` | str | `"must"` | 布尔逻辑 |
| `match_operator` | str | `"or"` | 词语间操作符 |
| `extra_filters` | list[dict] | `[]` | 额外过滤条件 |
| `suggest_info` | dict | `{}` | 建议信息 |
| `request_type` | str | `"search"` | 请求类型 (`"search"` / `"suggest"`) |
| `parse_hits` | bool | `True` | 是否解析命中结果 |
| `drop_no_highlights` | bool | `False` | 是否丢弃无高亮结果 |
| `add_region_info` | bool | `True` | 添加分区信息 |
| `add_highlights_info` | bool | `True` | 添加高亮汇总 |
| `is_explain` | bool | `False` | ES explain 模式 |
| `is_profile` | bool | `False` | ES profile 模式 |
| `is_highlight` | bool | `True` | 是否启用高亮 |
| `boost` | bool | `True` | 是否使用字段加权 |
| `boosted_fields` | dict | `SEARCH_BOOSTED_FIELDS` | 字段权重配置 |
| `use_script_score` | bool | `False` | 是否使用脚本评分 |
| `rank_method` | str | `"stats"` | 排序方法 |
| `score_threshold` | float | `None` | 最低分数阈值 |
| `limit` | int | `50` | 返回结果数上限 |
| `rank_top_k` | int | `50` | 排序后取 Top-K |
| `terminate_after` | int | `2000000` | ES terminate_after |
| `timeout` | float | `2` | 搜索超时（秒） |
| `verbose` | bool | `False` | 详细日志 |

### knn_search()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串 |
| `source_fields` | list[str] | `SOURCE_FIELDS` | 返回字段 |
| `extra_filters` | list[dict] | `[]` | 额外过滤 |
| `knn_field` | str | `"text_emb"` | 向量字段名 |
| `k` | int | `400` | 每分片返回近邻数 |
| `num_candidates` | int | `4000` | 每分片候选数 |
| `similarity` | float | `None` | 最低相似度阈值 |
| `parse_hits` | bool | `True` | 解析命中 |
| `add_region_info` | bool | `True` | 分区信息 |
| `rank_method` | str | `"stats"` | 排序方法 |
| `limit` | int | `50` | 结果上限 |
| `rank_top_k` | int | `50` | Top-K |
| `skip_ranking` | bool | `False` | 跳过排序 |
| `timeout` | float | `8` | 超时 |
| `verbose` | bool | `False` | 详细日志 |

### hybrid_search()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串 |
| `fusion_method` | str | `"rrf"` | 融合方法 (`"rrf"` / `"weighted"`) |
| `rrf_k` | int | `60` | RRF 常数 k |
| `word_weight` | float | `0.5` | 词语权重 (weighted 模式) |
| `vector_weight` | float | `0.5` | 向量权重 (weighted 模式) |
| `knn_k` | int | `400` | KNN 近邻数 |
| `knn_num_candidates` | int | `4000` | KNN 候选数 |
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
| `most_relevant_limit` | int | `10000` | 初始搜索范围 |
| `rank_method` | str | `"stats"` | 排序方法 |
| `rank_top_k` | int | `400` | 返回 Top-K |
| `group_owner_limit` | int | `25` | UP 主分组上限 |
| `knn_field` | str | `"text_emb"` | KNN 向量字段 |
| `knn_k` | int | `400` | KNN 近邻数 |
| `knn_num_candidates` | int | `4000` | KNN 候选数 |

### explore()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串 |
| `query_dsl_dict` | dict | `None` | 预构建的 DSL 字典 |
| `match_fields` | list[str] | `SEARCH_MATCH_FIELDS` | 匹配字段 |
| `match_type` | str | `"cross_fields"` | 匹配类型 |
| `extra_filters` | list[dict] | `[]` | 额外过滤 |
| `suggest_info` | dict | `{}` | 建议信息 |
| `boost` | bool | `True` | 字段加权 |
| `boosted_fields` | dict | `EXPLORE_BOOSTED_FIELDS` | 权重配置 |
| `verbose` | bool | `False` | 详细日志 |
| `most_relevant_limit` | int | `10000` | 初始搜索范围 |
| `rank_method` | str | `"stats"` | 排序方法 |
| `rank_top_k` | int | `400` | Top-K |
| `group_owner_limit` | int | `25` | UP 主分组上限 |
| `enable_rerank` | bool | `False` | 是否精排 |
| `rerank_max_hits` | int | `2000` | 精排候选上限 |
| `rerank_keyword_boost` | float | `1.5` | 关键词加权 |
| `rerank_title_keyword_boost` | float | `2.0` | 标题关键词加权 |
| `rerank_text_fields` | list[str] | `["title","tags","desc","owner.name"]` | 精排文本字段 |

### knn_explore()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串 |
| `extra_filters` | list[dict] | `[]` | 额外过滤 |
| `verbose` | bool | `False` | 详细日志 |
| `knn_field` | str | `"text_emb"` | 向量字段 |
| `knn_k` | int | `400` | 每分片近邻数 |
| `knn_num_candidates` | int | `4000` | 每分片候选数 |
| `similarity` | float | `None` | 最低相似度 |
| `enable_rerank` | bool | `True` | 是否精排 |
| `rerank_max_hits` | int | `2000` | 精排候选上限 |
| `rerank_keyword_boost` | float | `1.5` | 关键词加权 |
| `rerank_title_keyword_boost` | float | `2.0` | 标题关键词加权 |
| `rerank_text_fields` | list[str] | `["title","tags","desc","owner.name"]` | 精排文本字段 |
| `word_recall_enabled` | bool | `True` | 启用词语补充召回 |
| `word_recall_limit` | int | `1000` | 词语搜索上限 |
| `word_recall_timeout` | float | `3` | 词语搜索超时（秒） |
| `most_relevant_limit` | int | `10000` | 搜索范围 |
| `rank_method` | str | `"relevance"` | 排序方法 |
| `rank_top_k` | int | `400` | Top-K |
| `group_owner_limit` | int | `25` | UP 主分组上限 |
| `group_sort_field` | str | `"first_appear_order"` | UP 主排序字段 |

### hybrid_explore()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | — | 查询字符串 |
| `extra_filters` | list[dict] | `[]` | 额外过滤 |
| `suggest_info` | dict | `{}` | 建议信息 |
| `verbose` | bool | `False` | 详细日志 |
| `knn_field` | str | `"text_emb"` | 向量字段 |
| `knn_k` | int | `400` | KNN 近邻数 |
| `knn_num_candidates` | int | `4000` | KNN 候选数 |
| `rrf_k` | int | `60` | RRF 常数 |
| `fusion_method` | str | `"rrf"` | 融合方法 |
| `most_relevant_limit` | int | `10000` | 搜索范围 |
| `rank_method` | str | `"tiered"` | 排序方法 |
| `rank_top_k` | int | `400` | Top-K |
| `group_owner_limit` | int | `25` | UP 主分组上限 |
| `group_sort_field` | str | `"first_appear_order"` | UP 主排序字段 |
| `enable_rerank` | bool | `False` | 是否精排 |
| `rerank_max_hits` | int | `2000` | 精排候选上限 |

---

## 附录 C: 常量配置速查

> 配置文件：`elastics/videos/constants.py` 和 `ranks/constants.py`

### 搜索字段配置

```python
# 搜索时使用的匹配字段（.words 子字段，由 es_tok 分词器索引）
SEARCH_MATCH_FIELDS = ["title.words", "tags.words", "owner.name.words", "desc.words"]

# 字段权重（用于 boosted multi_match / cross_fields）
SEARCH_BOOSTED_FIELDS = {
    "title": 3, "title.words": 3,        # 标题权重最高
    "tags": 2.5, "tags.words": 2.5,      # 标签次之
    "owner.name": 2, "owner.name.words": 2,  # UP主名
    "desc": 0.1, "desc.words": 0.1,      # 简介权重较低
    "title.pinyin": 0.25,                # 拼音（v6 已移除）
    "tags.pinyin": 0.2,
    "owner.name.pinyin": 0.2,
    "desc.pinyin": 0.01,
}
```

### KNN 设置

```python
KNN_TEXT_EMB_FIELD = "text_emb"     # 向量字段
KNN_K = 400                         # 每分片近邻数
KNN_NUM_CANDIDATES = 4000           # 每分片候选数 (10× K)
KNN_TIMEOUT = 8                     # KNN 搜索超时
KNN_SIMILARITY = "hamming"          # 距离度量
KNN_LSH_BITN = 2048                 # LSH bit 数
```

### 词语补充召回

```python
KNN_WORD_RECALL_ENABLED = True      # 启用
KNN_WORD_RECALL_LIMIT = 1000        # 词语搜索上限
KNN_WORD_RECALL_TIMEOUT = 3         # 超时（秒）
```

### 精排配置

```python
RERANK_ENABLED = True               # 默认启用
RERANK_MAX_HITS = 2000              # 最大精排候选
RERANK_KEYWORD_BOOST = 1.5          # 一般关键词加权
RERANK_TITLE_KEYWORD_BOOST = 2.0    # 标题关键词加权
RERANK_TEXT_FIELDS = ["title", "tags", "desc", "owner.name"]
RERANK_TIMEOUT = 30                 # 精排超时
RERANK_MAX_PASSAGE_LENGTH = 4096    # 最大段落长度
```

---

## 附录 D: 排序策略配置

### stats — 统计排序

```python
# 分数融合公式: stats_score × pubdate_score × (relate_score³)
STAT_FIELDS = ["view", "favorite", "coin", "reply", "share", "danmaku"]  # 统计字段
STAT_LOGX_OFFSETS = {"view": 10, "favorite": 2, "coin": 2, ...}  # 对数偏移

# 时效性衰减
PUBDATE_SCORE_POINTS = [
    (0, 4.0),    # 今天
    (7, 1.0),    # 1 周
    (30, 0.6),   # 1 月
    (365, 0.3),  # 1 年
]
```

### relevance — 纯相关度

```python
RELEVANCE_MIN_SCORE = 0.4           # 最低分数
RELEVANCE_SCORE_POWER = 2.0         # 幂变换指数
HIGH_RELEVANCE_THRESHOLD = 0.85     # 高相关度阈值
HIGH_RELEVANCE_BOOST = 2.0          # 高相关度加权
```

### tiered — 分层排序

```python
TIERED_HIGH_RELEVANCE_THRESHOLD = 0.7   # 高相关区阈值
TIERED_SIMILARITY_THRESHOLD = 0.05      # 等价相关度阈值
TIERED_STATS_WEIGHT = 0.7               # 统计权重
TIERED_RECENCY_WEIGHT = 0.3             # 时效权重
```

### RRF — 倒数排序融合

```python
RRF_K = 60                          # RRF 常数
RRF_WEIGHTS = {
    "score": 5.0,                   # 相关度 (最高)
    "pubdate": 2.0,                 # 发布日期
    "stat.view": 1.0,               # 播放量
    "stat.favorite": 1.0,           # 收藏数
    "stat.coin": 1.0,               # 投币数
}
```
