# Owner Search 改进方案

> 版本：v2.0  
> 日期：2026-03-01  
> 状态：方案设计

---

## 目录

- [1. 现状分析](#1-现状分析)
  - [1.1 当前架构：基于 videos 索引的间接搜索](#11-当前架构基于-videos-索引的间接搜索)
  - [1.2 数据源现状](#12-数据源现状)
  - [1.3 核心痛点](#13-核心痛点)
- [2. 目标与需求](#2-目标与需求)
  - [2.1 搜索能力目标](#21-搜索能力目标)
  - [2.2 五大搜索维度](#22-五大搜索维度)
- [3. 方案总览](#3-方案总览)
- [4. 数据层：构建 Owner 索引](#4-数据层构建-owner-索引)
  - [4.1 数据聚合管线](#41-数据聚合管线)
  - [4.2 ES 索引 Mapping 设计](#42-es-索引-mapping-设计)
  - [4.3 数据更新策略](#43-数据更新策略)
- [5. 算法层：Owner 搜索算法设计](#5-算法层owner-搜索算法设计)
  - [5.1 名称匹配（Name Match）](#51-名称匹配name-match)
  - [5.2 影响力排序（Influence Scoring）](#52-影响力排序influence-scoring)
  - [5.3 创作质量评估（Quality Scoring）](#53-创作质量评估quality-scoring)
  - [5.4 活跃度评估（Activity Scoring）](#54-活跃度评估activity-scoring)
  - [5.5 关联领域匹配（Domain Relevance）](#55-关联领域匹配domain-relevance)
  - [5.6 关联用户（User Graph）](#56-关联用户user-graph)
  - [5.7 综合排序公式](#57-综合排序公式)
- [6. 系统设计层：Owner 搜索服务架构](#6-系统设计层owner-搜索服务架构)
  - [6.1 OwnerSearcher 服务设计](#61-ownersearcher-服务设计)
  - [6.2 与 VideoExplorer 联动](#62-与-videoexplorer-联动)
  - [6.3 LLM 工具集成](#63-llm-工具集成)
  - [6.4 前端 UI 设计](#64-前端-ui-设计)
- [7. 实施路线图](#7-实施路线图)
- [8. 风险与约束](#8-风险与约束)

---

## 1. 现状分析

### 1.1 当前架构：基于 videos 索引的间接搜索

当前系统中 **不存在独立的 Owner/User ES 索引**。所有 owner 搜索能力都是从 `bili_videos` 索引中间接推导的：

```
用户查询 "红警08"
    │
    ├─ VideoSearcherV2.suggest()
    │   └─ BM25 搜索 owner.name.words 字段 (boost=2.0)
    │   └─ analyze_suggest_for_authors() 从命中结果中统计 owner.name 频率
    │
    ├─ MultiLaneWordRecall (6 车道)
    │   └─ owner_name 车道: owner.name.words (boost=8.0) → 200 条
    │   └─ _tag_owner_matches(): token-set / CJK子串 / 通用子串 三重策略
    │
    ├─ _detect_owner_intent() → 集中度分析 → 判定 owner 意图
    ├─ _owner_focused_recall() → term 过滤 owner.name.keyword → 补充 50 条/owner
    │
    ├─ RecallPoolOptimizer → OwnerAnalysis → intent_strength
    ├─ DiversifiedRanker → owner_match_bonus (0.30×strength)
    │
    └─ AuthorGrouper → 按 owner.mid 分组 → 返回 "相关作者" 侧边栏
```

**关键路径上的组件：**

| 组件 | 位置 | 作用 |
|------|------|------|
| `check_author` (LLM 工具) | `llms/tools/executor.py` | 调用 suggest → `analyze_suggest_for_authors()` 检测查询是否匹配 UP 主 |
| `MultiLaneWordRecall.owner_name` | `recalls/word.py` | 专用车道，BM25 搜索 `owner.name.words^8.0` |
| `_tag_owner_matches()` | `recalls/word.py` | 三重策略标记 `_owner_matched` |
| `_detect_owner_intent()` | `recalls/manager.py` | token-set + CJK子串 + 通用子串匹配 + 集中度分析 |
| `_owner_focused_recall()` | `recalls/manager.py` | `owner.name.keyword` 精确过滤 + 合成分数注入 (p25) |
| `OwnerAnalysis` | `recalls/optimizer.py` | 计算 intent_strength/concentration/diversity |
| `DiversifiedRanker` | `ranks/diversified.py` | `OWNER_MATCH_BONUS=0.30`, `OWNER_RELEVANCE_FLOOR=0.75` |
| `AuthorGrouper` | `ranks/grouper.py` | 按 mid 分组，聚合 sum_view/sum_count/latest_pubdate |
| `ResultAuthorsList` | `bili-search-ui` | 前端 "相关作者" 侧边栏 |

### 1.2 数据源现状

**唯一可用数据源：MongoDB `videos` 集合**

| 集合 | `_id` | 主要字段 | 数据量 | 更新频率 |
|------|-------|----------|--------|----------|
| `videos` | bvid | owner.{mid, name}, stat.{view, like, coin, favorite, danmaku, reply, share}, title, tags, desc, pubdate, duration, tid, ptid, pic | ~10 亿 | 持续抓取 |
| `videos_tags` | bvid | tags 列表（已合并入 videos.tags） | ~10 亿 | 定期抓取 |

> **关于其他数据源的说明：**
> - `users_cards` — 历史上曾通过 B 站 Card API 抓取用户画像（face, sign, level, official, follower 等），但该 API 已被封禁，此 collection **不可用**。
> - `users_stats` — 通过 videos 聚合生成的用户统计，但设计不合理、聚合效率差，**计划废弃**。
> - 因此，本方案所有数据均 **仅来源于 `videos` 集合**。每个 video 文档中可用的 owner 信息仅有 `owner.mid` (long) 和 `owner.name` (string)。

**单个 video 文档结构（与 owner 相关的可用字段）：**

```json
{
  "bvid": "BV1xxx",
  "owner": { "mid": 666632703, "name": "某UP主" },
  "stat": {
    "view": 77, "danmaku": 0, "reply": 0,
    "favorite": 1, "coin": 0, "share": 0, "like": 1
  },
  "title": "...", "tags": "tag1, tag2, ...", "desc": "...",
  "pubdate": 1721384851, "duration": 61,
  "tid": 201, "ptid": 36,
  "stat_score": 0.35,
  "pic": "http://..."
}
```

**ES videos 索引中的 owner 字段：**

```json
"owner": {
  "properties": {
    "mid": { "type": "long" },
    "name": {
      "type": "text", "index": false,
      "fields": {
        "keyword": { "type": "keyword", "eager_global_ordinals": true },
        "words": { "type": "text", "analyzer": "chinese_analyzer" }
      }
    }
  }
}
```

仅包含 mid (long) 和 name (keyword + chinese_analyzer)。没有独立的用户画像数据（粉丝数、签名、头像、认证等均不可获取）。

### 1.3 核心痛点

#### P0: 无法直接搜索 Owner

- **没有 owner 索引**：必须先搜索 videos，再从结果中间接推测 owner
- **只能基于 owner.name 匹配**：无法按 mid 搜索；无法利用 owner 级别的统计数据进行排序
- **搜索结果是 videos 而非 owners**：`check_author` 返回的是视频命中中的 owner 频率分布，而非一个真正的 "用户搜索结果"
- **无独立用户画像数据**：Card API 已封禁，无法获取粉丝数、签名、头像、认证等信息；只能完全依赖 videos 中聚合出的数据

#### P1: 名称匹配不精准

- **Token-set 匹配的 false positive**：query "红警" 能匹配到 "红警HBK08"、"红警V神"、"红警魔鬼蓝天" — 无法区分谁是用户真正要找的
- **没有完整的 pinyin 匹配**：es_tok 虽然有 pinyin 分词，但 owner_name 车道未充分利用拼音搜索
- **短查询子串匹配的歧义**：query "米娜" 匹配到 6 个不同的 UP 主名称（如 "大聪明罗米娜"），但用户可能是在搜内容而非搜人
- **无法处理曾用名/别名**：UP 主改名后旧内容的 owner.name 可能不一致

#### P2: 排序效果差

- **缺乏 owner 维度的信号**：当前排序完全基于单个视频的 BM25 + stat + recency，没有 owner 级别的质量/影响力信号
- **AuthorGrouper 排序过于简单**：仅支持 `sum_view`、`sum_count`、`first_appear_order` 等简单聚合，不能综合评估
- **合成分数注入不够精确**：owner_focused_recall 注入 p25 合成分数，可能偏高或偏低

#### P3: 搜索维度单一

- 无法按 "领域/话题" 搜索 UP 主（如 "做黑神话悟空视频的 UP 主"）
- 无法按 "影响力" 排序（如 "播放量最高的游戏区 UP 主"）
- 无法按 "活跃度" 过滤（如 "最近一个月有更新的 UP 主"）
- 无法发现 "关联 UP 主"（如 "经常和影视飓风合作的 UP 主"）

#### P4: 性能瓶颈

- **间接搜索开销大**：通过 videos 索引搜索再聚合 owner，需要 6 车道并行 + 3 轮补充 + owner_focused_recall，延迟高
- **聚合成本高**：10 亿级文档上做 owner 维度的 terms 聚合非常昂贵
- **无法预计算**：每次搜索都要实时聚合 owner 的 view/count/pubdate 等统计量

---

## 2. 目标与需求

### 2.1 搜索能力目标

构建一套完整的 Owner 搜索系统，实现：

1. **直接搜索 Owner**：通过独立的 owners 索引，支持直接返回 owner 列表（而非 videos 列表）
2. **多维度搜索**：支持名称、影响力、质量、活跃度、领域、关联用户 6 个维度的搜索和排序
3. **与 videos 搜索联动**：owner 搜索结果可用于过滤/增强 videos 搜索
4. **LLM 工具增强**：让 LLM 能直接查找 UP 主并基于 UP 主返回推荐

### 2.2 五大搜索维度

| 维度 | 信号来源 | 使用场景 |
|------|----------|----------|
| **影响力** | 总视频数、总播放量、总获赞数、总投币数 | "做游戏的大 UP 主" |
| **创作质量** | 平均收藏率、平均投币率、收藏/播放比、平均 stat_score | "质量最高的科技区 UP 主" |
| **活跃度** | 最新视频日期、发布频率（视频数/时间跨度） | "最近还在更新的 UP 主" |
| **关联领域** | 视频的 title/tags/desc 关键词；tags 分布；向量 embedding | "做黑神话悟空内容的 UP 主" |
| **关联用户** | 视频中提及的 @用户、合作视频、tags 中出现的其他用户名 | "和影视飓风合作过的 UP 主" |

> **关于 "粉丝数" 的说明：** 由于 B 站 Card API 已封禁，无法获取粉丝数（follower）。影响力维度完全基于视频统计数据（播放/互动/作品量）。实际上，视频统计数据是比粉丝数更可靠的 "内容影响力" 信号——粉丝数可能有水分，但播放量和互动数据更真实地反映创作者的实际影响力。

---

## 3. 方案总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         方案三层架构                                 │
├──────────────┬──────────────────────┬───────────────────────────────┤
│   数据层     │      算法层          │         系统设计层             │
│              │                      │                               │
│ ▸ MongoDB    │ ▸ 名称匹配           │ ▸ OwnerSearcher 服务          │
│   聚合管线   │ ▸ 影响力评分         │ ▸ 与 VideoExplorer 联动       │
│ ▸ ES owners  │ ▸ 创作质量评分       │ ▸ LLM 工具集成                │
│   索引设计   │ ▸ 活跃度评分         │ ▸ 前端 UI 设计                │
│ ▸ 增量更新   │ ▸ 领域相关性         │ ▸ DSL 扩展                    │
│   策略       │ ▸ 关联用户图谱       │                               │
│              │ ▸ 综合排序           │                               │
└──────────────┴──────────────────────┴───────────────────────────────┘
```

---

## 4. 数据层：构建 Owner 索引

### 4.1 数据聚合管线

**唯一数据源：MongoDB `videos` 集合**。所有 owner 数据通过对 videos 按 `owner.mid` 聚合产生。

```
┌────────────────────────────────────────────────┐
│              MongoDB videos 集合                │
│              (~10亿 docs)                       │
│  每条: owner.{mid, name}, stat.{view, like,...} │
│        title, tags, desc, pubdate, duration,    │
│        tid, ptid, stat_score, pic               │
└─────────────────────┬──────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │  Stage 1: MongoDB     │
          │  聚合管线              │
          │  $sort + $group       │
          │  by owner.mid         │
          └───────────┬───────────┘
                      │
          ┌───────────┴───────────┐
          │  Stage 2: Python      │
          │  后处理计算            │
          │  ▸ 评分计算            │
          │  ▸ 标签/关键词提取    │
          │  ▸ 关联用户提取       │
          └───────────┬───────────┘
                      │
          ┌───────────┴───────────┐
          │  Stage 3: 向量计算     │
          │  (可延后到 Phase 3)    │
          │  ▸ owner_emb 编码     │
          └───────────┬───────────┘
                      │
                      ▼
          ┌────────────────────────┐
          │  ES: bili_owners_v1     │
          │  (~6000万 docs)        │
          └────────────────────────┘
```

#### Stage 1: MongoDB 聚合管线

全量聚合方案 — 利用 MongoDB aggregation pipeline 按 `owner.mid` 分组：

```python
AGG_PIPELINE = [
    # 0. 预排序：按 pubdate 降序，使 $first 取到最新的
    {"$sort": {"pubdate": -1}},

    # 1. 按 owner.mid 分组聚合
    {"$group": {
        "_id": "$owner.mid",
        "name":             {"$first": "$owner.name"},     # 最新视频的 name (可能改名)
        # ---- 基础统计 ----
        "total_videos":     {"$sum": 1},
        "total_duration":   {"$sum": "$duration"},
        # ---- 播放/互动汇总 ----
        "total_view":       {"$sum": "$stat.view"},
        "total_like":       {"$sum": "$stat.like"},
        "total_coin":       {"$sum": "$stat.coin"},
        "total_favorite":   {"$sum": "$stat.favorite"},
        "total_danmaku":    {"$sum": "$stat.danmaku"},
        "total_reply":      {"$sum": "$stat.reply"},
        "total_share":      {"$sum": "$stat.share"},
        # ---- 时间统计 ----
        "latest_pubdate":   {"$max": "$pubdate"},
        "earliest_pubdate": {"$min": "$pubdate"},
        "latest_bvid":      {"$first": "$bvid"},
        # ---- 质量统计 ----
        "total_stat_score": {"$sum": "$stat_score"},
        # ---- 区域统计 (众数需后处理) ----
        "tid_list":         {"$push": "$tid"},
        "ptid_list":        {"$push": "$ptid"},
        # ---- 领域统计 (标签文本需后处理提取) ----
        "all_tags":         {"$push": "$tags"},
        # ---- 代表作 (最新一条的 pic 和 bvid) ----
        "latest_pic":       {"$first": "$pic"},
    }},

    # 2. 基础字段重映射
    {"$project": {
        "_id": 1,
        "name": 1,
        "total_videos": 1, "total_duration": 1,
        "total_view": 1, "total_like": 1, "total_coin": 1,
        "total_favorite": 1, "total_danmaku": 1,
        "total_reply": 1, "total_share": 1,
        "latest_pubdate": 1, "earliest_pubdate": 1, "latest_bvid": 1,
        "total_stat_score": 1,
        "tid_list": 1, "ptid_list": 1,
        "all_tags": 1, "latest_pic": 1,
    }},
]
```

> **注意：** `$push "$tags"` 和 `$push "$tid"` 会为每个 owner 收集所有 tags 文本和 tid 列表。对于视频数极多的 UP 主（如 >10000 条），单个聚合文档可能较大。需要在 Python 后处理中做截断和采样（见 Stage 2）。
>
> **替代方案：** 如果内存压力过大，可以放弃在 MongoDB pipeline 中 `$push` 全部 tags，改为在 Python 层面分批 cursor 遍历该 owner 的视频，流式提取 tags 和 tid。全量聚合时使用 `allowDiskUse=True`。

#### Stage 2: Python 后处理计算

MongoDB pipeline 输出的原始聚合结果传入 Python 做进一步计算：

```python
def build_owner_doc(agg_result: dict, now_ts: int) -> dict:
    """将 MongoDB 聚合结果转化为 ES owner 文档"""
    mid = agg_result["_id"]
    total_videos = agg_result["total_videos"]
    total_view = agg_result["total_view"]
    total_like = agg_result["total_like"]
    total_coin = agg_result["total_coin"]
    total_favorite = agg_result["total_favorite"]
    latest_pubdate = agg_result["latest_pubdate"]
    earliest_pubdate = agg_result["earliest_pubdate"]
    total_stat_score = agg_result["total_stat_score"]

    # ---- 影响力指标 ----
    influence_score = compute_influence(
        total_view, total_videos, total_like, total_coin
    )

    # ---- 质量指标 ----
    avg_view = total_view / max(total_videos, 1)
    avg_favorite_rate = total_favorite / max(total_view, 1)
    avg_coin_rate = total_coin / max(total_view, 1)
    avg_like_rate = total_like / max(total_view, 1)
    avg_stat_score = total_stat_score / max(total_videos, 1)
    quality_score = compute_quality(
        avg_favorite_rate, avg_coin_rate, avg_like_rate, avg_stat_score, total_videos
    )

    # ---- 活跃度指标 ----
    days_span = max((latest_pubdate - earliest_pubdate) / 86400, 1)
    publish_freq = total_videos / days_span  # 视频/天
    days_since_last = max((now_ts - latest_pubdate) / 86400, 0)
    activity_score = compute_activity(
        days_since_last, publish_freq, total_videos, days_span
    )

    # ---- 主要分区 (众数) ----
    primary_tid = compute_mode(agg_result["tid_list"])
    primary_ptid = compute_mode(agg_result["ptid_list"])

    # ---- 领域标签提取 ----
    # all_tags 是字符串列表，每条形如 "tag1, tag2, tag3"
    all_tags_flat = flatten_tags(agg_result["all_tags"])  # → ["tag1", "tag2", ...]
    top_tags, tag_counts = extract_top_tags(all_tags_flat, top_k=30)
    top_tags_text = ", ".join(top_tags)

    # ---- 关联用户提取 ----
    # 需要单独扫描该 owner 的视频 desc/title/tags; 
    # 全量聚合时跳过此步 (在 Stage 2.5 单独处理)
    mentioned_names = []
    mentioned_mids = []

    return {
        "_id": mid,
        "mid": mid,
        "name": agg_result["name"],
        # 影响力
        "total_videos": total_videos,
        "total_duration": agg_result["total_duration"],
        "total_view": total_view,
        "total_like": total_like,
        "total_coin": total_coin,
        "total_favorite": total_favorite,
        "total_danmaku": agg_result["total_danmaku"],
        "total_reply": agg_result["total_reply"],
        "total_share": agg_result["total_share"],
        "influence_score": influence_score,
        # 质量
        "avg_view": avg_view,
        "avg_favorite_rate": avg_favorite_rate,
        "avg_coin_rate": avg_coin_rate,
        "avg_like_rate": avg_like_rate,
        "avg_stat_score": avg_stat_score,
        "quality_score": quality_score,
        # 活跃度
        "latest_pubdate": latest_pubdate,
        "earliest_pubdate": earliest_pubdate,
        "latest_bvid": agg_result["latest_bvid"],
        "publish_freq": publish_freq,
        "days_since_last": int(days_since_last),
        "activity_score": activity_score,
        # 领域
        "top_tags": top_tags_text,
        "primary_tid": primary_tid,
        "primary_ptid": primary_ptid,
        # 代表作封面 (用作 avatar 替代)
        "latest_pic": agg_result.get("latest_pic", ""),
        # 关联用户 (后续填充)
        "mentioned_names": " ".join(mentioned_names),
        "mentioned_mids": mentioned_mids,
        # 元信息
        "index_at": now_ts,
    }
```

**辅助函数：**

```python
def flatten_tags(all_tags: list[str], max_total: int = 5000) -> list[str]:
    """展开所有视频的 tags 字符串为去重标签列表。
    max_total 限制处理的标签总数，防止超大 UP 主导致内存溢出。"""
    tags = []
    for tag_str in all_tags:
        if not tag_str:
            continue
        for tag in tag_str.split(","):
            tag = tag.strip()
            if tag:
                tags.append(tag)
            if len(tags) >= max_total:
                return tags
    return tags

def extract_top_tags(tags: list[str], top_k: int = 30) -> tuple[list[str], dict]:
    """统计标签频率，返回 top-K 标签和完整计数"""
    from collections import Counter
    counter = Counter(tags)
    top = counter.most_common(top_k)
    return [t for t, _ in top], dict(top)

def compute_mode(values: list) -> int:
    """计算众数（出现频率最高的值）"""
    from collections import Counter
    if not values:
        return 0
    counter = Counter(v for v in values if v is not None)
    return counter.most_common(1)[0][0] if counter else 0
```

#### Stage 2.5: 关联用户提取（可与 Stage 2 解耦）

关联用户提取需要逐视频扫描文本内容，不适合在 MongoDB aggregation 中完成：

```python
def extract_mentions_for_owner(videos_cursor, known_owner_names: dict) -> dict:
    """扫描一个 owner 的所有视频，提取 @提及 和 tags 中出现的其他 UP 主名称。
    
    known_owner_names: dict[name, mid] — 预构建的已知 UP 主名称→mid 映射表
    """
    mentions = Counter()  # {target_mid: count}
    
    for video in videos_cursor:
        text = f"{video.get('desc', '')} {video.get('title', '')}"
        tags_text = video.get("tags", "")
        
        # 方式1: @提及
        at_mentions = re.findall(r'@([\w\-_]+)', text)
        for name in at_mentions:
            target_mid = known_owner_names.get(name)
            if target_mid:
                mentions[target_mid] += 1
        
        # 方式2: tags 中出现已知 UP 主名称
        for tag in tags_text.split(","):
            tag = tag.strip()
            target_mid = known_owner_names.get(tag)
            if target_mid and target_mid != video["owner"]["mid"]:
                mentions[target_mid] += 1
    
    return mentions
```

> **性能考虑：** 全量构建时，可以先完成 Stage 1+2 导入基本数据，关联用户作为后续批量任务单独运行。`known_owner_names` 字典从 Stage 1 的聚合结果构建（name → mid），消耗约 ~2-4GB 内存（6000 万条 name→mid 映射）。

#### Stage 3: 领域向量计算

Owner 级别的语义向量，用于 "搜索特定领域的 UP 主"：

```python
# 方案 A: 文本编码 (推荐)
# 拼接 name + top_tags 的文本直接编码
owner_text = f"{name} {top_tags_text}"
owner_embedding = embed_model.encode(owner_text)

# 方案 B: 加权聚合视频向量 (备选)
# 取该 owner 的 top-K 视频的 text_emb，以 stat_score 为权重加权平均
video_embeddings = fetch_top_k_video_embeddings(mid, top_k=50)
owner_embedding = weighted_average(video_embeddings, weights=stat_scores)
```

> **推荐方案 A** — 文本编码更简单、信号更纯净。方案 B 需要从 RocksDB 批量读取视频向量，增加 I/O 开销和复杂度。方案 A 可在 Stage 2 结束后直接批量计算。

### 4.2 ES 索引 Mapping 设计

```json
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "analysis": {
      "analyzer": {
        "chinese_analyzer": {
          "type": "custom",
          "tokenizer": "es_tok"
        }
      }
    }
  },
  "mappings": {
    "properties": {
      // === 身份信息 ===
      "mid":  { "type": "long" },
      "name": {
        "type": "text",
        "index": false,
        "fields": {
          "keyword": { "type": "keyword" },
          "words":   { "type": "text", "analyzer": "chinese_analyzer" }
        }
      },

      // === 影响力指标 (全部从 videos 聚合) ===
      "total_videos": { "type": "integer" },
      "total_view":   { "type": "long" },
      "total_like":   { "type": "long" },
      "total_coin":   { "type": "long" },
      "total_favorite": { "type": "long" },
      "total_danmaku":  { "type": "long" },
      "total_reply":    { "type": "long" },
      "total_share":    { "type": "long" },
      "total_duration": { "type": "long" },
      "influence_score": { "type": "half_float" },

      // === 创作质量指标 ===
      "avg_view":          { "type": "float" },
      "avg_favorite_rate": { "type": "half_float" },
      "avg_coin_rate":     { "type": "half_float" },
      "avg_like_rate":     { "type": "half_float" },
      "avg_stat_score":    { "type": "half_float" },
      "quality_score":     { "type": "half_float" },

      // === 活跃度指标 ===
      "latest_pubdate":   { "type": "long" },
      "earliest_pubdate": { "type": "long" },
      "latest_bvid":      { "type": "keyword" },
      "publish_freq":     { "type": "half_float" },
      "days_since_last":  { "type": "integer" },
      "activity_score":   { "type": "half_float" },

      // === 领域信息 ===
      "top_tags": {
        "type": "text",
        "index": false,
        "fields": {
          "words": { "type": "text", "analyzer": "chinese_analyzer" }
        }
      },
      "primary_tid":  { "type": "integer" },
      "primary_ptid": { "type": "integer" },

      // === 代表作封面 (替代不可用的 face 头像) ===
      "latest_pic": { "type": "keyword", "index": false },

      // === 领域向量 (Phase 3 补充) ===
      "owner_emb": {
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
      },

      // === 关联用户 ===
      "mentioned_mids": { "type": "long" },
      "mentioned_names": {
        "type": "text",
        "index": false,
        "fields": {
          "words": { "type": "text", "analyzer": "chinese_analyzer" }
        }
      },

      // === 元信息 ===
      "index_at":  { "type": "long" },
      "update_at": { "type": "long" }
    }
  }
}
```

**设计说明：**

| 字段 | 类型选择理由 |
|------|--------------|
| `name.keyword` | 精确匹配（用于过滤和聚合） |
| `name.words` | es_tok 中文分词（支持 "红警08" → "红警" + "08" token-set 匹配） |
| `top_tags.words` | 领域标签搜索（如 "黑神话悟空"） |
| `owner_emb` | 语义向量搜索（用于 "做类似内容的 UP 主"） |
| `influence_score` / `quality_score` / `activity_score` | 预计算评分，避免实时聚合 |
| `mentioned_names.words` | 关联用户搜索（如 "经常提到 xxx"） |
| `latest_pic` | 最新视频封面，作为 UP 主 "头像" 的替代展示 |
| `half_float` | 各种 score/rate 用 half_float 节约空间（6000 万 docs，空间敏感） |

**与 v1 方案对比 — 移除的字段：**

| 移除字段 | 原因 |
|----------|------|
| `face` (头像 URL) | 依赖 users_cards (Card API 已封禁)，改用 `latest_pic` 替代 |
| `sign` (个性签名) | 依赖 users_cards，无法获取 |
| `level` (用户等级) | 依赖 users_cards，无法获取 |
| `sex` (性别) | 依赖 users_cards，无法获取 |
| `official` (认证信息) | 依赖 users_cards，无法获取 |
| `follower` (粉丝数) | 依赖 users_cards，无法获取 |
| `top_keywords` (标题关键词) | 简化设计：`top_tags` 已足够覆盖领域信息，减少聚合复杂度 |
| `tag_distribution` | 简化设计：用 `top_tags` 替代完整分布 |
| `collab_mids/names` | 简化设计：统一使用 `mentioned_mids/names`；合作关系与 @提及本质相同 |

**索引规模估算：**

| 项目 | 值 |
|------|-----|
| 文档数量 | ~6000 万 |
| 单文档大小（不含向量）| ~600 Bytes |
| 向量字段 (2048 bit) | 256 Bytes |
| 索引总大小估算 | ~50-60 GB（含 3 shard × 1 replica） |
| 对比：videos 索引 | ~10 亿 docs, 数百 GB |

### 4.3 数据更新策略

#### 全量构建（初次）

```
1. MongoDB aggregation pipeline: $sort pubdate desc → $group by owner.mid
   - 使用 allowDiskUse=True
   - cursor 流式遍历聚合结果
2. Python 后处理: compute scores, extract tags, build owner_doc
3. ES bulk upsert: batch_size=5000
4. (可选) 向量计算: 批量编码 owner_emb → 单独 bulk update
```

**全量预估耗时：**

| 阶段 | 预估耗时 | 瓶颈 |
|------|----------|------|
| MongoDB 聚合 ($sort + $group on ~10 亿 docs) | 6-12 小时 | I/O 和 allowDiskUse 临时空间 |
| Python 后处理 + ES bulk upsert (6000 万 owners) | 2-4 小时 | 批量写入 ES 吞吐 |
| 向量计算 6000 万 × ~5ms/doc | ~83 小时 → 需并行 | GPU/CPU 计算 |

**优化策略：**
- Phase 1 先不计算 embedding（向量字段留空），先构建基础索引验证搜索效果
- MongoDB aggregation 可替换为 **Python cursor 流式聚合**：`videos.find().sort("owner.mid", 1)` → 在 Python 中按 mid 分组累加，避免 MongoDB $push 导致的内存压力。此方案更灵活，且可以在迭代中流式写入 ES，不需要等全部聚合完成

> **推荐全量构建方案：Python cursor 流式聚合**
> ```python
> cursor = videos.find({}, projection).sort("owner.mid", 1).batch_size(10000)
> current_mid = None
> accumulator = None
> 
> for doc in cursor:
>     mid = doc["owner"]["mid"]
>     if mid != current_mid:
>         if accumulator:
>             owner_doc = build_owner_doc(accumulator)
>             bulk_buffer.append(owner_doc)
>             if len(bulk_buffer) >= 5000:
>                 es_bulk_upsert(bulk_buffer)
>                 bulk_buffer.clear()
>         current_mid = mid
>         accumulator = new_accumulator(doc)
>     else:
>         accumulator.add(doc)
> ```
> 这种方案的优势：
> - 内存可控 — 每次只保持一个 owner 的累加器在内存中
> - 流式写入 — 不需要等全部聚合完成
> - 灵活性高 — Python 中可以做任意复杂的计算
> - 无 16MB 文档限制 — 避免 MongoDB $push 对超大 UP 主的限制

#### 增量更新

复用现有的 `actions/elastic_videos_indexer.py` 的定时触发模式：

```
每 30-60 分钟:
1. 从 bv_flags 找到最近更新的 video bvids
2. db.videos.distinct("owner.mid", {"bvid": {"$in": updated_bvids}})
   → 提取受影响的 owner.mid 列表
3. 对每个受影响的 mid:
   db.videos.aggregate([
       {"$match": {"owner.mid": mid}},
       {"$group": {...同全量管线...}}
   ])
4. build_owner_doc() → ES upsert
```

**增量优势：** 每次仅更新 ~数千到数万个 owner（取决于新增视频量），远小于全量 6000 万。

**替代方案 — 基于 ES videos 索引的增量聚合：**

对于已在 ES videos 索引中的大部分数据，也可以直接用 ES 的 terms aggregation 按 owner.mid 聚合，避免回查 MongoDB：

```python
# 用 ES 聚合一批 mid 的统计数据 (适合小批量增量)
{
    "size": 0,
    "query": {"terms": {"owner.mid": [mid1, mid2, ...]}},
    "aggs": {
        "by_owner": {
            "terms": {"field": "owner.mid", "size": len(mids)},
            "aggs": {
                "total_view": {"sum": {"field": "stat.view"}},
                "total_like": {"sum": {"field": "stat.like"}},
                "latest_pubdate": {"max": {"field": "pubdate"}},
                ...
            }
        }
    }
}
```

这种方案对小批量增量更新更高效（无需回查 MongoDB），但对 tags 提取不如 MongoDB cursor 灵活。

#### 低频更新字段

| 字段 | 更新策略 |
|------|----------|
| `owner_emb` | 每天或每周重计算（领域变化缓慢） |
| `mentioned_mids/names` | 每周批量重算（需全量扫描 desc/tags） |
| `days_since_last` | 每次增量更新时更新（基于 latest_pubdate 和当前时间） |
| `activity_score` | 每次增量更新时重算（因为 days_since_last 变化） |

---

## 5. 算法层：Owner 搜索算法设计

### 5.1 名称匹配（Name Match）

在独立索引上，名称匹配比从 videos 间接匹配更精准、更高效：

#### 5.1.1 多层匹配策略

| 层级 | 字段 | 分析器 | 匹配方式 | 示例 |
|------|------|--------|----------|------|
| L1 | `name.keyword` | keyword | 精确匹配 | "影视飓风" → 完全一致 |
| L2 | `name.words` | es_tok chinese | Token BM25 | "红警08" → token-set {红警, 08} 匹配 "红警HBK08" |
| L3 | `top_tags.words` | es_tok chinese | 领域标签 | "黑神话悟空" → tags 含该关键词的 UP 主 |
| L4 | `mentioned_names.words` | es_tok chinese | 关联用户 | "影视飓风" → 经常提及该 UP 主的其他 UP 主 |

> **注意：** v1.0 中的 `name.pinyin` (拼音匹配)、`sign.words` (签名搜索)、`official.title.words` (认证搜索) 已移除 — 前者需要 ES 拼音分析器（es_tok 暂不支持，可后续扩展），后两者依赖不可用的 users_cards 数据。

#### 5.1.2 查询构建

```python
def build_owner_name_query(query: str, boost_config: dict = None) -> dict:
    """构建 owner 名称多字段查询"""
    default_boosts = {
        "name.keyword": 50.0,   # 精确匹配最高优先级
        "name.words":   10.0,   # 分词匹配
        "top_tags.words": 3.0,  # 领域标签辅助
        "mentioned_names.words": 1.5,  # 关联用户辅助
    }
    boosts = boost_config or default_boosts

    return {
        "bool": {
            "should": [
                {"term": {"name.keyword": {"value": query, "boost": boosts["name.keyword"]}}},
                {"match": {"name.words": {"query": query, "boost": boosts["name.words"]}}},
                {"match": {"top_tags.words": {"query": query, "boost": boosts["top_tags.words"]}}},
                {"match": {"mentioned_names.words": {"query": query, "boost": boosts["mentioned_names.words"]}}},
            ],
            "minimum_should_match": 1,
        }
    }
```

#### 5.1.3 前缀补全

利用 ES 的 `prefix` 查询或 bili-search-algo 中的 `PrefixMatcher`（marisa-trie）提供输入联想：

```python
# ES prefix query (在索引上直接执行)
{"prefix": {"name.keyword": {"value": "影视", "boost": 20.0}}}

# 或结合 completion suggester (需要额外的 completion 字段)
```

### 5.2 影响力排序（Influence Scoring）

#### 评分公式设计

```python
def compute_influence(total_view, total_videos, total_like, total_coin):
    """
    影响力 = 曝光量 × 创作规模 × 社区认可
    使用对数缩放避免极端值支配
    
    注意：由于 Card API 已封禁，无法获取粉丝数（follower）。
    影响力完全基于视频统计数据。总播放量 + 总互动量是比粉丝数
    更真实的 "内容影响力" 信号。
    """
    # 曝光维度: log10(total_view + 1) / log10(max_view + 1) → [0, 1]
    view_score = log_normalize(total_view, max_val=1e10)

    # 规模维度: log10(total_videos + 1) / log10(max_videos + 1) → [0, 1]
    scale_score = log_normalize(total_videos, max_val=10000)

    # 点赞维度: log10(total_like + 1) / log10(max_like + 1) → [0, 1]
    like_score = log_normalize(total_like, max_val=1e8)

    # 投币维度: 投币是B站特有的深度认可信号
    coin_score = log_normalize(total_coin, max_val=1e7)

    # 加权融合
    influence = (
        0.40 * view_score +
        0.20 * scale_score +
        0.25 * like_score +
        0.15 * coin_score
    )
    return round(influence, 4)
```

> 这里使用 `log_normalize` 而非线性归一化，因为互联网数据具有极端的幂律分布（少数头部 UP 主占据绝大部分流量），对数缩放可以让中腰部 UP 主也有合理的分数。

### 5.3 创作质量评估（Quality Scoring）

```python
def compute_quality(avg_favorite_rate, avg_coin_rate, avg_like_rate, avg_stat_score, total_videos):
    """
    质量 = 互动深度 × 内容价值
    收藏和投币是比播放和点赞更深度的互动信号
    """
    # 收藏率 [0, 1]: 典型范围 0.001 ~ 0.05
    fav_score = bounded_normalize(avg_favorite_rate, low=0.001, high=0.03)

    # 投币率 [0, 1]: 典型范围 0.0005 ~ 0.02
    coin_score = bounded_normalize(avg_coin_rate, low=0.0005, high=0.015)

    # 点赞率 [0, 1]: 典型范围 0.01 ~ 0.10
    like_score = bounded_normalize(avg_like_rate, low=0.01, high=0.08)

    # 综合质量分 (已由 blux.doc_score 计算)
    stat_quality = min(avg_stat_score / 100, 1.0)

    # 样本量可信度 — 视频数越多，质量评估越可靠
    confidence = min(total_videos / 20, 1.0)  # 20 个视频以上才完全可信

    # 加权融合
    raw_quality = (
        0.35 * fav_score +
        0.25 * coin_score +
        0.20 * like_score +
        0.20 * stat_quality
    )

    # 应用可信度衰减 — 低视频数的 UP 主质量分打折
    quality = raw_quality * (0.3 + 0.7 * confidence)
    return round(quality, 4)
```

### 5.4 活跃度评估（Activity Scoring）

```python
def compute_activity(days_since_last, publish_freq, total_videos, days_span):
    """
    活跃度 = 近期活跃 × 持续产出
    """
    # 最近发布衰减: 0天=1.0, 30天=0.7, 90天=0.3, 365天=0.05
    recency = math.exp(-days_since_last / 60)  # 指数衰减, τ=60天

    # 发布频率 [0, 1]: 每天1条=1.0, 每周1条=0.7, 每月1条=0.3
    freq_score = bounded_normalize(publish_freq, low=1/90, high=1)

    # 持续性: 至少活跃 180 天以上的 UP 主更有价值
    persistence = min(days_span / 180, 1.0)

    # 作品量门槛: 至少 5 个视频才有意义
    volume_gate = min(total_videos / 5, 1.0)

    activity = (
        0.45 * recency +
        0.25 * freq_score +
        0.15 * persistence +
        0.15 * volume_gate
    )
    return round(activity, 4)
```

### 5.5 关联领域匹配（Domain Relevance）

这是最核心的能力 — 让用户能通过 **内容关键词** 找到对应领域的 UP 主。

#### 5.5.1 文本匹配方式

```python
# 用户搜索 "黑神话悟空" → 搜索 top_tags
{
    "bool": {
        "should": [
            {"match": {"top_tags.words": {"query": "黑神话悟空", "boost": 3.0}}},
            {"match": {"name.words": {"query": "黑神话悟空", "boost": 1.5}}},
        ]
    }
}
```

#### 5.5.2 向量语义匹配

当文本匹配不足时（如 "讲解古代兵器的 UP 主"），使用 KNN 向量搜索：

```python
# 将查询编码为向量
query_emb = embed_model.encode("讲解古代兵器的UP主")

# KNN 搜索 owner_emb 字段
{
    "knn": {
        "field": "owner_emb",
        "query_vector": query_emb,
        "k": 50,
        "num_candidates": 200
    }
}
```

#### 5.5.3 混合检索

文本 BM25 + 向量 KNN 双路召回后 RRF 融合（与 videos_explore 的混合搜索设计一致）：

```python
# 两路并行
text_hits = es.search(body=text_query, size=100)
vector_hits = es.search(body=knn_query, size=100)

# RRF 融合
fused = reciprocal_rank_fusion(text_hits, vector_hits, k=60)
```

### 5.6 关联用户（User Graph）

#### 5.6.1 关联关系提取

从视频数据中提取 UP 主之间的关联关系：

```python
def extract_user_relations(videos_cursor):
    """从视频的 title/desc/tags 中提取 @提及和合作信号"""
    relations = defaultdict(Counter)  # {mid_a: Counter({mid_b: count})}

    for video in videos_cursor:
        mid = video["owner"]["mid"]
        text = f"{video.get('title', '')} {video.get('desc', '')} {video.get('tags', '')}"

        # 方式1: 提取 @ 提及 (B站视频描述中常见 @username)
        mentions = re.findall(r'@([\w\-_]+)', text)
        for mention in mentions:
            # 查找 mention 对应的 mid
            target_mid = resolve_username_to_mid(mention)
            if target_mid:
                relations[mid][target_mid] += 1

        # 方式2: tags 中出现其他已知 UP 主名称
        for tag in video.get("tags", "").split(","):
            tag = tag.strip()
            target_mid = known_owner_names.get(tag)
            if target_mid and target_mid != mid:
                relations[mid][target_mid] += 1

    return relations
```

#### 5.6.2 存储格式

```python
# 每个 owner 文档中存储 top-K 关联用户 (合并合作和提及为统一的 mentioned)
owner_doc["mentioned_mids"]  = [mid for mid, _ in top_mentions[:20]]
owner_doc["mentioned_names"] = " ".join([name for _, name in top_mentions[:20]])
```

#### 5.6.3 查询方式

```python
# "和影视飓风合作过的 UP 主" / "经常提到影视飓风的 UP 主"
target_mid = search_owner_by_name("影视飓风")  # → 946974
{
    "bool": {
        "should": [
            {"term": {"mentioned_mids": target_mid}},
            {"match": {"mentioned_names.words": "影视飓风"}},
        ],
        "minimum_should_match": 1,
    }
}
```

### 5.7 综合排序公式

当用户搜索 UP 主时，最终排序由多个维度加权融合：

```python
def compute_owner_rank_score(
    name_match_score: float,   # BM25 名称匹配 [0, ~50]
    domain_score: float,       # 领域相关性 [0, 1] (text match + vector)
    influence_score: float,    # 影响力 [0, 1]
    quality_score: float,      # 创作质量 [0, 1]
    activity_score: float,     # 活跃度 [0, 1]
    query_type: str = "name",  # "name" / "domain" / "mixed"
) -> float:
    """
    根据查询类型动态调整权重
    """
    WEIGHT_PROFILES = {
        "name": {
            # 名称查询 — 名称匹配最重要，影响力用于消歧
            "name_match": 0.50, "domain": 0.05, "influence": 0.25,
            "quality": 0.10, "activity": 0.10,
        },
        "domain": {
            # 领域查询 — 领域相关性最重要，影响力和质量辅助排序
            "name_match": 0.05, "domain": 0.35, "influence": 0.25,
            "quality": 0.20, "activity": 0.15,
        },
        "mixed": {
            # 混合查询 — 均衡
            "name_match": 0.25, "domain": 0.20, "influence": 0.25,
            "quality": 0.15, "activity": 0.15,
        },
    }

    w = WEIGHT_PROFILES[query_type]

    # 归一化 name_match_score (BM25 分数范围不确定)
    name_norm = min(name_match_score / 30.0, 1.0)

    final_score = (
        w["name_match"] * name_norm +
        w["domain"]     * domain_score +
        w["influence"]  * influence_score +
        w["quality"]    * quality_score +
        w["activity"]   * activity_score
    )
    return final_score
```

**查询类型自动判定：**

```python
def detect_owner_query_type(query: str, name_hits: list, domain_hits: list) -> str:
    """
    根据名称命中和领域命中的情况判断查询类型:
    - name:   查询精确匹配某个 UP 主名称 (如 "影视飓风")
    - domain: 查询描述一个领域/话题 (如 "黑神话悟空 UP 主")
    - mixed:  两者都有 (如 "红警" — 既可能是 UP 主名也可能是话题)
    """
    has_exact_name = any(hit["name.keyword"] == query for hit in name_hits)
    has_strong_name = len(name_hits) > 0 and name_hits[0]["_score"] > 20.0
    has_domain_hits = len(domain_hits) > 3

    if has_exact_name or has_strong_name:
        return "name" if not has_domain_hits else "mixed"
    elif has_domain_hits:
        return "domain"
    else:
        return "mixed"
```

---

## 6. 系统设计层：Owner 搜索服务架构

### 6.1 OwnerSearcher 服务设计

在 `bili-search` 中新建 `elastics/owners/` 模块，与 `elastics/videos/` 平行：

```
elastics/
├── owners/
│   ├── __init__.py
│   ├── constants.py      # Owner 搜索常量 (字段、boost、limit)
│   ├── searcher.py        # OwnerSearcher — 核心搜索逻辑
│   ├── scorer.py          # Owner 排序评分器
│   └── hits.py            # Owner 搜索结果解析
├── videos/
│   ├── searcher_v2.py
│   ├── explorer.py
│   └── ...
└── structure.py
```

**OwnerSearcher 核心接口：**

```python
class OwnerSearcher:
    """Owner 搜索服务 — 独立于 VideoSearcher"""

    def __init__(self, es_client, index_name="bili_owners_v1"):
        self.es = es_client
        self.index = index_name

    def search(self, query: str, sort_by: str = "relevance",
               filters: dict = None, limit: int = 20) -> dict:
        """主搜索入口 — 返回 owner 列表"""
        ...

    def search_by_name(self, name: str, limit: int = 10) -> dict:
        """名称精确/模糊搜索"""
        ...

    def search_by_domain(self, query: str, mode: str = "hybrid",
                         limit: int = 20) -> dict:
        """领域搜索 — 文本 + 向量混合检索"""
        ...

    def search_by_relation(self, mid: int, limit: int = 10) -> dict:
        """关联用户搜索 — 查找合作/提及关系"""
        ...

    def get_owner(self, mid: int) -> dict:
        """根据 mid 获取单个 owner 详情"""
        ...

    def suggest(self, prefix: str, limit: int = 10) -> dict:
        """输入联想/自动补全 — 用于搜索框"""
        ...

    def top_owners(self, sort_by: str = "influence",
                   tid: int = None, limit: int = 50) -> dict:
        """排行榜 — 按指定维度排序的 TOP owner"""
        ...
```

### 6.2 与 VideoExplorer 联动

Owner 搜索不仅是一个独立功能，还应增强现有的 videos 搜索体验：

#### 6.2.1 替换 check_author

当前 `check_author` 从 videos 间接推测 author ← 改为直接在 owners 索引精确查询：

```python
# BEFORE: 间接搜索 (从 videos suggest 结果推测)
def _check_author_old(self, args):
    suggest_result = self.search_client.suggest(query=name)
    return analyze_suggest_for_authors(suggest_result, query=name)

# AFTER: 直接搜索 owners 索引
def _check_author_new(self, args):
    name = args.get("name", "")
    owner_result = self.owner_searcher.search_by_name(name, limit=5)
    return {
        "found": len(owner_result["hits"]) > 0,
        "owners": [
            {
                "mid": hit["mid"],
                "name": hit["name"],
                "total_videos": hit["total_videos"],
                "total_view": hit["total_view"],
                "influence_score": hit["influence_score"],
                "top_tags": hit.get("top_tags", ""),
                "latest_pic": hit.get("latest_pic", ""),
                "score": hit["_score"],
            }
            for hit in owner_result["hits"]
        ],
    }
```

#### 6.2.2 增强 Owner Intent Detection

在 `RecallManager._detect_owner_intent()` 中，利用 owners 索引做二次确認：

```python
# BEFORE: 纯粹基于 recall pool 中 owner.name 的频率分析
matched_owners = analyze_from_pool(pool)

# AFTER: 先在 owners 索引中精确搜索，得到明确结果
owner_hits = self.owner_searcher.search_by_name(query, limit=3)
if owner_hits and owner_hits["hits"][0]["_score"] > 20.0:
    # 强匹配 — 确认是 owner 查询
    owner_info = owner_hits["hits"][0]
    return OwnerIntentResult(
        detected=True,
        confidence=0.95,
        owners=[owner_info],
        source="owners_index",  # 标记来源
    )
# 否则退回到现有的 pool 分析逻辑
```

#### 6.2.3 增强 AuthorGrouper

用 owners 索引的数据丰富 author groups：

```python
# BEFORE: AuthorGrouper 只聚合当前搜索结果的少量字段
group = {
    "mid": mid, "name": name,
    "sum_view": sum_view, "sum_count": sum_count,
}

# AFTER: 从 owners 索引补充完整画像
owner_doc = self.owner_searcher.get_owner(mid)
group = {
    **group,
    "total_videos": owner_doc.get("total_videos", 0),
    "total_view": owner_doc.get("total_view", 0),
    "influence_score": owner_doc.get("influence_score", 0),
    "quality_score": owner_doc.get("quality_score", 0),
    "activity_score": owner_doc.get("activity_score", 0),
    "top_tags": owner_doc.get("top_tags", ""),
    "latest_pic": owner_doc.get("latest_pic", ""),  # 代替不可用的 face
}
```

### 6.3 LLM 工具集成

#### 6.3.1 新增 search_owners 工具

```python
SEARCH_OWNERS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_owners",
        "description": (
            "搜索B站UP主/用户。可以按名称、领域、影响力等维度搜索。"
            "返回UP主列表，含作品数、播放量、领域标签等信息。"
            "用于查找特定UP主或发现某个领域的创作者。"
            "示例: 按名称搜索 '影视飓风'，按领域搜索 '黑神话悟空 游戏区'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索UP主的查询词（名称、领域关键词等）",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "influence", "quality", "activity", "total_view"],
                    "description": "排序方式：relevance(相关性) influence(影响力) quality(创作质量) activity(活跃度) total_view(总播放量)",
                },
            },
            "required": ["query"],
        },
    },
}
```

#### 6.3.2 升级 check_author 工具

改为直接使用 owners 索引，响应更快、信息更丰富：

```python
CHECK_AUTHOR_TOOL_V2 = {
    "type": "function",
    "function": {
        "name": "check_author",
        "description": (
            "检查输入是否匹配B站UP主。返回匹配的UP主列表（含UID、作品数、播放量、领域标签等）。"
            "用于判断用户意图：是搜索关键词还是查找特定UP主。"
            "建议与 search_videos 在同一轮并行调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要检查的名称或关键词",
                },
            },
            "required": ["name"],
        },
    },
}
```

#### 6.3.3 工具组合策略

当 LLM 收到 "推荐几个做黑神话悟空内容的 UP 主" 这样的请求时：

```
Round 1: search_owners(query="黑神话悟空", sort_by="influence")
    → 返回 top-10 做黑神话内容的 UP 主 (每个含 name, mid, total_view, total_videos, top_tags)

Round 2 (可选): search_videos(queries=["黑神话悟空 :user=UP主A q=vwr", ":user=UP主B :date<=30d"])
    → 返回各 UP 主的代表作
```

### 6.4 前端 UI 设计

#### 6.4.1 搜索结果页增强

现有的 `ResultAuthorsList` 只显示头像 + 名称。改进后：

```
┌─────────────────────────────────────────────────────┐
│  相关作者                                  [展开全部] │
├─────────────────────────────────────────────────────┤
│  ┌──────┐  影视飓风                                    │
│  │ pic  │  598个视频 · 25.3亿播放 · 影响力 0.92        │
│  │      │  每周更新 · 科技/数码/影视                  │
│  └──────┘  [查看全部视频]  [查看主页]                 │
│                                                      │
│  ┌──────┐  飓多多StormCrew                           │
│  │ pic  │  312个视频 · 4.2亿播放 · 影响力 0.78         │
│  │      │  每周更新 · 科技/Vlog                       │
│  └──────┘  [查看全部视频]  [查看主页]                 │
└─────────────────────────────────────────────────────┘
```

> **设计说明：** `pic` 处显示 UP 主最新视频的封面图 (`latest_pic`)，作为头像的替代。由于 Card API 已封禁，无法获取用户头像 (face)。视频封面同样具有较好的辨识度。显示的指标全部来自视频统计聚合，不依赖任何外部数据源。

#### 6.4.2 独立 Owner 搜索页（可选 - 后续优化）

```
搜索框: [黑神话悟空 UP 主]  排序: [影响力 ▾]

┌────────────────────────────────────────────┐
│  1. 黑猴的名义 · 246个视频 · 8200万播放      │
│     黑神话悟空, 游戏, 攻略                   │
│     最近更新: 3天前                          │
│     代表作: [视频卡片] [视频卡片]            │
├────────────────────────────────────────────┤
│  2. GameHubs · 1.2k个视频 · 18.5亿播放        │
│     游戏解说, 黑神话悟空, 原神               │
│     最近更新: 1天前                          │
│     代表作: [视频卡片] [视频卡片]            │
└────────────────────────────────────────────┘
```

#### 6.4.3 Smart Suggest 增强

现有的 `smartSuggestService.ts` 已支持 `author` 类型的 suggestion。改进：

- 从 owners 索引获取 suggest 数据（而非仅从搜索结果中提取）
- 显示播放量/视频数/领域标签作为 suggestion 的副信息
- 支持拼音输入直接联想 UP 主名称

---

## 7. 实施路线图

### Phase 1: 数据基础（预计 1-2 周）

| 任务 | 优先级 | 依赖 | 说明 |
|------|--------|------|------|
| 1.1 实现 videos → owner 聚合管线 | P0 | - | Python cursor 流式聚合 + OwnerDocBuilder（替代已废弃的 user_stat aggregator） |
| 1.2 设计并创建 ES owners 索引 | P0 | - | 实现 mapping, settings, 测试基本写入 |
| 1.3 实现全量 owner 导入管线 | P0 | 1.1, 1.2 | MongoDB videos cursor → OwnerDocBuilder → ES bulk upsert |
| 1.4 实现增量更新调度器 | P1 | 1.3 | 每 30-60 分钟增量更新 |

### Phase 2: 核心搜索（预计 2-3 周）

| 任务 | 优先级 | 依赖 | 说明 |
|------|--------|------|------|
| 2.1 实现 OwnerSearcher 基础搜索 | P0 | Phase 1 | name match + influence sort |
| 2.2 实现 influence/quality/activity 评分 | P0 | Phase 1 | score 计算 + 预写入 ES |
| 2.3 实现领域搜索 (text match) | P0 | 2.1 | top_tags.words 匹配 |
| 2.4 替换 check_author 为 owners 索引查询 | P1 | 2.1 | LLM 工具层改造 |
| 2.5 增强 AuthorGrouper | P1 | 2.1 | 从 owners 索引补充 author group 信息 |

### Phase 3: 高级功能（预计 2-3 周）

| 任务 | 优先级 | 依赖 | 说明 |
|------|--------|------|------|
| 3.1 实现 owner_emb 向量计算和索引 | P1 | Phase 2 | 领域语义搜索 |
| 3.2 实现混合检索 (text + vector + RRF) | P1 | 3.1 | 完整的领域搜索能力 |
| 3.3 实现关联用户提取和查询 | P2 | Phase 2 | @提及分析 |
| 3.4 新增 search_owners LLM 工具 | P1 | Phase 2 | LLM 层新工具 |
| 3.5 增强 owner intent detection | P2 | 2.1 | 利用 owners 索引做二次确认 |

### Phase 4: 前端 & 优化（预计 1-2 周）

| 任务 | 优先级 | 依赖 | 说明 |
|------|--------|------|------|
| 4.1 增强 ResultAuthorsList 组件 | P1 | Phase 2 | 显示视频数、播放量、领域标签、更新频率 |
| 4.2 增强 Smart Suggest 的 owner 联想 | P2 | Phase 2 | 拼音/前缀联想 + 副信息 |
| 4.3 性能优化: 热门 owner 缓存 | P2 | Phase 2 | Redis 缓存 top-1000 owner |
| 4.4 独立 Owner 搜索页（可选）| P3 | Phase 3 | 完整的 UP 主搜索页面 |

### 里程碑

| 里程碑 | 完成标志 | 预计时间 |
|--------|----------|----------|
| M1: Owner 索引可查 | 6000 万 owner 文档已导入 ES，基本名称搜索可用 | Phase 1 完成 |
| M2: 核心搜索上线 | OwnerSearcher 在生产环境提供搜索服务 | Phase 2 完成 |
| M3: 语义搜索可用 | 向量检索 + 混合搜索 + 关联用户 | Phase 3 完成 |
| M4: 完整体验 | 前端增强 + LLM 工具 + 性能优化 | Phase 4 完成 |

---

## 8. 风险与约束

### 技术风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 全量聚合耗时过长（10 亿 docs） | 高 | 阻塞 Phase 1 | Python cursor 流式聚合 + allowDiskUse + 分批写入 ES |
| owners 索引空间超预期 | 中 | 需要更多磁盘 | 先不索引 embedding (Phase 1), 使用 half_float 节约空间 |
| 唯一数据源局限 | 中 | 缺少粉丝数/头像/签名等用户画像信息 | 用 total_view/latest_pic 替代；视频统计数据事实上更客观 |
| owner_emb 质量不佳 | 中 | 领域搜索效果差 | 方案 A (文本编码) 简单直接，方案 B (聚合向量) 作为备选 |
| 增量更新遗漏 | 低 | 数据不一致 | 定期全量重刷 + md5 变化检测 |
| MongoDB $group 内存溢出 | 中 | 超大 UP 主的 tags/tid_list 过大 | 改用 Python 流式聚合方案（推荐），避免 16MB 文档限制 |

### 约束条件

| 约束 | 说明 |
|------|------|
| **唯一数据源** | 所有 owner 数据仅来自 MongoDB `videos` 集合。`users_cards` (Card API 已封禁) 和 `users_stats` (设计不合理) 均不可用 |
| ES 集群资源 | 在现有集群上额外增加 ~50-60GB 的 owners 索引 |
| MongoDB 资源 | 全量聚合管线需要读取 10 亿 docs，需在低峰期执行 |
| 依赖 es_tok 插件 | owners 索引需要 chinese_analyzer — 复用现有 es_tok 插件 |
| 向量计算资源 | 6000 万 owner embedding 的计算需要 GPU 或大量 CPU 时间 |
| 无粉丝数信号 | 影响力排序完全基于视频统计（total_view, total_like, total_coin），无法直接排序粉丝数 |

### 不做的事情（Out of Scope）

| 项目 | 原因 |
|------|------|
| 获取用户画像数据 (face/sign/level/official/follower) | Card API 已封禁，结构性不可获取 |
| UP 主视频时间线浏览 | 属于 "用户主页" 功能，不在搜索范围内 |
| UP 主推荐算法（协同过滤） | 需要用户行为数据，目前不具备 |
| B 站官方 UP 主搜索 API 接入 | 不依赖官方搜索，自建能力 |
| users_stats 集合的维护或扩展 | 该集合设计不合理，将被新的 owner 索引完全替代 |
