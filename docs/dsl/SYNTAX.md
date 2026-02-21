# DSL Syntax Reference

## Basic Structure

A DSL query is a single-line expression combining keywords, filters, and boolean operators.

```
<query> = <expr> | <expr> <bool_op> <expr> | (<query>)
<expr>  = <word_expr> | <filter_expr> | <constraint_expr>
```

## Keywords (word_expr)

Plain text or quoted strings for full-text search.

| Syntax | Description | Example |
|--------|-------------|---------|
| `text` | Search for text | `影视飓风` |
| `"text"` | Exact phrase search | `"deep learning"` |
| `k=text` | Explicit keyword | `k=你好` |
| `k!=[a,b]` | Must NOT match keywords | `k!=[你好,世界]` |
| `text?` | Optional/fuzzy match | `hello?` |
| `text~N` | Fuzzy edit distance | `hello~2` |

## Token Constraints (constraint_expr)

Token constraints use `+` and `-` prefixes to require or exclude specific tokens via `es_tok_constraints`.

| Syntax | Description | Example |
|--------|-------------|---------|
| `+token` | Must have token | `+影视飓风` |
| `-token` | Must NOT have token | `-广告` |
| `!token` | Same as `-token` | `!广告` |
| `+"quoted"` | Must have quoted token | `+"影视飓风"` |

### Constraint Boolean Combinations

Multiple constraints default to AND. Use `&`, `|`, `()` for complex logic.

```
+A +B        → AND(A, B)
+A -B        → AND(A, NOT B)
(+A & +B)    → AND(A, B)
(+A | +B)    → OR(A, B)
(+A & +B) | (+C & -D) → OR(AND(A,B), AND(C, NOT D))
```

### Mixed Queries

Constraints can be combined with keyword search and filters:

```
世界 +影视飓风 -广告        → search "世界", must have "影视飓风", exclude "广告"
影视飓风 +小米 v>10k        → search "影视飓风", must have "小米", views > 10k
```

## Date Filters (date_expr)

| Syntax | Description | Example |
|--------|-------------|---------|
| `d=1d` | Past 1 day | `date=1d` |
| `d=2024` | Year 2024 | `d=2024` |
| `d=2024-01-01` | Specific date | `dt=2024-01/01` |
| `d>=2024-01` | Since Jan 2024 | `d>=2024-01` |
| `d=[2024,2025-01]` | Date range | `dt==[2024,2025-01]` |
| `d=this_week` | Current week | `d=tw` |
| `d=last_month` | Last month | `d=lm` |
| `d=past_week` | Past week | `d=pw` |

**Key aliases**: `date`, `dt`, `d`, `rq`

## User Filters (user_expr)

| Syntax | Description | Example |
|--------|-------------|---------|
| `u=name` | Match user | `u=影视飓风` |
| `u!=[a,b]` | Exclude users | `u!=[影视飓风,何同学]` |
| `@name` | Match user (@ syntax) | `@影视飓风` |
| `@!name` | Exclude user (@ syntax) | `@!影视飓风` |

**Key aliases**: `user`, `up`, `u`

## User ID Filters (uid_expr)

| Syntax | Description | Example |
|--------|-------------|---------|
| `uid=123` | Match user ID | `mid=1234` |
| `uid!=[1,2]` | Exclude user IDs | `uid!=[123,456]` |

**Key aliases**: `uid`, `mid`, `ud`

## Statistics Filters (stat_expr)

| Syntax | Description | Example |
|--------|-------------|---------|
| `v>=10k` | Views ≥ 10,000 | `:view>=10k` |
| `v=[1w,10w)` | Views in range | `:vw=[1w,10w)` |
| `lk>100` | Likes > 100 | `:like>100` |
| `cn>=25` | Coins ≥ 25 | `:coin>=25` |

**Stat keys**: `view`/`v`, `like`/`lk`, `coin`/`cn`, `favorite`/`fv`, `reply`/`rp`, `danmaku`/`dm`, `share`/`sh`

**Units**: `k`/`K` = 1,000; `w`/`W` = 10,000; `m`/`M` = 1,000,000

## Duration Filters (dura_expr)

| Syntax | Description | Example |
|--------|-------------|---------|
| `t>30` | Duration > 30s | `dura>30` |
| `t<=1h` | Duration ≤ 1 hour | `t<=1h` |
| `t=[30,1m30s]` | Duration range | `t=[30,1m30s]` |

**Key aliases**: `duration`, `dura`, `dr`, `time`, `t`

**Units**: `d` (days), `h` (hours), `m` (minutes), `s` (seconds)

## Video ID Filters (bvid_expr)

| Syntax | Description | Example |
|--------|-------------|---------|
| `bv=BV1xx` | Match BV ID | `bvid=BV1xxxx` |
| `av=12345` | Match AV ID | `avid=12345` |
| `bv=[BV1,BV2]` | Match multiple | `bv=[BV1xx,BV2yy]` |

## Region Filters (region_expr)

| Syntax | Description | Example |
|--------|-------------|---------|
| `rg=动画` | Match region | `region=动画` |
| `rg=(影视,动画)` | Match regions | `region=(影视,动画,音乐)` |
| `rid!=[1,24]` | Exclude region IDs | `rid!=[1,24,153]` |

## Query Mode (qmod_expr)

Controls search method. Default is hybrid (`wv`).

| Syntax | Mode | Description |
|--------|------|-------------|
| `q=w` | word | Word-based retrieval only |
| `q=v` | vector | Vector/KNN retrieval only |
| `q=wv` | hybrid | Word + vector (default) |
| `q=wr` | word+rerank | Word search with reranking |
| `q=vr` | vector+rerank | Vector search with reranking |
| `q=wvr` | full | Hybrid search with reranking |

## Boolean Operators

| Operator | Description | Example |
|----------|-------------|---------|
| (space) | Implicit AND (co) | `hello world` |
| `&&` or `&` | Explicit AND | `hello && world` |
| `\|\|` or `\|` | OR | `hello \|\| world` |
| `()` | Grouping | `(hello \|\| world) && test` |

## Operator Precedence

From highest to lowest:
1. `()` — Parentheses
2. Atom expressions (keywords, filters)
3. `&&` / `&` — AND
4. (space) — Implicit AND (co-occurrence)
5. `||` / `|` — OR

## Complete Examples

```
影视飓风 v>10k :coin>=25 u=[飓多多StormCrew, 影视飓风]
```
Search "影视飓风", views > 10k, coins ≥ 25, from specified users.

```
(影视飓风 || 飓多多) && d=2024 v>=1w
```
Search "影视飓风" OR "飓多多", in 2024, views ≥ 10k.

```
世界 +影视飓风 -广告 v>10k d=1m
```
Search "世界", must have token "影视飓风", exclude "广告", views > 10k, past month.

```
(+影视飓风 & +小米) | (+小米 & -苹果) q=wv
```
Complex constraint: (have 影视飓风 AND 小米) OR (have 小米 AND NOT 苹果), hybrid search.
