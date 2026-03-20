# DSL Usage Guide

## Quick Start

The DSL enables concise, one-line queries that combine text search with filters.

### Basic Search
```
影视飓风                    # Simple keyword search
"deep learning"             # Exact segment search
影视飓风 风光摄影             # Multiple keywords (implicit AND)
影视飓风 || 飓多多            # OR search
```

### Adding Filters
```
影视飓风 v>10k              # Views > 10,000
影视飓风 d=2024             # Published in 2024
影视飓风 u=飓多多StormCrew    # By specific user
影视飓风 t>5m               # Duration > 5 minutes
```

### Token Constraints
Use `+`/`-`/`"..."` to express exact segments:
```
影视飓风 +小米              # Must contain exact segment "小米"
影视飓风 -广告              # Must NOT contain exact segment "广告"
"影视飓风"                 # Exact segment, but not forced MUST / MUST_NOT
+影视飓风 -广告 -推广        # Must contain "影视飓风", exclude "广告" and "推广"
```

### Breaking Change And Migration

上游 `es-tok` 已经把 `es_tok_query_string` 收敛成最小 DSL。bili-search 也已经同步切换到新语义：

- `+片段` / `-片段` / `"片段"` 会保留在 `es_tok_query_string.query` 里，由插件执行 analyzer-aware exact matching。
- 旧的 `have_token` 路径只保留给真正的 `es_tok_constraints` 业务过滤；不要再把 `+/-` 预先改写成它。
- bili-search 侧已经停止向 `es_tok_query_string` 发送 `type` 这类旧参数。

调用方迁移时请注意：

- 以前依赖 Lucene `query_string` 的 `field:term`、`foo*`、`[a TO b]`、`AND/OR/NOT` 关键字语义，需要迁移到 DSL 的外层字段/过滤表达式。
- 以前把 `+/-` 当成普通词或 must_not 子句理解的代码，需要改成 exact segment 语义。

### Complex Queries
```
(影视飓风 || 飓多多) && d=2024 v>=1w
影视飓风 v>10k :coin>=25 u=[飓多多StormCrew, 亿点点不一样]
(+影视飓风 & +小米) | (+小米 & -苹果)
```

## For LLM Agents

The DSL is designed for programmatic generation by LLM agents during search and reasoning tasks.

### Query Refinement Flow

1. Start with a broad query: `影视飓风`
2. Add filters to narrow results: `影视飓风 v>10k d=2024`
3. Add user constraints: `影视飓风 v>10k d=2024 u!=广告号`
4. Adjust constraints: `影视飓风 v>10k +小米 d=2024`
5. Try alternative: `(影视飓风 || 飓多多) v>10k +小米 d=2024`

### Constraint-Based Filtering

Exact constraints (`+`/`-`/`"..."`) are handled by `es_tok_query_string` itself:
- `+token`: Must contain the exact segment.
- `-token`: Must NOT contain the exact segment.
- `"token"`: Exact segment without forcing MUST / MUST_NOT.

They are especially useful for:

- **Precision**: Ensure results contain or exclude specific analyzer-aware exact segments
- **Iterative refinement**: Add/remove constraints without changing the main query

### Query Mode Control

Control the search method with `q=`:
```
影视飓风 q=w      # Word search only (fastest)
影视飓风 q=v      # Vector/semantic search only
影视飓风 q=wv     # Hybrid: word + vector (default, best recall)
影视飓风 q=wvr    # Hybrid + rerank (best precision, slower)
```

## Module Structure

The DSL module is a top-level module at `dsl/`:
```python
from dsl.elastic import DslExprToElasticConverter
from dsl.rewrite import DslExprRewriter
from dsl.filter import QueryDslDictFilterMerger
```

### Key Classes

- `DslExprToElasticConverter`: Main converter from DSL string to Elasticsearch query
- `DslExprRewriter`: Handles query rewriting and word expansion
- `QueryDslDictFilterMerger`: Merges filter clauses with ES queries
- `DslLarkParser`: Low-level Lark grammar parser
- `DslSyntaxChecker`: Validates DSL syntax

### Usage Example

```python
from dsl.elastic import DslExprToElasticConverter

converter = DslExprToElasticConverter()

# Simple query
result = converter.expr_to_dict("影视飓风 v>10k")

# With exact constraints preserved inside es_tok_query_string
result = converter.expr_to_dict("+影视飓风 -广告 v>10k")
```
