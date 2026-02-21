# DSL Usage Guide

## Quick Start

The DSL enables concise, one-line queries that combine text search with filters.

### Basic Search
```
影视飓风                    # Simple keyword search
"deep learning"             # Exact phrase search
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
Use `+`/`-` to require or exclude specific tokens:
```
影视飓风 +小米              # Must have token "小米" in title/tags
影视飓风 -广告              # Must NOT have token "广告"
+影视飓风 -广告 -推广        # Must have "影视飓风", exclude "广告" and "推广"
```

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

Token constraints (`+`/`-`) map to `es_tok_constraints` for efficient token-level filtering.
They work independently from keyword search and are especially useful for:

- **Precision**: Ensure results contain or exclude specific tokens
- **KNN filtering**: Applied as pre-filters during vector search
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

# With constraints
result = converter.expr_to_dict("+影视飓风 -广告 v>10k")

# Custom constraint fields
result = converter.expr_to_dict(
    "+影视飓风",
    constraint_fields=["title.words", "tags.words"]
)
```
