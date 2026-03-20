# DSL Design

## Overview

The DSL (Domain-Specific Language) module provides a concise, one-line query syntax for the bili-search engine. It is designed to be:

1. **Human-friendly**: Easy to read, write, and understand
2. **LLM-friendly**: Compact and unambiguous for AI agents to generate and adjust
3. **Powerful**: Supports complex search queries with filters, boolean logic, and token constraints

## Architecture

```
dsl/
├── __init__.py
├── syntax.lark          # Lark grammar definition
├── parse.py             # Lark parser wrapper
├── node.py              # DSL tree node classes
├── constants.py         # Constants and type definitions
├── elastic.py           # DSL tree → Elasticsearch query converter
├── filter.py            # Query DSL dict filter merger
├── rewrite.py           # Query rewriting and expansion
├── check.py             # Syntax validation
├── test.py              # Internal test cases
└── fields/              # Per-field expression converters
    ├── bool.py           # Boolean clause reducer
    ├── bvid.py           # Video ID (av/bv) expressions
    ├── constraint.py     # Token constraint expressions (+/-/!)
    ├── date.py           # Date/time expressions
    ├── dura.py           # Duration expressions
    ├── qmod.py           # Query mode expressions
    ├── stat.py           # Statistics expressions (view, like, etc.)
    ├── umid.py           # User ID (uid/mid) expressions
    ├── user.py           # Username expressions
    └── word.py           # Keyword/word expressions
```

## Processing Pipeline

```
Input Query String
    ↓
[1] Lark Parser (syntax.lark → parse.py)
    ↓ Lark Tree
[2] DSL Tree Builder (node.py: DslTreeBuilder)
    ↓ DslNode tree
[3] Expression Grouper (node.py: DslTreeExprGrouper)
    ↓ DslExprNode tree (grouped by priority)
[4] Tree Flatter (node.py: DslExprTreeFlatter)
    ↓ Flattened DslExprNode tree
[5] Constraint Extractor (fields/constraint.py)
    ↓ Constraints + cleaned tree
[6] Elastic Converter (elastic.py + fields/*.py)
    ↓ Elasticsearch query dict
[7] Filter Merger (filter.py)
    ↓ Final merged query
```

## Key Design Decisions

### Boolean Expression Priority
```
atom > pa (parentheses) > and > co (implicit AND) > or
```

### Token Constraints
Words with `+`/`-` prefixes are treated as token constraints:
- `+token` → kept inside `es_tok_query_string` and interpreted as an analyzer-aware exact segment requirement
- `-token` → kept inside `es_tok_query_string` and interpreted as an analyzer-aware exact segment exclusion
- `"token"` → same exact-segment semantics without forcing MUST / MUST_NOT

主查询和 `constraint_filter` 都统一走 `es_tok_query_string`，确保 analyzer 行为、field coverage 和 exact 语义一致。旧的 `es_tok_constraints.have_token` 路径只保留给真正的 token-level 业务过滤，不再承担 `+/-` 的文本语义。

这次变更是破坏性的：

- 不再把 `+/-` 从查询文本里抽成 `have_token`。
- 不再给 `es_tok_query_string` 发送 `type` 这类旧参数。
- analyzer-split 中文片段会通过 exact-segment + position-aware phrase fallback 保留“不可拆分”的含义，而不是退化成松散 token 组合。

```
世界 +影视飓风 -广告    → search "世界" + exact include "影视飓风" + exact exclude "广告"
```

### Query Mode (qmod)
Controls the search method: word search, vector KNN, or hybrid:
```
q=w    → word-only search
q=v    → vector-only KNN search
q=wv   → hybrid (word + vector, default)
q=wvr  → hybrid + rerank
```

### User Exclusion
User exclusion uses `must_not` in Elasticsearch, properly preserved in both
word search and KNN search paths via `get_all_bool_clauses_from_query_dsl_dict`.
