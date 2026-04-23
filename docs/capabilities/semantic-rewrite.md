# Semantic Rewrite And Intent Architecture

## Scope

This document describes the current semantic rewrite and intent-recognition
architecture shared across bili-search and es-tok after the recent refactor.

The goal is to stop growing search quality through scattered code constants,
and instead move deterministic knowledge into versioned assets and index-time
semantic state.

## What Moved Out Of Code

### 1. Term alias normalization in bili-search

- Local alias rewrite is no longer embedded as `_TERM_ALIAS_MAP` in Python.
- Alias rules now live in `llms/intent/assets/term_aliases.json`.
- `llms/intent/semantic_assets.py` is the stable loader boundary.
- `llms.intent.focus.rewrite_known_term_aliases()` only performs token-aware
  matching and replacement; it no longer owns the alias data itself.

### 2. Owner intent policy in bili-search

- Owner intent thresholds, blocked markers, source-label rules, and model-code
  patterns no longer live as searcher-local constants.
- They now live in `elastics/videos/assets/owner_intent_policy.json`.
- `elastics/videos/owner_intent_policy.py` exposes a cached
  `OwnerIntentPolicy` object used by `VideoSearcherV2`.
- The searcher keeps the control flow, but policy values are now external and
  testable without editing the main retrieval pipeline.

## es-tok Semantic Store

### 1. Query-time abstraction

es-tok semantic expansion no longer depends directly on a single static tuning
loader. The semantic branch now flows through a common interface:

- `SemanticExpansionStore`
- `SemanticExpansionRule`
- `SemanticQueryExpansionSuggester`

The default resource-backed rule set is still provided by
`QueryExpansionTuning`, but it now implements the same store contract as any
future snapshot-backed or hybrid provider.

### 2. Index-time incremental semantic snapshot

es-tok now includes a first index-time semantic snapshot pipeline:

- `SemanticSnapshotManager`
- `SemanticSnapshotIndexListener`
- `IndexSemanticExpansionSnapshot`

Current behavior:

1. Successful `postIndex` events decode the indexed `_source`.
2. Only `title`, `tags`, and `rtags` are used for the first snapshot stage.
3. Each document gets a lightweight fingerprint derived from normalized
   `title + tags`.
4. Re-indexing the same document with an unchanged fingerprint is skipped.
5. Changed documents remove their old contribution before adding the new one.
6. Successful deletes remove the document contribution from the snapshot.
7. Query-time `semantic` expansion merges static rule expansions with dynamic
   `doc_cooccurrence` expansions from the live snapshot.

### 3. Noise control in the first snapshot version

The first snapshot version deliberately treats title-derived terms as anchors,
not primary expansion targets.

This keeps useful mappings such as:

- `ComfyUI -> 教程`
- `教程 -> 讲解`

while avoiding low-value expansions such as:

- `教程 -> ComfyUI 教程`
- `教程 -> ComfyUI`

## Why This Matters

Before this refactor, semantic knowledge was split across three unrelated
surfaces:

- static es-tok query expansion tuning
- local bili-search alias rewrites
- searcher-local owner intent thresholds and regexes

That meant search quality drifted whenever one surface changed and the others
did not.

The new structure makes the boundaries explicit:

- deterministic rules belong in asset files
- dynamic semantic evidence belongs in the semantic snapshot store
- retrieval/orchestration code should consume policy/store objects instead of
  embedding rule tables inline

## Next Steps

The current implementation is the first operational slice, not the final form.

Planned follow-ups:

1. Persist semantic snapshots across node restarts instead of rebuilding only
   from live index traffic.
2. Rebuild snapshot state on shard start for existing docs, not just new writes.
3. Improve document semantic extraction beyond `title/tags/rtags`, especially
   for long Chinese titles and multi-part topics.
4. Feed snapshot-backed semantic evidence into owner intent and query planning,
   not only token expansion.
5. Evaluate whether some of the remaining owner intent heuristics should move
   from policy assets into learned or corpus-driven priors.