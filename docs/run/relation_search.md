# Relation Search Notes

## Scope

`search_app` relation APIs are thin wrappers over canonical ES-TOK REST endpoints under `/{index}/_es_tok/*`.

Implemented routes:

- `/related_tokens_by_tokens`
- `/related_owners_by_tokens`
- `/related_videos_by_videos`
- `/related_owners_by_videos`
- `/related_videos_by_owners`
- `/related_owners_by_owners`

## Practical Usage

- Use `related_owners_by_tokens` when the user wants creators for a topic. This is a better fit than trying to infer a creator via `suggest` heuristics.
- Use `search_videos` for exact timeline/listing tasks such as `:user=影视飓风 :date<=15d`.
- Use graph relation endpoints only when you already have reliable seeds like `bvids` or `mids`.
- Keep Google search in `llms` only. It is a supplement for external facts, not part of the search service contract.

## ES-TOK Findings

- The canonical endpoint shape from `es-tok/docs/01_API.md` is stable enough to wrap directly; no extra compatibility aliases were needed.
- XML tool parsing must tolerate `>` inside quoted DSL fragments like `:view>=1w`, otherwise multi-query commands get truncated. The handler now uses a quoted-attribute-aware regex.
- Relation APIs should return compact, LLM-oriented payloads instead of raw plugin responses. The executor now normalizes owners/videos/tokens into small stable shapes.
- `related_owners_by_tokens` is the correct replacement for the old `check_author` heuristic in creator-discovery flows. Exact creator timeline queries should still go straight to `search_videos` with `:user=`.