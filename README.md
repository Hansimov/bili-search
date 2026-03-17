# Bili-Search

Backend of Bili Search Engine (https://blbl.top) developped by Hansimov.

## Runtime Notes

- `search_app` now exposes ES-TOK relation endpoints:
	- `/related_tokens_by_tokens`
	- `/related_owners_by_tokens`
	- `/related_videos_by_videos`
	- `/related_owners_by_videos`
	- `/related_videos_by_owners`
	- `/related_owners_by_owners`
- `llms` can call local Google Hub for web search via `search_google`, but this is intentionally not exposed as a `search_app` HTTP endpoint.
- Google Hub base URL defaults to `http://127.0.0.1:18100` and can be overridden with `BILI_GOOGLE_HUB_BASE_URL`.
- Google Hub timeout defaults to `45s` and can be overridden with `BILI_GOOGLE_HUB_TIMEOUT`.

See [docs/run/search_app_service.md](/home/asimov/repos/bili-search/docs/run/search_app_service.md) for service operations and [docs/run/relation_search.md](/home/asimov/repos/bili-search/docs/run/relation_search.md) for relation-search notes.