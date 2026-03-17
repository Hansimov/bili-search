# Search App Service Runbook

## Start

```bash
PYTHONPATH=. python -m apps.search_app_cli start -m dev -p 21031 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

## Status And Logs

```bash
PYTHONPATH=. python -m apps.search_app_cli status -m dev -p 21031 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
PYTHONPATH=. python -m apps.search_app_cli logs -n 120
PYTHONPATH=. python -m apps.search_app_cli logs -f
```

## Health And Capabilities

```bash
curl -sS http://127.0.0.1:21031/health | jq
curl -sS http://127.0.0.1:21031/capabilities | jq
```

## Search Checks

```bash
curl -sS -X POST http://127.0.0.1:21031/suggest -H 'Content-Type: application/json' -d '{"query":"黑神话","limit":1}' | jq
curl -sS -X POST http://127.0.0.1:21031/explore -H 'Content-Type: application/json' -d '{"query":"黑神话 q=vwr"}' | jq '.query,.status'
curl -sS -X POST http://127.0.0.1:21031/related_owners_by_tokens -H 'Content-Type: application/json' -d '{"text":"黑神话悟空","size":5}' | jq '.owners[:3]'
curl -sS -X POST http://127.0.0.1:21031/related_videos_by_owners -H 'Content-Type: application/json' -d '{"mids":[946974],"size":3}' | jq '.videos[:3]'
```

## LLM Checks

Direct app endpoint:

```bash
curl -sS -X POST http://127.0.0.1:21031/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"请用一句话介绍黑神话悟空是什么。"}],"stream":false}' | jq '.choices[0].message.content'
```

Remote llms CLI through search_app:

```bash
PYTHONPATH=. python -m llms.cli --llm-config gpt --search-base-url http://127.0.0.1:21031 -q "推荐几条高播放的黑神话悟空视频"
```

Google Hub is used only inside `llms` tool execution. By default it targets `http://127.0.0.1:18100` and can be overridden with `BILI_GOOGLE_HUB_BASE_URL`:

```bash
BILI_GOOGLE_HUB_BASE_URL=http://127.0.0.1:18100 BILI_GOOGLE_HUB_TIMEOUT=45 PYTHONPATH=. python -m llms.cli --llm-config gpt --search-base-url http://127.0.0.1:21031 -q "Gemini 2.5 最近有哪些官方更新，B站上有没有解读视频？"
```

## Runtime Pytest

```bash
PYTHONPATH=. BILI_SEARCH_RUNTIME_URL=http://127.0.0.1:21031 pytest -q tests/llm/test_runtime_integration.py -k 'not runtime_chat_completion'
PYTHONPATH=. BILI_SEARCH_RUNTIME_URL=http://127.0.0.1:21031 BILI_SEARCH_RUNTIME_LLM=1 pytest -q tests/llm/test_runtime_integration.py
```

## Stop

```bash
PYTHONPATH=. python -m apps.search_app_cli stop
```