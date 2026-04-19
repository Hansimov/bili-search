# 测试与联调手册

本文档用于记录 bili-search 当前推荐的本地测试、运行联调和清理流程。

## 统一配置来源

默认运行参数统一来自 `configs/envs.json.search_app`。当前关键字段包括：`host`、`port`、`elastic_index`、`elastic_env_name`、`llm_config`。

优先级是：`CLI 参数 > BILI_SEARCH_APP_* 环境变量 > configs/envs.json.search_app`。

推荐做法：

- 常规 `bsdk` 命令直接继承配置，不要反复手写 `-p/-ei/-ev/-lc`。
- 如果你修改了 `configs/envs.json.search_app.port`，下面的 `curl` / `pytest` URL 也应该跟着同一个值走。

先在当前 shell 里读取默认值：

```bash
APP_PORT=$(jq -r '.search_app.port' configs/envs.json)
APP_INDEX=$(jq -r '.search_app.elastic_index' configs/envs.json)
APP_ES_ENV=$(jq -r '.search_app.elastic_env_name' configs/envs.json)
APP_LLM=$(jq -r '.search_app.llm_config' configs/envs.json)
```

## 本地启动服务

后台启动：

```bash
bsdk start --runtime local
```

前台启动：

```bash
bsdk start --runtime local --foreground
```

前台热重载：

```bash
bsdk start --runtime local --foreground --reload
```

如果端口已被旧进程占用，可以在启动前加上 `-k`：

```bash
bsdk start --runtime local -k
```

## Docker 启动服务

```bash
bsdk start
```

默认情况下，上述命令会统一继承 `configs/envs.json.search_app` 中的 `port`、`elastic_index`、`elastic_env_name` 和 `llm_config`。只有临时覆盖时，才需要显式传 `-p/-ei/-ev/-lc`。

例如：

```bash
bsdk start --runtime local -p 21011
bsdk start -p 21011 -ei bili_videos_dev7 -ev elastic_test -lc deepseek
```

查看实例：

```bash
bsdk ps
bsdk ps --all
bsdk status
```

## 等待服务就绪

```bash
sleep 8 && curl -sS "http://127.0.0.1:${APP_PORT}/health" | jq
curl -sS "http://127.0.0.1:${APP_PORT}/capabilities" | jq
```

期望健康检查返回：

```json
{
  "status": "ok",
  "search_service": "integrated",
  "llm_model": "<configured default large model>"
}
```

## 查看状态与日志

本地服务：

```bash
bsdk status --runtime local
bsdk ps --runtime local
bsdk logs --runtime local -f
bsdk logs --runtime local -n 80
```

Docker 服务：

```bash
bsdk status
bsdk logs -f -n 120
bsdk prune
```

## 基础接口测试

```bash
curl -sS "http://127.0.0.1:${APP_PORT}/health" | jq

curl -sS -X POST "http://127.0.0.1:${APP_PORT}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":false}' \
  | jq
```

## 流式输出测试

基础流式输出：

```bash
curl -sS -N -X POST "http://127.0.0.1:${APP_PORT}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"说一句话"}],"stream":true}' \
  | while IFS= read -r line; do
      echo "$(date +%T.%3N) $line"
    done
```

工具调用流式输出：

```bash
curl -sS -N -X POST "http://127.0.0.1:${APP_PORT}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"搜索Python编程视频"}],"stream":true}' \
  | while IFS= read -r line; do
      TS=$(date +%T.%3N)
      if   echo "$line" | grep -q '"reasoning_content"'; then echo "$TS [THINK]"
      elif echo "$line" | grep -q '"retract_content"';   then echo "$TS [RETRACT]"
      elif echo "$line" | grep -q '"tool_events"';       then echo "$TS [TOOL]"
      elif echo "$line" | grep -q '"content".*[^"]"';   then echo "$TS [CONTENT]"
      elif echo "$line" | grep -q '"finish_reason": "stop"'; then echo "$TS [DONE]"
      fi
    done
```

思考模式：

```bash
curl -sS -N -X POST "http://127.0.0.1:${APP_PORT}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"分析一下B站学习类视频的趋势"}],"stream":true,"thinking":true}' \
  | head -80
```

## 中止流式请求

```bash
STREAM_ID="从第一个 SSE 事件中读取"

curl -sS -X POST "http://127.0.0.1:${APP_PORT}/chat/abort" \
  -H "Content-Type: application/json" \
  -d "{\"stream_id\": \"$STREAM_ID\"}"
```

## 运行测试

完整回归：

```bash
pytest -q
```

运行时集成测试：

```bash
BILI_SEARCH_RUNTIME_URL="http://127.0.0.1:${APP_PORT}" pytest -q tests/llm/test_runtime_integration.py -k 'not runtime_chat_completion'
BILI_SEARCH_RUNTIME_URL="http://127.0.0.1:${APP_PORT}" BILI_SEARCH_RUNTIME_LLM=1 pytest -q tests/llm/test_runtime_integration.py
```

说明：`tests/llm/test_runtime_integration.py` 的完整 live 套件包含多个 90s 级别的 HTTP case，以及一个更长的流式 case；在 `pytest -q` 下看起来可能会长时间没有输出，但这通常不是卡死。排查或回归时，优先使用可见进度和分批执行：

```bash
BILI_SEARCH_RUNTIME_URL="http://127.0.0.1:${APP_PORT}" BILI_SEARCH_RUNTIME_LLM=1 pytest -vv -s tests/llm/test_runtime_integration.py -k 'test_runtime_health_and_capabilities or test_runtime_chat_completion'
BILI_SEARCH_RUNTIME_URL="http://127.0.0.1:${APP_PORT}" BILI_SEARCH_RUNTIME_LLM=1 pytest -vv -s tests/llm/test_runtime_integration.py -k 'known_video_summary_transcript or multi_creator_productivity_comparison'
```

常用定向测试：

```bash
pytest -q tests/test_bsdk_local_cli.py tests/test_bsdk_cli.py
pytest -q tests/test_secret_hygiene.py
pytest -q tests/llm/test_search_service.py tests/llm/test_app.py
```

`es_tok_query_string` 破坏性改动后的推荐回归入口：

```bash
pytest -q elastics/tests/test_es_tok_query_smoke.py
pytest -q elastics/tests/test_videos.py -k 'es_tok_query'
python -m elastics.tests.benchmark_es_tok_exact_segments
conda run -n ai python debugs/profile_es_tok_exclusion_path.py
```

说明：

- `test_es_tok_query_smoke.py` 是 reload 插件后最快的 live smoke suite。
- `test_videos.py -k 'es_tok_query'` 覆盖 DSL 构造、constraint_filter 保真和更多真实索引 case。
- `benchmark_es_tok_exact_segments` 用来对比 plain / quoted / `+` / `-` exact query 的真实性能。
- `profile_es_tok_exclusion_path.py` 用来拆 `-exact` 慢路径里 ES 查询、script_score 和 `track_total_hits` 的成本。

## 清理服务

停止本地后台实例：

```bash
bsdk stop --runtime local
```

停止 Docker 实例：

```bash
bsdk down
bsdk prune
```

确认端口是否释放：

```bash
lsof -i:"${APP_PORT}"
```

## 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `[Errno 98] address already in use` | 旧进程未退出，端口被占用 | 启动时加 `-k`，或 `lsof -t -i:${APP_PORT} -sTCP:LISTEN \| xargs -r kill -9` |
| `curl: (7) Failed to connect` | 服务尚未完成启动 | 先执行 `bsdk status --runtime local` / `bsdk status`，再查看日志 |
| `bsdk status` 首次健康检查失败 | 容器已启动，但应用仍在加载 | 等待数秒后再次执行 `bsdk status` |
| 流式输出一次性全部到达 | 上游流式返回异常或前端未按 SSE 消费 | 用带时间戳的 `curl -N` 直接验证 |
| `stream_id not found` | 请求已经结束或 `stream_id` 错误 | 可忽略，重新发起流式请求即可 |