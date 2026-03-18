# bssv 使用说明

`bssv` 用来管理当前主机上的 bili-search 服务进程。默认以后台方式运行，也可以通过 `--foreground` 在前台直接启动，适合本地调试。

默认按 `Asia/Shanghai` 时区输出时间。

## 启动服务

后台启动：

```bash
bssv start -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

前台启动：

```bash
bssv start --foreground -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

前台热重载：

```bash
bssv start --foreground --reload -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

## 查看状态、端口和日志

```bash
bssv status -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
bssv ps
bssv ps --all
bssv logs -n 120 -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
bssv logs -f -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

`bssv ps` 会列出后台受管服务的端口、PID 和启动时间；`--all` 会额外显示失效的 PID 记录。

## 健康检查与能力接口

```bash
curl -sS http://127.0.0.1:21001/health | jq
curl -sS http://127.0.0.1:21001/capabilities | jq
```

## 运行时集成测试

```bash
BILI_SEARCH_RUNTIME_URL=http://127.0.0.1:21001 pytest -q tests/llm/test_runtime_integration.py -k 'not runtime_chat_completion'
BILI_SEARCH_RUNTIME_URL=http://127.0.0.1:21001 BILI_SEARCH_RUNTIME_LLM=1 pytest -q tests/llm/test_runtime_integration.py
```

## 停止服务

```bash
bssv stop -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```