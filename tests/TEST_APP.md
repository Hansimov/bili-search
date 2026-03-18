# TEST_APP — 实时调试 search service 流程记录

本文档记录了通过 `bssv` 在前台或后台启动 search service、实时调用 API 进行调试、以及调试完成后清理进程的完整流程。

---

## 1. 启动后台服务

```bash
# 开发模式（带 LLM 配置）
bssv start -m dev -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

> 如果需要在当前 shell 前台调试，请改用 `bssv start --foreground ...`。

**等待服务就绪：**

```bash
sleep 10 && curl -s http://localhost:21001/health
# 期望输出：{"status":"ok","search_service":"integrated","llm_model":"gpt-5.2"}
```

---

## 2. 查找并 kill 旧进程

当端口已被占用时，手动清理：

```bash
# 查找占用端口 21001 的进程
lsof -i:21001 -sTCP:LISTEN
# 或
ss -tlnp | grep 21001

# 一键 kill（只杀 LISTEN 状态）
lsof -t -i:21001 -sTCP:LISTEN | xargs -r kill -9

# 验证端口已释放
lsof -i:21001
```

---

## 3. 查看实时日志

```bash
# 查看后台进程的输出
bssv logs -f -m dev -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt

# 查看最后 50 行
bssv logs -n 50 -m dev -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

---

## 4. 健康检查 & 基础 API 测试

```bash
# 健康检查
curl -s http://localhost:21001/health | python3 -m json.tool

# 非流式 chat（同步等待完整回答）
curl -s -X POST http://localhost:21001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":false}' \
  | python3 -m json.tool
```

---

## 5. 测试流式输出（SSE streaming）

### 5a. 基础测试（带时间戳，验证是否逐 token 输出）

```bash
curl -s -N -X POST http://localhost:21001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"说一句话"}],"stream":true}' \
  | while IFS= read -r line; do
      echo "$(date +%T.%3N) $line"
    done
```

**期望效果：** 每个 token 之间有 50–200ms 的自然间隔，而非所有 chunk 在同一毫秒内全部到达。

### 5b. 带工具调用的流式测试（验证 retract_content + 工具事件）

```bash
curl -s -N -X POST http://localhost:21001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"搜索Python编程视频"}],"stream":true}' \
  | while IFS= read -r line; do
      TS=$(date +%T.%3N)
      if   echo "$line" | grep -q '"reasoning_content"'; then echo "$TS [THINK]"
      elif echo "$line" | grep -q '"retract_content"';   then echo "$TS [RETRACT]"
      elif echo "$line" | grep -q '"tool_events"';       then echo "$TS [TOOL]"
      elif echo "$line" | grep -q '"content".*[^"]"';    then echo "$TS [CONTENT]"
      elif echo "$line" | grep -q '"finish_reason": "stop"'; then echo "$TS [DONE]"
      fi
    done
```

**期望事件顺序：**
1. `[CONTENT]` 分析文本实时逐 token 输出
2. `[RETRACT]` 后端检测到工具命令，通知前端清空 content
3. `[THINK]` 分析文本转移到思考区域（reasoning_content）
4. `[TOOL]` 工具调用事件（pending → completed）
5. `[CONTENT]` 最终回答实时逐 token 输出
6. `[DONE]`

### 5c. 思考模式测试（thinking=true）

```bash
curl -s -N -X POST http://localhost:21001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"分析一下B站学习类视频的趋势"}],"stream":true,"thinking":true}' \
  | while IFS= read -r line; do
      echo "$(date +%T.%3N) $line"
    done | head -80
```

---

## 6. 中止流式请求（abort）

```bash
# 先获取 stream_id（第一个 SSE 事件），再发送 abort
STREAM_ID="从第一个 SSE 事件中读取"

curl -s -X POST http://localhost:21001/chat/abort \
  -H "Content-Type: application/json" \
  -d "{\"stream_id\": \"$STREAM_ID\"}"
```

---

## 7. 调试完成后清理

```bash
# 推荐：通过 bssv 停止后台实例
bssv stop -m dev -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt

# 验证已清理
lsof -i:21001
```

---

## 8. 常用启动命令速查

```bash
# 生产模式前台
bssv start --foreground

# 生产模式后台（指定索引）
bssv start -ei bili_videos_pro1 -ev elastic_pro

# 开发模式前台
bssv start --foreground -m dev -ei bili_videos_dev6 -ev elastic_dev

# 开发模式 + LLM（deepseek）
bssv start --foreground -m dev -ei bili_videos_dev6 -ev elastic_dev -lc deepseek

# 开发模式 + LLM（gpt）+ 后台运行
bssv start -m dev -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

---

## 9. 故障排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `[Errno 98] address already in use` | 旧进程未退出，端口被占用 | `lsof -t -i:21001 -sTCP:LISTEN \| xargs -r kill -9`（或重新启动时自动处理） |
| `curl: (7) Failed to connect` | 服务未启动或启动中 | `tail -f /tmp/backend.log`，等待 `Application startup complete` |
| 流式输出一次性全部到达 | 旧版本 fake-streaming | 升级后端代码（实时 LLM streaming，含 look-ahead buffer） |
| `stream_id not found` for abort | 请求已结束或 stream_id 错误 | 正常，可忽略 |
