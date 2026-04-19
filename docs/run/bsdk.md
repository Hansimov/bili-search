# bsdk 使用说明

`bsdk` 是统一的 bili-search 运行管理 CLI，通过 `--runtime local|docker` 同时覆盖本地服务进程和 Docker 实例。

## 相关文件

- Compose 文件：[docker/docker-compose.yml](/home/asimov/repos/bili-search/docker/docker-compose.yml)
- 运行时镜像：[docker/Dockerfile](/home/asimov/repos/bili-search/docker/Dockerfile)
- 基础镜像：[docker/Dockerfile.base](/home/asimov/repos/bili-search/docker/Dockerfile.base)
- 默认环境文件：[docker/.env](/home/asimov/repos/bili-search/docker/.env)
- 示例环境文件：[docker/.env.example](/home/asimov/repos/bili-search/docker/.env.example)

默认按 `Asia/Shanghai` 时区输出时间。

首次配置时，可以将 [docker/.env.example](/home/asimov/repos/bili-search/docker/.env.example) 的内容复制到 [docker/.env](/home/asimov/repos/bili-search/docker/.env) 再按需修改；敏感配置请单独填写到本地 `configs/secrets.json`。

## 转写服务配置

如果当前环境需要使用视频音频转写，请在本地 `configs/secrets.json` 中确保存在如下配置：

```json
"bili_store": {
  "endpoint": "http://YOUR_BILI_STORE_HOST:21501",
  "timeout": 60
}
```

修改 `configs/secrets.json` 后，必须重启当前受管后端进程；否则运行中的 `21001` 不会重新加载 transcript 配置。

重启完成后，用下面命令确认 transcript 能力已开启：

```bash
curl -sS http://127.0.0.1:21001/capabilities | jq '.supports_transcript_lookup'
```

期望返回 `true`。

## 速查

```bash
# 本地前台调试
bsdk start --runtime local --foreground -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7

# Docker 启动当前工作区代码
bsdk start -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7

# 默认重启：同步代码，只重启容器内 app
bsdk restart -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7

# 重建整个容器
bsdk restart --restart-scope container -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7

# 只做 docker restart，不同步代码、不重建镜像
bsdk restart --restart-scope container --no-sync-code -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7

# 查看状态 / 列表 / 日志 / 健康
bsdk status -p 21001
bsdk ps --all
bsdk logs -f -n 120 -p 21001
bsdk check -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7

# 查看本地受管服务
bsdk status --runtime local -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7
bsdk ps --runtime local --all
bsdk prune
```

## 通过 blbl-dash 管理 local-dev 后端

如果 `21001` 已经纳入 `blbl-dash` 管控，优先通过 `bldash` 执行动作，不要手工起 `uvicorn`。这样可以保留统一的状态、wait probe、operation 记录和日志定位。

当前 `search.backend-local` 的受管重启底层实际调用的是：

```bash
bsdk restart --runtime local -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7
```

但日常操作推荐始终从 `blbl-dash` 入口执行。

### 后端单服务重启

适用场景：

- 只改了 `bili-search` 后端 Python 代码或本地配置。
- 只需要让 `21001` 重新加载 `configs/secrets.json`、LLM 配置或 transcript 配置。
- 不希望顺带重启 `21002` 前端。

推荐流程：

```bash
cd /home/asimov/repos/blbl-dash

# 先看当前状态
/home/asimov/miniconda3/envs/ai/bin/bldash service status search.backend-local --output json

# 先 dry-run，确认会走受管 backend restart
/home/asimov/miniconda3/envs/ai/bin/bldash service restart search.backend-local \
  --db var/blbl-dash.sqlite3 \
  --dry-run \
  --output json

# 再执行真实重启
/home/asimov/miniconda3/envs/ai/bin/bldash service restart search.backend-local \
  --db var/blbl-dash.sqlite3 \
  --output json
```

重启后验证：

```bash
curl -sS http://127.0.0.1:21001/health | jq
curl -sS http://127.0.0.1:21001/capabilities | jq
```

### local-dev 整链重启

适用场景：

- 后端和前端都需要一起重启。
- 需要按 `local-dev` 链的标准顺序执行依赖检查、backend restart、frontend restart。

```bash
cd /home/asimov/repos/blbl-dash

/home/asimov/miniconda3/envs/ai/bin/bldash chain restart search.local-dev \
  --db var/blbl-dash.sqlite3 \
  --dry-run \
  --output json

/home/asimov/miniconda3/envs/ai/bin/bldash chain restart search.local-dev \
  --db var/blbl-dash.sqlite3 \
  --output json
```

`chain restart` 当前会先检查 dev Elasticsearch，再重启 `search.backend-local`，最后重启 `search.ui-dev`。

### local-dev 更新到当前工作区代码

适用场景：

- 需要让 `local-dev` 链按当前工作区代码重新刷新受管服务。
- 希望使用 `blbl-dash` 的标准 update helper，而不是自己逐个 stop/start。

```bash
cd /home/asimov/repos/blbl-dash

/home/asimov/miniconda3/envs/ai/bin/bldash chain update search.local-dev \
  --db var/blbl-dash.sqlite3 \
  --dry-run \
  --output json

/home/asimov/miniconda3/envs/ai/bin/bldash chain update search.local-dev \
  --db var/blbl-dash.sqlite3 \
  --output json
```

当前 `chain update search.local-dev` 会先检查 dev Elasticsearch，再按 helper 顺序刷新 backend 和 frontend 到当前工作区代码。

## Docker 构建

```bash
# 构建基础镜像
bsdk build-base

# 仅构建服务镜像
bsdk build -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7

# 使用主源 + 回退源
bsdk build \
  -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7 \
  --pip-index-url https://mirrors.ustc.edu.cn/pypi/simple \
  --pip-extra-index-url https://pypi.org/simple
```

当前 Docker 依赖链会优先使用 `--pip-index-url`。如果 wheel 构建失败，会自动切换到 `--pip-extra-index-url` 再重试一次。

## 代码来源

```bash
# 当前工作区
bsdk start -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7

# 本地 Git 历史版本
bsdk start \
  --source local-git \
  --git-ref HEAD~1 \
  -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7

# 远端 Git 仓库
bsdk start \
  --source remote-git \
  --git-url https://github.com/hansimov/bili-search.git \
  --git-ref main \
  -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7
```

## 运行说明

- `--runtime local`：管理当前主机上的 bili-search 本地服务进程，适合开发调试。
- `--runtime docker`：管理 Docker 里的 bili-search 实例，适合部署和隔离运行。默认值是 `docker`。
- `bsdk ps` 在 `docker` 模式下列出容器，在 `local` 模式下列出本地受管服务。
- `bsdk prune` 会清理 stale 本地 PID 记录和 exited Docker 容器，但不会停止当前运行中的实例。
- 对已启动的 Docker 实例，`stop`、`down`、`restart`、`status`、`logs` 这类命令通常只需要端口号就足够反查目标容器，其他运行参数无需重复输入。
- `bsdk status` 在 `docker` 模式下会额外显示 app 级别的启动时间和运行时长。对于 `--restart-scope app`，容器本身的启动时间不会变化，应以 `App Started At` 和 `App Uptime` 为准。
- 当前实例使用 host network，因此 Docker 原生 `docker ps` 的 `PORTS` 可能为空；`bsdk status` / `bsdk ps` 会补齐实际监听端口。

## 注意事项

- 如果 `21001` 已经由 `blbl-dash` 管理，不要再手工执行 `python -m uvicorn ... --port 21001`。这样容易产生 orphan 进程，导致控制面状态和真实运行态不一致。
- 改完 `configs/secrets.json` 这类本地配置后，优先使用 `bldash service restart search.backend-local`；不要只做 capability 推断，不重启进程。
- 只改后端时优先做 backend 单服务重启；只有前端也需要刷新时，才使用 `bldash chain restart search.local-dev`。
- 计划刷新整条 local-dev 开发链到当前工作区代码时，优先使用 `bldash chain update search.local-dev`，不要手工分别 stop/start backend 和 frontend。
- 当修改了容器启动链本身，例如 `service.container_supervisor`、Dockerfile、依赖安装方式或 console script，优先使用 `--restart-scope container`，不要只做 app-scope restart。
- `--restart-scope app` 更适合纯 Python 业务代码变更；它不会替换已经在容器里运行的 supervisor 进程。
- Docker 构建如果遇到镜像源抖动，可保留 USTC 作为主源，同时配置官方 PyPI 作为 `--pip-extra-index-url` 回退源。
- `build-base`、`build`、`config`、`down` 只支持 `--runtime docker`。`--foreground`、`--reload`、`--kill` 只支持 `--runtime local`。

## 参数参考

- `--runtime local|docker`：选择管理本地服务进程还是 Docker 实例，默认 `docker`。
- `-p/--port`：实例端口。对已启动的 Docker 实例，通常只用端口就足够定位目标容器。
- `-ei/--elastic-index`：搜索视频索引名。
- `-ev/--elastic-env-name`：`configs/secrets.json` 中的 Elasticsearch 环境名。
- `-lc/--llm-config`：LLM 配置名。
- `--source workspace|local-git|remote-git`：Docker 构建使用的代码来源。
- `--git-repo --git-ref --git-url`：Git 源相关参数。
- `--restart-scope app|container`：重启 app 进程还是重启/重建整个容器。
- `--sync-code/--no-sync-code`：重启前是否同步当前代码，默认同步。
- `--no-build`：`start` 时跳过 `docker compose up --build`。
- `--no-base-build`：跳过基础镜像构建。
- `--pip-index-url`：Docker 构建的主 pip 源。
- `--pip-extra-index-url`：主源失败时的回退 pip 源。
- `--pip-trusted-host`：主 pip 源对应的 trusted host。
- `--pip-retries`：Docker wheel 下载重试次数。
- `--pip-timeout`：Docker wheel 下载单次请求超时时间。
- `--lines/-n`：`logs` 输出的尾部行数。
- `--follow/-f`：持续跟随日志。
- `--timeout`：`status` / `check` 健康检查超时时间。

## 其他命令

```bash
# 渲染 compose 配置
bsdk config -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc minimax-m2.7

# 删除实例
bsdk down -p 21001

# 清理历史残留
bsdk prune
```