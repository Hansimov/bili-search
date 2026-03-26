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

## 运行模式

- `--runtime local`：管理当前主机上的 bili-search 本地服务进程，适合开发调试。
- `--runtime docker`：管理 Docker 里的 bili-search 实例，适合部署和隔离运行。默认值是 `docker`。

本地模式示例：

```bash
bsdk start --runtime local --foreground -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
bsdk start --runtime local -k -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
bsdk status --runtime local -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
bsdk ps --runtime local --all
```

Docker 模式示例：

```bash
bsdk start -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
bsdk status -p 21001
bsdk ps --all
```

## Docker 构建

```bash
bsdk build-base
```

## 仅构建服务镜像

```bash
bsdk build -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

## 从当前工作区启动

```bash
bsdk start -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

## 从本地 Git 提交启动

```bash
bsdk start \
  --source local-git \
  --git-ref HEAD~1 \
  -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

## 从远端 Git 启动

```bash
bsdk start \
  --source remote-git \
  --git-url https://github.com/hansimov/bili-search.git \
  --git-ref main \
  -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

## 查看状态、端口和日志

```bash
bsdk status -p 21001
bsdk ps
bsdk ps --all
bsdk logs -f -n 120 -p 21001
bsdk status --runtime local -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
bsdk logs --runtime local -f -n 120 -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
bsdk down -p 21001
```

`bsdk ps` 在 `docker` 模式下列出容器，在 `local` 模式下列出本地受管服务。`docker` 模式的 `stop`、`down`、`restart`、`status`、`logs` 这类面向已启动实例的命令，端口号已经足够唯一定位目标实例，其他运行参数不再需要重复输入。

`bsdk status` 在 `docker` 模式下额外会显示 app 级别的启动时间和运行时长。对于 `--restart-scope app` 这类容器内重启，容器本身的 `CREATED` / `Up ...` 不会变化，这是预期行为，应以 `App Started At` 和 `App Uptime` 为准。

当前实例使用 host network，因此 Docker 的 `PORTS` 列为空也是预期行为；实际监听端口会单独显示在 `App Port` 一行。

## 重启策略

默认重启命令：

```bash
bsdk restart -p 21001
```

它等价于：

```bash
bsdk restart -p 21001 --restart-scope app --sync-code
```

含义：

- `--restart-scope app`：只重启容器内的 bili-search app 进程，不重建整个 Docker 容器。
- `--restart-scope container`：重启或重建整个 bili-search 容器。
- `--sync-code`：重启前同步最新代码，默认开启。app-scope 同步会跳过宿主机挂载的 `configs/` 和 `logs/`，避免覆盖只读配置和变化中的日志文件。
- `--no-sync-code`：重启前不做代码同步。

常见组合：

```bash
# 默认：同步最新代码，然后只重启容器内 app
bsdk restart -p 21001

# 只重启容器内 app，但不同步代码
bsdk restart -p 21001 --restart-scope app --no-sync-code

# 基于最新代码重建整个容器
bsdk restart -p 21001 --restart-scope container --sync-code

# 仅对现有容器做 docker restart，不同步代码、不重建镜像
bsdk restart -p 21001 --restart-scope container --no-sync-code
```

## 渲染 Compose 配置

```bash
bsdk config -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```

`build-base`、`build`、`config`、`down` 只支持 `--runtime docker`。`--foreground`、`--reload`、`--kill` 只支持 `--runtime local`。