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

## 速查

```bash
# 本地前台调试
bsdk start --runtime local --foreground -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek

# Docker 启动当前工作区代码
bsdk start -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek

# 默认重启：同步代码，只重启容器内 app
bsdk restart -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek

# 重建整个容器
bsdk restart --restart-scope container -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek

# 只做 docker restart，不同步代码、不重建镜像
bsdk restart --restart-scope container --no-sync-code -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek

# 查看状态 / 列表 / 日志 / 健康
bsdk status -p 21001
bsdk ps --all
bsdk logs -f -n 120 -p 21001
bsdk check -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek

# 查看本地受管服务
bsdk status --runtime local -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek
bsdk ps --runtime local --all
```

## Docker 构建

```bash
# 构建基础镜像
bsdk build-base

# 仅构建服务镜像
bsdk build -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek

# 使用主源 + 回退源
bsdk build \
  -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek \
  --pip-index-url https://mirrors.ustc.edu.cn/pypi/simple \
  --pip-extra-index-url https://pypi.org/simple
```

当前 Docker 依赖链会优先使用 `--pip-index-url`。如果 wheel 构建失败，会自动切换到 `--pip-extra-index-url` 再重试一次。

## 代码来源

```bash
# 当前工作区
bsdk start -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek

# 本地 Git 历史版本
bsdk start \
  --source local-git \
  --git-ref HEAD~1 \
  -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek

# 远端 Git 仓库
bsdk start \
  --source remote-git \
  --git-url https://github.com/hansimov/bili-search.git \
  --git-ref main \
  -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek
```

## 运行说明

- `--runtime local`：管理当前主机上的 bili-search 本地服务进程，适合开发调试。
- `--runtime docker`：管理 Docker 里的 bili-search 实例，适合部署和隔离运行。默认值是 `docker`。
- `bsdk ps` 在 `docker` 模式下列出容器，在 `local` 模式下列出本地受管服务。
- 对已启动的 Docker 实例，`stop`、`down`、`restart`、`status`、`logs` 这类命令通常只需要端口号就足够反查目标容器，其他运行参数无需重复输入。
- `bsdk status` 在 `docker` 模式下会额外显示 app 级别的启动时间和运行时长。对于 `--restart-scope app`，容器本身的启动时间不会变化，应以 `App Started At` 和 `App Uptime` 为准。
- 当前实例使用 host network，因此 Docker 原生 `docker ps` 的 `PORTS` 可能为空；`bsdk status` / `bsdk ps` 会补齐实际监听端口。

## 注意事项

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
bsdk config -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc deepseek

# 删除实例
bsdk down -p 21001
```