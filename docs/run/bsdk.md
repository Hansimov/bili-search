# bsdk 使用说明

`bsdk` 用来管理 Docker 方式运行的 bili-search 服务实例。它复用 `bssv` 的运行时参数，并额外提供 compose、镜像、源码来源和宿主机挂载目录等参数。

## 相关文件

- Compose 文件：[docker/docker-compose.yml](/home/asimov/repos/bili-search/docker/docker-compose.yml)
- 运行时镜像：[docker/Dockerfile](/home/asimov/repos/bili-search/docker/Dockerfile)
- 基础镜像：[docker/Dockerfile.base](/home/asimov/repos/bili-search/docker/Dockerfile.base)
- 默认环境文件：[docker/.env](/home/asimov/repos/bili-search/docker/.env)
- 示例环境文件：[docker/.env.example](/home/asimov/repos/bili-search/docker/.env.example)

默认按 `Asia/Shanghai` 时区输出时间。

首次配置时，可以将 [docker/.env.example](/home/asimov/repos/bili-search/docker/.env.example) 的内容复制到 [docker/.env](/home/asimov/repos/bili-search/docker/.env) 再按需修改；敏感配置请单独填写到本地 `configs/secrets.json`。

## 构建基础镜像

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
bsdk down -p 21001
```

`bsdk ps` 会列出当前机器上的 bili-search 容器，包括端口、状态、启动时间和运行时长；`--all` 会额外显示已退出容器。对于 `stop`、`down`、`restart`、`status`、`logs` 这类面向已启动实例的命令，端口号已经足够唯一定位目标实例，其他运行参数不再需要重复输入。

## 渲染 Compose 配置

```bash
bsdk config -p 21001 -ei bili_videos_dev6 -ev elastic_dev -lc gpt
```