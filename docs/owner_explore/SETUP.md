# Owner Search Setup

## 目标

本目录记录 owner 搜索在 `bili-search` 仓库内可直接落地和验证的部分：

- `elastics/owners` 的独立检索逻辑
- owner 画像并入视频搜索结果里的作者分组
- LLM 工具链中的 `check_author` / `search_owners`
- `tests/owner_search` 中的单元测试和轻量集成测试

## 环境前提

1. 安装 `requirements.txt` 中的 Python 依赖。
2. 配置可用的 Elasticsearch 环境变量与密钥，`configs/secrets.json` 中需要至少包含视频索引对应环境。
3. 如果要连真实 owners 索引，需要在应用环境里提供 `elastic_owners_index`。
4. 如果要创建 owners 索引，先在 `bili-scraper` 侧准备好 mapping：
   `converters/elastic/owner_index_settings_v1.py`
5. 如果 owners 索引使用 `chinese_analyzer`，需要先在 ES 上安装并加载 `es-tok` 插件。
6. `bili-scraper` 侧现已具备 owners 构建入口：
   - `workers.elastic_owners.commander`
   - `actions.elastic_owners_indexer`

## 当前仓库内已接通的点

1. `OwnerSearcher.search()`
   现在会做 name/domain 双路召回，并在 `sort_by=relevance` 时做 owner 级融合排序。
2. `VideoExplorer.group_hits_by_owner()`
   如果存在 `owner_searcher`，会额外读取 owners 索引中的画像字段并补进 author list。
3. `ChatHandler`
   现在支持解析 `<search_owners .../>`，不再只是 `ToolExecutor` 内部实现。

## 当前未在本仓库直接完成的部分

1. 从 MongoDB 全量/增量构建 owners 索引。
2. 跨仓库运行 `bili-scraper` 或 `bili-search-algo` 的真实生产脚本。
3. 真实 ES / Mongo / embedding 服务的长流程联调。

其中第 1 项在本轮已经把 `bili-scraper` 的命令入口和 action 补齐，但仍需要在真实环境执行。
