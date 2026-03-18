# 关系能力说明

## 范围

`search_app` 的关系能力接口，本质上是对 ES-TOK 标准 REST 接口 `/{index}/_es_tok/*` 的一层轻包装。

当前已实现的路由有：

- `/related_tokens_by_tokens`
- `/related_owners_by_tokens`
- `/related_videos_by_videos`
- `/related_owners_by_videos`
- `/related_videos_by_owners`
- `/related_owners_by_owners`

## 实际使用建议

- 当用户是按话题找创作者时，优先使用 `related_owners_by_tokens`，比用 `suggest` 去猜作者更合适。
- 当用户要做明确的作者时间线或列表查询时，仍然应直接使用 `search_videos`，例如 `:user=影视飓风 :date<=15d`。
- 只有在你已经拿到可靠种子，例如 `bvids` 或 `mids` 时，才适合调用图关系接口。
- Google 搜索只应保留在 `llms` 侧作为外部事实补充，不属于搜索服务本身的 HTTP 能力合同。

## 当前结论

- 参考 `es-tok/docs/01_API.md` 后，现有标准接口形态已经足够稳定，可以直接封装，不需要再加兼容别名。
- XML 工具命令解析必须容忍带引号属性中的 `>`，例如 `:view>=1w`，否则多工具命令会被错误截断；当前处理器已经改成引号感知的正则解析。
- 关系接口返回值不应直接透传底层插件响应，而应整理成更紧凑、更稳定、面向 LLM 的 owners/videos/tokens 结构。
- 在“找某类创作者”这类意图中，`related_owners_by_tokens` 已经可以替代早期的 `check_author` 启发式；如果用户明确要查某个作者的时间线，仍应直接走带 `:user=` 的 `search_videos`。