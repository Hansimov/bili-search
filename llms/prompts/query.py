from llms.prompts.intro import now_ymd, yesterday_ymd

QUERY_SYNTAX = f"""[SYNTAX] 搜索引擎 query 的语法：
- `(<关键词>)* (:<过滤器><操作符><值>)*`
也就是说，关键词之间用空格分隔，过滤器以冒号`:`起始，后接操作符和值。
- NOTE: `(...)*` 表示可以出现0次或多次，`(...)+` 表示至少出现1次，`(...)?` 表示不出现或出现1次。

合法的过滤器名和操作符如下：

1. 数据类型的过滤器：
    - 支持的关键字： `view`(播放/观看数), `like`(点赞/喜欢数), `coin`(投币数), `danmaku`(弹幕数), `reply`(回复/评论数),  `favorite`(收藏/星标数), `share`(分享/转发数)
    - 支持的数据格式：
        - 数字：必须为整数，禁止使用科学计数法
        - 数字+单位：合法的单位有 `k` (千), `w` (万), `m` (百万)
    - 支持的操作符：
        - 单向操作符：`>`, `>=`, `<`, `<=`, `=`
        - 区间操作符：`=[num1,num2]`, `=(num1,num2]`, `=[num1,num2)`, `=(num1,num2)`
            - NOTE: `()`表示开区间，`[]`表示闭区间
    - 例如：`:view>1000`，`:coin>=1000`，`:danmaku<1000`，`:reply=[100, 1k]`, `:view=(1k,10w]`

2. 时间类型的过滤器：
    - 支持的关键字：`date`
    - 支持的日期格式：
        - 具体日期：`YYYY-MM-DD.HH`, `YYYY-MM-DD`, `YYYY-MM`, `YYYY`, `MM-DD`, `MM-DD.HH`
            - NOTE： 对于 `MM-DD` 和 `MM-DD.HH`，只有时间是今年的时候才可以这样表示
        - 数字+单位：合法的单位有 `hour`/`h`(小时), `day`/`d` (天), `week`/`w` (周), `month`/`m` (月), `year`/`y` (年)
    - 支持的操作符：
        - 单向操作符：`>`, `>=`, `<`, `<=`, `=`。例如：
            - `>1d` 表示距现在的day数超过1天，`<=3week` 表示距现在的week数不超过3星期，`=1mon` 表示距现在1个月以内，`<3y` 表示距现在小于3年
            - `=2023` 表示从 `2023-01-01 00:00:00` 到 `2023-12-31 23:59:59`
            - `=<今年的YYYY>` 表示从今年的 `01-01 00:00:00` 到现在
            - `=2022-10` 表示从 `2022-10-01 00:00:00` 到 `2022-10-31 23:59:59`
            - `=2024-03-02` 表示从 `2024-03-02 00:00:00` 到 `2024-03-02 23:59:59`
        - 区间操作符：`=[date1,date2]`, `=(date1,date2]`, `=[date1,date2)`, `=(date1,date2)`。例如：
            - NOTE: `()`表示开区间，`[]`表示闭区间
            - `=[2021,2023]`，表示从 `2021-01-01 00:00:00` 到 `2023-12-31 23:59:59`
            - `=(2021-01,2023]`，表示从 `2021-02-01 00:00:00` 到 `2023-12-31 23:59:59`
            - `=[2021-10-10,2023-03-20)`，表示从 `2021-10-10 00:00:00` 到 `2023-03-19 23:59:59`
            - `=[1d,3d]`, 表示距离现在的day数在1天到3天之间
            - `=(2week,1month]`, 表示距离现在的week大于2周，month小于等于1个月
            - `=[1y,3y)`, 表示距离现在的year数大于等于1年，小于3年

3. 视频作者昵称的过滤器：
    - 支持的关键字：`name`
    - 支持的操作符：`=`
    - 支持的值：（昵称必须完整，切不允许空格）
        - 单个用户名。例如：`:name=影视飓风`
        - 多个用户名，用逗号`,`分开。例如：`:name=影视飓风,飓多多StormCrew,亿点点不一样`

4. 视频作者uid的过滤器：
    - 支持的关键字：`uid`
    - 支持的操作符：`=`
    - 支持的值：（uid必须为数字）
        - 单个uid。例如：`:uid=642389251`
        - 多个uid，用逗号`,`分开。例如：`:uid=642389251,946974,1780480185`
[/SYNTAX]
"""

INSTRUCT_TO_QUERY_TOOL_DESC = f"""[TOOL_DESC] `instruct_to_query`:
- DESCRIPTION: 分析用户的提问或指令，识别和提取出用户意图，然后转换成符合本搜索引擎语法的语句。需结合搜索引擎的 SYNTAX 使用。
[/TOOL_DESC]
"""

INSTRUCT_TO_QUERY_TOOL_EXAMPLE = f"""[TOOL_EXAMPLE] `instruct_to_query`:
- Example 1:
    USER: 影视飓风最近有什么新视频？
    ASSISTANT:
        [think] 影视飓风可能是一个视频作者昵称或者视频系列，应当调用工具 `check_author` 来确认用户的意图是是搜索对应的文本，还是视频作者。 [/think]
        [check_author] `影视飓风` [/check_author]
    USER: `chekc_author` 的返回结果为：
        ```json
        {{
            "intension": "search_author",
            "author": {{
                "name": "影视飓风",
                "uid": 946974
            }}
        }}
        ```
    ASSISTANT:
        [think] 可知用户想搜索是昵称为“影视飓风”的作者，因此需要使用昵称过滤器；同时用户强调了“最近”，所以需要加上日期过滤器，“最近”这个词比较模糊，暂时可以理解为最近15天 [/think]
        [query] `:name=影视飓风 :date<=15d` [/query]
    USER: 能不能再近一点？
    ASSISTANT:
        [think] 看样子用户认为15天内还不够近，所以需要再缩短时间范围，暂时理解为7天内 [/think]
        [query] `:name=影视飓风 :date<=7d` [/query]
    USER: 不必限定“影视飓风”是昵称
    ASSISTANT:
        [think] 好的，不用昵称过滤器，而是只作为普通关键词进行搜索 [/think]
        [query] `影视飓风 :date<=7d` [/query]

- Example 2:
    USER: 推荐一点黑神话的视频
    ASSISTANT:
        [think] 黑神话可能是一个视频作者昵称或者视频系列，应当调用工具 `check_author` 来确认用户的意图是是想搜索对应的文本，还是想搜索视频作者。 [/think]
        [check_author] `影视飓风` [/check_author]
    USER: `chekc_author` 的返回结果为：
        ```json
        {{ "intension": "search_text" }}
        ```
    ASSISTANT:
        [think] 这里的关键词是 `黑神话`，因为本搜索引擎的主体就是视频，所以这里不需要额外加上 `视频` 这个词 [/think]
        [query] `黑神话` [/query]
    USER: 能不能来点热度高的
    ASSISTANT:
        [think] 用户想要热度高的视频，可以理解为播放量高，所以需要加上 `view` 过滤器 [/think]
        [query] `黑神话 :view>=10w` [/query]
    USER: 我希望质量和活跃度也高
    ASSISTANT:
        [think] 质量主要体现在 coin (投币数) 和 favorite (收藏数) 上，活跃度主要体现在 danmaku (弹幕数) 和 reply (评论数) 上，所以需要加上这些过滤器；考虑到不要太复杂，暂时只新增 coin 和 danmaku 的过滤器 [/think]
        [query] `黑神话 :view>=10w :coin>=1k :danmaku>=100` [/query]

- Example 3:
    USER: 红警08今天发了什么视频
    ASSISTANT:
        [think] 红警08可能是一个视频作者昵称或者视频系列，应当调用工具 `check_author` 来确认用户的意图是是想搜索对应的文本，还是想搜索视频作者。 [/think]
        [check_author] `红警 08` [/check_author]
    USER: `chekc_author` 的返回结果为：
        ```json
        {{
            "intension": "search_author",
            "author": {{
                "name": "红警HBK08",
                "uid": 1629347259
            }}
        }}
        ```
    ASSISTANT:
        [think] 可知用户想搜索是昵称为`红警HBK08`的作者，因此需要使用昵称过滤器；同时用户强调了“今天”，所以需要加上日期过滤器，已知今天是 {now_ymd} [/think]
        [query] `:name=红警HBK08 :date={now_ymd}` [/query]
    USER: 那么昨天呢
    ASSISTANT:
        [think] 昨天的日期是 {yesterday_ymd} [/think]
        [query] `:name=红警HBK08 :date={yesterday_ymd}` [/query]
    USER: 我还想加上月亮3的视频
    ASSISTANT:
        [think] 用户还想搜索昵称和“月亮3”相关的作者的视频，所以需要调用 `check_author` 来判断完整的昵称 [/think]
        [check_author] `月亮3` [/check]
    USER: `check_author` 的返回结果为：
        ```json
        {{
            "intension": "search_author",
            "author": {{
                "name": "红警月亮3",
                "uid": 674510452
            }}
        }}
        ```
    ASSISTANT:
        [think] 可知用户还想加上昵称为`红警月亮3`的作者的视频内容，故需要添加进昵称过滤器中 [/think]
        [query] `:name=红警HBK08,红警月亮3 :date={yesterday_ymd}` [/query]
[/TOOL]
"""
