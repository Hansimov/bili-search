CHECK_AUTHOR_TOOL_DESC = """[TOOL_DESC] `check_author`:
- DESCRIPTION: 根据用户输入的语句，和数据库的检索结果，来判断用户是下列哪种情况：
    (a) 想搜索该语句中的关键词文本; (b) 想搜索和关键词相近的昵称对应的视频作者。
- INPUT:
    ```json
    {
        "query": <str>,
        "hits": [<dict>] # 每个元素是一个视频的信息，包括 bvid, owner, title 等字段
    }
    ```
- OUTPUT:
(a) 想搜索该语句中的关键词文本:
    ```json
    {
        "intension": "search_text"
    }
    ```
(b) 想搜索和关键词相近的昵称对应的视频作者:
    ```json
    {
        "intension": "search_author",
        "author": [{"name": <owner.name>, "uid": <owner.mid>}]
    }
    ```
[/TOOL_DESC]
"""

CHECK_AUTHOR_TOOL_EXAMPLE = """[TOOL_EXAMPLE] `check_author`:
- EXAMPLE 1:

USER:
```json
{
    "query" : "影视飓风",
    "hits"  : [{
        "bvid"        : "BV1Yf421e72c",
        "owner"       : {
            "mid"  : 946974,
            "name" : "影视飓风"
        },
        "title"       : "内部绝密视频首次公开？影视飓风900万粉丝Q&A！"
    }, {
        "bvid"        : "BV1Cx2JYSEBb",
        "owner"       : {
            "mid"  : 946974,
            "name" : "影视飓风"
        },
        "title"       : "清晰度不如4年前！视频变糊是你的错觉吗？"
    }, {
        "bvid"        : "BV1Ec411z7j2",
        "owner"       : {
            "mid"  : 946974,
            "name" : "影视飓风"
        },
        "title"       : "我们买了一颗国产卫星。",
    }]
}
```

ASSISTANT: [think] 从数据库检索的结果可以发现，搜索到的前3个视频的作者都是“影视飓风”，并且用户的输入对该作者昵称具有强指向性，因此可以判断用户想要搜索的是该作者。[/think]
[output]
```json
{
    "intension": "search_author",
    "author": {"name": "影视飓风", "uid": 946974}
}
```
[/output]

- EXAMPLE 2:

USER:
```json
{
    "query" : "黑神话",
    "hits"  : [{
        "bvid"  : "BV1oH4y1c7Kk",
        "owner" : {
            "mid"  : 642389251,
            "name" : "黑神话悟空"
        },
        "title" : "《黑神话：悟空》最终预告 | 8月20日，重走西游"
    }, {
        "bvid"  : "BV114421U75X",
        "owner" : {
            "mid"  : 585267,
            "name" : "-纯黑-"
        },
        "title" : "纯黑《黑神话：悟空》全程无伤攻略解说 第一期"
    }, {
        "bvid"  : "BV1Ti421a7dv",
        "owner" : {
            "mid"  : 652239032,
            "name" : "IGN中国"
        },
        "title" : "【IGN】10分，《黑神话：悟空》评测：踏平坎坷成大道"
    }, {
        "bvid"  : "BV1AE4m1d7XT",
        "owner" : {
            "mid"  : 360046931,
            "name" : "11的游戏世界"
        },
        "title" : "【黑神话：悟空】4K 最高画质 全收集 全结局 全剧情流程通关攻略 西游背景动作角色扮演游戏 首款国产3A大作【完结】"
    }]
}
```

ASSISTANT: [think] 从数据库检索的结果可以发现，搜索到的前4个视频的作者并不相同，并且用户的输入对该作者昵称没有强指向性，因此可以判断用户想要搜索的是该关键词文本。[/think]
[output]
```json
{
    "intension": "search_text"
}
```
[/output]

[/TOOL_EXAMPLE]
"""
