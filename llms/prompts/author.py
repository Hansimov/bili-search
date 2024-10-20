CHECK_AUTHOR_TOOL_DESC = """[TOOL_DESC] `check_author`:
- DESCRIPTION: 根据用户输入的语句，返回数据库的检索结果，用于判断用户输入语句的意图：
    (a) 想搜索该语句中的关键词文本; (b) 想搜索和关键词相近的昵称对应的视频作者。
- OUTPUT:
    ```json
    {
        "query": <str>,
        "highlighted_keywords": <list[dict]>, # key: keyword, value: count
        "related_authors": <list[dict]> # key: author name, value: {"uid": <int>, "count": <int>}, and if author name is highlighted by query, the "highlighted" is True
    }
    ```
[/TOOL_DESC]
"""

CHECK_AUTHOR_TOOL_EXAMPLE = """[TOOL_EXAMPLE] `check_author`:
[/TOOL_EXAMPLE]
"""
