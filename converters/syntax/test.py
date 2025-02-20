date_queries = [
    "日期=1d",
    "dt= =1小时",
    "d< = 2wk",
    "rq = 2024",
    "d=2024-01/01",
    "d==[2024, 2025-01]",
    "d==[,2024, 3d,,）",
    "d=past 3 day",
    "d=this_天",
    "d=this_h",
    "d=[last_d,]",
]
user_queries = [
    "u=影视飓风",
    "u==咬人猫=",
    "用户!==飓多多StormCrew",
    "@！-LKs-,红警HBK08，",
]
uid_queries = [
    "uid>1234",
    "uid=123,456,789",
    "mid! =[123,456]",
]
stat_queries = [
    "view<1000",
    "v>10k",
    ":播放=[1万,10w)",
]
region_queries = [
    "r=动画",
    "fenqu=影视,动画,音乐",
    "rid=1,24,153",
    "r- =影视,动画,153",
]
word_queries = [
    "k=你好",
    'k="世界,你好"',
    "k!=[你好，世界]",
    "k-=[你好,世界]",
    'k=["你好,世界","再见，故乡"]',
    "k=3-0",
]

queries = [
    # *date_queries,
    # *user_queries,
    # *uid_queries,
    # *stat_queries,
    # *region_queries,
    *word_queries,
]
