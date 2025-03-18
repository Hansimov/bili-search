from tclogger import logger, dict_to_str
from converters.dsl.rewrite import DslExprRewriter

date_queries = [
    "日期=1d",
    "dt= =1小时",
    "d< = 2wk",
    "rq = 2024",
    "d=2024-01/01",
    "d==[2024, 2025-01]",
    "d==[,2024, 3d,,）",
    "d=this_天",
    "d=this_h",
    "d=[last_d,]",
]
user_queries = [
    "u=影视飓风",
    "u==咬人猫=",
    "用户 ! = ,飓多多StormCrew,何同学，影视飓风",
    "@！-LKs-,  ，红警HBK08，，红警月亮3,,",
]
uid_queries = [
    "uid>1234",
    "uid=123,456,789",
    "mid! =[123,456]",
]
stat_queries = [
    "view<1000",
    ":v>=10k",
    ":点赞 = = 【1k,10k]",
    "： 播放 <= [ 1万,10w )",
    ": dz 》 = = [ 1万,10w ）",
]
region_queries = [
    "r = 动画",
    "fenqu=影视,动画,音乐",
    "rid ! = 1,24, 153",
    "r- = =影视,动画,153",
]
word_queries = [
    "k=你好",
    'k="世界,你好"',
    "k!=[你好，世界]",
    "k-=[你好,世界]",
    'k=["你好,世界","再见，故乡"]',
    "k=3-0",
    '你好 世界 "再见，故乡"',
    '"你好，故乡"',
    '“你好，"故乡"”',
]
bool_queries = [
    "你好 这是 世界",
    "hello && world",
    "hello | | world & & nothing",
    "(hello || world)",
    "(hello || world) 你好",
    "(hello || world) && nothing",
    "find nothing && ((hello | world) && anything)",
    "(find nothing) || ((hello | world) && anything)",
    "(hello world) (find nothing) (((",
    "hello || world || boy",
    "( ( find nothing ) )",
    "( ( find nothing",
    "你好 这是 世界 ()",
    "( ( (",
    "你好 这是 (( 世界 (",
]

comp_queries = [
    "影视飓风 v>10k :coin>=25 u=,飓多多StormCrew, 亿点点不一样 风光摄影",
    "影视飓风 v>10k :coin>=25 u=,飓多多 StormCrew, 亿点点不一样 ,, 影视飓风",
    # "影视飓风 v>10k :coin>=25 u=[,) , 何同学",
    "(影视飓风 || 飓多多 || TIM 李四维 && 青青 && k-=LKS) (v>=1w || :coin>=25)",
    # "(影视飓风 || 飓多多 || TIM )",
    ":date=2024-01 :view>=1w",
    ":date=2024-01-01 yingshi",
    "《影视飓风》 :date=2024-01-01 :view>=1w",
]

rewrite_queries = [
    "yingshi ju",
    "hongjing (08 | 月亮3)",
    "hongjing 08 2024 :view>=1w",
]


queries = [
    # *date_queries,
    # *user_queries,
    # *uid_queries,
    # *stat_queries,
    # *region_queries,
    # *word_queries,
    # *bool_queries,
    # *comp_queries,
    *rewrite_queries,
]


def test_rewriter():
    rewriter = DslExprRewriter()
    for query in rewrite_queries:
        query_info = rewriter.get_query_info(query)
        logger.mesg(dict_to_str(query_info), indent=2)


if __name__ == "__main__":
    test_rewriter()

    # python -m converters.dsl.test
