from collections import defaultdict
from tclogger import logger, logstr, dict_to_str, brk

from converters.dsl.node import DslExprNode
from converters.dsl.rewrite import DslExprRewriter
from converters.dsl.elastic import DslExprToElasticConverter

date_queries = [
    ["date=1d", ("date_expr", 1)],
    ["dt= =1h", ("date_expr", 1)],
    ["d< = 2wk", ("date_expr", 1)],
    ["d= 2024", ("date_expr", 1)],
    ["d=2024-01/01", ("date_expr", 1)],
    ["dt==[2024, 2025-01]", ("date_expr", 1)],
    ["date==[,2024, 3d,,)", ("date_expr", 1)],
    ["d=this_d", ("date_expr", 1)],
    ["d=this_h", ("date_expr", 1)],
    ["d=[last_d,]", ("date_expr", 1)],
]
user_queries = [
    ["u=影视飓风", ("user_expr", 1)],
    ["u==咬人猫=", ("user_expr", 1)],
    ["user!=(飓多多StormCrew,何同学，影视飓风)", ("user_expr", 1)],
    ['@!["-LKs-",  ，红警HBK08，，红警月亮3,,]', ("user_expr", 1)],
]
uid_queries = [
    ["uid=1234", ("uid_expr", 1)],
    ["uid=[123,456,789)", ("uid_expr", 1)],
    ["mid! =[123,456]", ("uid_expr", 1)],
]
stat_queries = [
    ["bf<1000", ("stat_expr", 1)],
    [":v>=10k", ("stat_expr", 1)],
    [":dz= = [1k,10k]", ("stat_expr", 1)],
    [":vw <= [ 1w,10w )", ("stat_expr", 1)],
    [":lk>= = [ 1w,10w )", ("stat_expr", 1)],
]
dura_queries = [
    ["dura>30", ("dura_expr", 1)],
    ["t<=1h", ("dura_expr", 1)],
    ["t=[30,1m30s]", ("dura_expr", 1)],
    ["d=1d t>5h v>1w", (("date_expr", 1), ("dura_expr", 1), ("stat_expr", 1))],
]
region_queries = [
    ["rg = 动画", ("region_expr", 1)],
    ["region=(影视,动画,音乐)", ("region_expr", 1)],
    ["rid ! = [1,24, 153]", ("region_expr", 1)],
    ["rg- = =(影视,动画,153]", ("region_expr", 1)],
]
word_queries = [
    ["k=你好", ("word_expr", 1)],
    ['k="世界,你好"', ("word_expr", 1)],
    ["k!=[你好，世界]", ("word_expr", 1)],
    ["k-=[你好,世界]", ("word_expr", 1)],
    ['k=["你好,世界","再见，故乡"]', ("word_expr", 1)],
    ["k=3-0", ("word_expr", 1)],
    ['你好 世界 "再见，故乡"', ("word_expr", 3)],
    ['"你好，故乡"', ("word_expr", 1)],
    ['“你好，"故乡"”', ("word_expr", 1)],
]
bool_queries = [
    ["你好 这是 世界 u=hello", [("user_expr", 1), ("word_expr", 3)], [("co", 1)]],
    ["hello && world", [("word_expr", 2)], [("and", 1)]],
    ["hello | | world & & nothing", [("word_expr", 3)], [("and", 1), ("or", 1)]],
    ["(hello || world)", [("word_expr", 2)], [("or", 1), ("pa", 1)]],
    ["(hello || world) 你好", [("word_expr", 3)], [("co", 1), ("or", 1), ("pa", 1)]],
    [
        "(hello || world) && nothing",
        [("word_expr", 3)],
        [("and", 1), ("or", 1), ("pa", 1)],
    ],
    [
        "find nothing && ((hello | world) && anything)",
        [("word_expr", 5)],
        [("and", 2), ("co", 1), ("or", 1), ("pa", 2)],
    ],
    [
        "(find nothing) || ((hello | world) && anything)",
        [("word_expr", 5)],
        [("or", 2), ("co", 1), ("and", 1), ("pa", 3)],
    ],
    [
        "(hello world) (find nothing) (((",
        [("word_expr", 4)],
        [("co", 3), ("pa", 2)],
    ],
    ["hello || world || boy", [("word_expr", 3)], [("or", 2)]],
    ["( ( find nothing ) )", [("word_expr", 2)], [("co", 1), ("pa", 2)]],
    ["你好 这是 世界 ()", [("word_expr", 3)], [("co", 1), ("pa", 1)]],
    ["( ( (", [], []],
    ["( ( find nothing", [("word_expr", 2)], [("co", 1)]],  # FLAT PASSED
    ["你好 这是 (( 世界 (", [("word_expr", 3)], [("co", 2)]],  # FLAT PASSED
]

comp_queries = [
    [
        "影视飓风 v>10k :coin>=25 u=[飓多多StormCrew, 亿点点不一样] 风光摄影",
        [("user_expr", 1), ("word_expr", 2), ("stat_expr", 2)],
        [("co", 1)],
    ],
    ['+"影视飓风" !“李四维”', [("word_expr", 2)], [("co", 1)]],
    [
        "影视飓风 v>10k :coin>=25 u=[飓多多StormCrew, 亿点点不一样 ,, 影视飓风]",
        [("user_expr", 1), ("word_expr", 1), ("stat_expr", 2)],
        [("co", 1)],
    ],
    [
        "影视飓风 v>10k :coin>=25 u=[,) , 何同学",
        [("user_expr", 1), ("word_expr", 2), ("stat_expr", 2)],
        [("co", 1)],
    ],
    [
        "(影视飓风 || 飓多多 || TIM 李四维 && 青青 && k-=LKS) (v>=1w || :coin>=25)",
        [("word_expr", 6), ("stat_expr", 2)],
        [("co", 2), ("and", 2), ("or", 3), ("pa", 2)],
    ],
    [
        "(影视飓风 || 飓多多 || TIM )",
        [("word_expr", 3)],
        [("or", 2), ("pa", 1)],
    ],
    [
        ":date=2024-01 :view>=1w",
        [("date_expr", 1), ("stat_expr", 1)],
        [("co", 1)],
    ],
    [
        ":date=2024-01-01 yingshi",
        [("date_expr", 1), ("word_expr", 1)],
        [("co", 1)],
    ],
    [
        "《影视飓风》 :date=2024-01/01 :view>=1w ",
        [("date_expr", 1), ("word_expr", 1), ("stat_expr", 1)],
        [("co", 1)],
    ],
    [
        '(+“雷军” || +"小米") (+"影视飓风" || +"tim")',
        [("word_expr", 4)],
        [("co", 1), ("or", 2), ("pa", 2)],
    ],
    ['"deep learning"~', [("word_expr", 1)], []],
]

rewrite_queries = [
    "yingshi ju",
    "hongjing (08 | 月亮3)",
    "hongjing 08 2024 :view>=1w",
]


queries_of_atoms = [
    *date_queries,
    *user_queries,
    *uid_queries,
    *stat_queries,
    *dura_queries,
    *region_queries,
    *word_queries,
]
queries_of_bools = [
    # *bool_queries,
    # *comp_queries,
    # *rewrite_queries,
]


def test_rewriter():
    rewriter = DslExprRewriter()
    for query in rewrite_queries:
        query_info = rewriter.get_query_info(query)
        logger.mesg(dict_to_str(query_info), indent=2)


def list_to_tuple(lst: list) -> tuple:
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0]
    else:
        lst = sorted(lst, key=lambda x: x[0])
        return tuple(lst)


def get_atoms_info(node: DslExprNode) -> tuple:
    atom_expr_keys = node.get_all_atom_childs_expr_keys()
    atoms_info = defaultdict(int)
    for key in atom_expr_keys:
        atoms_info[key] += 1
    atoms_info = sorted(atoms_info.items(), key=lambda x: x[0])
    return list_to_tuple(atoms_info)


def get_bools_info(node: DslExprNode) -> tuple:
    bool_expr_keys = node.get_all_bool_childs_keys()
    bools_info = defaultdict(int)
    for key in bool_expr_keys:
        bools_info[key] += 1
    bools_info = sorted(bools_info.items(), key=lambda x: x[0])
    return list_to_tuple(bools_info)


def test_expr_tree():
    converter = DslExprToElasticConverter()
    okay_mark = logstr.okay("✓")
    fail_mark = logstr.fail("×")

    for query, correct_atoms_info in queries_of_atoms:
        expr_tree = converter.construct_expr_tree(query)
        atoms_info = get_atoms_info(expr_tree)
        if atoms_info == correct_atoms_info:
            logger.mesg(f"{okay_mark} {query}:", end=" ")
            logger.okay(f"{atoms_info}")
            # logger.okay(dict_to_str(expr_tree.yaml()))
        else:
            logger.fail(f"{fail_mark} {query}:")
            logger.okay(f"  * {correct_atoms_info}")
            logger.fail(f"  * {atoms_info}")
            logger.fail(dict_to_str(expr_tree.yaml()))
            break
    for query, correct_atoms_info, correct_bools_info in queries_of_bools:
        correct_atoms_info = list_to_tuple(correct_atoms_info)
        correct_bools_info = list_to_tuple(correct_bools_info)
        expr_tree = converter.construct_expr_tree(query)
        atoms_info = get_atoms_info(expr_tree)
        bools_info = get_bools_info(expr_tree)
        flat_tree = converter.flatter.flatten(expr_tree)
        atoms_finfo = get_atoms_info(flat_tree)
        bools_finfo = get_bools_info(flat_tree)
        if atoms_info == correct_atoms_info and bools_info == correct_bools_info:
            logger.mesg(f"{okay_mark} {brk(query)}:", end=" ")
            logger.okay(f"{atoms_info}, {bools_info}")
        elif atoms_finfo == correct_atoms_info and bools_finfo == correct_bools_info:
            logger.mesg(f"{okay_mark} {brk(query)}:", end=" ")
            logger.hint(f"{atoms_finfo}, {bools_finfo}")
        else:
            logger.fail(f"{fail_mark} {brk(query)}:")
            logger.okay(f"{okay_mark} {correct_atoms_info}")
            if atoms_info != correct_atoms_info:
                logger.fail(f"{fail_mark} {atoms_info}")
            logger.okay(f"{okay_mark} {correct_bools_info}")
            if bools_info != correct_bools_info:
                logger.fail(f"{fail_mark} {bools_info}")
            logger.fail(dict_to_str(expr_tree.yaml()))
            logger.fail(flat_tree.yaml())
            elastic_dict = converter.expr_tree_to_dict(expr_tree)
            logger.fail(dict_to_str(elastic_dict))
            break


if __name__ == "__main__":
    # test_rewriter()
    test_expr_tree()

    # python -m converters.dsl.test
