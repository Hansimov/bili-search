test_date_strs = [
    "2014",
    "2014-02",
    "2014/08/12",
    "08/12",
    "2014-08-12.12",
    "this_year",
    "this_month",
    "this_week",
    "this_day",
    "this_hour",
    "last_year",
    "last_month",
    "last_week",
    "last_day",
    "last_hour",
    "1 year",
    "2 months",
    "3 weeks",
    "4 days",
    "5 hours",
    # "past year",
    # "past.month",
    # "past-week",
    # "过去一天",
    # "past_hour",
]


test_date_list_strs = [
    "(3d, 1d]",
    "[1d, 3d)",
    "[2014, ]",
    "[2014, 2015]",
    "[2014/08/12, 2015/08)",
    "(02/16, 02/25]",
    "[2/16, 2/25)",
    "[6h, 12h]",
]


test_user_strs = [
    "@=咬人猫=",
    "@!-LKs-",
    '@["=咬人猫=", -LKs-, 影视飓风]',
    "user-==咬人猫=",
    "u==咬人猫=",
    "作者!=[=咬人猫=, -LKs-, 影视飓风]",
    "用户-==咬人猫=, -LKs-, 红警HBK08",
    "@![=咬人猫=, -LKs-, 影视飓风]",
]


test_umid_strs = [
    "uid=946974",
    "uid=[946974, 1629347259]",
    "uid!=946974",
    "uid!=[946974, 1629347259]",
]

test_text_strs = [
    # "(A B) || C D || E F? G",
    # "红警HBK08",
    # "k=红警HBK08",
    # "k-=红警HBK08",
    "红警 (小块地 || 冰天雪地 1v1? 2v2? || 冰天雪地 4v4? k!=[3v3,5v5]) d=1d",
    # "deepseek !=[满血]"
    # "红警 (小块地 && 冰天雪地)",
    # "红警 (小块地 冰天雪地)",
    # "k=[小块地,冰天雪地]",
    # '红警 !=["小块地",冰天雪地]',
    # "红警 (@![红警HBK08,红警月亮3])",
]

test_comb_strs = [
    "(小块地 || 冰天) @[红警HBK08,红警月亮3]",
]
