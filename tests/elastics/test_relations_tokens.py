from elastics.relations.tokens import sanitize_related_token_options
from elastics.relations.tokens import sanitize_related_token_result


def test_sanitize_related_token_result_strips_boilerplate_suffix_and_blocked_markers():
    result = sanitize_related_token_result(
        "【一只小雪莉ovo】寄明月~ 点点关注不错过 持续更新系列中",
        {
            "options": [
                {
                    "text": "超自然行动组创作激励计划 点点关注不错过 持续更新系列中",
                    "type": "doc_cooccurrence",
                    "score": 816.6,
                    "doc_freq": 8,
                },
                {
                    "text": "__pyf__guan|关注不错过",
                    "type": "prefix",
                    "score": 583.8,
                    "doc_freq": 11,
                },
            ]
        },
    )

    assert result["options"] == [
        {
            "text": "超自然行动组创作激励计划",
            "type": "doc_cooccurrence",
            "score": 816.6,
            "doc_freq": 8,
        }
    ]


def test_sanitize_related_token_options_filters_short_prefix_noise_for_long_query():
    options = sanitize_related_token_options(
        "【一只小雪莉ovo】寄明月~ 点点关注不错过 持续更新系列中",
        [
            {"text": "一只", "type": "prefix", "doc_freq": 193867, "score": 1271.5},
            {"text": "一只小", "type": "prefix", "doc_freq": 25159, "score": 697.7},
            {"text": "一只猫", "type": "prefix", "doc_freq": 2986, "score": 332.2},
            {"text": "寄明月 翻唱", "type": "rewrite", "doc_freq": 9, "score": 220.0},
        ],
    )

    assert options == [
        {"text": "寄明月 翻唱", "type": "rewrite", "doc_freq": 9, "score": 220.0}
    ]


def test_sanitize_related_token_options_filters_short_digit_prefix_noise():
    options = sanitize_related_token_options(
        "月栖乐序 高能音乐挑战赛",
        [
            {"text": "月1", "type": "prefix", "doc_freq": 285925, "score": 1051.5},
            {"text": "月2", "type": "prefix", "doc_freq": 214521, "score": 843.8},
            {
                "text": "全能音乐挑战赛",
                "type": "cooccurrence",
                "doc_freq": 465,
                "score": 708.3,
            },
        ],
    )

    assert options == [
        {
            "text": "全能音乐挑战赛",
            "type": "cooccurrence",
            "doc_freq": 465,
            "score": 708.3,
        }
    ]


def test_sanitize_related_token_options_filters_low_freq_source_echo_noise():
    options = sanitize_related_token_options(
        "月栖乐序 高能音乐挑战赛",
        [
            {
                "text": "月栖乐序 音乐现场",
                "type": "doc_cooccurrence",
                "doc_freq": 8,
                "score": 816.6,
            },
            {
                "text": "月栖乐序 演唱会",
                "type": "doc_cooccurrence",
                "doc_freq": 8,
                "score": 690.6,
            },
            {
                "text": "全能音乐挑战赛",
                "type": "cooccurrence",
                "doc_freq": 465,
                "score": 708.3,
            },
        ],
    )

    assert options == [
        {
            "text": "全能音乐挑战赛",
            "type": "cooccurrence",
            "doc_freq": 465,
            "score": 708.3,
        }
    ]


def test_sanitize_related_token_options_keeps_tail_already_present_in_source_query():
    options = sanitize_related_token_options(
        "周杰伦 演唱会",
        [
            {
                "text": "周杰伦 演唱会",
                "type": "doc_cooccurrence",
                "doc_freq": 8,
                "score": 420.0,
            }
        ],
    )

    assert options == [
        {
            "text": "周杰伦 演唱会",
            "type": "doc_cooccurrence",
            "doc_freq": 8,
            "score": 420.0,
        }
    ]
