from tclogger import logger
from converters.query.pinyin import ChinesePinyinizer
from converters.highlight.pinyin import PinyinHighlighter
from converters.highlight.count import HighlightsCounter


def test_pinyinize():
    pinyinizer = ChinesePinyinizer()
    text = "chang城"
    res = pinyinizer.text_to_pinyin_combinations(text)
    logger.success(res)


def test_highlights_counter():
    highlighter = PinyinHighlighter()
    # query = "vlog alibaba"
    # text = "【Vlog】在阿里巴巴达摩院工作是什么样的体验？"
    # query = "ali"
    # text = "给百大UP主加上特效，这可太炸裂了！【百大UP主颁奖】"
    query = "影视飓风 xiangsu"
    text = "【影 视 飓 风】4万块的1亿像素中画幅？"
    res_text = highlighter.highlight(query, text, tag="hit", verbose=True)
    logger.mesg(f"Merged highlighted text:", end=" ")
    logger.success(res_text)
    counter = HighlightsCounter()
    count_res = counter.extract_highlighted_keywords(res_text)
    logger.success(count_res)


def test_pinyin_highlighter():
    query_text_list = [
        ("changcheng", "万里长城永不倒"),
        ("影视j", "影视飓风"),
        ("线性", "【熟肉】线 性代数的本质 - 02 - 线性组合、张成的空间与基"),
    ]
    for query, text in query_text_list:
        highlighter = PinyinHighlighter()
        res_text = highlighter.highlight(query, text, tag="hit", verbose=True)
        logger.success(res_text)


if __name__ == "__main__":
    # test_pinyinize()
    # test_highlights_counter()
    test_pinyin_highlighter()

    # python -m converters.highlight.tests
