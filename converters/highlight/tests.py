from tclogger import logger
from converters.highlight.pinyin import PinyinHighlighter
from converters.highlight.count import HighlightsCounter


def test_pinyin_highlighter():
    highlighter = PinyinHighlighter()
    # query = "vlog alibaba"
    # text = "【Vlog】在阿里巴巴达摩院工作是什么样的体验？"
    # query = "ali"
    # text = "给百大UP主加上特效，这可太炸裂了！【百大UP主颁奖】"
    query = "影视飓风 xiangsu"
    text = "【影视飓风】4万块的1亿像素中画幅？"
    res_text = highlighter.highlight(query, text, tag="hit", verbose=True)
    logger.mesg(f"Merged highlighted text:", end=" ")
    logger.success(res_text)
    counter = HighlightsCounter()
    count_res = counter.extract_highlighted_keywords(res_text)
    logger.success(count_res)


if __name__ == "__main__":
    test_pinyin_highlighter()
    # python -m converters.highlight.tests
