from tclogger import logger
from typing import Union

from converters.query.pinyin import ChinesePinyinizer
from converters.highlight.merge import HighlightMerger


class PinyinHighlighter:
    def __init__(self):
        self.pinyinizer = ChinesePinyinizer()

    def calc_pinyin_offsets(self, pinyins: list[str]):
        # Example:
        #   pinyins = ["zai", "a", "li", "ba", "ba"]
        # then pinyin_offsets are:
        #   [(0,3), (3,4), (4,6), (6,8), (8,10)]
        pinyin_offsets = []
        start_offset = 0
        end_offset = 0
        for pinyin in pinyins:
            end_offset = start_offset + len(pinyin)
            pinyin_offsets.append((start_offset, end_offset))
            start_offset = end_offset
        return pinyin_offsets

    def get_matched_indices(
        self, query_pinyin: str, text_pinyins: list[str]
    ) -> list[int]:
        # Example:
        #   query_pinyin = "ali"
        #   text_pinyins = ["zai", "a", "li", "ba", "ba"]
        # then matched indices are: [1, 2]
        matched_indices = []
        text_pinyin_offsets = self.calc_pinyin_offsets(text_pinyins)
        text_pinyin_start_offsets = [
            start_offset for (start_offset, end_offset) in text_pinyin_offsets
        ]

        text_pinyin_str = "".join(text_pinyins)
        matched_start_index = text_pinyin_str.find(query_pinyin)
        matched_end_index = matched_start_index + len(query_pinyin)

        if (
            matched_start_index >= 0
            and matched_start_index in text_pinyin_start_offsets
        ):
            logger.mesg(f"Matched index: {matched_start_index}, {matched_end_index}")

            for i, (start_offset, end_offset) in enumerate(text_pinyin_offsets):
                if (
                    start_offset >= matched_start_index
                    and start_offset <= matched_end_index - 1
                    and text_pinyins[i]
                ):
                    matched_indices.append(i)

        return matched_indices

    def highlight_keyword(
        self, keyword: str, text: str, tag: str = "em", verbose: bool = False
    ) -> Union[str, None]:
        if not keyword:
            return None

        logger.enter_quiet(not verbose)

        text_segs = self.pinyinizer.text_to_segs(text)
        keyword_pinyin = self.pinyinizer.text_to_pinyin_str(keyword)
        text_pinyins = self.pinyinizer.text_to_pinyin_segs(text)

        logger.mesg(f"Keyword : {keyword}")
        logger.mesg(f"Text    : {text}")
        logger.mesg(f"Segs    : {text_segs}")
        logger.mesg(f"Keyword Pinyin : {keyword_pinyin}")
        logger.mesg(f"Text Pinyin    : {text_pinyins}")

        matched_indices = self.get_matched_indices(keyword_pinyin, text_pinyins)
        if not matched_indices:
            logger.exit_quiet(not verbose)
            return None

        matched_index_start = matched_indices[0]
        matched_index_end = matched_indices[-1]
        logger.mesg(f"Matched indices: {matched_indices}")

        matched_segs = [text_segs[i] for i in matched_indices]
        logger.mesg(f"Matched segs: {matched_segs}")

        highlighted_text_segs = text_segs[matched_index_start : matched_index_end + 1]
        highlighted_text = f"<{tag}>" + "".join(highlighted_text_segs) + f"</{tag}>"
        combined_highlighted_text = (
            "".join(text_segs[:matched_index_start])
            + highlighted_text
            + "".join(text_segs[matched_index_end + 1 :])
        )
        logger.mesg(f"Highlighted text: {combined_highlighted_text}")

        logger.exit_quiet(not verbose)

        return combined_highlighted_text

    def highlight(self, query: str, text: str, tag: str = "em", verbose: bool = False):
        keywords = query.split()
        highlighted_texts = []
        for keyword in keywords:
            highlighted_text = self.highlight_keyword(
                keyword, text, tag=tag, verbose=verbose
            )
            if highlighted_text:
                highlighted_texts.append(highlighted_text)
        if highlighted_texts:
            merger = HighlightMerger()
            res_text = merger.merge(text, highlighted_texts, tag=tag)
        else:
            res_text = None
        return res_text
