import re
from copy import deepcopy
from tclogger import logger
from typing import Union

from converters.query.pinyin import ChinesePinyinizer
from converters.highlight.merge import HighlightMerger


class PinyinHighlighter:
    def __init__(self):
        self.pinyinizer = ChinesePinyinizer()
        self.letter_pattern = re.compile(r"[a-zA-Z]+")

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
        istart, iend = None, None
        q_pinyin = query_pinyin
        for idx, text_pinyin in enumerate(text_pinyins):
            is_prefix = q_pinyin.startswith(text_pinyin)
            is_suffix = text_pinyin.startswith(q_pinyin)
            if is_suffix and iend is None:
                iend = idx
            if is_prefix:
                if istart is None:
                    istart = idx
                q_pinyin = q_pinyin[len(text_pinyin) :]
            if (
                not is_prefix
                and not is_suffix
                and self.letter_pattern.match(text_pinyin)
            ):
                q_pinyin = query_pinyin
                istart, iend = None, None
            if istart is not None and iend is not None:
                matched_indices.append((istart, iend))
                q_pinyin = query_pinyin
                istart, iend = None, None

        if matched_indices:
            logger.success(f"Matched indices: ({matched_indices})")
        else:
            logger.mesg("No matched index found!")

        return matched_indices

    def highlight_keyword(
        self, keyword: str, text: str, tag: str = "hit", verbose: bool = False
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

        htext_segs = deepcopy(text_segs)
        for istart, iend in matched_indices:
            htext_segs[istart] = f"<{tag}>" + htext_segs[istart]
            htext_segs[iend] = htext_segs[iend] + f"</{tag}>"
        htext_str = "".join(htext_segs)
        logger.mesg(f"Highlighted text: {htext_str}")

        logger.exit_quiet(not verbose)

        return htext_str

    def highlight(
        self,
        keywords: Union[str, list[str]],
        text: str,
        tag: str = "hit",
        verbose: bool = False,
    ):
        if isinstance(keywords, str):
            keywords = [keyword.lower() for keyword in keywords.split()]
        else:
            keywords = [keyword.lower() for keyword in keywords]
        highlighted_texts = []
        for keyword in keywords:
            highlighted_text = self.highlight_keyword(
                keyword, text, tag=tag, verbose=verbose
            )
            if highlighted_text:
                highlighted_texts.append(highlighted_text)
        if highlighted_texts:
            merger = HighlightMerger()
            res_text = merger.extract_and_merge(text, highlighted_texts, tag=tag)[
                "merged"
            ]
        else:
            res_text = None
        return res_text
