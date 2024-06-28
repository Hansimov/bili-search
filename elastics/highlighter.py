from pypinyin import lazy_pinyin
from tclogger import logger
from typing import Union


class PinyinHighlighter:
    def filter_chars_from_pinyin(self, pinyins: list[str]) -> list[str]:
        res = []
        for pinyin in pinyins:
            pinyin_chars = "".join([ch for ch in pinyin if ch.isalnum()]).lower()
            res.append(pinyin_chars)
        return res

    def text_to_pinyins(self, text: str) -> list[str]:
        pinyins = lazy_pinyin(text)
        pinyins = self.filter_chars_from_pinyin(pinyins)
        return pinyins

    def text_to_pinyin_str(self, text: str) -> str:
        return "".join(self.text_to_pinyins(text))

    def calc_pinyin_offsets(self, pinyins: list[str]):
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
        # then the matched indices are [1, 2]
        matched_indices = []
        text_pinyin_offsets = self.calc_pinyin_offsets(text_pinyins)

        text_pinyin_str = "".join(text_pinyins)
        matched_start_index = text_pinyin_str.find(query_pinyin)
        matched_end_index = matched_start_index + len(query_pinyin)

        if matched_start_index >= 0:
            logger.mesg(f"Matched index: {matched_start_index}, {matched_end_index}")

            for i, (start_offset, end_offset) in enumerate(text_pinyin_offsets):
                if (
                    start_offset >= matched_start_index
                    and start_offset <= matched_end_index - 1
                    and text_pinyins[i]
                ):
                    matched_indices.append(i)

        return matched_indices

    def highlight(
        self, query: str, text: str, tag: str = "em", verbose: bool = False
    ) -> Union[str, None]:
        if not query:
            return None

        logger.enter_quiet(not verbose)
        text_segs = list(text)

        query_pinyin = self.text_to_pinyin_str(query)
        text_pinyins = lazy_pinyin(text_segs)
        text_pinyins = self.filter_chars_from_pinyin(text_pinyins)

        logger.mesg(f"Query : {query}")
        logger.mesg(f"Text  : {text}")
        logger.mesg(f"Segs  : {text_segs}")
        logger.mesg(f"Query Pinyin : {query_pinyin}")
        logger.mesg(f"Text Pinyin  : {text_pinyins}")

        matched_indices = self.get_matched_indices(query_pinyin, text_pinyins)
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


if __name__ == "__main__":
    highlighter = PinyinHighlighter()
    query = "vlog"
    text = "【Vlog】在阿里巴巴达摩院工作是什么样的体验？"
    text_highlighted = highlighter.highlight(query, text, verbose=True)
    logger.mesg(f"Highlighted text:", end=" ")
    logger.success(text_highlighted)

    # python -m elastics.highlighter
