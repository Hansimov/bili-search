import re

from pypinyin import lazy_pinyin
from tclogger import logger
from typing import Union
from converters.query.punct import Puncter


class HighlightMerger:
    def merge(self, text: str, htexts: list[str], tag: str = "em"):
        """Merge all highlighted text segments into one."""
        if not text or not htexts:
            return None

        pattern = f"<{tag}>(.*?)</{tag}>"
        highlighted_segements = set()
        for htext in htexts:
            segments = re.findall(pattern, htext)
            highlighted_segements.update(segments)

        res_text = text
        for seg in highlighted_segements:
            res_text = res_text.replace(seg, f"<{tag}>{seg}</{tag}>")
        return res_text


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
        text_segs = list(text)

        keyword_pinyin = self.text_to_pinyin_str(keyword)
        text_pinyins = lazy_pinyin(text_segs)
        text_pinyins = self.filter_chars_from_pinyin(text_pinyins)

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


class HighlightsCounter:
    def __init__(self):
        self.puncter = Puncter()

    def extract_highlighted_keywords(
        self, htext: str, tag="hit", remove_puncts: bool = True
    ) -> dict:
        pattern = f"<{tag}>(.*?)</{tag}>"
        highlighted_keywords = {}
        match = re.findall(pattern, htext)
        for m in match:
            if remove_puncts:
                m = self.puncter.remove(m)
            highlighted_keywords[m] = highlighted_keywords.get(m, 0) + 1
        return highlighted_keywords

    def count_keywords(
        self,
        hits: list[dict],
        exclude_fields: list = ["pubdate_str"],
        use_score: bool = False,
        threshold: int = 2,
    ) -> dict:
        res = {}
        for hit in hits:
            merged_highlights = hit.get("merged_highlights", {})
            if use_score:
                hit_score = hit.get("score", 1)
            else:
                hit_score = 1
            for field, text in merged_highlights.items():
                if field in exclude_fields:
                    continue
                if text:
                    if isinstance(text, list):
                        htext = text[0]
                    else:
                        htext = text
                    highlighted_keywords = self.extract_highlighted_keywords(htext)
                    res[field] = res.get(field, {})
                    for keyword, keyword_count in highlighted_keywords.items():
                        keyword = keyword.lower().replace(" ", "")
                        res[field][keyword] = (
                            res[field].get(keyword, 0) + keyword_count * hit_score
                        )
        res = {
            field: {k: v for k, v in keywords.items() if v >= threshold}
            for field, keywords in res.items()
        }
        res = {
            field: dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True))
            for field, keywords in res.items()
        }
        return res

    def count_authors(self, hits: list[dict], threshold: int = 2) -> dict:
        res = {}
        for hit in hits:
            owner = hit.get("owner", {})
            name = owner.get("name", None)
            uid = owner.get("mid", None)
            if name in res.keys():
                res[name]["count"] += 1
            else:
                res[name] = {"uid": uid, "count": 1}
                if "owner.name" in hit["merged_highlights"].keys():
                    res[name]["highlighted"] = True
        res = {name: info for name, info in res.items() if info["count"] >= threshold}
        res = dict(
            sorted(
                res.items(),
                key=lambda item: (item[1].get("highlighted", False), item[1]["count"]),
                reverse=True,
            )
        )
        return res


if __name__ == "__main__":
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

    # python -m elastics.highlighter
