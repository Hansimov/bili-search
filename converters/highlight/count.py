import re

from typing import Union
from tclogger import get_now, tcdatetime

from converters.query.punct import Puncter
from converters.query.pinyin import ChinesePinyinizer
from converters.times import DateFormatChecker


class HighlightsCounter:
    def __init__(self):
        self.puncter = Puncter()
        self.pinyinizer = ChinesePinyinizer()
        self.date_checker = DateFormatChecker()
        now = get_now()
        self.next_year_start_dt = tcdatetime(year=now.year + 1, month=1, day=1)

    def remove_date_qwords(self, qwords: list[str]) -> list[str]:
        return [
            qword
            for qword in qwords
            if not self.date_checker.is_in_date_range(
                qword, start="2009-09-09", end=self.next_year_start_dt
            )
        ]

    def extract_highlighted_keywords(
        self, htext: str, tag="hit", remove_puncts: bool = True
    ) -> dict[str, int]:
        pattern = f"<{tag}>\s*([^\s]*?)\s*</{tag}>"
        highlighted_keywords = {}
        match = re.findall(pattern, htext)
        for m in match:
            if remove_puncts:
                m = self.puncter.remove(m)
            highlighted_keywords[m] = highlighted_keywords.get(m, 0) + 1
        return highlighted_keywords

    def qword_match_hword(self, qword: str, hword: str) -> dict[str, bool]:
        is_match = {"prefix": False, "full": False, "middle": False}
        qword_str = qword.lower().strip()
        hword_str = hword.lower().strip()
        qword_pinyin = self.pinyinizer.text_to_pinyin_str(qword)
        hword_pinyin = self.pinyinizer.text_to_pinyin_str(hword)
        is_match = {
            "prefix": qword_str.startswith(hword_str)
            or hword_pinyin.startswith(qword_pinyin),
            "full": (qword_str == hword_str) or (hword_pinyin == qword_pinyin),
            "middle": (qword_str in hword_str) or (qword_pinyin in hword_pinyin),
        }
        return is_match

    def sort_hwords_by_qwords(self, qwords: list[str], hwords: list[str]) -> dict:
        hword_with_qword_idx: list[tuple] = []
        qword_hword_dict: dict[str, dict[str, int]] = {}
        for hword in hwords:
            is_hword_matched = False
            for idx, qword in enumerate(qwords):
                if self.qword_match_hword(qword, hword)["prefix"]:
                    hword_with_qword_idx.append((hword, idx))
                    qword_hword_dict[qword] = qword_hword_dict.get(qword, {})
                    qword_hword_dict[qword][hword] = (
                        qword_hword_dict[qword].get(hword, 0) + 1
                    )
                    is_hword_matched = True
                    break
            if not is_hword_matched:
                hword_with_qword_idx.append((hword, len(qwords)))
        sorted_hwords: list[str] = [
            hword for hword, _ in sorted(hword_with_qword_idx, key=lambda x: x[1])
        ]
        res = {
            "dict": qword_hword_dict,
            "list": sorted_hwords,
            "str": " ".join(sorted_hwords),
        }
        return res

    def extract_hwords_containing_all_qwords(
        self, qword_hword_count: dict[str, dict[str, int]]
    ) -> dict[str, int]:
        qwords = list(qword_hword_count.keys())
        hwords_containing_all_qwords: dict[str, int] = {}
        for hword_count_dict in qword_hword_count.values():
            for hword, hword_count in hword_count_dict.items():
                is_hword_containing_all_qwords = True
                for qword in qwords:
                    if not self.qword_match_hword(qword, hword)["middle"]:
                        is_hword_containing_all_qwords = False
                        break
                if is_hword_containing_all_qwords:
                    hwords_containing_all_qwords[hword] = hword_count
        return hwords_containing_all_qwords

    def count_hword_by_qword(
        self,
        qwords: list[str],
        hword_count_of_hits: list[dict[str, int]],
        hit_scores: list[int] = [],
        threshold: int = 2,
    ) -> dict[str, dict[str, int]]:
        """return: `{<qword>: {<hword>: <count>}, ...}`"""
        res: dict[str, dict[str, int]] = {}
        if not hit_scores:
            hit_scores = [1] * len(hword_count_of_hits)
        for hword_count_dict, hit_score in zip(hword_count_of_hits, hit_scores):
            for hword, hword_count in hword_count_dict.items():
                hword = hword.lower().replace(" ", "")
                for qword in qwords:
                    if (
                        len(qwords) == 1
                        or self.qword_match_hword(qword, hword)["prefix"]
                    ):
                        res[qword] = res.get(qword, {})
                        res[qword][hword] = (
                            res[qword].get(hword, 0) + hword_count * hit_score
                        )
        res = {
            qword: {k: v for k, v in hwords.items() if v >= threshold}
            for qword, hwords in res.items()
        }
        res = {
            qword: dict(sorted(hwords.items(), key=lambda x: x[1], reverse=True))
            for qword, hwords in res.items()
        }
        return res

    def count_hwords_str_by_hit(
        self,
        qwords: list[str],
        hword_count_of_hits: list[dict[str, int]],
        hwords_containing_all_qwords: dict[str, int] = {},
        hit_scores: list[int] = [],
        threshold: int = 2,
    ) -> dict[str, int]:
        """return: `{<sorted_hwords_str_of_hit>: <count>, ...}`"""
        res = {}
        if not hit_scores:
            hit_scores = [1] * len(hword_count_of_hits)
        for hword_count_dict, hit_score in zip(hword_count_of_hits, hit_scores):
            hwords_of_hit = [
                hword
                for hword in hword_count_dict.keys()
                if hword not in hwords_containing_all_qwords
            ]
            hwords_sort_res = self.sort_hwords_by_qwords(qwords, hwords_of_hit)
            sorted_hwords_str = hwords_sort_res["str"]
            qword_hword_dict = hwords_sort_res["dict"]
            if len(list(qword_hword_dict.keys())) >= len(qwords):
                res[sorted_hwords_str] = res.get(sorted_hwords_str, 0) + hit_score
            for word in hwords_containing_all_qwords.keys():
                if word in hword_count_dict:
                    res[word] = res.get(word, 0) + hit_score
        res = {k: v for k, v in res.items() if v >= threshold}
        res = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
        return res

    def count_keywords(
        self,
        query: str,
        hits: list[dict],
        exclude_fields: list = ["pubdate_str"],
        use_score: bool = False,
        threshold: int = 2,
    ) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
        qwords = self.remove_date_qwords(query.split())
        hword_count_of_hits: list[dict[str, int]] = []
        hit_scores: list[Union[int, float]] = []
        for hit in hits:
            merged_highlights = hit.get("merged_highlights", {})
            hit_score = hit.get("score", 1) if use_score else 1
            hit_scores.append(hit_score)
            hword_count_of_hit: dict[str, int] = {}
            for field, text in merged_highlights.items():
                if field in exclude_fields or not text:
                    continue
                htext = text[0] if isinstance(text, list) else text
                hword_count_of_field = self.extract_highlighted_keywords(htext)
                for field_hword, field_hword_count in hword_count_of_field.items():
                    hword_count_of_hit[field_hword] = (
                        hword_count_of_hit.get(field_hword, 0) + field_hword_count
                    )
            hword_count_of_hits.append(hword_count_of_hit)

        res_by_qword = self.count_hword_by_qword(
            qwords, hword_count_of_hits, hit_scores=hit_scores, threshold=threshold
        )
        hwords_containing_all_qwords = self.extract_hwords_containing_all_qwords(
            res_by_qword
        )
        res_by_hit = self.count_hwords_str_by_hit(
            qwords,
            hword_count_of_hits,
            hwords_containing_all_qwords=hwords_containing_all_qwords,
            hit_scores=hit_scores,
            threshold=threshold,
        )
        return res_by_qword, res_by_hit

    def count_authors(
        self,
        hits: list[dict],
        threshold: int = 2,
        threshold_level: int = 0,
        top_k: int = 8,
    ) -> dict:
        if len(hits) <= 20:
            threshold = 2
        elif len(hits) <= 100:
            threshold = 3
        else:
            threshold = 4
        threshold -= threshold_level
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
        res = {
            name: info
            for name, info in res.items()
            if info["count"] >= threshold
            or (info.get("highlighted", False) and info["count"] >= threshold - 1)
        }
        if not res and threshold_level == 0:
            res = self.count_authors(hits, threshold_level=1, top_k=top_k)
        else:
            res = dict(
                sorted(
                    res.items(),
                    key=lambda item: (
                        item[1].get("highlighted", False),
                        item[1]["count"],
                    ),
                    reverse=True,
                )
            )
            res = dict(list(res.items())[:top_k])
        return res
