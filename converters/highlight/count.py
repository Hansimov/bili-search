from collections import defaultdict
from typing import Union, Literal

from converters.query.punct import Puncter
from converters.query.pinyin import ChinesePinyinizer


class HighlightsCounter:
    def __init__(self):
        self.puncter = Puncter(non_specials="-")
        self.pinyinizer = ChinesePinyinizer()

    def qword_match_hword(self, qword: str, hword: str) -> dict[str, bool]:
        is_match = {"prefix": False, "full": False, "middle": False}
        qword_str = qword.lower().strip()
        hword_str = hword.lower().strip()
        qword_pinyin = self.pinyinizer.text_to_pinyin_str(qword).lower().strip()
        hword_pinyin = self.pinyinizer.text_to_pinyin_str(hword).lower().strip()
        is_match = {
            "prefix": hword_str.startswith(qword_str)
            or hword_pinyin.startswith(qword_pinyin),
            "full": (hword_str == qword_str) or (hword_pinyin == qword_pinyin),
            "middle": (qword_str in hword_str) or (qword_pinyin in hword_pinyin),
        }
        return is_match

    def get_matched_qwords_with_hword(
        self,
        hword: str,
        qwords: list[str],
        match_part: Literal["prefix", "full", "middle"] = "middle",
    ) -> list[str]:
        """get all matched qwords with hword"""
        matched_qwords = []
        for qword in qwords:
            if self.qword_match_hword(qword, hword)[match_part]:
                matched_qwords.append(qword)
        return matched_qwords

    def filter_hword_qword_tuples(
        self,
        hword_qword_tuples: list[tuple[str, str, int]],
        qword_hword_count: dict[str, dict[str, int]],
    ) -> list[tuple[str, str, int]]:
        """This function takes input of `hword_qword_tuples` (list[tuple[str,str,int]]) and `qword_hword_count` (dict[str,dict[str,int]]).
        It filters the most-related hword with highest count for each qword.
        """
        qword_hword_dict: dict[str, tuple[str, int, int]] = {}
        for hword, qword, qword_idx in hword_qword_tuples:
            if not qword:
                continue
            hword_count = qword_hword_count[qword].get(hword, 0)
            if qword not in qword_hword_dict:
                qword_hword_dict[qword] = (hword, hword_count, qword_idx)
            else:
                _, old_hword_count, _ = qword_hword_dict[qword]
                if hword_count > old_hword_count:
                    qword_hword_dict[qword] = (hword, hword_count, qword_idx)
        new_hword_qword_tuples = [
            (hword, qword, qword_idx)
            for hword, qword, qword_idx in qword_hword_dict.values()
        ]
        new_hword_qword_tuples = sorted(new_hword_qword_tuples, key=lambda x: x[-1])
        return new_hword_qword_tuples

    def filter_hwords_by_qwords(
        self,
        qwords: list[str],
        hwords: list[str],
        qword_hword_count: dict[str, dict[str, int]],
    ) -> dict:
        """This function takes input of `qwords` (list), `hwords` (list), and `qword_hword_count` (dict[str,dict[str,int]]).
        It returns dict with keys:
        - `qword_hword_dict`(dict): `{<qword>: {<hword>: <count>}, ...}`
        - `hword_qword_tuples`(list[tuple]): `[(<hword>, <qword>, <qword_idx>), ...]`
        - `hwords_list`(list): `[<hword>, ...]`
        - `hwords_str`(str): `"hword1 hword2 ..."`
        This is used to find the related qword of each given hword, and record qword index in original qwords list.
        """
        hword_qword_tuples: list[tuple[str, str, int]] = []
        qword_hword_dict: dict[str, dict[str, int]] = {}
        for hword in hwords:
            is_hword_matched = False
            for qword_idx, qword in enumerate(qwords):
                if self.qword_match_hword(qword, hword)["prefix"]:
                    hword_qword_tuples.append((hword, qword, qword_idx))
                    qword_hword_dict[qword] = qword_hword_dict.get(qword, {})
                    qword_hword_dict[qword][hword] = (
                        qword_hword_dict[qword].get(hword, 0) + 1
                    )
                    is_hword_matched = True
                    break
            if not is_hword_matched:
                hword_qword_tuples.append((hword, None, len(qwords)))
        hword_qword_tuples = self.filter_hword_qword_tuples(
            hword_qword_tuples, qword_hword_count
        )
        hwords_list = [hword for hword, _, _ in hword_qword_tuples]
        res = {
            "dict": qword_hword_dict,
            "tuples": hword_qword_tuples,
            "list": hwords_list,
            "str": " ".join(hwords_list),
        }
        return res

    def extract_hwords_containing_all_qwords(
        self, qword_hword_count: dict[str, dict[str, int]]
    ) -> dict[str, int]:
        """This function takes input of `qword_hword_count` (dict[str,dict[str,int]]).
        It returns `hwords_containing_all_qwords` (dict[str,int]), and the hwords are the ones that match all qwords in different parts.
        This is used to exclude the hwords that not contain all qwords, to reduce redundancy of overlapped text parts among the rewrited query keywords.
        """
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

    def get_hword_count_of_hits(
        self,
        hits: list[dict],
        exclude_fields: list = [],
        ignore_case: bool = True,
        remove_punct: bool = True,
    ) -> list[dict[str, int]]:
        """Example of output:
        ```json
        [
            { "红警": 2, "08": 2 },
            { "红警": 1, "08": 1 },
            { "红警08": 1, "红警": 1, "08": 2 },
            ...
        ]
        ```
        """
        hword_count_of_hits: list[dict[str, int]] = []
        for hit in hits:
            segged_highlights = hit.get("highlights", {}).get("segged", {})
            hword_count_of_hit: dict[str, int] = {}
            for field, segs in segged_highlights.items():
                if field in exclude_fields or not segs:
                    continue
                if ignore_case:
                    segs = [seg.lower() for seg in segs]
                if remove_punct:
                    segs = [self.puncter.remove(seg) for seg in segs]
                for seg in segs:
                    hword_count_of_hit[seg] = hword_count_of_hit.get(seg, 0) + 1
            hword_count_of_hits.append(hword_count_of_hit)
        return hword_count_of_hits

    def get_hit_scores(
        self, hits: list[dict], use_score: bool = False
    ) -> list[Union[int, float]]:
        hit_scores: list[Union[int, float]] = []
        for hit in hits:
            hit_score = hit.get("score", 1) if use_score else 1
            hit_scores.append(hit_score)
        return hit_scores

    def calc_qword_hword_count_of_hit(
        self,
        qwords: list[str],
        hword_count_of_hit: dict[str, int],
        hit_score: Union[int, float] = 1,
        match_part: Literal["prefix", "full", "middle"] = "prefix",
        res: dict[str, dict[str, int]] = None,
    ) -> dict[str, dict[str, int]]:
        """Example of output:
        ```json
        {
            "hongjing": { "红警": 1, "红警08": 1 },
            "08": { "08": 2 }
        }
        ```
        """
        if res is None:
            res = {}
        for hword, hword_count in hword_count_of_hit.items():
            hword = hword.lower().replace(" ", "")
            for qword in qwords:
                if len(qwords) == 1 or self.qword_match_hword(qword, hword)[match_part]:
                    res[qword] = res.get(qword, {})
                    res[qword][hword] = (
                        res[qword].get(hword, 0) + hword_count * hit_score
                    )
        return res

    def calc_qword_hword_count(
        self,
        qwords: list[str],
        hword_count_of_hits: list[dict[str, int]],
        hit_scores: list[int] = [],
        threshold: int = 2,
    ) -> dict[str, dict[str, int]]:
        """This function takes input of `qwords` (list) and `hword_count_of_hits` (list of dict[str,int]).
        Each item in `hword_count_of_hits` is a dict that stores appeared hwords and their counts in each hit.
        It returns `qword_hword_count` (dict[str,dict[str,int]]), which stores the total hwords and counts that match each qword.
        Example of `qword_hword_count`:
        ```json
        {
            "hongjing": {
                "红警"   : 52,
                "红警08" : 4
            },
            "08": {
                "08": 33
            }
        }
        ```
        """
        res: dict[str, dict[str, int]] = {}
        for hword_count_of_hit, hit_score in zip(hword_count_of_hits, hit_scores):
            res = self.calc_qword_hword_count_of_hit(
                qwords,
                hword_count_of_hit=hword_count_of_hit,
                hit_score=hit_score,
                res=res,
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

    def calc_hword_qwords_maps(
        self, qwords: list[str], qword_hword_count: dict[str, dict[str, int]]
    ) -> dict[str, list[str]]:
        """Example of `hword_qwords_maps`:
        ```json
        {
            "红警": ['hongjing'],
            "红警小块地": ['hongjing', 'xiaokuaidi'],
            "08": ['08'],
            "小块地": ['xiaokuaidi'],
            "小快递": ['xiaokuaidi']
        },
        ```
        """
        hwords = [
            hword
            for hword_count_dict in qword_hword_count.values()
            for hword in hword_count_dict.keys()
        ]
        hword_qwords_maps = {
            hword: sorted(self.get_matched_qwords_with_hword(hword, qwords=qwords))
            for hword in hwords
        }
        return hword_qwords_maps

    def calc_hwords_str_count(
        self,
        qwords: list[str],
        hword_count_of_hits: list[dict[str, int]],
        qword_hword_count: dict[str, dict[str, int]] = {},
        hit_scores: list[int] = [],
        threshold: int = 2,
    ) -> dict[str, int]:
        """This function takes input of `qwords` (list), `hword_count_of_hits` (list of dict[str,int]), and `qword_hword_count` (dict[str,dict[str,int]]).
        It returns `hwords_str_count` (dict[str,int]) which stores the count of hwords_str (joined by space), and each hwords_str is a group of hwords that appear simultaneously at same hit.
        This is for replacing qwords with (fixed or corrected) hwords, that also considers that the different hword groups should appear at same hit, which avoids incorrect mixing of hwords in different contexts.
        Example of `hwords_str_count`:
        ```json
        {
            "红警 08": 20,
            "红警08": 3
        }
        """
        res = {}
        hwords_containing_all_qwords = self.extract_hwords_containing_all_qwords(
            qword_hword_count
        )
        for hword_count_dict, hit_score in zip(hword_count_of_hits, hit_scores):
            hwords_of_hit = hword_count_dict.keys()
            separated_hwords_of_hit = [
                hword
                for hword in hwords_of_hit
                if hword not in hwords_containing_all_qwords
            ]
            filter_hwords_res = self.filter_hwords_by_qwords(
                qwords, separated_hwords_of_hit, qword_hword_count
            )
            sorted_hwords_str = filter_hwords_res["str"]
            qword_hword_dict = filter_hwords_res["dict"]
            if sorted_hwords_str.strip() and (
                len(list(qword_hword_dict.keys())) >= len(qwords)
            ):
                res[sorted_hwords_str] = res.get(sorted_hwords_str, 0) + hit_score
            for word in hwords_containing_all_qwords.keys():
                if word in hwords_of_hit:
                    res[word] = res.get(word, 0) + hit_score
        res = {k: v for k, v in res.items() if v >= threshold}
        res = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
        return res

    def calc_group_hword_count_of_hit(
        self, qwords: list[str], hword_count_of_hit: dict[str, int], hit_score: int = 1
    ) -> dict:
        """Examples of `group_hword_qword_count_of_hit` with query `Hongjing 08 2024 xiaokuaidi`:
        - {'group_hwords': ('08', '红警'), 'count': 1, 'hword_qwords': {'08': ['08'], '红警': ['hongjing']}}
        - {'group_hwords': ('08', '小块地', '红警'), 'count': 1, 'hword_qwords': {'08': ['08'], '小块地': ['xiaokuaidi'], '红警': ['hongjing']}}
        - {'group_hwords': ('08', '小快递', '红警小块地'), 'count': 1, 'hword_qwords': {'08': ['08'], '小快递': ['xiaokuaidi'], '红警小块地': ['hongjing', 'xiaokuaidi']}}
        """
        qword_hword_count_of_hit = self.calc_qword_hword_count_of_hit(
            qwords,
            hword_count_of_hit=hword_count_of_hit,
            hit_score=hit_score,
            match_part="prefix",
        )
        # sort qword_hword_count_of_hit by hword_count
        qword_hword_count_of_hit = {
            qword: dict(
                sorted(hword_count_of_hit.items(), key=lambda x: x[1], reverse=True)
            )
            for qword, hword_count_of_hit in qword_hword_count_of_hit.items()
        }
        # # pick hword with highest hword_count for each qword
        # qword_hword_count_of_hit = {
        #     qword: dict(list(hword_count_of_hit.items())[:1])
        #     for qword, hword_count_of_hit in qword_hword_count_of_hit.items()
        # }
        """Examples of `qword_hword_count_of_hit` with query `Hongjing 08 2024 xiaokuaidi`:
        - {'hongjing': {'红警': 2}, '08': {'08': 1}}
        - {'hongjing': {'红警': 3}, '08': {'08': 1}, 'xiaokuaidi': {'小块地': 1}}
        - {'hongjing': {'红警小块地': 1}, '08': {'08': 1}, 'xiaokuaidi': {'小快递': 1}}
        """
        # store qwords and count for each hword
        group_hwords_count_dict = defaultdict(int)
        for qword, hword_count_dict in qword_hword_count_of_hit.items():
            for hword, hword_count in hword_count_dict.items():
                group_hwords_count_dict[hword] += hword_count
        # group hwords to tuple, and get matched qwords for each hword
        group_hwords = tuple(sorted(list(group_hwords_count_dict.keys())))
        group_hwords_count = min(
            group_hwords_count_dict[hword] for hword in group_hwords
        )
        group_hword_qword_count_of_hit = {
            "group_hwords": group_hwords,
            "count": group_hwords_count,
        }
        print(group_hwords, group_hwords_count)
        return group_hword_qword_count_of_hit

    def calc_group_hwords_count(
        self,
        qwords: list[str],
        hword_count_of_hits: list[dict[str, int]],
        qword_hword_count: dict[str, dict[str, int]] = {},
        hit_scores: list[int] = [],
        threshold: int = 2,
    ) -> dict[tuple, dict]:
        """Example of `hwords_qwords_count`:
        ```json
        {
            ('08', '小块地', '红警'): 4,
            ('08', '小块地', '红警', '红警小块地'): 1,
            ('08', '红警', '红警小块地'): 1,
            ('08', '小快递', '红警', '红警小块地'): 1
        },
        ```
        """
        group_hwords_count = defaultdict(int)
        qwords = list(qword_hword_count.keys())
        for hword_count_of_hit, hit_score in zip(hword_count_of_hits, hit_scores):
            group_hword_count_of_hit = self.calc_group_hword_count_of_hit(
                qwords, hword_count_of_hit=hword_count_of_hit, hit_score=hit_score
            )
            group_hwords_of_hit = group_hword_count_of_hit["group_hwords"]
            count_of_hit = group_hword_count_of_hit["count"]
            group_hwords_count[group_hwords_of_hit] += count_of_hit
        # sort group_hwords_count by count
        group_hwords_count = dict(
            sorted(dict(group_hwords_count).items(), key=lambda x: x[1], reverse=True)
        )
        return group_hwords_count

    def count_keywords(
        self,
        qwords: list[str],
        hits: list[dict],
        exclude_fields: list = ["pubdate_str"],
        use_score: bool = False,
        threshold: int = 2,
        ignore_case: bool = True,
        remove_punct: bool = True,
    ) -> dict:
        """Example of output:
        ```json
        {
            "qword_hword_count": {
                "08": {
                    "08": 33
                },
                "hongjing": {
                    "红警"   : 52,
                    "红警08" : 4
                }
            },
            "hwords_str_count": {
                "红警 08" : 20,
                "红警08"  : 3
            }
        }
        ```
        """
        if ignore_case:
            qwords = [qword.lower() for qword in qwords]
        hword_count_of_hits = self.get_hword_count_of_hits(
            hits,
            exclude_fields=exclude_fields,
            ignore_case=ignore_case,
            remove_punct=remove_punct,
        )
        hit_scores = self.get_hit_scores(hits, use_score=use_score)
        qword_hword_count = self.calc_qword_hword_count(
            qwords, hword_count_of_hits, hit_scores=hit_scores, threshold=1
        )
        hword_qwords_maps = self.calc_hword_qwords_maps(qwords, qword_hword_count)
        hword_func_params = {
            "qwords": qwords,
            "hword_count_of_hits": hword_count_of_hits,
            "qword_hword_count": qword_hword_count,
            "hit_scores": hit_scores,
            "threshold": threshold,
        }
        group_hwords_count = self.calc_group_hwords_count(**hword_func_params)
        hwords_str_count = self.calc_hwords_str_count(**hword_func_params)
        res = {
            "qword_hword_count": qword_hword_count,
            "hword_qwords_maps": hword_qwords_maps,
            "group_hwords_count": group_hwords_count,
            "hwords_str_count": hwords_str_count,
        }
        return res

    def count_authors(
        self,
        hits: list[dict],
        threshold: int = 2,
        threshold_level: int = 0,
        top_k: int = 8,
    ) -> dict:
        """Example of output:
        ```json
        {
            "红警HBK08": {
                "uid"         : 1629347259,
                "count"       : 20,
                "highlighted" : True
            }
        }
        ```
        """
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
                if "owner.name" in hit["highlights"]["merged"].keys():
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
