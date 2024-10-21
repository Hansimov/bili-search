import re

from converters.query.punct import Puncter
from converters.query.filter import QueryFilterExtractor


class HighlightsCounter:
    def __init__(self):
        self.puncter = Puncter()
        self.split_query = QueryFilterExtractor().split_keyword_and_filter_expr

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

    def qword_match_hword(self, qword: str, hword: str):
        return qword == hword

    def count_keywords(
        self,
        query: str,
        hits: list[dict],
        exclude_fields: list = ["pubdate_str"],
        use_score: bool = False,
        threshold: int = 2,
    ) -> dict:
        res = {}
        qwords = self.split_query(query)["keywords"]
        for hit in hits:
            merged_highlights = hit.get("merged_highlights", {})
            hit_score = hit.get("score", 1) if use_score else 1
            for field, text in merged_highlights.items():
                if field in exclude_fields or not text:
                    continue
                htext = text[0] if isinstance(text, list) else text
                hwords = self.extract_highlighted_keywords(htext)
                for hword, hword_count in hwords.items():
                    hword = hword.lower().replace(" ", "")
                    for qword in qwords:
                        if len(qwords) == 1 or self.qword_match_hword(qword, hword):
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
