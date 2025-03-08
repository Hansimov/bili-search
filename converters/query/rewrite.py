class QueryRewriter:
    def rewrite(
        self,
        query_info: dict = {},
        suggest_info: dict = {},
        threshold: int = 2,
        append_date: bool = True,
    ) -> dict:
        """Exampe of output:
        ```json
        {
            "query": "hongjing 08 2024",
            "list": ["红警 08 2024", "红警08 2024"],
            "tuples": [("红警 08 2024", 20), ("红警08 2024", 3)],
            "dict": {},
            "rewrited: True
        }
        ```
        """
        res = {
            "query": query_info.get("query", ""),
            "list": [],
            "tuples": [],
            "dict": {},
            "rewrited": False,
        }
        if not suggest_info:
            return res
        hwords_str_count = suggest_info.get("hwords_str_count", {})
        if not hwords_str_count:
            return res
        keywords_date = query_info.get("keywords_date", [])
        keywords_date_str = " ".join(keywords_date)
        hwords_str_count_tuples = [
            (k, v) for k, v in hwords_str_count.items() if v >= threshold
        ]
        rewrite_hwords_str_list = []
        if hwords_str_count_tuples:
            rewrite_hwords_str_tuples = sorted(
                hwords_str_count_tuples, key=lambda x: x[1], reverse=True
            )
            for hwords_str, count in hwords_str_count_tuples:
                rewrite_hwords_str_list.append(hwords_str)
        else:
            rewrite_hwords_str_tuples = []
            rewrite_hwords_str_list = [" ".join(query_info.get("keywords_body"))]
        if append_date and keywords_date:
            rewrite_hwords_str_tuples = [
                (f"{hwords_str} {keywords_date_str}", count)
                for hwords_str, count in hwords_str_count_tuples
            ]
            rewrite_hwords_str_list = [
                f"{hwords_str} {keywords_date_str}"
                for hwords_str in rewrite_hwords_str_list
            ]
        res.update(
            {
                "list": rewrite_hwords_str_list,
                "tuples": rewrite_hwords_str_tuples,
                "rewrited": True,
            }
        )
        return res
