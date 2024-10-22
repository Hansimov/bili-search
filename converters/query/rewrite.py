from copy import deepcopy


class QueryRewriter:
    def rewrite(
        self, query_keywords: list[str], suggest_info: dict = {}, threshold: int = 2
    ) -> list[str]:
        if not suggest_info:
            return query_keywords
        query_keywords = deepcopy(query_keywords)
        suggest_wordict = suggest_info.get("highlighted_keywords", {})
        if not suggest_wordict:
            return query_keywords
        for idx, qword in enumerate(query_keywords):
            choices = suggest_wordict.get(qword, {})
            if choices:
                choices = dict(
                    sorted(choices.items(), key=lambda x: x[1], reverse=True)
                )
                new_qword, count = list(choices.items())[0]
                if new_qword != qword and count >= threshold:
                    query_keywords[idx] = new_qword

        return query_keywords
