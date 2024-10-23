from copy import deepcopy


class QueryRewriter:
    def rewrite(
        self, query_keywords: list[str], suggest_info: dict = {}, threshold: int = 2
    ) -> list[str]:
        if not suggest_info:
            return query_keywords
        qwords = deepcopy(query_keywords)
        suggest_wordict = suggest_info.get("highlighted_keywords", {})
        if not suggest_wordict:
            return qwords
        for idx, qword in enumerate(qwords):
            choices = suggest_wordict.get(qword, {})
            if choices:
                choices = dict(
                    sorted(choices.items(), key=lambda x: x[1], reverse=True)
                )
                best_qword, count = list(choices.items())[0]
                same_qword_count = choices.get(qword, 0)
                if same_qword_count >= threshold:
                    continue
                if best_qword != qword and count >= threshold:
                    qwords[idx] = best_qword

        return qwords
