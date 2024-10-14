import math

from datetime import datetime
from tclogger import get_now_ts
from typing import Literal, Union

from converters.times import DateFormatChecker
from converters.query.pinyin import ChinesePinyinizer


class MultiMatchQueryDSLConstructor:
    def __init__(self) -> None:
        self.pinyinizer = ChinesePinyinizer()

    def remove_boost_from_fields(self, fields: list[str]) -> list[str]:
        return [field.split("^", 1)[0] for field in fields]

    def deboost_field(self, field: str):
        return field.split("^", 1)[0]

    def is_pinyin_field(self, field: str):
        return self.deboost_field(field).endswith(".pinyin")

    def remove_fields_from_fields(
        self, fields_to_remove: Union[str, list], fields: list[str]
    ) -> list[str]:
        if isinstance(fields_to_remove, str):
            fields_to_remove = [fields_to_remove]
        clean_fields = []
        for field in fields:
            for field_to_remove in fields_to_remove:
                if not field.startswith(field_to_remove):
                    clean_fields.append(field)
        return clean_fields

    def is_field_in_fields(self, field_to_check: str, fields: list[str]) -> bool:
        for field in fields:
            if field.startswith(field_to_check):
                return True
        return False

    def construct_match_clause(
        self,
        query_type: Literal["multi_match", "combined_fields"],
        keyword: str,
        match_fields: list[str],
        match_type: Literal["phrase_prefix", "bool_prefix"] = "phrase_prefix",
        match_operator: Literal["or", "and"] = "or",
    ):
        return {
            query_type: {
                "query": keyword,
                "fields": match_fields,
                "type": match_type,
                "operator": match_operator,
            }
        }

    def construct_query_for_date_keyword(
        self,
        keyword: str,
        date_match_fields: list[str],
        checker: DateFormatChecker,
        date_fields: list[Literal["pubdate_str"]] = ["pubdate_str"],
        match_type: Literal["phrase_prefix", "bool_prefix"] = "phrase_prefix",
        match_operator: Literal["or", "and"] = "or",
    ):
        clause = {
            "bool": {
                "should": [],
                "minimum_should_match": 1,
            }
        }

        if date_fields:
            keyword_date = checker.rewrite(
                keyword, sep="-", check_format=False, use_current_year=True
            )
            clause["bool"]["should"].append(
                self.construct_match_clause(
                    "multi_match",
                    keyword_date,
                    date_match_fields,
                    match_type="bool_prefix",
                    match_operator=match_operator,
                )
            )

        non_date_fields = self.remove_fields_from_fields(date_fields, date_match_fields)
        if non_date_fields and checker.matched_format == "%Y":
            clause["bool"]["should"].append(
                self.construct_match_clause(
                    "multi_match",
                    keyword,
                    non_date_fields,
                    match_type=match_type,
                    match_operator=match_operator,
                )
            )
        return clause

    def construct_combined_fields_clauses(
        self,
        keyword: str,
        match_fields: list[str],
        match_type: Literal["phrase_prefix", "bool_prefix"] = "phrase_prefix",
        combined_fields_list: list[list[str]] = [],
    ) -> list[dict]:
        """NOTE: Only text fields are supported, and they must all have the same search analyzer.
        See also:
        - https://www.elastic.co/guide/en/elasticsearch/reference/8.14/query-dsl-combined-fields-query.html#combined-field-top-level-params
        """
        clauses = []
        for combined_fields in combined_fields_list:
            if all(
                self.is_field_in_fields(cfield, match_fields)
                for cfield in combined_fields
            ):
                combined_fields_fullnames = [
                    mfield
                    for mfield in match_fields
                    if any(mfield.startswith(cfield) for cfield in combined_fields)
                ]
                clauses.append(
                    self.construct_match_clause(
                        "combined_fields",
                        keyword,
                        combined_fields_fullnames,
                        match_type=match_type,
                        match_operator="and",
                    )
                )
                # remove combined_fields_with_pinyin from fields_with_pinyin
                for cfield in combined_fields_fullnames:
                    match_fields.remove(cfield)
        return clauses

    def construct_query_for_text_keyword(
        self,
        keyword: str,
        match_fields: list[str],
        match_type: Literal["phrase_prefix", "bool_prefix"] = "phrase_prefix",
        match_operator: Literal["or", "and"] = "or",
        combined_fields_list: list[list[str]] = [],
    ):
        clause = {
            "bool": {
                "should": [],
                "minimum_should_match": 1,
            }
        }

        fields_with_pinyin = [
            field for field in match_fields if self.is_pinyin_field(field)
        ]
        if fields_with_pinyin:
            keyword_pinyin = self.pinyinizer.convert(keyword)

            combined_fields_clauses = self.construct_combined_fields_clauses(
                keyword_pinyin,
                fields_with_pinyin,
                match_type=match_type,
                combined_fields_list=combined_fields_list,
            )
            match_clause = self.construct_match_clause(
                "multi_match",
                keyword_pinyin,
                fields_with_pinyin,
                match_type=match_type,  # "bool_prefix" is better on recall, but slower
                match_operator=match_operator,
            )
            clause["bool"]["should"].extend(combined_fields_clauses)
            clause["bool"]["should"].append(match_clause)

        fields_without_pinyin = [
            field for field in match_fields if not self.is_pinyin_field(field)
        ]
        if fields_without_pinyin:
            combined_fields_clauses = self.construct_combined_fields_clauses(
                keyword,
                fields_without_pinyin,
                match_type=match_type,
                combined_fields_list=combined_fields_list,
            )
            match_clause = self.construct_match_clause(
                "multi_match",
                keyword,
                fields_without_pinyin,
                match_type=match_type,
                match_operator=match_operator,
            )
            clause["bool"]["should"].extend(combined_fields_clauses)
            clause["bool"]["should"].append(match_clause)

        return clause

    def construct(
        self,
        query: str,
        match_fields: list[str],
        date_match_fields: list[str],
        date_fields: list[Literal["pubdate_str"]] = ["pubdate_str"],
        match_bool: Literal["must", "should"] = "must",
        match_type: Literal["phrase_prefix", "bool_prefix"] = "phrase_prefix",
        match_operator: Literal["or", "and"] = "or",
        combined_fields_list: list[list[str]] = [],
    ) -> dict:
        query_dsl_dict = {"bool": {match_bool: []}}
        query_keywords = query.split()
        checker = DateFormatChecker()
        match_non_date_fields = self.remove_fields_from_fields(
            date_fields, match_fields
        )

        for keyword in query_keywords:
            checker.init_year_month_day()
            is_keyword_date_format = checker.is_in_date_range(
                keyword, start="2009-09-09", end=datetime.now(), verbose=False
            )
            if is_keyword_date_format:
                clause = self.construct_query_for_date_keyword(
                    keyword,
                    date_match_fields,
                    checker,
                    match_type=match_type,
                    match_operator=match_operator,
                )
            else:
                clause = self.construct_query_for_text_keyword(
                    keyword,
                    match_non_date_fields,
                    match_type=match_type,
                    match_operator=match_operator,
                    combined_fields_list=combined_fields_list,
                )
            query_dsl_dict["bool"][match_bool].append(clause)

        return query_dsl_dict


class ScriptScoreQueryDSLConstructor:
    def log_func(
        self, field: str, min_input: float = 2, min_output: float = 0.0
    ) -> str:
        func_str = f"(Math.log10(Math.max({field}, {min_input})) + {min_output})"
        return func_str

    def pow_func(
        self,
        field: str,
        power: float = 1,
        min_value: float = 1,
        power_precision: int = 4,
    ) -> str:
        if power == 1.0:
            func_str = f"Math.max({field}, {min_value})"
        else:
            func_str = (
                f"Math.pow(Math.max({field}, {min_value}), {power:.{power_precision}f})"
            )
        return func_str

    def field_to_var(self, field: str):
        return field.replace(".", "_")

    def assign_var(
        self, field: str, get_value_func: str = None, default_value: float = 0
    ):
        field_var = self.field_to_var(field)
        if not get_value_func:
            get_value_func = f"doc['{field}'].value"
        new_var = ""
        if field == "pubdate":
            default_value = 1262275200
            get_value_func = "doc['pubdate'].value"
            new_var = f"\ndouble pubdate_decay;\n"
        return (
            f"double {field_var} = (doc['{field}'].size() > 0) ? {get_value_func} : {default_value};"
            f"{new_var}"
        )

    def assign_var_of_pubdate_decay(
        self,
        field: str = "pubdate",
        now_ts_field: str = "params.now_ts",
        half_life_days: int = 7,
        max_life_days: int = 365,
        min_value: float = 0.2,
        max_value: float = 1.0,
    ) -> str:
        """f(x) = k / (1 + (x / b)^power) + min_value
            where k, b is unknown.
        if power == 1.0, then we have:
            f(x) = k / (1 + x / b) + min_value

        a. when x->inf, f(x) -> min_value
        b. when x->0,   f(x) -> max_value, we get:
            k = max_value - min_value
        c. when x=half_life_days, f(x) -> max_value/2, we get:
            k / (1 + half_life_days / b) + min_value = (max_value + min_value) / 2
            b = half_life_days

        Here is the guide of finetuning the params:
        1. b (half_life_days) is higher, decay is slower
        2. (max_value - min_value) is higher, the effects of pubdate is larger
        3. max_life_days is higher, the window of recent videos is larger
        """
        seconds_per_day = 86400
        pass_seconds_str = f"({now_ts_field} - {field})"
        pass_days_str = f"{pass_seconds_str}/{seconds_per_day}"
        k = max_value - min_value
        b = half_life_days
        pass_days_scale = round(seconds_per_day * half_life_days * b, 2)
        reciprocal_str = f"{k} / (1 + {pass_seconds_str} / {pass_days_scale})"
        max_life_seconds = max_life_days * seconds_per_day
        func_str = f"""if ({pass_seconds_str} > {max_life_seconds}) {{
            pubdate_decay = {min_value};
        }} else {{
            pubdate_decay = {reciprocal_str} + {min_value};
        }}"""
        return func_str

    def assign_var_of_relevance_score(
        self, min_relevance_score: float = 0.01, down_scale: float = 100
    ):
        score_str = f"Math.max(_score, {min_relevance_score}) / {down_scale}"
        assign_str = f"\ndouble r_score = {score_str};\n"
        return assign_str

    def get_script_source_by_powers(self):
        assign_vars = []
        stat_powers = {
            "stat.view": 0.1,
            "stat.like": 0.1,
            "stat.coin": 0.25,
            "stat.favorite": 0.15,
            "stat.reply": 0.2,
            "stat.danmaku": 0.15,
            "stat.share": 0.2,
        }
        stat_pow_ratio = (
            1 / math.sqrt(len(stat_powers.keys())) / sum(stat_powers.values())
        )
        for field in list(stat_powers.keys()) + ["pubdate"]:
            assign_vars.append(self.assign_var(field))
        assign_vars_str = "\n".join(assign_vars)
        assign_vars_str += self.assign_var_of_pubdate_decay()
        stat_func_str = " * ".join(
            self.pow_func(self.field_to_var(field), field_power * stat_pow_ratio, 1)
            for field, field_power in stat_powers.items()
        )
        score_str = self.pow_func("_score", 1, 1)
        func_str = f"return ({stat_func_str}) * pubdate_decay * {score_str};"
        script_source = f"{assign_vars_str}\n{func_str}"
        return script_source

    def get_script_source_by_stats(
        self,
        stat_fields: list[
            Literal[
                "stat.view",
                "stat.like",
                "stat.coin",
                "stat.favorite",
                "stat.danmaku",
                "stat.reply",
                "stat.share",
            ]
        ] = ["stat.like", "stat.coin", "stat.danmaku", "stat.reply"],
    ):
        assign_vars = []
        for field in stat_fields + ["pubdate"]:
            assign_vars.append(self.assign_var(field))
        assign_vars_str = "\n".join(assign_vars)
        assign_vars_str += self.assign_var_of_pubdate_decay(
            half_life_days=7, max_life_days=30, min_value=0.25, max_value=1.0
        )
        assign_vars_str += self.assign_var_of_relevance_score(
            min_relevance_score=0.01, down_scale=100
        )
        stat_func_str = " * ".join(
            f"{self.log_func(self.field_to_var(field))}" for field in stat_fields
        )
        func_str = f"return {stat_func_str} * pubdate_decay * r_score;"
        script_source = f"{assign_vars_str}\n{func_str}"
        return script_source

    def construct(self, query_dsl_dict: dict) -> dict:
        script_score_dsl_dict = {
            "script_score": {
                "query": query_dsl_dict,
                "script": {
                    # "source": self.get_script_source_by_powers(),
                    "source": self.get_script_source_by_stats(),
                    "params": {
                        "now_ts": get_now_ts(),
                    },
                },
            }
        }
        return script_score_dsl_dict

    def construct_rrf(
        self,
        query_dsl_dict: dict,
        window_size: int = 50,
        k: int = 20,
    ):
        """NOTE: Only available for subscription users. See more:
        - https://www.elastic.co/subscriptions

        Otherwise, an error will be raised:
        - AuthorizationException(403, 'security_exception',
            'current license is non-compliant for [Reciprocal Rank Fusion (RRF)]')
        """
        script_danmaku = {
            "script_score": {
                "query": query_dsl_dict,
                "script": {"source": f"doc['danmaku'].value"},
            }
        }
        script_coin = {
            "script_score": {
                "query": query_dsl_dict,
                "script": {"source": f"doc['coin'].value"},
            }
        }
        rrf_dsl_dict = {
            "retriever": {
                "rrf": {
                    "retrievers": [script_danmaku, script_coin],
                    "rank_window_size": window_size,
                    "rank_constant": k,
                }
            }
        }
        return rrf_dsl_dict


if __name__ == "__main__":
    from tclogger import logger, logstr, dict_to_str
    from elastics.videos.constants import SEARCH_BOOSTED_FIELDS
    from elastics.videos.constants import SEARCH_COMBINED_FIELDS_LIST

    match_fields_default = ["title", "owner.name", "desc", "tags"]
    match_fields_words = [f"{field}.words" for field in match_fields_default]
    match_fields_pinyin = [f"{field}.pinyin" for field in match_fields_default]
    match_fields = match_fields_words + match_fields_pinyin + ["pubdate_str"]
    date_match_fields = match_fields_words + ["pubdate_str"]

    for idx, mfield in enumerate(match_fields):
        if SEARCH_BOOSTED_FIELDS.get(mfield):
            boost = SEARCH_BOOSTED_FIELDS[mfield]
            match_fields[idx] = f"{mfield}^{boost}"

    queries = ["秋葉aaaki 2024", "影视飓feng 2024-01"]
    constructor = MultiMatchQueryDSLConstructor()
    for query in queries:
        logger.note(f"> [{logstr.mesg(query)}]:")
        query_dsl_dict = constructor.construct(
            query,
            match_fields=match_fields,
            date_match_fields=date_match_fields,
            combined_fields_list=SEARCH_COMBINED_FIELDS_LIST,
        )
        logger.success(dict_to_str(query_dsl_dict, add_quotes=True), indent=2)

    # python -m converters.query.dsl
