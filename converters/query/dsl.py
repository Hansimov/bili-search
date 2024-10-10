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
                {
                    "multi_match": {
                        "query": keyword_date,
                        "type": "bool_prefix",
                        "fields": date_fields,
                        "operator": match_operator,
                    }
                }
            )
        non_date_fields = self.remove_fields_from_fields(date_fields, date_match_fields)
        if non_date_fields and checker.matched_format == "%Y":
            clause["bool"]["should"].append(
                {
                    "multi_match": {
                        "query": keyword,
                        "type": match_type,
                        "fields": non_date_fields,
                        "operator": match_operator,
                    }
                }
            )
        return clause

    def construct_query_for_text_keyword(
        self,
        keyword: str,
        match_fields: list[str],
        match_type: Literal["phrase_prefix", "bool_prefix"] = "phrase_prefix",
        match_operator: Literal["or", "and"] = "or",
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
            clause["bool"]["should"].append(
                {
                    "multi_match": {
                        "query": keyword_pinyin,
                        "type": match_type,
                        "fields": fields_with_pinyin,
                        "operator": match_operator,
                    }
                }
            )
        fields_without_pinyin = [
            field for field in match_fields if not self.is_pinyin_field(field)
        ]
        clause["bool"]["should"].append(
            {
                "multi_match": {
                    "query": keyword,
                    "type": match_type,
                    "fields": fields_without_pinyin,
                    "operator": match_operator,
                }
            }
        )
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
    ) -> dict:
        query_dsl_dict = {"bool": {"must": []}}
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
                )
            query_dsl_dict["bool"][match_bool].append(clause)

        return query_dsl_dict


class ScriptScoreQueryDSLConstructor:
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

    def log_func(self, field: str, min_value: float = 2) -> str:
        func_str = f"Math.log10(Math.max({field}, {min_value}))"
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

    def assign_var_of_pubdate_decay(
        self,
        field: str = "pubdate",
        now_ts_field: str = "params.now_ts",
        half_life_days: int = 7,
        max_life_days: int = 365,
        power: float = 1.5,
        min_value: float = 0.2,
    ) -> str:
        pass_seconds_str = f"({now_ts_field} - {field})"
        seconds_per_day = 86400
        pass_days = f"{pass_seconds_str}/{seconds_per_day}"
        scaled_pass_days = f"{pass_days}/{half_life_days}"
        power_str = self.pow_func(scaled_pass_days, power=power, min_value=0.1)
        reciprocal_str = f"(1 - {min_value}) / (1 + {power_str})"
        func_str = f"""if ({pass_days} > {max_life_days}) {{
            pubdate_decay = {min_value};
        }} else {{
            pubdate_decay = {reciprocal_str} + {min_value};
        }}"""
        return func_str

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
            half_life_days=7, power=1, max_life_days=90, min_value=0.15
        )
        stat_func_str = " * ".join(
            f"{self.log_func(self.field_to_var(field))}" for field in stat_fields
        )
        score_str = "Math.max(_score, 1)"
        func_str = f"return {stat_func_str} * pubdate_decay * {score_str} / 1e2;"
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

    match_fields_default = ["title", "owner.name", "desc", "tags"]
    match_fields_words = [f"{field}.words" for field in match_fields_default]
    match_fields_pinyin = [f"{field}.pinyin" for field in match_fields_default]
    match_fields = match_fields_words + match_fields_pinyin + ["pubdate_str"]
    date_match_fields = match_fields_words + ["pubdate_str"]

    queries = ["秋葉aaaki 2024", "影视飓feng 2024-01"]
    constructor = MultiMatchQueryDSLConstructor()
    for query in queries:
        logger.note(f"> [{logstr.mesg(query)}]:")
        query_dsl_dict = constructor.construct(
            query,
            match_fields=match_fields,
            date_match_fields=date_match_fields,
        )
        logger.success(dict_to_str(query_dsl_dict, add_quotes=True), indent=2)

    # python -m converters.query.dsl
