import math

from tclogger import get_now, get_now_ts
from typing import Literal, Union

from converters.times import DateFormatChecker
from converters.query.pinyin import ChinesePinyinizer
from converters.field.date import DateFieldConverter
from converters.query.punct import HansChecker
from converters.query.field import is_pinyin_field, is_field_in_fields
from converters.query.field import remove_fields_from_fields
from converters.query.field import remove_suffixes_from_fields
from elastics.videos.constants import SEARCH_MATCH_TYPE

STAT_FIELD_TYPE = Literal[
    "stat.view",
    "stat.like",
    "stat.coin",
    "stat.favorite",
    "stat.danmaku",
    "stat.reply",
    "stat.share",
]
SCORED_STAT_FIELDS = ["stat.like", "stat.coin", "stat.danmaku", "stat.reply"]


class MultiMatchQueryDSLConstructor:
    def __init__(self) -> None:
        self.pinyinizer = ChinesePinyinizer()
        self.date_field_converter = DateFieldConverter()
        self.hans_checker = HansChecker()

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
        date_fields: list[Literal["pubdate"]] = ["pubdate"],
        match_type: Literal["phrase_prefix", "bool_prefix"] = "phrase_prefix",
        match_operator: Literal["or", "and"] = "or",
    ):
        clause = {"bool": {"should": [], "minimum_should_match": 1}}

        if date_fields:
            _, start_ts, end_ts = self.date_field_converter.get_date_ts_range(keyword)
            clause["bool"]["should"].extend(
                [
                    {"range": {date_field: {"gte": start_ts, "lte": end_ts}}}
                    for date_field in date_fields
                ]
            )

        non_date_fields = remove_fields_from_fields(date_fields, date_match_fields)
        if non_date_fields:
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
        """NOTE: Currently this method is not used indeed.
        And only text fields are supported, and they must all have the same search analyzer.
        See also:
        - https://www.elastic.co/guide/en/elasticsearch/reference/8.14/query-dsl-combined-fields-query.html#combined-field-top-level-params
        """
        clauses = []
        for combined_fields in combined_fields_list:
            if all(
                is_field_in_fields(cfield, match_fields) for cfield in combined_fields
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

        fields_with_pinyin = [field for field in match_fields if is_pinyin_field(field)]
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
            field for field in match_fields if not is_pinyin_field(field)
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
        date_fields: list[Literal["pubdate"]] = ["pubdate"],
        match_bool: Literal["must", "should"] = "must",
        match_type: Literal["phrase_prefix", "bool_prefix"] = "phrase_prefix",
        match_operator: Literal["or", "and"] = "or",
        combined_fields_list: list[list[str]] = [],
    ) -> dict:
        query_dsl_dict = {"bool": {match_bool: []}}
        query_keywords = query.split()
        checker = DateFormatChecker()
        match_non_date_fields = remove_fields_from_fields(date_fields, match_fields)
        match_non_date_fields_without_suffix = remove_suffixes_from_fields(
            match_non_date_fields, suffixes=[".words"]
        )
        date_match_fields_without_suffix = remove_suffixes_from_fields(
            date_match_fields, suffixes=[".words"]
        )

        for keyword in query_keywords:
            is_keywod_no_hans = self.hans_checker.no_hans(keyword)
            checker.init_year_month_day()
            is_keyword_date_format = checker.is_in_date_range(
                keyword, start="2009-09-09", end=get_now(), verbose=False
            )
            if is_keyword_date_format:
                clause = self.construct_query_for_date_keyword(
                    keyword,
                    (
                        date_match_fields_without_suffix
                        if is_keywod_no_hans
                        else date_match_fields
                    ),
                    checker,
                    match_type=match_type,
                    match_operator=match_operator,
                )
            else:
                clause = self.construct_query_for_text_keyword(
                    keyword,
                    (
                        match_non_date_fields_without_suffix
                        if is_keywod_no_hans
                        else match_non_date_fields
                    ),
                    match_type=match_type,
                    match_operator=match_operator,
                    combined_fields_list=combined_fields_list,
                )
            query_dsl_dict["bool"][match_bool].append(clause)

        return query_dsl_dict


class ScriptScoreQueryDSLConstructor:
    def log_func(
        self, field: str, min_input: float = 10, min_output: float = 0.0
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
            # 2010-01-01 00:00:00, the beginning of most videos in Bilibili
            default_value = 1262275200
            get_value_func = "doc['pubdate'].value"
            new_var = f"\ndouble pubdate_decay;\n"
        return (
            f"double {field_var} = (doc['{field}'].size() > 0) ? {get_value_func} : {default_value};"
            f"{new_var}"
        )

    def assign_var_of_pubdate_decay_by_reciprocal(
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

    def assign_var_of_pubdate_decay_by_interpolation(
        self,
        field: str = "pubdate",
        now_ts_field: str = "params.now_ts",
        points: list[tuple[float, float]] = [
            *[(0, 4.0), (7, 1.0)],
            *[(30, 0.6), (365, 0.3)],
        ],
        inf_value: float = 0.25,
        # power: float = 1,
    ) -> str:
        """
        Creates a decay function that fits the provided points using linear interpolation.
        Args:
            field (str): The field name for publication date.
            now_ts_field (str): The current timestamp field.
            points (list[tuple[float, float]]): List of (x, y) points representing the decay curve.

        Returns:
            str: An ElasticSearch Painless script to handle pubdate decay.
        """
        seconds_per_day = 86400
        pass_seconds_str = f"({now_ts_field} - {field})"
        pass_days_var = f"double pass_days = {pass_seconds_str}/{seconds_per_day};\n"
        func_str = pass_days_var

        # Create condition blocks for each interval between points
        conditions = []
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]

            # Calculate the slope for linear interpolation
            slope = round((y2 - y1) / (x2 - x1) if x2 != x1 else 0, 6)
            intercept = round(y1 - slope * x1, 6)

            # Convert days to seconds for comparison in script
            condition = f"""if (pass_days < {x2}) {{
                pubdate_decay = {slope} * pass_days + {intercept};
            }}"""
            conditions.append(condition)

        # Handling the case where the date is beyond the maximum x value
        conditions.append(
            f"""{{
                pubdate_decay = {inf_value};
            }}"""
        )

        # Combine conditions
        func_str += " else ".join(conditions)

        # # Apply power function to finetune the weight of pubdate_decay
        # pow_pubdate_decay_str = self.pow_func(
        #     "pubdate_decay", power=power, min_value=inf_value
        # )
        # func_str += f"\npubdate_decay = {pow_pubdate_decay_str};\n"
        return func_str

    def assign_var_of_relevance_score(
        self, power: float = 2, min_value: float = 0.0001, down_scale: float = 100
    ):
        if SEARCH_MATCH_TYPE in ["phrase_prefix", "cross_fields"]:
            dow_score_str = f"(_score / {down_scale})"
            pow_score_str = self.pow_func(
                dow_score_str, power=power, min_value=min_value
            )
            assign_str = f"\ndouble relevance_score = {pow_score_str};\n"
        else:
            assign_str = f"\ndouble relevance_score = (_score + {min_value});\n"
        return assign_str

    def assign_var_of_stats_score(
        self,
        stat_fields: list[STAT_FIELD_TYPE] = SCORED_STAT_FIELDS,
        power: float = 2,
        min_value: float = 1,
    ):
        stats_score_str = " * ".join(
            f"{self.log_func(self.field_to_var(field))}" for field in stat_fields
        )
        stats_score_str = self.pow_func(
            stats_score_str, power=power, min_value=min_value
        )
        assign_str = f"\ndouble stats_score = {stats_score_str};\n"
        return assign_str

    def score_threshold_script(self, name: str = "score_threshold"):
        return f"if (_score < params.{name}) {{ return 0; }}\n"

    def get_script_source_by_powers(self):
        """Deprecated. Use `assign_var_of_stats_score()` + `get_script_source_by_stats()` instead."""
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
        assign_vars_str += self.assign_var_of_pubdate_decay_by_reciprocal()
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
        stat_fields: list[STAT_FIELD_TYPE] = SCORED_STAT_FIELDS,
    ):
        assign_vars = []
        for field in stat_fields + ["pubdate"]:
            assign_vars.append(self.assign_var(field))
        assign_vars_str = "\n".join(assign_vars)
        assign_vars_str += self.assign_var_of_pubdate_decay_by_interpolation()
        assign_vars_str += self.assign_var_of_relevance_score()
        assign_vars_str += self.assign_var_of_stats_score(stat_fields=stat_fields)
        func_str = f"return stats_score * pubdate_decay * relevance_score;"
        script_source = f"{assign_vars_str}\n{func_str}"
        return script_source

    def construct(
        self,
        query_dsl_dict: dict,
        only_script: bool = False,
        score_threshold: float = None,
        combine_type: Literal["sort", "wrap"] = "sort",
    ) -> dict:
        script_source = self.get_script_source_by_stats()
        if score_threshold is not None:
            score_thresold_name = "score_threshold"
            score_threshold_script = self.score_threshold_script(score_thresold_name)
            script_source = f"{score_threshold_script}{script_source}"
            script_dict = {
                "source": script_source,
                "params": {
                    "now_ts": get_now_ts(),
                    score_thresold_name: score_threshold,
                },
            }
        else:
            script_dict = {
                "source": script_source,
                "params": {"now_ts": get_now_ts()},
            }
        if only_script:
            return script_dict
        if combine_type == "wrap":
            script_score_dict = {
                "script_score": {"query": query_dsl_dict, "script": script_dict}
            }
        else:
            script_sort_dict = {
                "_script": {
                    "type": "number",
                    "script": script_dict,
                    "order": "desc",
                }
            }
            script_score_dict = {
                "query": query_dsl_dict,
                "track_scores": True,
                "sort": [script_sort_dict],
            }
        return script_score_dict

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
    match_fields = match_fields_words + match_fields_pinyin
    date_match_fields = match_fields_words

    for idx, mfield in enumerate(match_fields):
        if SEARCH_BOOSTED_FIELDS.get(mfield):
            boost = SEARCH_BOOSTED_FIELDS[mfield]
            match_fields[idx] = f"{mfield}^{boost}"

    # queries = ["秋葉aaaki 2024", "影视飓feng 2024-01"]
    queries = ["雷军 are you ok 2024"]
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
