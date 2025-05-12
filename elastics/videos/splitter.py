import math
import re

from sedb import ElasticOperator
from tclogger import logger, logstr, dict_get, dict_to_str, brk, chars_len
from typing import Union

from configs.envs import ELASTIC_ENVS
from elastics.videos.constants import VIDEOS_INDEX_DEFAULT, SEARCH_ANALYZER_NAME
from elastics.videos.constants import SEARCH_MATCH_FIELDS_WORDS
from llms.agents.entity import QueryEntityExtractor

# https://github.com/infinilabs/analysis-ik/blob/master/core/src/main/java/org/wltea/analyzer/core/Lexeme.java
CHINESE_ANALYZER_TYPES = [
    *["english", "arabic", "letter"],
    *["cn_word", "cn_char", "other_cjk"],
    *["count", "type_cnum", "type_cquan"],
    "unknown",
]

RE_CH_CJK = r"[\u4E00-\u9FFF\u3040-\u30FF]"
PT_CH_CJK = re.compile(RE_CH_CJK)


def calc_cjk_char_len(text: str) -> int:
    return sum(1 for char in text if PT_CH_CJK.match(char))


def calc_pmi(counts: list[int], co_count: int, total_count: int) -> float:
    """Calc pointwise mutual infomation for words.
    See: https://en.wikipedia.org/wiki/Pointwise_mutual_information
    """
    if co_count <= 0 or any(count <= 0 for count in counts):
        return 0.0
    prod_val = math.prod(counts)
    pow_total_count = math.pow(total_count, len(counts) - 1)
    pmi = math.log(co_count * pow_total_count / prod_val)
    return pmi


class QuerySplitter:
    """Split query to well-segmented words."""

    def __init__(self, index_name: str = VIDEOS_INDEX_DEFAULT):
        self.index_name = index_name
        self.es = ElasticOperator(
            ELASTIC_ENVS,
            connect_msg=f"{logstr.mesg(self.__class__.__name__)} -> {logstr.mesg(brk('elastic'))}",
        )
        self.entity_extractor = QueryEntityExtractor()

    def get_total_doc_count(self) -> int:
        """Get total doc count from ES. Useful for calc PMI."""
        es_res = self.es.client.count(index=self.index_name)
        es_res_dict = es_res.body
        if es_res_dict:
            total_count = es_res_dict.get("count", 0)
            return total_count
        else:
            return 0

    def format_token_dicts(self, token_dicts: list[dict]) -> list[dict]:
        for token_dict in token_dicts:
            token_dict["beg"] = token_dict.pop("start_offset", 0) - 1
            token_dict["end"] = token_dict.pop("end_offset", 0) - 1
            token_dict["pos"] = token_dict.pop("position", 0)
        token_dicts.sort(key=lambda x: (x.get("end", 0), -x.get("beg", 0)))
        return token_dicts

    def get_token_idxs(self, token_dicts: list[dict]) -> dict[int, str]:
        res: list[dict] = {}
        for token_dict in token_dicts:
            beg = token_dict.get("beg", 0)
            token = token_dict.get("token", "")
            for ch in token:
                if beg not in res:
                    res[beg] = ch
                beg += 1
        return res

    def get_token_bounds(self, token_dicts: list[dict]) -> dict[tuple[int, int], str]:
        res: dict[tuple[int, int], str] = {}
        for token_dict in token_dicts:
            beg = token_dict.get("beg", 0)
            end = token_dict.get("end", 0)
            token = token_dict.get("token", "")
            res[(beg, end)] = token
        return res

    def analyze(self, query: str, is_sort: bool = True) -> list[dict]:
        """Get tokens by analyzer from ES.
        Example output:
        ```py
        {
            "tokens": [
                {"token": "我",   "beg": 0, "end": 1, "type": "CN_CHAR", "pos": 0},
                {"token": "的",   "beg": 1, "end": 2, "type": "CN_CHAR", "pos": 1},
                {"token": "世界", "beg": 2, "end": 4, "type": "CN_WORD", "pos": 2},
                ...
            ],
            "idxs": {
                0: "我", 1: "的", 2: "世", 3: "界", ...
            }
        }
        ```
        """
        analyze_body = {"text": query, "analyzer": SEARCH_ANALYZER_NAME}
        es_res = self.es.client.indices.analyze(
            index=self.index_name, body=analyze_body
        )
        es_res_dict = es_res.body
        if es_res_dict:
            token_dicts: list[dict] = es_res_dict.get("tokens", [])
            token_dicts = self.format_token_dicts(token_dicts)
            token_idxs = self.get_token_idxs(token_dicts)
            token_bounds = self.get_token_bounds(token_dicts)
            res = {"tokens": token_dicts, "idxs": token_idxs, "bounds": token_bounds}
            print(dict_to_str(res))
            return res
        else:
            return {"tokens": [], "idxs": {}}

    def get_match_clause(self, word: str) -> dict:
        return {
            "multi_match": {
                "query": word,
                "type": "phrase",
                "fields": SEARCH_MATCH_FIELDS_WORDS,
            }
        }

    def get_words_filter_clause(
        self, words: Union[str, list[str]], is_concat: bool = False
    ) -> dict:
        if isinstance(words, str):
            return {words: self.get_match_clause(words)}
        if len(words) < 1:
            return {}
        elif len(words) == 1 or is_concat:
            words_str = "".join(words)
            return {words_str: self.get_match_clause(words_str)}
        else:
            must_clause = [self.get_match_clause(word) for word in words]
            filter_key = "▂".join(words)
            filter_val = {"bool": {"must": must_clause}}
            return {filter_key: filter_val}

    def agg_counts(self, words_list: Union[list[str], list[list[str]]]) -> dict:
        """Get words counts from ES."""
        if not words_list:
            return {}
        agg_filters = {}
        for words in words_list:
            agg_filters.update(self.get_words_filter_clause(words, is_concat=True))
        agg_name = "words_counts"
        agg_body = {
            "size": 0,
            "track_total_hits": False,
            "aggs": {agg_name: {"filters": {"filters": agg_filters}}},
        }
        print(dict_to_str(agg_body, add_quotes=True))
        es_res = self.es.client.search(index=self.index_name, body=agg_body)
        es_res_dict = es_res.body
        if es_res:
            agg_res = dict_get(es_res_dict, f"aggregations.{agg_name}.buckets", {})
            words_counts = {
                k: agg_res.get(k, {}).get("doc_count", 0) for k in agg_filters.keys()
            }
            return words_counts
        else:
            return {}

    def get_ngram_tokens(
        self,
        token_dicts: list[dict],
        max_cjk_chars_len: int = 7,
        max_doc_count: int = None,
        max_ngram_num: int = 5,
    ) -> list[list[dict]]:
        """Shingle tokens to get n-grams. Used to get all possible combinations of words."""
        res: list[list[dict]] = []
        ...

    def split(self, query: str) -> list[str]:
        total_doc_count = self.get_total_doc_count()
        logger.line(f"total_doc_count: {total_doc_count}")
        token_dicts = self.analyze(query)
        # # extract entity from query
        # entity_dict = self.entity_extractor.get_entity(query)
        # logger.line("entity_dict:")
        # logger.okay(dict_to_str(entity_dict))
        # get first-level analyzed tokens and counts
        tokens = [token_dict.get("token", "") for token_dict in token_dicts["tokens"]]
        words_counts = self.agg_counts(tokens)

        ...

        return words_counts
