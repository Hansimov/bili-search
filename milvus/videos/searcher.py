import numpy as np
from sedb import MilvusOperator
from tclogger import logger, logstr, dict_to_str, brk

from configs.envs import MILVUS_ENVS
from milvus.videos.constants import MILVUS_OUTPUT_FIELDS
from remotes.fasttext import FasttextModelRunnerClient


class MilvusVideoSearcher:
    def __init__(self, collection: str, verbose: bool = False):
        self.collection = collection
        self.fasttext_client = FasttextModelRunnerClient("doc")
        self.runner = self.fasttext_client.runner
        self.init_milvus()
        self.verbose = verbose

    def init_milvus(self):
        self.milvus = MilvusOperator(
            configs=MILVUS_ENVS,
            connect_msg=f"{logstr.mesg(self.__class__.__name__)} -> {logstr.mesg(brk('milvus'))}",
        )
        self.milvus_client = self.milvus.client

    def search(
        self,
        query: str,
        anns_field: str,
        filter: str = None,
        output_fields: list[str] = MILVUS_OUTPUT_FIELDS,
        search_params: dict = None,
        limit: int = 10,
        timeout: float = None,
    ) -> list[dict]:
        query_info = self.runner.calc_tokens_and_weights_of_sentence(
            query, max_char_len=3
        )
        logger.mesg(dict_to_str(query_info), indent=2)
        query_vector = self.runner.calc_stretch_query_vector(query).astype(np.float16)
        milvus_search_params = {
            "collection_name": self.collection,
            "data": [query_vector],
            "anns_field": anns_field,
            "filter": filter,
            "output_fields": output_fields,
            "search_params": search_params,
            "limit": limit,
            "timeout": timeout,
            "round_decimal": 4,
        }
        results = self.milvus_client.search(**milvus_search_params)
        return results


def test_milvus_video_searcher():
    searcher = MilvusVideoSearcher("videos")
    queries = ["哪吒大电影 魔童降世", "星露谷", "deepsek 小白教程", "华为汽车"]
    for query in queries:
        res = searcher.search(
            query,
            anns_field="title_vec",
            filter="stats_arr[0] > 1000",
            limit=5,
        )
        logger.note(f"> {query}:")
        for item in res:
            logger.mesg(dict_to_str(item))


if __name__ == "__main__":
    test_milvus_video_searcher()

    # python -m milvus.videos.searcher
