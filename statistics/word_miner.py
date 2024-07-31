import hanlp
import hanlp.pretrained
import jieba

from collections import Counter
from pprint import pformat
from tclogger import logger
from typing import Literal, Union

from elastics.video_searcher import VideoSearcher


class WordMiner:
    HANLP_TOKENIZE_MODELS = {
        "coarse": hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH,
        "fine": hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH,
    }
    JIEBA_TOKENIZE_METHODS = {
        "cut": jieba.lcut,
        "cut_for_search": jieba.lcut_for_search,
    }

    def __init__(self):
        self.tokenizer = None

    def load_tokenizer(
        self,
        engine: Literal["jieba", "hanlp"] = "jieba",
        hanlp_level: Literal["coarse", "fine"] = "coarse",
        jieba_method: Literal["cut", "cut_for_search"] = "cut",
    ):
        self.engine = engine
        if engine == "hanlp":
            self.tokenizer = hanlp.load(
                WordMiner.HANLP_TOKENIZE_MODELS[hanlp_level.lower()]
            )
        else:
            self.tokenizer = WordMiner.JIEBA_TOKENIZE_METHODS[jieba_method.lower()]

    def tokenize(self, text: Union[str, list[str]], verbose: bool = False, **kwargs):
        logger.enter_quiet(not verbose)
        logger.note(f"> Tokenizing:", end=" ")
        logger.mesg(f"[{text}]")
        if isinstance(text, list):
            tokenize_res = [self.tokenizer(subtext, **kwargs) for subtext in text]
        else:
            tokenize_res = self.tokenizer(text, **kwargs)
        logger.success(tokenize_res)
        logger.exit_quiet(not verbose)
        return tokenize_res

    def count(self, words: list, verbose: bool = False):
        logger.enter_quiet(not verbose)
        if any(isinstance(word, list) for word in words):
            words = [word for sublist in words for word in sublist]

        counter = Counter(words)
        word_counts = dict(
            sorted(counter.items(), key=lambda item: item[1], reverse=True)
        )
        word_counts = {word: count for word, count in word_counts.items() if count >= 3}
        logger.success(pformat(word_counts, sort_dicts=False, compact=True))
        logger.exit_quiet(not verbose)
        return word_counts


if __name__ == "__main__":
    query = "影视飓风"
    searcher = VideoSearcher("bili_videos_dev")

    search_results = searcher.search(
        query,
        source_fields=["title", "owner.name", "desc", "pubdate_str", "stat"],
        boost=True,
        use_script_score=True,
        detail_level=1,
        limit=200,
        verbose=False,
    )
    titles = [
        " | ".join([res["title"], res["desc"], res["owner"]["name"]])
        for res in search_results["hits"]
    ]
    # titles = ["影视飓风我们团队本期视频使用了高速相机拍摄"]

    miner = WordMiner()
    # miner.load_tokenizer(engine="hanlp", hanlp_level="coarse")
    miner.load_tokenizer(engine="jieba", jieba_method="cut_for_search")
    tokens_list = miner.tokenize(titles, verbose=False)
    word_counts = miner.count(tokens_list, verbose=True)

    # python -m statistics.word_miner
