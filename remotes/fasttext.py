import Pyro5.api
import Pyro5.core
import Pyro5.server

from tclogger import logger, logstr, dict_to_str
from typing import Literal, Union

from configs.envs import PYRO_ENVS

PYRO_NS = {
    "word": "fasttext_model_runner_word",
    "doc": "fasttext_model_runner_doc",
}


class FasttextModelRunnerClient:
    def __init__(
        self,
        model_class: Literal["word", "doc"] = "doc",
        host: str = PYRO_ENVS["host"],
        port: int = PYRO_ENVS["port"],
        verbose: bool = False,
    ):
        self.model_class = model_class
        self.nameserver = PYRO_NS.get(model_class, "fasttext_model_runner")
        self.uri = f"PYRO:{self.nameserver}@{host}:{port}"
        self.runner = Pyro5.api.Proxy(self.uri)
        self.verbose = verbose


def test_fasttext_model_runner_client():
    client = FasttextModelRunnerClient()
    sent = "我爱北京天安门"
    max_char_len = 3
    tokens = client.runner.preprocessor_func("preprocess", sent, max_char_len=None)
    tokens_splits, split_idxs = client.runner.preprocessor_func(
        "get_tokens_splits_and_idxs", tokens, max_char_len=max_char_len
    )
    logger.mesg(f"{sent}: {logstr.success(tokens)}")
    logger.file(f"* {tokens_splits}")
    logger.file(f"* {split_idxs}")

    sentence_info = client.runner.calc_tokens_and_weights_of_sentence(
        sent, max_char_len=3
    )
    logger.mesg(dict_to_str(sentence_info), indent=2)


if __name__ == "__main__":
    test_fasttext_model_runner_client()

    # python -m remotes.fasttext
