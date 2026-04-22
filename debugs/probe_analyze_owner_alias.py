from __future__ import annotations

import argparse
import json

from elastics.structure import analyze_tokens
from elastics.videos.constants import ELASTIC_DEV, ELASTIC_VIDEOS_DEV_INDEX
from elastics.videos.searcher_v2 import VideoSearcherV2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("texts", nargs="+", help="Texts to analyze")
    parser.add_argument(
        "--analyzers",
        nargs="+",
        default=["chinese_analyzer", "owner_suggest_analyzer"],
        help="Analyzer names configured on the live index",
    )
    parser.add_argument("--index", default=ELASTIC_VIDEOS_DEV_INDEX)
    parser.add_argument("--elastic-env", default=ELASTIC_DEV)
    args = parser.parse_args()

    searcher = VideoSearcherV2(
        index_name=args.index,
        elastic_env_name=args.elastic_env,
    )

    for analyzer in args.analyzers:
        print(f"ANALYZER: {analyzer}")
        for text in args.texts:
            tokens = analyze_tokens(
                searcher.es.client, args.index, text, analyzer=analyzer
            )
            print(f"TEXT: {text}")
            print(json.dumps(tokens, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
