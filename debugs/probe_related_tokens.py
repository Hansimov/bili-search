from __future__ import annotations

import argparse
import json

from elastics.relations import RelationsClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--mode", default="correction")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--scan-limit", type=int, default=128)
    parser.add_argument("--index", default="bili_videos_dev6")
    parser.add_argument("--elastic-env", default="elastic_dev")
    args = parser.parse_args()

    client = RelationsClient(
        args.index,
        elastic_env_name=args.elastic_env,
    )
    result = client.related_tokens_by_tokens(
        text=args.text,
        mode=args.mode,
        size=args.size,
        scan_limit=args.scan_limit,
        use_pinyin=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
