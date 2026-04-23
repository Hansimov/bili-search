from __future__ import annotations

import argparse
import json

from tclogger import logger

from elastics.structure import analyze_tokens
from elastics.structure import build_auto_constraint_filter
from elastics.structure import select_covering_tokens
from elastics.videos.constants import ELASTIC_DEV
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX
from elastics.videos.explorer import VideoExplorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="袁启 采访")
    parser.add_argument(
        "--qmods",
        nargs="+",
        default=["v", "vwr"],
        help="Explore qmod values to compare",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["title.words", "tags.words", "owner.name.words", "desc.words"],
        help="Fields used by auto constraint filter",
    )
    parser.add_argument("--limit", type=int, default=10)
    return parser


def get_explorer() -> VideoExplorer:
    return VideoExplorer(
        index_name=ELASTIC_VIDEOS_DEV_INDEX,
        elastic_env_name=ELASTIC_DEV,
    )


def extract_main_step(res: dict) -> dict:
    for step in res.get("data", []):
        if step.get("name") in {"knn_search", "hybrid_search", "most_relevant_search"}:
            return step
    return {}


def format_hits(hits: list[dict], limit: int = 5) -> list[dict]:
    rows = []
    for hit in hits[:limit]:
        owner = hit.get("owner") or {}
        rows.append(
            {
                "owner": owner.get("name", ""),
                "title": hit.get("title", ""),
                "recall_lanes": hit.get("_recall_lanes", []),
                "score": hit.get("score"),
            }
        )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    explorer = get_explorer()

    logger.note(f"> Query: {args.query}")
    tokens = analyze_tokens(explorer.es.client, explorer.index_name, args.query)
    logger.file("TOKENS")
    print(json.dumps(tokens, ensure_ascii=False, indent=2))

    covering = select_covering_tokens(tokens, args.query)
    logger.file(f"COVERING: {covering}")

    constraint_query = explorer._resolve_vector_auto_constraint_query(args.query)
    logger.file(f"CONSTRAINT_QUERY: {constraint_query}")

    auto_constraint = build_auto_constraint_filter(
        es_client=explorer.es.client,
        index_name=explorer.index_name,
        query=constraint_query,
        fields=args.fields,
    )
    logger.file("AUTO_CONSTRAINT")
    print(json.dumps(auto_constraint, ensure_ascii=False, indent=2))

    for qmod in args.qmods:
        for enabled in (True, False):
            logger.note(f"\n=== q={qmod} auto_constraint={enabled} ===")
            res = explorer.unified_explore(
                query=f"{args.query} q={qmod}",
                auto_constraint=enabled,
                verbose=False,
            )
            step = extract_main_step(res)
            output = step.get("output") or {}
            logger.file(
                {
                    "step": step.get("name"),
                    "qmod": output.get("qmod"),
                    "total_hits": output.get("total_hits"),
                    "return_hits": output.get("return_hits"),
                    "retry_info": res.get("retry_info"),
                    "top_hits": format_hits(output.get("hits") or [], limit=args.limit),
                }
            )


if __name__ == "__main__":
    main()
