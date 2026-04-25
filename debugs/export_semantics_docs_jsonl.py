from __future__ import annotations

import argparse
import json
import sys

from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from debugs.build_live_case_corpus import collect_docs
from debugs.probe_semantic_snapshot_live import (
    fetch_mongo_docs as fetch_recent_window_mongo_docs,
    parse_time,
)
from elastics.videos.constants import ELASTIC_DEV, ELASTIC_VIDEOS_DEV_INDEX
from elastics.videos.searcher_v2 import VideoSearcherV2


MONGO_SOURCE_FIELDS = {
    "_id": 0,
    "bvid": 1,
    "title": 1,
    "desc": 1,
    "tags": 1,
    "rtags": 1,
    "tid": 1,
    "owner": 1,
    "stat": 1,
    "insert_at": 1,
    "pubdate": 1,
    "stat_score": 1,
}


def collapse_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def useful_text(doc: dict) -> bool:
    title = collapse_text(doc.get("title"))
    desc = collapse_text(doc.get("desc"))
    return len(title) >= 6 or len(desc) >= 16


def append_unique_docs(target: list[dict], docs: list[dict], seen: set[str]) -> None:
    for doc in docs:
        bvid = doc.get("bvid")
        if not bvid or bvid in seen or not useful_text(doc):
            continue
        seen.add(bvid)
        target.append(doc)


def fetch_mongo_docs(
    searcher: VideoSearcherV2,
    limit: int,
    sort_clause: dict,
    match_clause: dict | None = None,
) -> list[dict]:
    filters = {
        "title": {"$exists": True, "$type": "string", "$ne": ""},
        **(match_clause or {}),
    }
    pipeline = [
        {"$match": filters},
        {"$sort": sort_clause},
        {"$limit": limit},
        {"$project": MONGO_SOURCE_FIELDS},
    ]
    return list(
        searcher.mongo.get_agg_cursor(
            "videos", pipeline, batch_size=min(max(limit, 8), 256)
        )
    )


def collect_docs_from_mongo(
    searcher: VideoSearcherV2,
    fetch_size: int,
    tid_count: int,
    docs_per_tid: int,
    min_view: int,
) -> list[dict]:
    combined: list[dict] = []
    seen: set[str] = set()

    base_match = {"stat.view": {"$gte": min_view}} if min_view > 0 else None
    for docs in (
        fetch_mongo_docs(searcher, fetch_size, {"insert_at": -1}, base_match),
        fetch_mongo_docs(searcher, fetch_size, {"stat.view": -1}, base_match),
        fetch_mongo_docs(searcher, fetch_size, {"stat_score": -1}, base_match),
    ):
        append_unique_docs(combined, docs, seen)

    candidate_tids = [
        tid
        for tid, _count in Counter(
            doc.get("tid") for doc in combined if doc.get("tid") is not None
        ).most_common(tid_count)
    ]

    for tid in candidate_tids:
        tid_match = {"tid": tid}
        if min_view > 0:
            tid_match["stat.view"] = {"$gte": min_view}
        append_unique_docs(
            combined,
            fetch_mongo_docs(searcher, docs_per_tid, {"insert_at": -1}, tid_match),
            seen,
        )
        append_unique_docs(
            combined,
            fetch_mongo_docs(searcher, docs_per_tid, {"stat.view": -1}, tid_match),
            seen,
        )
    return combined


def doc_view(doc: dict) -> int:
    stat = doc.get("stat") or {}
    view = stat.get("view") if isinstance(stat, dict) else None
    if view is None:
        view = doc.get("stat.view")
    try:
        return int(view or 0)
    except (TypeError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a larger cross-domain semantic docs corpus as JSONL"
    )
    parser.add_argument("--index", default=ELASTIC_VIDEOS_DEV_INDEX)
    parser.add_argument("--elastic-env", default=ELASTIC_DEV)
    parser.add_argument("--fetch-size", type=int, default=120)
    parser.add_argument("--tid-count", type=int, default=24)
    parser.add_argument("--docs-per-tid", type=int, default=18)
    parser.add_argument("--max-docs", type=int, default=360)
    parser.add_argument("--min-view", type=int, default=5000)
    parser.add_argument(
        "--start", help="Optional recent-window start time, e.g. 2026-04-01"
    )
    parser.add_argument(
        "--end", help="Optional recent-window end time, e.g. 2026-04-25"
    )
    parser.add_argument("--doc-limit", type=int, default=120)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Target JSONL path for exported source docs",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        help="Optional summary JSON path",
    )
    args = parser.parse_args()

    searcher = VideoSearcherV2(index_name=args.index, elastic_env_name=args.elastic_env)
    source = "elastic"
    if args.start and args.end:
        start_ts = parse_time(args.start)
        end_ts = parse_time(args.end)
        if start_ts > end_ts:
            raise ValueError("start must be earlier than end")
        docs = fetch_recent_window_mongo_docs(
            searcher,
            start_ts=start_ts,
            end_ts=end_ts,
            doc_limit=args.doc_limit,
            min_view=args.min_view,
        )
        source = "mongo_recent_window"
    else:
        docs = collect_docs(
            searcher,
            fetch_size=args.fetch_size,
            tid_count=args.tid_count,
            docs_per_tid=args.docs_per_tid,
        )
        docs = [doc for doc in docs if doc_view(doc) >= args.min_view]
        if not docs:
            docs = collect_docs_from_mongo(
                searcher,
                fetch_size=args.fetch_size,
                tid_count=args.tid_count,
                docs_per_tid=args.docs_per_tid,
                min_view=args.min_view,
            )
            source = "mongo"
    if args.max_docs > 0:
        docs = docs[: args.max_docs]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    tid_counter = Counter(doc.get("tid") for doc in docs if doc.get("tid") is not None)
    summary = {
        "index": args.index,
        "elastic_env": args.elastic_env,
        "source": source,
        "doc_count": len(docs),
        "tid_count": len(tid_counter),
        "top_tids": tid_counter.most_common(24),
        "output_jsonl": str(args.output_jsonl),
    }

    print(
        f"source={summary['source']} docs={summary['doc_count']}"
        f" tids={summary['tid_count']} output={args.output_jsonl}"
    )
    for tid, count in summary["top_tids"][:12]:
        print(f"tid={tid} count={count}")

    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
