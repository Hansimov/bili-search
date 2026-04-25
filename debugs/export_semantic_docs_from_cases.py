from __future__ import annotations

import argparse
import json
import sys

from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def collapse_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def normalize_tag_list(value: object) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    elif value is None:
        raw_items = []
    else:
        raw_items = str(value).replace("#", " ").split(",")
    return [collapse_text(item) for item in raw_items if collapse_text(item)]


def normalize_doc(seed: dict, case: dict) -> dict:
    owner = seed.get("owner") if isinstance(seed.get("owner"), dict) else {}
    tags = seed.get("tags") if isinstance(seed.get("tags"), list) else []
    return {
        "bvid": seed.get("bvid"),
        "title": collapse_text(seed.get("title")),
        "tid": seed.get("tid"),
        "owner": {
            "mid": owner.get("mid"),
            "name": collapse_text(owner.get("name")),
        },
        "tags": [collapse_text(tag) for tag in tags if collapse_text(tag)],
        "stat": seed.get("stat") if isinstance(seed.get("stat"), dict) else {},
        "case_id": case.get("id"),
        "case_category": case.get("category"),
        "search_query": collapse_text(case.get("search_query")),
    }


def normalize_probe_doc(source_doc: dict, case: dict) -> dict:
    return {
        "bvid": source_doc.get("bvid"),
        "title": collapse_text(source_doc.get("title")),
        "tid": source_doc.get("tid"),
        "owner": {
            "mid": None,
            "name": collapse_text(source_doc.get("owner")),
        },
        "tags": normalize_tag_list(source_doc.get("tags")),
        "rtags": normalize_tag_list(source_doc.get("rtags")),
        "stat": {"view": source_doc.get("view")},
        "case_id": case.get("term"),
        "case_category": "probe_report",
        "search_query": collapse_text(case.get("term")),
    }


def load_case_like_items(paths: list[Path]) -> list[dict]:
    items: list[dict] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            items.extend(item for item in data if isinstance(item, dict))
            continue
        if isinstance(data, dict) and isinstance(data.get("cases"), list):
            items.extend(item for item in data["cases"] if isinstance(item, dict))
            continue
        raise ValueError(f"unsupported input JSON structure: {path}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert live case corpus seed docs to semantic docs JSONL"
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Input case corpus JSON files"
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path)
    args = parser.parse_args()

    cases = load_case_like_items(args.inputs)
    docs_by_bvid: dict[str, dict] = {}
    for case in cases:
        if isinstance(case.get("seed"), dict):
            doc = normalize_doc(case["seed"], case)
        elif isinstance(case.get("source_doc"), dict):
            doc = normalize_probe_doc(case["source_doc"], case)
        else:
            continue
        bvid = collapse_text(doc.get("bvid"))
        if not bvid or not doc.get("title"):
            continue
        docs_by_bvid[bvid] = doc

    docs = sorted(
        docs_by_bvid.values(),
        key=lambda item: (item.get("tid") or 0, item.get("bvid") or ""),
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    tid_counter = Counter(doc.get("tid") for doc in docs if doc.get("tid") is not None)
    summary = {
        "input_count": len(args.inputs),
        "case_count": len(cases),
        "doc_count": len(docs),
        "tid_count": len(tid_counter),
        "top_tids": tid_counter.most_common(24),
        "output_jsonl": str(args.output_jsonl),
    }
    print(
        f"cases={summary['case_count']} docs={summary['doc_count']} tids={summary['tid_count']} output={args.output_jsonl}"
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
