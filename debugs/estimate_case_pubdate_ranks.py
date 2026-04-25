from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from configs.envs import MONGO_ENVS


DEFAULT_CASES_PATH = Path(__file__).with_name("live_case_corpus_10x10.json")


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a case list: {path}")
    return [case for case in payload if isinstance(case, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate seed bvid positions for pubdate desc indexing"
    )
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--collection", default="videos")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()

    from sedb import MongoOperator

    operator = MongoOperator(MONGO_ENVS, connect_msg="from rank estimator", indent=0)
    collection = operator.db[args.collection]
    cases = load_cases(args.cases_path)
    seed_cases: dict[str, dict[str, Any]] = {}
    for case in cases:
        bvid = str((case.get("seed") or {}).get("bvid") or "").strip()
        if bvid and bvid not in seed_cases:
            seed_cases[bvid] = case

    ranks: list[dict[str, Any]] = []
    for bvid, case in seed_cases.items():
        doc = collection.find_one(
            {"bvid": bvid},
            {"_id": 0, "bvid": 1, "title": 1, "pubdate": 1, "owner.name": 1},
        )
        if not doc:
            ranks.append(
                {
                    "bvid": bvid,
                    "category": case.get("category"),
                    "found_in_mongo": False,
                }
            )
            continue
        filter_dict: dict[str, Any] = {"pubdate": {"$gt": int(doc.get("pubdate") or 0)}}
        if args.start or args.end:
            pubdate_range: dict[str, int] = {}
            if args.start:
                pubdate_range["$gte"] = args.start
            if args.end:
                pubdate_range["$lte"] = args.end
            filter_dict = {"$and": [{"pubdate": pubdate_range}, filter_dict]}
        newer = collection.count_documents(filter_dict)
        ranks.append(
            {
                "bvid": bvid,
                "category": case.get("category"),
                "found_in_mongo": True,
                "rank_pubdate_desc": newer + 1,
                "pubdate": int(doc.get("pubdate") or 0),
                "title": doc.get("title"),
                "owner": ((doc.get("owner") or {}).get("name") or ""),
            }
        )

    ranks.sort(key=lambda item: item.get("rank_pubdate_desc") or 10**18)
    found = [item for item in ranks if item.get("found_in_mongo")]
    print(f"unique_bvids={len(seed_cases)}")
    print(f"found_in_mongo={len(found)}")
    if found:
        print(f"min_rank={min(item['rank_pubdate_desc'] for item in found)}")
        print(f"max_rank={max(item['rank_pubdate_desc'] for item in found)}")
    for item in ranks:
        if item.get("found_in_mongo"):
            print(
                "RANK "
                f"{item['rank_pubdate_desc']} {item['bvid']} "
                f"category={item['category']} pubdate={item['pubdate']} "
                f"title={item['title']} owner={item['owner']}"
            )
        else:
            print(f"MISSING_MONGO {item['bvid']} category={item['category']}")


if __name__ == "__main__":
    main()
