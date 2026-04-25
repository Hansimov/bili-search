from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlunparse

from configs.envs import ELASTIC_DEV_ENVS
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX


DEFAULT_CASES_PATH = Path(__file__).with_name("live_case_corpus_10x10.json")


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a case list: {path}")
    return [case for case in payload if isinstance(case, dict)]


def elastic_endpoint() -> str:
    endpoint = str(ELASTIC_DEV_ENVS.get("endpoint") or "").rstrip("/")
    if endpoint:
        return endpoint.replace("https://localhost", "https://127.0.0.1")
    host = ELASTIC_DEV_ENVS.get("host") or "127.0.0.1"
    if host == "localhost":
        host = "127.0.0.1"
    port = ELASTIC_DEV_ENVS.get("port") or 9200
    return urlunparse(("https", f"{host}:{port}", "", "", "", ""))


def post_json(url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    username = ELASTIC_DEV_ENVS.get("username") or "elastic"
    password = os.environ.get("ELASTIC_PASSWORD") or ELASTIC_DEV_ENVS.get("password")
    if not password or str(password).startswith("YOUR_"):
        password = os.environ.get("ELASTIC_PASSWORD")
    if username and password:
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def chunked(values: list[str], size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def mget_docs(index: str, bvids: list[str], timeout: int) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    includes = "bvid,title,owner.name,pubdate"
    url = f"{elastic_endpoint()}/{index}/_mget?_source_includes={includes}"
    for batch in chunked(bvids, 500):
        result = post_json(
            url,
            {"ids": batch},
            timeout=timeout,
        )
        for doc in result.get("docs") or []:
            if doc.get("found"):
                found[str(doc.get("_id") or "")] = doc.get("_source") or {}
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Check live case bvid coverage in ES")
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--index", default=ELASTIC_VIDEOS_DEV_INDEX)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--missing-limit", type=int, default=30)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    cases = load_cases(args.cases_path)
    bvids = sorted(
        {
            str((case.get("seed") or {}).get("bvid") or "").strip()
            for case in cases
            if (case.get("seed") or {}).get("bvid")
        }
    )
    found = mget_docs(args.index, bvids, args.timeout)
    missing = [bvid for bvid in bvids if bvid not in found]

    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "found": 0})
    for case in cases:
        category = str(case.get("category") or "unknown")
        bvid = str((case.get("seed") or {}).get("bvid") or "").strip()
        if not bvid:
            continue
        grouped[category]["total"] += 1
        if bvid in found:
            grouped[category]["found"] += 1

    print(f"index={args.index}")
    print(f"unique_bvids={len(bvids)}")
    print(f"found={len(found)}")
    print(f"missing={len(missing)}")
    for category in sorted(grouped):
        total = grouped[category]["total"]
        count = grouped[category]["found"]
        print(f"{category} found={count}/{total}")

    for bvid in missing[: args.missing_limit]:
        source_case = next(
            (case for case in cases if (case.get("seed") or {}).get("bvid") == bvid),
            {},
        )
        seed = source_case.get("seed") or {}
        print(
            "MISSING "
            f"{bvid} category={source_case.get('category')} "
            f"title={seed.get('title')} owner={(seed.get('owner') or {}).get('name')}"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "index": args.index,
                    "unique_bvids": len(bvids),
                    "found": sorted(found),
                    "missing": missing,
                    "by_category": grouped,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
