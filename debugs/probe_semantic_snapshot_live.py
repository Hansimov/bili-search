from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from elastics.relations import RelationsClient
from elastics.relations.client import ES_COMPAT_MEDIA_TYPE
from elastics.videos.constants import ELASTIC_DEV, ELASTIC_VIDEOS_DEV_INDEX
from elastics.videos.searcher_v2 import VideoSearcherV2


LOCAL_TZ = ZoneInfo("Asia/Shanghai")
TERM_SPLIT_RE = re.compile(r"[,，/|#\s]+")
TITLE_SPLIT_RE = re.compile(r"[\s,，。！？!?、|/:：()（）\[\]【】《》<>\"'`~]+")
MONGO_REPLAY_FIELDS = {
    "_id": 0,
    "bvid": 1,
    "title": 1,
    "desc": 1,
    "tid": 1,
    "ptid": 1,
    "tname": 1,
    "tags": 1,
    "rtags": 1,
    "owner": 1,
    "pic": 1,
    "duration": 1,
    "stat": 1,
    "pubdate": 1,
    "insert_at": 1,
}


def parse_time(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        raise ValueError("time value cannot be empty")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return int(
                datetime.strptime(text, fmt).replace(tzinfo=LOCAL_TZ).timestamp()
            )
        except ValueError:
            continue
    raise ValueError(f"unsupported time format: {text}")


def collapse_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def normalize_term(value: object) -> str:
    text = collapse_text(value).lower()
    if not text:
        return ""
    return text


def iter_tag_terms(doc: dict) -> list[str]:
    terms: list[str] = []
    for field_name in ("tags", "rtags"):
        raw_value = doc.get(field_name)
        if raw_value is None:
            continue
        if isinstance(raw_value, list):
            raw_terms = raw_value
        else:
            raw_terms = TERM_SPLIT_RE.split(str(raw_value))
        for raw_term in raw_terms:
            term = normalize_term(raw_term)
            if 2 <= len(term) <= 24:
                terms.append(term)
    return terms


def iter_title_terms(doc: dict) -> list[str]:
    title = collapse_text(doc.get("title"))
    if not title:
        return []
    normalized_title = normalize_term(title)
    terms: list[str] = []
    if 2 <= len(normalized_title) <= 24:
        terms.append(normalized_title)
    for part in TITLE_SPLIT_RE.split(title):
        term = normalize_term(part)
        if 2 <= len(term) <= 24:
            terms.append(term)
    return terms


def extract_probe_terms(doc: dict, max_terms: int) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for term in [*iter_tag_terms(doc), *iter_title_terms(doc)]:
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def build_recent_docs_query(
    start_ts: int, end_ts: int, doc_limit: int, min_view: int
) -> dict:
    filters: list[dict] = [
        {"range": {"pubdate": {"gte": start_ts, "lte": end_ts}}},
    ]
    if min_view > 0:
        filters.append({"range": {"stat.view": {"gte": min_view}}})
    return {
        "size": doc_limit,
        "sort": [
            {"stat.view": {"order": "desc", "missing": "_last"}},
            {"pubdate": {"order": "desc"}},
        ],
        "_source": [
            "bvid",
            "title",
            "tags",
            "rtags",
            "owner.name",
            "stat.view",
            "pubdate",
        ],
        "query": {"bool": {"filter": filters}},
    }


def build_mongo_replay_pipeline(
    start_ts: int,
    end_ts: int,
    doc_limit: int,
    min_view: int,
) -> list[dict]:
    match_clause: dict = {
        "pubdate": {"$gte": start_ts, "$lte": end_ts},
    }
    if min_view > 0:
        match_clause["stat.view"] = {"$gte": min_view}
    return [
        {"$match": match_clause},
        {"$sort": {"stat.view": -1, "pubdate": -1}},
        {"$limit": doc_limit},
        {"$project": MONGO_REPLAY_FIELDS},
    ]


def fetch_recent_docs(
    searcher: VideoSearcherV2,
    start_ts: int,
    end_ts: int,
    doc_limit: int,
    min_view: int,
) -> list[dict]:
    response = searcher.es.client.search(
        index=searcher.index_name,
        body=build_recent_docs_query(start_ts, end_ts, doc_limit, min_view),
    )
    body = response.body if hasattr(response, "body") else response
    return [hit.get("_source") or {} for hit in body.get("hits", {}).get("hits", [])]


def fetch_mongo_docs(
    searcher: VideoSearcherV2,
    start_ts: int,
    end_ts: int,
    doc_limit: int,
    min_view: int,
) -> list[dict]:
    cursor = searcher.mongo.get_agg_cursor(
        "videos",
        build_mongo_replay_pipeline(start_ts, end_ts, doc_limit, min_view),
        batch_size=min(max(doc_limit, 8), 256),
    )
    return list(cursor)


def replay_docs_to_es(searcher: VideoSearcherV2, docs: list[dict]) -> list[str]:
    replayed_bvids: list[str] = []
    for doc in docs:
        bvid = collapse_text(doc.get("bvid"))
        if not bvid:
            continue
        searcher.es.client.index(index=searcher.index_name, id=bvid, document=doc)
        replayed_bvids.append(bvid)
    if replayed_bvids:
        searcher.es.client.indices.refresh(index=searcher.index_name)
    return replayed_bvids


def request_related_tokens(
    client: RelationsClient,
    *,
    term: str,
    options_limit: int,
    scan_limit: int,
    request_timeout: float,
) -> dict:
    path = f"/{client.index_name}/_es_tok/related_tokens_by_tokens"
    payload = {
        "text": term,
        "mode": "semantic",
        "fields": ["title.words", "tags.words", "owner.name.words"],
        "size": options_limit,
        "scan_limit": scan_limit,
        "use_pinyin": True,
    }
    try:
        es_client = client.es.client.options(request_timeout=request_timeout)
        response = es_client.perform_request(
            method="POST",
            path=path,
            body=payload,
            headers={
                "Accept": ES_COMPAT_MEDIA_TYPE,
                "Content-Type": ES_COMPAT_MEDIA_TYPE,
            },
        )
        if hasattr(response, "body"):
            response = response.body
        if isinstance(response, dict):
            return response
        return dict(response)
    except Exception as exc:
        return {
            "error": str(exc),
            "relation": "related_tokens_by_tokens",
            **payload,
        }


def probe_terms(
    client: RelationsClient,
    docs: list[dict],
    *,
    max_terms_per_doc: int,
    options_limit: int,
    scan_limit: int,
    request_timeout: float,
) -> list[dict]:
    cases: list[dict] = []
    seen_terms: set[str] = set()
    for doc in docs:
        for term in extract_probe_terms(doc, max_terms_per_doc):
            if term in seen_terms:
                continue
            seen_terms.add(term)
            result = request_related_tokens(
                client,
                term=term,
                options_limit=options_limit,
                scan_limit=scan_limit,
                request_timeout=request_timeout,
            )
            options = [
                {
                    "text": option.get("text"),
                    "score": option.get("score"),
                    "type": option.get("type"),
                    "keywords": option.get("keywords"),
                }
                for option in (result.get("options") or [])[:options_limit]
            ]
            cases.append(
                {
                    "term": term,
                    "source_doc": {
                        "bvid": doc.get("bvid"),
                        "title": doc.get("title"),
                        "owner": (doc.get("owner") or {}).get("name"),
                        "view": ((doc.get("stat") or {}).get("view")),
                        "pubdate": doc.get("pubdate"),
                    },
                    "mode": result.get("mode"),
                    "error": result.get("error"),
                    "options": options,
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        default=ELASTIC_VIDEOS_DEV_INDEX,
        help="Elasticsearch videos index name",
    )
    parser.add_argument(
        "--elastic-env",
        default=ELASTIC_DEV,
        help="Elastic environment name in configs/secrets.json",
    )
    parser.add_argument(
        "--start",
        required=True,
        help='Start time, e.g. "2026-04-22 12:00:00" or "2026-04-22"',
    )
    parser.add_argument(
        "--end",
        required=True,
        help='End time, e.g. "2026-04-22 18:00:00" or "2026-04-23"',
    )
    parser.add_argument("--doc-limit", type=int, default=20)
    parser.add_argument("--min-view", type=int, default=5000)
    parser.add_argument(
        "--replay-limit",
        type=int,
        default=0,
        help="Replay this many real docs from Mongo into the target index before probing",
    )
    parser.add_argument("--max-terms-per-doc", type=int, default=4)
    parser.add_argument("--options-limit", type=int, default=5)
    parser.add_argument("--scan-limit", type=int, default=128)
    parser.add_argument("--probe-timeout", type=float, default=5.0)
    parser.add_argument("--output-json", help="Optional JSON report path")
    args = parser.parse_args()

    start_ts = parse_time(args.start)
    end_ts = parse_time(args.end)
    if start_ts > end_ts:
        raise ValueError("start must be earlier than end")

    searcher = VideoSearcherV2(index_name=args.index, elastic_env_name=args.elastic_env)
    relations_client = RelationsClient(args.index, elastic_env_name=args.elastic_env)

    replayed_bvids: list[str] = []
    if args.replay_limit > 0:
        mongo_docs = fetch_mongo_docs(
            searcher,
            start_ts=start_ts,
            end_ts=end_ts,
            doc_limit=args.replay_limit,
            min_view=args.min_view,
        )
        replayed_bvids = replay_docs_to_es(searcher, mongo_docs)

    docs = fetch_recent_docs(
        searcher,
        start_ts=start_ts,
        end_ts=end_ts,
        doc_limit=args.doc_limit,
        min_view=args.min_view,
    )
    cases = probe_terms(
        relations_client,
        docs,
        max_terms_per_doc=args.max_terms_per_doc,
        options_limit=args.options_limit,
        scan_limit=args.scan_limit,
        request_timeout=args.probe_timeout,
    )

    summary = {
        "index": args.index,
        "elastic_env": args.elastic_env,
        "start": args.start,
        "end": args.end,
        "replayed_bvids": replayed_bvids,
        "doc_count": len(docs),
        "case_count": len(cases),
        "non_empty_case_count": sum(1 for case in cases if case.get("options")),
        "cases": cases,
    }

    print(
        f"window={args.start} -> {args.end} replayed={len(replayed_bvids)} doc_count={len(docs)} case_count={len(cases)} non_empty={summary['non_empty_case_count']}"
    )
    for case in cases:
        doc = case["source_doc"]
        print(
            f"- term={case['term']}"
            f" | bvid={doc.get('bvid')}"
            f" | owner={doc.get('owner')}"
            f" | view={doc.get('view')}"
        )
        if case.get("error"):
            print(f"    error: {case['error']}")
            continue
        for option in case.get("options") or []:
            print(
                f"    -> {option.get('text')}"
                f" | score={option.get('score')}"
                f" | type={option.get('type')}"
            )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
