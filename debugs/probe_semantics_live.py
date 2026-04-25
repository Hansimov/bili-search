from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from elastics.relations import RelationsClient
from elastics.relations.client import ES_COMPAT_MEDIA_TYPE
from elastics.videos.constants import ELASTIC_DEV, ELASTIC_VIDEOS_DEV_INDEX


TERM_SPLIT_RE = r"[,，/|#\s]+"
TITLE_SPLIT_RE = r"[\s,，。！？!?、|/:：()（）\[\]【】《》<>\"'`~]+"


def collapse_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def normalize_term(value: object) -> str:
    return collapse_text(value).lower()


def split_text(pattern: str, value: object) -> list[str]:
    import re

    return [item for item in re.split(pattern, str(value or "")) if collapse_text(item)]


def iter_tag_terms(doc: dict) -> list[str]:
    terms: list[str] = []
    for field_name in ("tags", "rtags"):
        raw_value = doc.get(field_name)
        if raw_value is None:
            continue
        raw_terms = (
            raw_value
            if isinstance(raw_value, list)
            else split_text(TERM_SPLIT_RE, raw_value)
        )
        for raw_term in raw_terms:
            term = normalize_term(raw_term)
            if 2 <= len(term) <= 24:
                terms.append(term)
    return terms


def iter_title_terms(doc: dict) -> list[str]:
    title = collapse_text(doc.get("title"))
    if not title:
        return []
    terms: list[str] = []
    normalized_title = normalize_term(title)
    if 2 <= len(normalized_title) <= 24:
        terms.append(normalized_title)
    for part in split_text(TITLE_SPLIT_RE, title):
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
        return response if isinstance(response, dict) else dict(response)
    except Exception as exc:
        return {
            "error": str(exc),
            "relation": "related_tokens_by_tokens",
            **payload,
        }


def load_docs(path: Path, max_docs: int) -> list[dict]:
    docs: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        docs.append(json.loads(line))
        if max_docs > 0 and len(docs) >= max_docs:
            break
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe es-tok semantic suggestions from exported docs JSONL"
    )
    parser.add_argument("--index", default=ELASTIC_VIDEOS_DEV_INDEX)
    parser.add_argument("--elastic-env", default=ELASTIC_DEV)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=48)
    parser.add_argument("--max-terms-per-doc", type=int, default=3)
    parser.add_argument("--options-limit", type=int, default=5)
    parser.add_argument("--scan-limit", type=int, default=128)
    parser.add_argument("--probe-timeout", type=float, default=20.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    docs = load_docs(args.input_jsonl, args.max_docs)
    client = RelationsClient(args.index, elastic_env_name=args.elastic_env)

    cases: list[dict] = []
    seen_terms: set[str] = set()
    for doc in docs:
        for term in extract_probe_terms(doc, args.max_terms_per_doc):
            if term in seen_terms:
                continue
            seen_terms.add(term)
            result = request_related_tokens(
                client,
                term=term,
                options_limit=args.options_limit,
                scan_limit=args.scan_limit,
                request_timeout=args.probe_timeout,
            )
            options = [
                {
                    "text": option.get("text"),
                    "score": option.get("score"),
                    "type": option.get("type"),
                }
                for option in (result.get("options") or [])[: args.options_limit]
            ]
            cases.append(
                {
                    "term": term,
                    "tid": doc.get("tid"),
                    "title": doc.get("title"),
                    "owner": (doc.get("owner") or {}).get("name"),
                    "view": ((doc.get("stat") or {}).get("view")),
                    "error": result.get("error"),
                    "options": options,
                }
            )

    summary = {
        "index": args.index,
        "elastic_env": args.elastic_env,
        "doc_count": len(docs),
        "term_count": len(cases),
        "non_empty_case_count": sum(1 for case in cases if case.get("options")),
        "cases": cases,
    }

    print(
        f"docs={summary['doc_count']} term_count={summary['term_count']} non_empty={summary['non_empty_case_count']}"
    )
    for case in cases:
        print(
            f"- tid={case.get('tid')} term={case['term']}"
            f" | owner={case.get('owner')}"
            f" | view={case.get('view')}"
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
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
