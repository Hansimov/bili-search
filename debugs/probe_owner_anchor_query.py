from __future__ import annotations

import argparse
import json
from typing import Iterable

from tclogger import logger

from elastics.structure import analyze_tokens
from elastics.videos.constants import ELASTIC_DEV, ELASTIC_VIDEOS_DEV_INDEX
from elastics.videos.searcher_v2 import VideoSearcherV2


TARGET_OWNER = "红警HBK08"
OWNER_FIELDS = ["owner.name.words", "owner.name.suggest"]
FIELD_ANALYZERS = {
    "owner.name.words": "chinese_analyzer",
    "owner.name.suggest": "owner_suggest_analyzer",
}


def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def is_ascii_alnum(text: str) -> bool:
    return bool(text) and all(ord(ch) < 128 and ch.isalnum() for ch in text)


def contains_chinese_and_digits(text: str) -> bool:
    return contains_chinese(text) and any(ch.isdigit() for ch in text)


def compact_even_length_han_segment(text: str) -> str:
    codepoints = [ch for ch in text]
    if len(codepoints) < 4 or len(codepoints) > 8 or (len(codepoints) % 2) != 0:
        return text
    return "".join(ch for idx, ch in enumerate(codepoints) if idx % 2 == 0)


def compact_digit_bearing_chinese_alias_variant(text: str) -> str | None:
    if (
        not text
        or any(ch.isspace() for ch in text)
        or not any(ch.isdigit() for ch in text)
    ):
        return None

    changed = False
    parts: list[str] = []
    index = 0
    while index < len(text):
        ch = text[index]
        if "\u4e00" <= ch <= "\u9fff":
            end = index + 1
            while end < len(text) and "\u4e00" <= text[end] <= "\u9fff":
                end += 1
            segment = text[index:end]
            compacted = compact_even_length_han_segment(segment)
            if compacted != segment:
                changed = True
            parts.append(compacted)
            index = end
            continue
        parts.append(ch)
        index += 1

    variant = "".join(parts)
    return variant if changed and variant != text else None


def owner_query_variants(text: str) -> list[str]:
    variants: list[str] = []
    for candidate in [text, compact_digit_bearing_chinese_alias_variant(text)]:
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def ordered_unique_tokens(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def select_query_terms(seed_terms: list[str], text: str) -> list[str]:
    ordered_terms = sorted(seed_terms, key=lambda term: (-len(term), term))
    chinese_query = contains_chinese(text)
    selected: list[str] = []
    for term in ordered_terms:
        if chinese_query and is_ascii_alnum(term):
            continue
        if chinese_query and contains_chinese_and_digits(term):
            continue
        if any(existing.find(term) >= 0 for existing in selected):
            continue
        selected.append(term)
    return selected or [text.lower()]


def build_query(fields: list[str], selected_terms: list[str]) -> dict:
    should = []
    for field in fields:
        must = [{"prefix": {field: term}} for term in selected_terms if term]
        if must:
            should.append({"bool": {"must": must}})
    return {
        "size": 400,
        "track_total_hits": False,
        "_source": ["owner.name", "owner.mid", "title"],
        "query": {
            "bool": {
                "should": should,
                "minimum_should_match": 1,
            }
        },
    }


def collapse_owner_hits(hits: list[dict]) -> list[dict]:
    owners: dict[int, dict] = {}
    for hit in hits:
        source = hit.get("_source") or {}
        owner = source.get("owner") or {}
        mid = owner.get("mid")
        name = (owner.get("name") or "").strip()
        if not mid or not name:
            continue
        entry = owners.setdefault(
            int(mid),
            {
                "mid": int(mid),
                "name": name,
                "doc_count": 0,
                "max_score": float(hit.get("_score") or 0.0),
                "sample_titles": [],
            },
        )
        entry["doc_count"] += 1
        entry["max_score"] = max(entry["max_score"], float(hit.get("_score") or 0.0))
        title = (source.get("title") or "").strip()
        if (
            title
            and title not in entry["sample_titles"]
            and len(entry["sample_titles"]) < 3
        ):
            entry["sample_titles"].append(title)
    return sorted(
        owners.values(),
        key=lambda item: (-item["max_score"], -item["doc_count"], item["name"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--index", default=ELASTIC_VIDEOS_DEV_INDEX)
    parser.add_argument("--elastic-env", default=ELASTIC_DEV)
    args = parser.parse_args()

    searcher = VideoSearcherV2(index_name=args.index, elastic_env_name=args.elastic_env)

    for variant in owner_query_variants(args.text):
        logger.note(f"\nVARIANT: {variant}")
        seed_terms: list[str] = []
        for field in OWNER_FIELDS:
            analyzer = FIELD_ANALYZERS[field]
            analyzed = analyze_tokens(
                searcher.es.client, args.index, variant, analyzer=analyzer
            )
            tokens = [
                token.get("token", "") for token in analyzed if token.get("token")
            ]
            logger.mesg(f"  {field} tokens={tokens}")
            seed_terms.extend(tokens)

        seed_terms = ordered_unique_tokens(seed_terms)
        selected_terms = select_query_terms(seed_terms, variant)
        logger.mesg(f"  seed_terms={seed_terms}")
        logger.mesg(f"  selected_terms={selected_terms}")

        body = build_query(OWNER_FIELDS, selected_terms)
        response = searcher.submit_to_es(body, context=f"owner_anchor_probe:{variant}")
        hits = (response.get("hits") or {}).get("hits") or []
        owners = collapse_owner_hits(hits)
        target = next(
            (owner for owner in owners if owner["name"] == TARGET_OWNER), None
        )

        logger.mesg(
            f"  hit_count={len(hits)} unique_owners={len(owners)} target_found={bool(target)}"
        )
        if target:
            logger.success(f"  target={json.dumps(target, ensure_ascii=False)}")
        preview = owners[:10]
        logger.file(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
