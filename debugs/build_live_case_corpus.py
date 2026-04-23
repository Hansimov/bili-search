from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from elastics.videos.constants import ELASTIC_DEV, ELASTIC_VIDEOS_DEV_INDEX
from elastics.videos.searcher_v2 import VideoSearcherV2


SOURCE_FIELDS = [
    "bvid",
    "title",
    "tags",
    "desc",
    "tid",
    "owner.mid",
    "owner.name",
    "stat.view",
    "stat_score",
    "insert_at",
]
TYPO_SUBSTITUTIONS = {
    "里": "理",
    "理": "里",
    "影": "映",
    "映": "影",
    "猫": "毛",
    "毛": "猫",
    "总": "中",
    "中": "总",
    "动": "冻",
    "冻": "动",
    "学": "雪",
    "雪": "学",
    "生": "声",
    "声": "生",
}
TITLE_SPLIT_RE = re.compile(r"[\s,，。！？!?、|/:：()（）\[\]【】《》<>\"'`~]+")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]{1,15}")


def collapse_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def normalize_text(value: object) -> str:
    return collapse_text(value).lower()


def owner_info(doc: dict) -> dict:
    return doc.get("owner") or {}


def parse_tag_text(doc: dict) -> list[str]:
    tags = doc.get("tags", "")
    if isinstance(tags, list):
        items = tags
    else:
        items = str(tags).replace("#", " ").split(",")
    return [collapse_text(item) for item in items if collapse_text(item)]


def extract_topic_tokens(value: object) -> list[str]:
    normalized = normalize_text(value)
    if not normalized:
        return []

    parts: list[str] = []
    buffer: list[str] = []
    for char in normalized:
        if char.isalnum() or ord(char) > 127:
            buffer.append(char)
            continue
        if buffer:
            parts.append("".join(buffer))
            buffer = []
    if buffer:
        parts.append("".join(buffer))

    tokens: list[str] = []
    for part in parts:
        if len(part) < 2:
            continue
        tokens.append(part)
        if any(ord(char) > 127 for char in part) and len(part) <= 12:
            max_width = min(4, len(part))
            for width in range(2, max_width + 1):
                for index in range(0, len(part) - width + 1):
                    tokens.append(part[index : index + width])
    return tokens


def useful_text(doc: dict) -> bool:
    title = collapse_text(doc.get("title"))
    desc = collapse_text(doc.get("desc"))
    return len(title) >= 6 or len(desc) >= 16


def shorten_title(title: str, limit: int = 36) -> str:
    text = collapse_text(title)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def mutate_single_typo(text: str) -> str:
    if not text:
        return text
    for original, replacement in TYPO_SUBSTITUTIONS.items():
        if original in text:
            return text.replace(original, replacement, 1)
    return text


def has_mixed_script(text: str) -> bool:
    normalized = collapse_text(text)
    if not normalized:
        return False
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in normalized)
    has_ascii = any(char.isascii() and char.isalnum() for char in normalized)
    return has_cjk and has_ascii


def best_topic_token(doc: dict) -> str:
    candidates: list[str] = []
    for value in [doc.get("title"), *parse_tag_text(doc)]:
        candidates.extend(extract_topic_tokens(value))
    filtered = [token for token in candidates if 2 <= len(token) <= 12]
    if not filtered:
        return ""
    return sorted(filtered, key=lambda item: (-len(item), item))[0]


def best_fragment_token(doc: dict) -> str:
    candidates: list[str] = []
    for value in [doc.get("title"), *parse_tag_text(doc)]:
        candidates.extend(extract_topic_tokens(value))
    filtered = [
        token
        for token in candidates
        if 2 <= len(token) <= 4 and any("\u4e00" <= char <= "\u9fff" for char in token)
    ]
    if not filtered:
        return ""
    return sorted(filtered, key=lambda item: (-len(item), item))[0]


def best_mixed_script_query(doc: dict) -> str:
    title = collapse_text(doc.get("title"))
    tags = parse_tag_text(doc)
    for value in [title, *tags]:
        if has_mixed_script(value):
            return value

    ascii_tokens = ASCII_TOKEN_RE.findall(title)
    cjk_token = best_fragment_token(doc)
    if ascii_tokens and cjk_token:
        return collapse_text(f"{ascii_tokens[0]} {cjk_token}")
    return ""


def build_seed(doc: dict) -> dict:
    return {
        "bvid": doc.get("bvid"),
        "title": collapse_text(doc.get("title")),
        "tid": doc.get("tid"),
        "owner": owner_info(doc),
        "tags": parse_tag_text(doc),
    }


def make_case(
    category: str,
    doc: dict,
    *,
    search_query: str,
    chat_content: str,
    extra_tags: list[str] | None = None,
) -> dict | None:
    normalized_query = collapse_text(search_query)
    if len(normalized_query) < 2:
        return None
    return {
        "id": f"{category}_{doc.get('bvid')}",
        "category": category,
        "tags": [category, *(extra_tags or [])],
        "related_mode": "semantic",
        "search_query": normalized_query,
        "chat_messages": [{"role": "user", "content": collapse_text(chat_content)}],
        "seed": build_seed(doc),
    }


def build_title_exact_case(doc: dict) -> dict | None:
    title = shorten_title(collapse_text(doc.get("title")), 42)
    if len(title) < 6:
        return None
    return make_case(
        "title_exact",
        doc,
        search_query=title,
        chat_content=f"最近有哪些值得看的《{title}》相关视频？",
        extra_tags=["title", "hot-doc"],
    )


def build_title_tag_combo_case(doc: dict) -> dict | None:
    title = shorten_title(collapse_text(doc.get("title")), 24)
    tags = parse_tag_text(doc)
    if len(title) < 4 or not tags:
        return None
    combo = collapse_text(f"{title} {tags[0]}")
    return make_case(
        "title_tag_combo",
        doc,
        search_query=combo,
        chat_content=f"有没有讲 {combo} 的高质量视频？",
        extra_tags=["combo", "topic"],
    )


def build_tag_only_case(doc: dict) -> dict | None:
    tags = parse_tag_text(doc)
    if not tags:
        return None
    tag = tags[0]
    if len(tag) < 2:
        return None
    return make_case(
        "tag_only",
        doc,
        search_query=tag,
        chat_content=f"请在B站站内搜索，最近有哪些关于 {tag} 的热门视频？",
        extra_tags=["tag", "broad-topic"],
    )


def build_long_desc_case(doc: dict) -> dict | None:
    title = shorten_title(collapse_text(doc.get("title")), 24)
    desc = collapse_text(doc.get("desc"))
    if len(title) < 4 or len(desc) < 18:
        return None
    query = collapse_text(f"{title} {desc[:32]}")
    return make_case(
        "long_desc",
        doc,
        search_query=query,
        chat_content=f"请找一些和“{query}”最相关的视频。",
        extra_tags=["long", "desc-context"],
    )


def build_boilerplate_noise_case(doc: dict) -> dict | None:
    title = shorten_title(collapse_text(doc.get("title")), 30)
    if len(title) < 6:
        return None
    query = collapse_text(f"{title} 点点关注不错过 持续更新系列中")
    return make_case(
        "boilerplate_noise",
        doc,
        search_query=query,
        chat_content=f"忽略口播和套话，帮我找和 {title} 真正相关的视频。",
        extra_tags=["noise", "boilerplate"],
    )


def build_single_typo_case(doc: dict) -> dict | None:
    base = shorten_title(collapse_text(doc.get("title")), 30)
    if len(base) < 4:
        return None
    typo_query = mutate_single_typo(base)
    if typo_query == base:
        return None
    return make_case(
        "single_typo",
        doc,
        search_query=typo_query,
        chat_content=f"我可能打错了字，想找 {typo_query} 相关的视频。",
        extra_tags=["typo", "robustness"],
    )


def build_owner_recent_case(doc: dict) -> dict | None:
    owner_name = collapse_text(owner_info(doc).get("name"))
    if len(owner_name) < 2:
        return None
    return make_case(
        "owner_recent",
        doc,
        search_query=owner_name,
        chat_content=f"{owner_name} 最近发了什么值得看的视频？",
        extra_tags=["owner-intent", "recent"],
    )


def build_owner_topic_case(doc: dict) -> dict | None:
    owner_name = collapse_text(owner_info(doc).get("name"))
    topic = best_topic_token(doc)
    if len(owner_name) < 2 or len(topic) < 2:
        return None
    query = collapse_text(f"{owner_name} {topic}")
    return make_case(
        "owner_topic",
        doc,
        search_query=query,
        chat_content=f"{owner_name} 关于 {topic} 有哪些值得看的视频？",
        extra_tags=["owner-intent", "topic"],
    )


def build_topic_fragment_case(doc: dict) -> dict | None:
    fragment = best_fragment_token(doc)
    if len(fragment) < 2:
        return None
    return make_case(
        "topic_fragment",
        doc,
        search_query=fragment,
        chat_content=f"{fragment} 相关内容里有哪些比较值得看的视频？",
        extra_tags=["fragment", "short-query"],
    )


def build_mixed_script_case(doc: dict) -> dict | None:
    mixed_query = best_mixed_script_query(doc)
    if len(mixed_query) < 3 or not has_mixed_script(mixed_query):
        return None
    return make_case(
        "mixed_script",
        doc,
        search_query=mixed_query,
        chat_content=f"{mixed_query} 有哪些适合直接上手看的视频？",
        extra_tags=["mixed-script", "alias-surface"],
    )


CATEGORY_BUILDERS = {
    "title_exact": build_title_exact_case,
    "title_tag_combo": build_title_tag_combo_case,
    "tag_only": build_tag_only_case,
    "long_desc": build_long_desc_case,
    "boilerplate_noise": build_boilerplate_noise_case,
    "single_typo": build_single_typo_case,
    "owner_recent": build_owner_recent_case,
    "owner_topic": build_owner_topic_case,
    "topic_fragment": build_topic_fragment_case,
    "mixed_script": build_mixed_script_case,
}


def fetch_docs(
    searcher: VideoSearcherV2,
    size: int,
    sort_clause: list[dict],
    query: dict | None = None,
) -> list[dict]:
    body = {
        "size": size,
        "sort": sort_clause,
        "_source": SOURCE_FIELDS,
        "query": query or {"bool": {"filter": [{"exists": {"field": "title.words"}}]}},
    }
    response = searcher.es.client.search(index=searcher.index_name, body=body)
    body = response.body if hasattr(response, "body") else response
    return [hit.get("_source") or {} for hit in body.get("hits", {}).get("hits", [])]


def fetch_top_tids(searcher: VideoSearcherV2, size: int) -> list[int]:
    body = {
        "size": 0,
        "query": {"bool": {"filter": [{"exists": {"field": "title.words"}}]}},
        "aggs": {"top_tids": {"terms": {"field": "tid", "size": size}}},
    }
    response = searcher.es.client.search(index=searcher.index_name, body=body)
    body = response.body if hasattr(response, "body") else response
    return [
        bucket.get("key")
        for bucket in body.get("aggregations", {})
        .get("top_tids", {})
        .get("buckets", [])
    ]


def append_unique_docs(target: list[dict], docs: list[dict], seen: set[str]) -> None:
    for doc in docs:
        bvid = doc.get("bvid")
        if not bvid or bvid in seen or not useful_text(doc):
            continue
        seen.add(bvid)
        target.append(doc)


def collect_docs(
    searcher: VideoSearcherV2,
    fetch_size: int,
    tid_count: int,
    docs_per_tid: int,
) -> list[dict]:
    combined: list[dict] = []
    seen: set[str] = set()
    for docs in (
        fetch_docs(searcher, fetch_size, [{"insert_at": "desc"}]),
        fetch_docs(searcher, fetch_size, [{"stat.view": "desc"}]),
        fetch_docs(searcher, fetch_size, [{"stat_score": "desc"}]),
    ):
        append_unique_docs(combined, docs, seen)

    for tid in fetch_top_tids(searcher, tid_count):
        tid_query = {
            "bool": {
                "filter": [
                    {"exists": {"field": "title.words"}},
                    {"term": {"tid": tid}},
                ]
            }
        }
        append_unique_docs(
            combined,
            fetch_docs(searcher, docs_per_tid, [{"insert_at": "desc"}], tid_query),
            seen,
        )
        append_unique_docs(
            combined,
            fetch_docs(searcher, docs_per_tid, [{"stat.view": "desc"}], tid_query),
            seen,
        )
    return combined


def build_case_corpus(
    docs: list[dict], per_category: int
) -> tuple[list[dict], dict[str, int]]:
    cases: list[dict] = []
    category_counts: dict[str, int] = defaultdict(int)
    category_seen_queries: dict[str, set[str]] = {
        category: set() for category in CATEGORY_BUILDERS
    }

    for doc in docs:
        for category, builder in CATEGORY_BUILDERS.items():
            if category_counts[category] >= per_category:
                continue
            case = builder(doc)
            if not case:
                continue
            normalized_query = normalize_text(case["search_query"])
            if normalized_query in category_seen_queries[category]:
                continue
            category_seen_queries[category].add(normalized_query)
            cases.append(case)
            category_counts[category] += 1
        if all(count >= per_category for count in category_counts.values()) and len(
            category_counts
        ) == len(CATEGORY_BUILDERS):
            break

    missing = {
        category: per_category - category_counts.get(category, 0)
        for category in CATEGORY_BUILDERS
        if category_counts.get(category, 0) < per_category
    }
    if missing:
        missing_text = ", ".join(
            f"{key}:{value}" for key, value in sorted(missing.items())
        )
        raise RuntimeError(f"insufficient cases for categories: {missing_text}")

    cases.sort(key=lambda item: (item["category"], item["id"]))
    return cases, dict(sorted(category_counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a 10x10 live case corpus from hot docs"
    )
    parser.add_argument("--index", default=ELASTIC_VIDEOS_DEV_INDEX)
    parser.add_argument("--elastic-env", default=ELASTIC_DEV)
    parser.add_argument("--fetch-size", type=int, default=80)
    parser.add_argument("--tid-count", type=int, default=10)
    parser.add_argument("--docs-per-tid", type=int, default=12)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).with_name("live_case_corpus_10x10.json"),
    )
    args = parser.parse_args()

    searcher = VideoSearcherV2(index_name=args.index, elastic_env_name=args.elastic_env)
    docs = collect_docs(
        searcher,
        fetch_size=args.fetch_size,
        tid_count=args.tid_count,
        docs_per_tid=args.docs_per_tid,
    )
    cases, category_counts = build_case_corpus(docs, args.per_category)

    args.output_json.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"docs={len(docs)} cases={len(cases)} output={args.output_json}")
    for category, count in category_counts.items():
        print(f"{category}={count}")


if __name__ == "__main__":
    main()
