"""Run 50 live natural-language QA quality cases against bili-search.

The runner samples the generated 10x10 live corpus, probes search/explore/chat
endpoints concurrently, and writes both machine-readable and human-readable
quality reports.

Usage:
    python debugs/run_live_qa_quality.py
    python debugs/run_live_qa_quality.py --sample-size 50 --per-category 5
    python debugs/run_live_qa_quality.py --skip-chat --workers 12
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import statistics
import threading
import time
from typing import Any
import urllib.error
import urllib.request

from configs.envs import SEARCH_APP_ENVS


DEFAULT_BASE_URL = f"http://127.0.0.1:{SEARCH_APP_ENVS.get('port', 21001)}"
DEFAULT_CASES_PATH = Path(__file__).with_name("live_case_corpus_10x10.json")
DEFAULT_REPORTS_DIR = Path(__file__).with_name("live_case_reports")

STANDARD_NAMES = {
    "authority": "权威",
    "quality": "质量",
    "core_intent": "核心",
    "freshness": "时效",
    "relevance": "相关",
    "stability": "稳定",
}
FAIL_THRESHOLD = 0.66
WARN_THRESHOLD = 0.8
ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9._+-]{1,30}")
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
NOISE_PHRASES = [
    "点点关注不错过",
    "持续更新系列中",
    "三连",
    "关注",
    "点赞",
    "投币",
]


@dataclass(frozen=True)
class ProbeOptions:
    base_url: str
    limit: int
    timeout: int
    max_iterations: int
    thinking: bool
    skip_search: bool
    skip_related: bool
    skip_explore: bool
    skip_chat: bool
    chat_retries: int


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError(f"Expected a list of cases in {path}")
    return cases


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").lower().split()).strip()


def truncate_text(text: object, limit: int = 700) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def post_json(
    base_url: str, endpoint: str, payload: dict[str, Any], timeout: int
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def flatten_tool_names(tool_events: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for event in tool_events or []:
        names.extend(event.get("tools") or [])
    return names


def numeric(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def hit_owner_name(hit: dict[str, Any]) -> str:
    owner = hit.get("owner") or {}
    if isinstance(owner, dict):
        return str(owner.get("name") or "")
    return ""


def hit_summary(hit: dict[str, Any]) -> dict[str, Any]:
    stat = hit.get("stat") or {}
    return {
        "title": hit.get("title"),
        "owner": hit_owner_name(hit),
        "bvid": hit.get("bvid"),
        "tags": hit.get("tags"),
        "view": stat.get("view"),
        "like": stat.get("like"),
        "duration": hit.get("duration"),
        "pubdate": hit.get("pubdate"),
        "insert_at": hit.get("insert_at"),
        "score": hit.get("score"),
        "stat_score": hit.get("stat_score"),
        "rank_score": hit.get("rank_score"),
    }


def summarize_search(result: dict[str, Any], limit: int) -> dict[str, Any]:
    return {
        "query": result.get("query"),
        "total_hits": result.get("total_hits", 0),
        "return_hits": result.get("return_hits", len(result.get("hits") or [])),
        "query_info": result.get("query_info") or {},
        "rewrite_info": result.get("rewrite_info") or {},
        "semantic_rewrite_info": result.get("semantic_rewrite_info") or {},
        "intent_info": result.get("intent_info") or {},
        "retry_info": result.get("retry_info") or {},
        "top_hits": [hit_summary(hit) for hit in (result.get("hits") or [])[:limit]],
    }


def summarize_related(result: dict[str, Any], limit: int) -> dict[str, Any]:
    return {
        "mode": result.get("mode"),
        "options": [
            {
                "text": option.get("text"),
                "score": option.get("score"),
                "keywords": option.get("keywords"),
            }
            for option in (result.get("options") or [])[:limit]
        ],
    }


def summarize_explore(result: dict[str, Any], limit: int) -> dict[str, Any]:
    steps = result.get("data") or []
    first_output = (steps[0].get("output") or {}) if steps else {}
    group_output: dict[str, Any] = {}
    for step in steps:
        if step.get("name") == "group_hits_by_owner":
            group_output = step.get("output") or {}
            break
    return {
        "intent_info": result.get("intent_info") or {},
        "retry_info": result.get("retry_info") or {},
        "qmod": first_output.get("qmod"),
        "filter_only": first_output.get("filter_only", False),
        "total_hits": first_output.get("total_hits", 0),
        "return_hits": first_output.get("return_hits", len(first_output.get("hits") or [])),
        "top_hits": [hit_summary(hit) for hit in (first_output.get("hits") or [])[:limit]],
        "authors": [
            {
                "name": author.get("name"),
                "mid": author.get("mid"),
                "count": author.get("count"),
            }
            for author in (group_output.get("authors") or [])[:limit]
        ],
        "perf": result.get("perf") or first_output.get("perf") or {},
    }


def summarize_chat(result: dict[str, Any]) -> dict[str, Any]:
    message = (result.get("choices") or [{}])[0].get("message") or {}
    calls: list[dict[str, Any]] = []
    for event in result.get("tool_events") or []:
        for call in event.get("calls") or []:
            call_result = call.get("result") or {}
            calls.append(
                {
                    "type": call.get("type"),
                    "args": call.get("args") or {},
                    "status": call.get("status"),
                    "result_id": call.get("result_id"),
                    "query": call_result.get("query"),
                    "total_hits": call_result.get("total_hits"),
                    "top_hits": [
                        hit_summary(hit) for hit in (call_result.get("hits") or [])[:3]
                    ],
                }
            )
    return {
        "tools": flatten_tool_names(result.get("tool_events") or []),
        "calls": calls,
        "usage": result.get("usage") or {},
        "content": truncate_text(message.get("content") or ""),
    }


def should_retry_chat_error(message: str) -> bool:
    normalized = str(message or "").lower()
    return "timed out" in normalized or normalized.startswith("http 5")


def run_case(
    case: dict[str, Any],
    *,
    options: ProbeOptions,
    chat_semaphore: threading.Semaphore | None,
) -> dict[str, Any]:
    case_result: dict[str, Any] = {
        "id": case.get("id"),
        "category": case.get("category"),
        "tags": case.get("tags") or [],
        "search_query": case.get("search_query") or "",
        "seed": case.get("seed") or {},
    }

    if not options.skip_search and case_result["search_query"]:
        started = time.perf_counter()
        try:
            search_result = post_json(
                options.base_url,
                "/search",
                {"query": case_result["search_query"], "limit": options.limit},
                timeout=options.timeout,
            )
            case_result["search"] = summarize_search(search_result, options.limit)
            case_result["search_elapsed_s"] = round(time.perf_counter() - started, 2)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            case_result["search_error"] = f"HTTP {exc.code}: {body}"
        except Exception as exc:
            case_result["search_error"] = str(exc)

    if not options.skip_related and case_result["search_query"]:
        started = time.perf_counter()
        try:
            related_result = post_json(
                options.base_url,
                "/related_tokens_by_tokens",
                {
                    "text": case_result["search_query"],
                    "mode": case.get("related_mode") or "auto",
                    "size": options.limit,
                    "scan_limit": 128,
                    "use_pinyin": True,
                },
                timeout=options.timeout,
            )
            case_result["related"] = summarize_related(related_result, options.limit)
            case_result["related_elapsed_s"] = round(time.perf_counter() - started, 2)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            case_result["related_error"] = f"HTTP {exc.code}: {body}"
        except Exception as exc:
            case_result["related_error"] = str(exc)

    if not options.skip_explore and case_result["search_query"]:
        started = time.perf_counter()
        try:
            explore_result = post_json(
                options.base_url,
                "/explore",
                {"query": case_result["search_query"], "verbose": False},
                timeout=options.timeout,
            )
            case_result["explore"] = summarize_explore(explore_result, options.limit)
            case_result["explore_elapsed_s"] = round(time.perf_counter() - started, 2)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            case_result["explore_error"] = f"HTTP {exc.code}: {body}"
        except Exception as exc:
            case_result["explore_error"] = str(exc)

    if not options.skip_chat and case.get("chat_messages"):
        attempts = 0
        started: float | None = None
        semaphore = chat_semaphore or threading.Semaphore(1)
        while True:
            attempts += 1
            try:
                with semaphore:
                    if started is None:
                        started = time.perf_counter()
                    chat_result = post_json(
                        options.base_url,
                        "/chat/completions",
                        {
                            "messages": case["chat_messages"],
                            "stream": False,
                            "thinking": options.thinking,
                            "max_iterations": options.max_iterations,
                        },
                        timeout=options.timeout,
                    )
                case_result["chat"] = summarize_chat(chat_result)
                case_result["chat_elapsed_s"] = round(
                    time.perf_counter() - (started or time.perf_counter()),
                    2,
                )
                if attempts > 1:
                    case_result["chat_attempts"] = attempts
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                message = f"HTTP {exc.code}: {body}"
                if attempts <= options.chat_retries and should_retry_chat_error(message):
                    time.sleep(min(2.0, 0.5 * attempts))
                    continue
                case_result["chat_error"] = message
                if attempts > 1:
                    case_result["chat_attempts"] = attempts
                break
            except Exception as exc:
                message = str(exc)
                if attempts <= options.chat_retries and should_retry_chat_error(message):
                    time.sleep(min(2.0, 0.5 * attempts))
                    continue
                case_result["chat_error"] = message
                if attempts > 1:
                    case_result["chat_attempts"] = attempts
                break

    case_result["evaluation"] = evaluate_case(case_result)
    return case_result


def collect_hits(result: dict[str, Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    hits.extend((result.get("search") or {}).get("top_hits") or [])
    hits.extend((result.get("explore") or {}).get("top_hits") or [])
    return hits


def title_similarity(left: object, right: object) -> float:
    left_text = normalize_text(left)
    right_text = normalize_text(right)
    if not left_text or not right_text:
        return 0.0
    if left_text in right_text or right_text in left_text:
        return 1.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def semantic_tokens(text: object) -> set[str]:
    normalized = normalize_text(text)
    tokens = {token.lower() for token in ASCII_TOKEN_RE.findall(normalized)}
    for cjk_part in CJK_RE.findall(normalized):
        if len(cjk_part) <= 1:
            continue
        tokens.add(cjk_part)
        for width in (2, 3, 4):
            if len(cjk_part) < width:
                continue
            for index in range(0, len(cjk_part) - width + 1):
                tokens.add(cjk_part[index : index + width])
    return {token for token in tokens if token not in {"视频", "相关", "最近", "高质量"}}


def hit_text(hit: dict[str, Any]) -> str:
    return " ".join(
        str(part or "")
        for part in [
            hit.get("title"),
            hit.get("owner"),
            hit.get("tags"),
            hit.get("bvid"),
        ]
    )


def any_seed_bvid_in_hits(seed: dict[str, Any], hits: list[dict[str, Any]], top: int) -> bool:
    seed_bvid = str(seed.get("bvid") or "")
    if not seed_bvid:
        return False
    return any(str(hit.get("bvid") or "") == seed_bvid for hit in hits[:top])


def best_seed_title_score(seed: dict[str, Any], hits: list[dict[str, Any]], top: int) -> float:
    seed_title = seed.get("title") or ""
    return max((title_similarity(seed_title, hit.get("title")) for hit in hits[:top]), default=0.0)


def owner_match_score(seed: dict[str, Any], hits: list[dict[str, Any]], top: int) -> float:
    owner_name = normalize_text((seed.get("owner") or {}).get("name"))
    if not owner_name:
        return 0.0
    for index, hit in enumerate(hits[:top]):
        if owner_name and owner_name == normalize_text(hit.get("owner")):
            return 1.0 if index < 3 else 0.8
    return 0.0


def tag_overlap_score(seed: dict[str, Any], hits: list[dict[str, Any]], top: int) -> float:
    tags = [normalize_text(tag) for tag in (seed.get("tags") or []) if normalize_text(tag)]
    if not tags:
        return 0.0
    joined_hits = normalize_text(" ".join(hit_text(hit) for hit in hits[:top]))
    matched = [tag for tag in tags if tag and tag in joined_hits]
    return min(1.0, len(matched) / min(3, len(tags)))


def query_overlap_score(query: str, hits: list[dict[str, Any]], top: int) -> float:
    query_tokens = semantic_tokens(query)
    if not query_tokens:
        return 0.0
    hit_tokens = semantic_tokens(" ".join(hit_text(hit) for hit in hits[:top]))
    return min(1.0, len(query_tokens & hit_tokens) / max(1, min(8, len(query_tokens))))


def has_rewrite_evidence(result: dict[str, Any]) -> bool:
    search = result.get("search") or {}
    related = result.get("related") or {}
    rewrite_info = search.get("rewrite_info") or {}
    semantic_info = search.get("semantic_rewrite_info") or {}
    intent_info = search.get("intent_info") or {}
    return bool(
        rewrite_info.get("rewrited")
        or rewrite_info.get("rewrites")
        or semantic_info
        or intent_info
        or related.get("options")
    )


def top_hit_views(hits: list[dict[str, Any]], top: int) -> list[float]:
    return [numeric(hit.get("view")) for hit in hits[:top] if hit.get("view") is not None]


def top_hit_stat_scores(hits: list[dict[str, Any]], top: int) -> list[float]:
    return [
        numeric(hit.get("stat_score"))
        for hit in hits[:top]
        if hit.get("stat_score") is not None
    ]


def timestamp_presence_score(hits: list[dict[str, Any]], top: int) -> float:
    if not hits[:top]:
        return 0.0
    present = [
        1
        for hit in hits[:top]
        if hit.get("insert_at") is not None or hit.get("pubdate") is not None
    ]
    return len(present) / len(hits[:top])


def noise_penalty(query: str, hits: list[dict[str, Any]], top: int) -> float:
    if not any(phrase in query for phrase in NOISE_PHRASES):
        return 0.0
    joined = " ".join(hit_text(hit) for hit in hits[:top])
    noise_hits = sum(1 for phrase in NOISE_PHRASES if phrase in joined)
    return min(0.35, noise_hits * 0.15)


def chat_used_search(chat: dict[str, Any]) -> bool:
    tools = [str(tool) for tool in chat.get("tools") or []]
    return any("search" in tool or "explore" in tool or "video" in tool for tool in tools)


def chat_mentions_seed(chat: dict[str, Any], seed: dict[str, Any]) -> bool:
    content = normalize_text(chat.get("content") or "")
    if not content:
        return False
    seed_bvid = normalize_text(seed.get("bvid") or "")
    if seed_bvid and seed_bvid in content:
        return True
    seed_title = normalize_text(seed.get("title") or "")
    if seed_title and (seed_title in content or title_similarity(seed_title, content) >= 0.42):
        return True
    return False


def chat_negates_results(chat: dict[str, Any]) -> bool:
    content = normalize_text(chat.get("content") or "")
    return any(
        marker in content
        for marker in [
            "未找到",
            "没找到",
            "没有找到",
            "无相关",
            "均与",
            "无关",
            "未命中",
        ]
    )


def score_quality(hits: list[dict[str, Any]], relevance: float, seed_top3: bool) -> float:
    if not hits:
        return 0.0
    views = top_hit_views(hits, 5)
    stats = top_hit_stat_scores(hits, 5)
    view_score = 0.55 if not views else min(1.0, statistics.mean(views) / 50000)
    stat_score = 0.6 if not stats else min(1.0, statistics.mean(stats) / 0.55)
    exact_floor = 0.82 if seed_top3 else 0.0
    return round(
        max(
            exact_floor,
            relevance * 0.45,
            min(1.0, 0.35 + view_score * 0.25 + stat_score * 0.3 + relevance * 0.25),
        ),
        3,
    )


def score_core_intent(
    result: dict[str, Any],
    hits: list[dict[str, Any]],
    relevance: float,
    owner_score: float,
    query_score: float,
    seed_top3: bool,
    chat_contradiction: bool,
) -> float:
    category = result.get("category") or ""
    score = 1.0 if seed_top3 else max(relevance * 0.75, query_score)
    if category in {"owner_recent", "owner_topic"}:
        score = max(score, owner_score)
    score -= noise_penalty(result.get("search_query") or "", hits, 5)
    if chat_contradiction:
        score = min(score, 0.45)
    return round(max(0.0, min(1.0, score)), 3)


def score_authority(
    result: dict[str, Any],
    relevance: float,
    owner_score: float,
    tag_score: float,
) -> float:
    category = result.get("category") or ""
    rewrite_bonus = 0.15 if has_rewrite_evidence(result) else 0.0
    base = max(relevance, owner_score, tag_score)
    if category in {"single_typo", "mixed_script", "topic_fragment", "tag_only"}:
        base = max(base, 0.5 + rewrite_bonus)
    return round(min(1.0, base + rewrite_bonus), 3)


def score_freshness(result: dict[str, Any], hits: list[dict[str, Any]]) -> float:
    category = result.get("category") or ""
    timestamp_score = timestamp_presence_score(hits, 5)
    search = result.get("search") or {}
    explore = result.get("explore") or {}
    intent_text = json.dumps(
        {
            "search": search.get("intent_info") or {},
            "explore": explore.get("intent_info") or {},
            "query_info": search.get("query_info") or {},
        },
        ensure_ascii=False,
    )
    recency_signal = any(
        token in intent_text or token in result.get("search_query", "")
        for token in ["最新", "最近", "recent", "date", "pubdate"]
    )
    if category == "owner_recent":
        return round(min(1.0, timestamp_score * 0.65 + (0.35 if recency_signal else 0.0)), 3)
    return round(max(0.75, timestamp_score), 3)


def evaluate_case(result: dict[str, Any]) -> dict[str, Any]:
    seed = result.get("seed") or {}
    hits = collect_hits(result)
    top3_seed = any_seed_bvid_in_hits(seed, hits, 3)
    top10_seed = any_seed_bvid_in_hits(seed, hits, 10)
    title_score = best_seed_title_score(seed, hits, 10)
    owner_score = owner_match_score(seed, hits, 10)
    tag_score = tag_overlap_score(seed, hits, 10)
    query_score = query_overlap_score(result.get("search_query") or "", hits, 10)
    relevance = max(
        1.0 if top3_seed else 0.0,
        0.9 if top10_seed else 0.0,
        title_score,
        owner_score * 0.85,
        tag_score * 0.8,
        query_score * 0.75,
    )

    search_total = int((result.get("search") or {}).get("total_hits") or 0)
    explore_total = int((result.get("explore") or {}).get("total_hits") or 0)
    chat = result.get("chat") or {}
    chat_ok = bool(chat.get("content")) and (chat_used_search(chat) or bool(hits))
    chat_contradiction = bool(
        chat
        and (top10_seed or title_score >= 0.82)
        and chat_negates_results(chat)
        and not chat_mentions_seed(chat, seed)
    )
    error_keys = [key for key in result if key.endswith("_error")]

    scores = {
        "authority": score_authority(result, relevance, owner_score, tag_score),
        "quality": score_quality(hits, relevance, top3_seed),
        "core_intent": score_core_intent(
            result,
            hits,
            relevance,
            owner_score,
            query_score,
            top3_seed,
            chat_contradiction,
        ),
        "freshness": score_freshness(result, hits),
        "relevance": round(min(1.0, relevance), 3),
        "stability": 1.0
        if not error_keys and (search_total > 0 or explore_total > 0) and (chat_ok or "chat" not in result)
        else 0.0,
    }

    issues = []
    for key, score in scores.items():
        if score < FAIL_THRESHOLD:
            issues.append(f"{STANDARD_NAMES[key]}失败: score={score}")
        elif score < WARN_THRESHOLD:
            issues.append(f"{STANDARD_NAMES[key]}偏弱: score={score}")

    if error_keys:
        issues.append("接口错误: " + ", ".join(f"{key}={result[key]}" for key in error_keys))
    if not hits:
        issues.append("search/explore 均未返回可评估结果")
    if result.get("category") in {"owner_recent", "owner_topic"} and owner_score < WARN_THRESHOLD:
        issues.append("UP 主意图未稳定进入 top10")
    if result.get("category") == "boilerplate_noise" and noise_penalty(result.get("search_query") or "", hits, 5):
        issues.append("口播/套话噪声仍影响 top5 结果")
    if "chat" in result and not chat_ok:
        issues.append("chat 未稳定产出检索型回答或缺少工具证据")
    if chat_contradiction:
        issues.append("chat 与检索结果矛盾：检索已命中但回答否定相关结果")

    return {
        "scores": scores,
        "overall": round(sum(scores.values()) / len(scores), 3),
        "signals": {
            "seed_bvid_top3": top3_seed,
            "seed_bvid_top10": top10_seed,
            "best_title_similarity_top10": round(title_score, 3),
            "owner_match_top10": round(owner_score, 3),
            "tag_overlap_top10": round(tag_score, 3),
            "query_overlap_top10": round(query_score, 3),
            "search_total_hits": search_total,
            "explore_total_hits": explore_total,
            "chat_used_search_tool": chat_used_search(chat),
            "chat_mentions_seed": chat_mentions_seed(chat, seed),
            "chat_negates_results": chat_negates_results(chat),
            "chat_contradiction": chat_contradiction,
        },
        "issues": issues,
    }


def select_cases(
    cases: list[dict[str, Any]],
    *,
    case_ids: list[str] | None,
    sample_size: int,
    per_category: int,
) -> list[dict[str, Any]]:
    if case_ids:
        selected = []
        for case_id in case_ids:
            matched = next((case for case in cases if case.get("id") == case_id), None)
            if matched is None:
                raise ValueError(f"Unknown case id: {case_id}")
            selected.append(matched)
        return selected

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_category[str(case.get("category") or "unknown")].append(case)

    selected: list[dict[str, Any]] = []
    for category in sorted(by_category):
        selected.extend(by_category[category][:per_category])
    if len(selected) >= sample_size:
        return selected[:sample_size]

    selected_ids = {case.get("id") for case in selected}
    for case in cases:
        if case.get("id") in selected_ids:
            continue
        selected.append(case)
        if len(selected) >= sample_size:
            break
    return selected


def resolve_output_paths(output_json: Path | None, output_md: Path | None) -> tuple[Path, Path]:
    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_json or DEFAULT_REPORTS_DIR / f"live-qa-quality-{timestamp}.json"
    md_path = output_md or DEFAULT_REPORTS_DIR / f"live-qa-quality-{timestamp}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    return json_path, md_path


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(str(result.get("category") or "unknown") for result in results)
    score_lists: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for key, score in (result.get("evaluation") or {}).get("scores", {}).items():
            score_lists[key].append(float(score))

    warning_cases = 0
    failing_cases = 0
    for result in results:
        evaluation = result.get("evaluation") or {}
        scores = evaluation.get("scores") or {}
        issues = evaluation.get("issues") or []
        has_failure_score = any(float(score) < FAIL_THRESHOLD for score in scores.values())
        has_blocking_issue = any(
            "失败" in issue or "接口错误" in issue or "矛盾" in issue
            for issue in issues
        )
        has_warning = bool(issues) or numeric(evaluation.get("overall")) < WARN_THRESHOLD
        if has_failure_score or has_blocking_issue:
            failing_cases += 1
        elif has_warning:
            warning_cases += 1
    return {
        "total_cases": len(results),
        "category_counts": dict(category_counts),
        "score_means": {
            key: round(sum(values) / len(values), 3)
            for key, values in score_lists.items()
            if values
        },
        "failing_cases": failing_cases,
        "warning_cases": warning_cases,
        "pass_rate": round(
            (len(results) - failing_cases - warning_cases) / max(1, len(results)),
            3,
        ),
        "non_blocking_rate": round((len(results) - failing_cases) / max(1, len(results)), 3),
    }


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    results = payload["results"]
    lines = [
        "# Live 自然语言问答质量评测",
        "",
        f"- 运行时间：{payload['run']['started_at']}",
        f"- 服务地址：`{payload['run']['base_url']}`",
        f"- 用例数：{summary['total_cases']}",
        f"- 通过率：{summary['pass_rate']}",
        f"- 非阻塞率：{summary.get('non_blocking_rate')}",
        f"- 失败/警告：{summary.get('failing_cases')}/{summary.get('warning_cases')}",
        f"- 分类分布：`{json.dumps(summary['category_counts'], ensure_ascii=False)}`",
        "",
        "## 六项评测标准",
        "",
        "- 权威：明确术语、概念、UP 主和作品名能够被正确 rewrite、alias 或精确召回。",
        "- 质量：优先返回数据质量更高、作者更权威、标题/标签/内容更完整的结果。",
        "- 核心：自然语言、多个关键词和噪声输入下能识别核心意图，并做过滤、rewrite、expand。",
        "- 时效：新近热门文档保留时间信号，并能在“最近/最新”等查询中影响召回和排序。",
        "- 相关：结果与用户查询的主题、标题、标签、UP 主或目标视频高度相关。",
        "- 稳定：复杂、边缘、错别字和混合语种查询不应超时、报错或空结果失效。",
        "",
        "## 汇总分",
        "",
    ]
    for key, score in summary["score_means"].items():
        lines.append(f"- {STANDARD_NAMES.get(key, key)}：{score}")

    lines.extend(["", "## 实测问题", ""])
    issue_count = 0
    for result in results:
        evaluation = result.get("evaluation") or {}
        issues = evaluation.get("issues") or []
        if not issues:
            continue
        issue_count += 1
        scores = evaluation.get("scores") or {}
        top_hit = ((result.get("search") or {}).get("top_hits") or [{}])[0]
        lines.extend(
            [
                f"### {result.get('id')}",
                "",
                f"- 分类：`{result.get('category')}`",
                f"- 查询：{result.get('search_query')}",
                f"- 总分：{evaluation.get('overall')}",
                f"- 分项：`{json.dumps(scores, ensure_ascii=False)}`",
                f"- top1：{top_hit.get('title') or ''} / {top_hit.get('owner') or ''} / {top_hit.get('bvid') or ''}",
                "- 问题：" + "；".join(issues),
                "",
            ]
        )
    if issue_count == 0:
        lines.append("本轮未发现低于阈值的问题。")

    lines.extend(
        [
            "",
            "## 根因排查入口",
            "",
            "- 对 relevance/core_intent 低分 case，优先检查 `/search` 的 `query_info`、`rewrite_info`、`semantic_rewrite_info` 和 `/explore` 的 `intent_info`。",
            "- 对 quality 低分 case，优先检查 top hits 的 `stat.view`、`stat_score`、作者聚合和排序阶段是否丢失质量信号。",
            "- 对 freshness 低分 case，优先检查 `pubdate`、`insert_at` 是否进入 `_source`，以及“最近/最新”是否生成日期或排序意图。",
            "- 对 stability 低分 case，优先查看服务日志、LLM 工具调用链和超时/5xx 响应。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(render_markdown(payload), encoding="utf-8")


def build_report_payload(
    *,
    args: argparse.Namespace,
    selected_cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    started_at: str,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "run": {
            "started_at": started_at,
            "elapsed_s": round(elapsed_s, 2),
            "base_url": args.base_url,
            "cases_path": str(args.cases_path),
            "sample_size": len(selected_cases),
            "limit": args.limit,
            "workers": args.workers,
            "chat_workers": args.chat_workers,
            "skip_search": args.skip_search,
            "skip_related": args.skip_related,
            "skip_explore": args.skip_explore,
            "skip_chat": args.skip_chat,
        },
        "summary": summarize_results(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 50 live natural-language QA quality cases")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Service base URL")
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", action="append", help="Case ID to run; repeatable")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-related", action="store_true")
    parser.add_argument("--skip-explore", action="store_true")
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--chat-retries", type=int, default=1)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--chat-workers", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-incremental-write", action="store_true")
    args = parser.parse_args()

    selected_cases = select_cases(
        load_cases(args.cases_path),
        case_ids=args.case,
        sample_size=max(1, args.sample_size),
        per_category=max(1, args.per_category),
    )
    output_json, output_md = resolve_output_paths(args.output_json, args.output_md)
    options = ProbeOptions(
        base_url=args.base_url,
        limit=max(1, args.limit),
        timeout=max(1, args.timeout),
        max_iterations=max(1, args.max_iterations),
        thinking=args.thinking,
        skip_search=args.skip_search,
        skip_related=args.skip_related,
        skip_explore=args.skip_explore,
        skip_chat=args.skip_chat,
        chat_retries=max(0, args.chat_retries),
    )
    chat_semaphore = None if args.skip_chat else threading.Semaphore(max(1, args.chat_workers))
    started = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    results: list[dict[str, Any] | None] = [None] * len(selected_cases)

    print(
        f"Running {len(selected_cases)} live QA cases with workers={args.workers}, chat_workers={args.chat_workers}; report -> {output_json}"
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_index = {
            executor.submit(
                run_case,
                case,
                options=options,
                chat_semaphore=chat_semaphore,
            ): index
            for index, case in enumerate(selected_cases)
        }
        for completed_count, future in enumerate(as_completed(future_to_index), start=1):
            index = future_to_index[future]
            result = future.result()
            results[index] = result
            evaluation = result["evaluation"]
            print(
                f"[{completed_count}/{len(selected_cases)}] {result['id']} overall={evaluation['overall']} issues={len(evaluation['issues'])}"
            )
            if not args.quiet and evaluation["issues"]:
                for issue in evaluation["issues"]:
                    print(f"  - {issue}")
            if not args.no_incremental_write:
                completed_results = [item for item in results if item is not None]
                payload = build_report_payload(
                    args=args,
                    selected_cases=selected_cases,
                    results=completed_results,
                    started_at=started_at,
                    elapsed_s=time.perf_counter() - started,
                )
                write_json_report(output_json, payload)
                write_markdown_report(output_md, payload)

    final_results = [item for item in results if item is not None]
    payload = build_report_payload(
        args=args,
        selected_cases=selected_cases,
        results=final_results,
        started_at=started_at,
        elapsed_s=time.perf_counter() - started,
    )
    write_json_report(output_json, payload)
    write_markdown_report(output_md, payload)
    print(f"Saved JSON report to: {output_json}")
    print(f"Saved Markdown report to: {output_md}")
    print(f"Summary: {json.dumps(payload['summary'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
