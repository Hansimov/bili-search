"""Benchmark exact-segment es_tok_query_string behavior on the live dev index.

Measures end-to-end search latency for plain, quoted, required, and excluded
exact-segment queries through the bili-search searcher path.

Usage:
    python -m elastics.tests.benchmark_es_tok_exact_segments
"""

from __future__ import annotations

import statistics
import time

from tclogger import logger

from elastics.videos.constants import ELASTIC_DEV, ELASTIC_VIDEOS_DEV_INDEX
from elastics.videos.searcher_v2 import VideoSearcherV2

SOURCE_FIELDS = ["bvid", "title"]
ITERATIONS = 5
WARMUP_QUERY = "测试"

CASE_GROUPS = [
    {
        "label": "Split Chinese exact segment",
        "cases": [
            {
                "label": "plain",
                "query": "若生命将于明日落幕",
                "expected_bvid": "BV1v8w8zwEBQ",
            },
            {
                "label": "quoted exact",
                "query": '"若生命将于明日落幕"',
                "expected_bvid": "BV1v8w8zwEBQ",
            },
            {
                "label": "required exact",
                "query": "+若生命将于明日落幕",
                "expected_bvid": "BV1v8w8zwEBQ",
            },
            {
                "label": "keyword plus required",
                "query": "游戏音乐 +若生命将于明日落幕",
                "expected_bvid": "BV1v8w8zwEBQ",
            },
            {
                "label": "keyword plus excluded",
                "query": "游戏音乐 -若生命将于明日落幕",
                "unexpected_bvid": "BV1v8w8zwEBQ",
            },
        ],
    },
    {
        "label": "Named exact segment",
        "cases": [
            {
                "label": "plain",
                "query": "工藤晴香",
                "expected_bvid": "BV1gcwuzhEaX",
            },
            {
                "label": "quoted exact",
                "query": '"工藤晴香"',
                "expected_bvid": "BV1gcwuzhEaX",
            },
            {
                "label": "required exact",
                "query": "+工藤晴香",
                "expected_bvid": "BV1gcwuzhEaX",
            },
        ],
    },
]


def get_searcher() -> VideoSearcherV2:
    return VideoSearcherV2(
        index_name=ELASTIC_VIDEOS_DEV_INDEX,
        elastic_env_name=ELASTIC_DEV,
    )


def run_case(searcher: VideoSearcherV2, query: str) -> dict:
    start = time.perf_counter()
    res = searcher.search(
        query,
        source_fields=SOURCE_FIELDS,
        add_highlights_info=False,
        limit=10,
        timeout=5,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    hits = res.get("hits", [])
    bvids = [hit.get("bvid") for hit in hits if hit.get("bvid")]
    titles = [hit.get("title", "") for hit in hits[:3]]
    return {
        "elapsed_ms": elapsed_ms,
        "hit_count": len(hits),
        "bvids": bvids,
        "titles": titles,
    }


def summarize_timings(samples: list[float]) -> str:
    mean_ms = statistics.mean(samples)
    p95_ms = max(samples)
    return f"avg={mean_ms:.1f}ms max={p95_ms:.1f}ms samples={[round(s, 1) for s in samples]}"


def benchmark_group(searcher: VideoSearcherV2, group: dict) -> None:
    logger.file(f"\n{'=' * 72}")
    logger.file(group["label"])
    logger.file(f"{'=' * 72}")

    baseline_mean = None

    for case in group["cases"]:
        timings = []
        last_result = None
        for _ in range(ITERATIONS):
            last_result = run_case(searcher, case["query"])
            timings.append(last_result["elapsed_ms"])

        mean_ms = statistics.mean(timings)
        if baseline_mean is None:
            baseline_mean = mean_ms
            delta = "baseline"
        else:
            ratio = mean_ms / baseline_mean if baseline_mean else 0.0
            delta = f"{ratio:.2f}x vs baseline"

        logger.note(f"\n[{case['label']}] {case['query']}")
        logger.mesg(f"  {summarize_timings(timings)} | {delta}")
        logger.mesg(
            f"  hits={last_result['hit_count']} top_bvids={last_result['bvids'][:5]}"
        )
        for title in last_result["titles"]:
            logger.mesg(f"  title={title}")

        expected_bvid = case.get("expected_bvid")
        if expected_bvid:
            logger.mesg(
                f"  expected_present={expected_bvid in set(last_result['bvids'])} ({expected_bvid})"
            )
        unexpected_bvid = case.get("unexpected_bvid")
        if unexpected_bvid:
            logger.mesg(
                f"  expected_absent={unexpected_bvid not in set(last_result['bvids'])} ({unexpected_bvid})"
            )


def main() -> None:
    logger.note(
        "> Benchmarking es_tok_query_string exact-segment cases on live dev index..."
    )
    searcher = get_searcher()

    # Warm up the shared clients and caches.
    run_case(searcher, WARMUP_QUERY)

    for group in CASE_GROUPS:
        benchmark_group(searcher, group)


if __name__ == "__main__":
    main()
