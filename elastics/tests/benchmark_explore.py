"""Benchmark explore pipeline: compare auto-constraint ON vs OFF across 5 test queries.

Measures:
- Recall time (KNN + word supplement)
- Total pipeline time
- Result count and top-8 quality
- Auto-constraint token coverage

Usage:
    python -m elastics.tests.benchmark_explore          # full benchmark
    python -m elastics.tests.benchmark_explore --knn    # raw KNN timing only
"""

import json
import time
from tclogger import logger
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.explorer import VideoExplorer
from elastics.structure import (
    analyze_tokens,
    select_covering_tokens,
    build_auto_constraint_filter,
)

FIELDS = ["title.words", "tags.words"]
QUERIES = ["飓风营救", "通义实验室", "红警08", "小红书推荐系统", "吴恩达大模型"]

_explorer = None


def get_explorer():
    global _explorer
    if _explorer is None:
        _explorer = VideoExplorer(
            index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
        )
    return _explorer


def format_hit(hit, idx):
    title = hit.get("title", "")
    score = hit.get("score", hit.get("rank_score", 0))
    owner = hit.get("owner", {})
    name = owner.get("name", "") if isinstance(owner, dict) else ""
    bvid = hit.get("bvid", "")
    return f"  [{idx+1:2d}] score={score:8.4f} | {title[:40]:40s} | @{name[:12]}"


def timed_explore(explorer, query, mode="v", auto_constraint=True):
    """Run explore and return (hits, elapsed_ms, perf_info)."""
    full_query = f"{query} q={mode}"
    start = time.perf_counter()
    res = explorer.unified_explore(
        query=full_query,
        auto_constraint=auto_constraint,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Extract hits and perf from step data
    hits = []
    step_name = ""
    perf_info = res.get("perf", {})  # Top-level perf from finalize()
    for step in res.get("data", []):
        name = step.get("name", "")
        if name in ("knn_search", "hybrid_search", "most_relevant_search"):
            output = step.get("output", {})
            hits = output.get("hits", [])
            step_name = name
            # Merge recall_info into perf for reporting
            recall_info = output.get("recall_info", {})
            if recall_info:
                perf_info["recall_info"] = recall_info
            break
        # Also capture rerank info
        if name == "rerank":
            output = step.get("output", {})
            perf_info["rerank_ms"] = output.get("rerank_ms", "?")
            perf_info["reranked_count"] = output.get("reranked_count", "?")

    # Second pass for rerank (may come after main step)
    for step in res.get("data", []):
        if step.get("name") == "rerank":
            output = step.get("output", {})
            perf_info["rerank_ms"] = output.get("rerank_ms", "?")
            perf_info["reranked_count"] = output.get("reranked_count", "?")
            break

    return hits, elapsed_ms, perf_info, step_name


def run_benchmark():
    """Benchmark all 5 queries, comparing auto-constraint ON vs OFF."""
    explorer = get_explorer()

    # Warm up
    logger.hint("Warming up explorer...")
    timed_explore(explorer, "测试", auto_constraint=False)
    timed_explore(explorer, "测试", auto_constraint=True)

    results = []

    for q in QUERIES:
        # Token analysis
        tokens = analyze_tokens(explorer.es.client, ELASTIC_VIDEOS_DEV_INDEX, q)
        covering = select_covering_tokens(tokens, q)
        cf = build_auto_constraint_filter(
            explorer.es.client, ELASTIC_VIDEOS_DEV_INDEX, q, fields=FIELDS
        )

        logger.file(f"\n{'='*72}")
        logger.file(f"QUERY: {q}")
        logger.file(f"  covering tokens: {covering}")
        if cf:
            constraints = cf.get("es_tok_constraints", {}).get("constraints", [])
            tokens_str = [
                c.get("have_token", [None])[0] for c in constraints if "have_token" in c
            ]
            logger.file(f"  constraint tokens: {tokens_str}")
        logger.file(f"{'='*72}")

        # Run 3 iterations each for stable timing
        N = 3
        off_times = []
        on_times = []
        off_hits = None
        on_hits = None
        off_perf = {}
        on_perf = {}

        for i in range(N):
            hits, ms, perf, _ = timed_explore(explorer, q, auto_constraint=False)
            off_times.append(ms)
            if i == N - 1:
                off_hits = hits
                off_perf = perf

            hits, ms, perf, _ = timed_explore(explorer, q, auto_constraint=True)
            on_times.append(ms)
            if i == N - 1:
                on_hits = hits
                on_perf = perf

        avg_off = sum(off_times) / N
        avg_on = sum(on_times) / N
        speedup = avg_off / avg_on if avg_on > 0 else float("inf")

        logger.note(
            f"\n  [NO CONSTRAINT] avg={avg_off:.0f}ms  (times: {[f'{t:.0f}' for t in off_times]})"
        )
        if off_perf:
            recall_ms = off_perf.get("recall_ms", "?")
            fetch_ms = off_perf.get("fetch_ms", "?")
            highlight_ms = off_perf.get("highlight_ms", "?")
            rerank_ms = off_perf.get("rerank_ms", "?")
            total_ms = off_perf.get("total_ms", "?")
            candidates = off_perf.get("recall_candidates", "?")
            recall_info = off_perf.get("recall_info", {})
            lanes = {}
            noise_info = None
            for k, v in recall_info.items():
                if k == "_noise_filter":
                    noise_info = v
                else:
                    lanes[k] = f"{v.get('hit_count',0)} hits/{v.get('took_ms',0):.0f}ms"
            logger.mesg(
                f"    recall={recall_ms}ms  fetch={fetch_ms}ms  highlight={highlight_ms}ms  rerank={rerank_ms}ms  total={total_ms}ms  candidates={candidates}"
            )
            if lanes:
                logger.mesg(f"    lanes: {lanes}")
            if noise_info:
                logger.mesg(
                    f"    noise_filter: kept={noise_info.get('kept','?')} removed={noise_info.get('removed','?')} gate={noise_info.get('gate','?')}"
                )
        logger.mesg(f"    returned: {len(off_hits)} hits")
        for i, hit in enumerate(off_hits[:5]):
            logger.mesg(format_hit(hit, i))

        logger.success(
            f"\n  [AUTO-CONSTRAINT] avg={avg_on:.0f}ms  (times: {[f'{t:.0f}' for t in on_times]})"
        )
        if on_perf:
            recall_ms = on_perf.get("recall_ms", "?")
            fetch_ms = on_perf.get("fetch_ms", "?")
            highlight_ms = on_perf.get("highlight_ms", "?")
            rerank_ms = on_perf.get("rerank_ms", "?")
            total_ms = on_perf.get("total_ms", "?")
            candidates = on_perf.get("recall_candidates", "?")
            recall_info = on_perf.get("recall_info", {})
            lanes = {}
            noise_info = None
            for k, v in recall_info.items():
                if k == "_noise_filter":
                    noise_info = v
                else:
                    lanes[k] = f"{v.get('hit_count',0)} hits/{v.get('took_ms',0):.0f}ms"
            logger.mesg(
                f"    recall={recall_ms}ms  fetch={fetch_ms}ms  highlight={highlight_ms}ms  rerank={rerank_ms}ms  total={total_ms}ms  candidates={candidates}"
            )
            if lanes:
                logger.mesg(f"    lanes: {lanes}")
            if noise_info:
                logger.mesg(
                    f"    noise_filter: kept={noise_info.get('kept','?')} removed={noise_info.get('removed','?')} gate={noise_info.get('gate','?')}"
                )
        logger.mesg(f"    returned: {len(on_hits)} hits")
        for i, hit in enumerate(on_hits[:5]):
            logger.mesg(format_hit(hit, i))

        # Compare top-5 bvids
        off_bvids = [h.get("bvid") for h in off_hits[:5]]
        on_bvids = [h.get("bvid") for h in on_hits[:5]]
        overlap = len(set(off_bvids) & set(on_bvids))
        logger.file(f"\n  ⚡ speedup: {speedup:.2f}x  |  top-5 overlap: {overlap}/5")

        results.append(
            {
                "query": q,
                "covering": covering,
                "off_avg_ms": round(avg_off),
                "on_avg_ms": round(avg_on),
                "speedup": round(speedup, 2),
                "off_hits": len(off_hits),
                "on_hits": len(on_hits),
                "off_recall_ms": off_perf.get("recall_ms", "?"),
                "on_recall_ms": on_perf.get("recall_ms", "?"),
                "top5_overlap": overlap,
            }
        )

    # Summary table
    logger.file(f"\n{'='*72}")
    logger.file("BENCHMARK SUMMARY")
    logger.file(f"{'='*72}")
    logger.file(
        f"{'Query':<20} {'OFF(ms)':>8} {'ON(ms)':>8} {'Speedup':>8} {'OFF#':>5} {'ON#':>5} {'Top5∩':>6}"
    )
    logger.file(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*6}")
    for r in results:
        logger.file(
            f"{r['query']:<20} {r['off_avg_ms']:>8} {r['on_avg_ms']:>8} "
            f"{r['speedup']:>7.2f}x {r['off_hits']:>5} {r['on_hits']:>5} "
            f"{r['top5_overlap']:>5}/5"
        )

    # Recall timing details
    logger.file(f"\nRecall timing (internal):")
    logger.file(f"{'Query':<20} {'OFF recall':>12} {'ON recall':>12}")
    for r in results:
        logger.file(
            f"{r['query']:<20} {str(r['off_recall_ms']):>12} {str(r['on_recall_ms']):>12}"
        )


def run_knn_benchmark():
    """Benchmark raw KNN search timing (no full pipeline)."""
    from elastics.videos.searcher_v2 import VideoSearcherV2

    searcher = VideoSearcherV2(ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV)

    logger.hint("Warming up KNN...")
    searcher.knn_search("测试")

    for q in QUERIES:
        tokens = analyze_tokens(searcher.es.client, ELASTIC_VIDEOS_DEV_INDEX, q)
        covering = select_covering_tokens(tokens, q)
        cf = build_auto_constraint_filter(
            searcher.es.client, ELASTIC_VIDEOS_DEV_INDEX, q, fields=FIELDS
        )

        logger.file(f"\n{'='*60}")
        logger.file(f"KNN: {q}  →  covering: {covering}")
        logger.file(f"{'='*60}")

        N = 3

        # Without constraint
        times = []
        for _ in range(N):
            start = time.perf_counter()
            res = searcher.knn_search(
                q, k=1000, num_candidates=10000, skip_ranking=True
            )
            times.append((time.perf_counter() - start) * 1000)
        avg_off = sum(times) / N
        hits_off = len(res.get("hits", []))

        # With constraint
        times = []
        for _ in range(N):
            start = time.perf_counter()
            res = searcher.knn_search(
                q, k=1000, num_candidates=10000, constraint_filter=cf, skip_ranking=True
            )
            times.append((time.perf_counter() - start) * 1000)
        avg_on = sum(times) / N
        hits_on = len(res.get("hits", []))

        speedup = avg_off / avg_on if avg_on > 0 else float("inf")
        logger.success(
            f"  OFF: {avg_off:.0f}ms ({hits_off} hits)  |  "
            f"ON: {avg_on:.0f}ms ({hits_on} hits)  |  "
            f"speedup: {speedup:.2f}x"
        )


if __name__ == "__main__":
    import sys

    if "--knn" in sys.argv:
        run_knn_benchmark()
    else:
        run_benchmark()
