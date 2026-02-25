"""Live integration tests for LLM chat — requires real LLM + Elasticsearch.

Tests complex, verifiable scenarios that push the LLM's reasoning boundaries.
Each test validates content quality, tool usage efficiency, and token budget.

Usage:
    python -m tests.llm.test_live_chat
    python -m tests.llm.test_live_chat --verbose
    python -m tests.llm.test_live_chat --test 1     # run specific test
    python -m tests.llm.test_live_chat --elastic-index bili_videos_dev6 --elastic-env-name elastic_dev
"""

import argparse
import json
import re
import time
import sys

from tclogger import logger


# ============================================================
# Test cases — complex, verifiable
# ============================================================

TEST_CASES = [
    {
        "id": 1,
        "name": "multi_up_comparison",
        "description": "多UP主对比 — 需要并行搜索不同UP主并综合对比分析",
        "query": "对比一下老番茄和影视飓风最近一个月发布的视频，谁更高产？",
        "checks": {
            # Content should mention both UP主
            "content_contains": ["老番茄", "影视飓风"],
            # Should have bilibili links
            "content_pattern": r"bilibili\.com/video/BV",
            # Should use multi-query or :user= filter
            "max_tokens": 20000,
        },
    },
    {
        "id": 2,
        "name": "cross_topic_reasoning",
        "description": "跨领域推理 — 需要搜索不同关键词并综合分析结果",
        "query": "B站上关于AI的视频和关于芯片的视频，哪个领域最近一周更火？用播放量数据说明",
        "checks": {
            "content_contains": ["AI", "芯片"],
            # Should have numeric data (播放量)
            "content_pattern": r"\d+[.\d]*[万亿]",
            "max_tokens": 20000,
        },
    },
    {
        "id": 3,
        "name": "complex_filter_chain",
        "description": "复杂过滤链 — 精确过滤条件+排除词+UP主检测的综合能力",
        "query": "何同学最近3个月播放量超过100万的视频有哪些？不要包含广告和恰饭内容",
        "checks": {
            # Should detect "何同学" as UP主
            "content_contains": ["何同学"],
            "content_pattern": r"bilibili\.com/video/BV",
            "max_tokens": 20000,
        },
    },
]


# ============================================================
# Test runner
# ============================================================


def run_test_case(handler, test_case: dict, verbose: bool = False) -> dict:
    """Run a single test case and return validation results."""
    tc_id = test_case["id"]
    name = test_case["name"]
    query = test_case["query"]
    checks = test_case["checks"]

    logger.note(f"\n{'=' * 60}")
    logger.note(f"[TEST {tc_id}] {name}")
    logger.mesg(f"  {test_case['description']}")
    logger.mesg(f"  Query: {query}")

    messages = [{"role": "user", "content": query}]
    start_time = time.perf_counter()

    try:
        result = handler.handle(messages=messages)
    except Exception as e:
        logger.warn(f"  × Handler error: {e}")
        return {"name": name, "status": "ERROR", "error": str(e)}

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    prompt = usage.get("prompt_tokens", 0)

    logger.mesg(f"  Time: {elapsed_ms}ms")
    logger.mesg(
        f"  Tokens: {total_tokens} (prompt={prompt}, " f"cache_hit={cache_hit})"
    )

    # --- Validations ---
    failures = []

    # Check content_contains
    for keyword in checks.get("content_contains", []):
        if keyword not in content:
            failures.append(f"content missing '{keyword}'")

    # Check content_pattern
    pattern = checks.get("content_pattern")
    if pattern and not re.search(pattern, content):
        failures.append(f"content does not match pattern '{pattern}'")

    # Check token budget
    max_tokens = checks.get("max_tokens", 30000)
    if total_tokens > max_tokens:
        failures.append(f"token budget exceeded: {total_tokens} > {max_tokens}")

    # Check not empty
    if len(content) < 50:
        failures.append(f"content too short: {len(content)} chars")

    # Check no DSML leakage
    if "DSML" in content or "function_calls" in content:
        failures.append("DSML markup leaked into content")

    # Report
    if failures:
        status = "FAIL"
        logger.warn(f"  Failures:")
        for f in failures:
            logger.warn(f"    × {f}")
    else:
        status = "PASS"
        logger.success(f"  All checks passed")

    if verbose:
        logger.mesg(f"\n  --- Content ({len(content)} chars) ---")
        print(content[:600])
        if len(content) > 600:
            print(f"  ... ({len(content) - 600} chars truncated)")

    return {
        "name": name,
        "status": status,
        "failures": failures,
        "elapsed_ms": elapsed_ms,
        "total_tokens": total_tokens,
        "content_length": len(content),
    }


def main():
    parser = argparse.ArgumentParser(description="Live LLM Chat Tests")
    parser.add_argument(
        "--llm-config",
        type=str,
        default="deepseek",
        help="LLM config name",
    )
    parser.add_argument(
        "--elastic-index",
        type=str,
        default=None,
        help="Elastic videos index name",
    )
    parser.add_argument(
        "--elastic-env-name",
        type=str,
        default=None,
        help="Elastic env name in secrets.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show full response content",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        help="Run specific test by ID (1, 2, or 3)",
    )
    args = parser.parse_args()

    # Lazy imports
    from configs.envs import SEARCH_APP_ENVS
    from elastics.videos.searcher_v2 import VideoSearcherV2
    from elastics.videos.explorer import VideoExplorer
    from llms.llm_client import create_llm_client
    from llms.tools.executor import SearchService
    from llms.chat.handler import ChatHandler

    elastic_index = args.elastic_index
    elastic_env_name = args.elastic_env_name
    if not elastic_index:
        idx = SEARCH_APP_ENVS.get("elastic_index", {})
        elastic_index = idx.get("prod", idx) if isinstance(idx, dict) else idx

    logger.note("> Initializing live test environment...")
    logger.mesg(f"  LLM: {args.llm_config}, Index: {elastic_index}")

    video_searcher = VideoSearcherV2(elastic_index, elastic_env_name=elastic_env_name)
    video_explorer = VideoExplorer(elastic_index, elastic_env_name=elastic_env_name)

    handler = ChatHandler(
        llm_client=create_llm_client(args.llm_config, verbose=args.verbose),
        search_client=SearchService(
            video_searcher, video_explorer, verbose=args.verbose
        ),
        verbose=args.verbose,
    )

    # Select tests
    cases = TEST_CASES
    if args.test:
        cases = [tc for tc in TEST_CASES if tc["id"] == args.test]
        if not cases:
            logger.warn(f"× Unknown test ID: {args.test}")
            sys.exit(1)

    # Run tests
    results = []
    for tc in cases:
        result = run_test_case(handler, tc, verbose=args.verbose)
        results.append(result)

    # Summary
    logger.note(f"\n{'=' * 60}")
    logger.note("[SUMMARY]")
    logger.note("=" * 60)
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    for r in results:
        status_fn = logger.success if r["status"] == "PASS" else logger.warn
        status_fn(
            f"  {r['name']}: {r['status']} "
            f"({r.get('elapsed_ms', 0)}ms, "
            f"{r.get('total_tokens', 0)} tokens, "
            f"{r.get('content_length', 0)} chars)"
        )
    logger.note(f"\n  {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

    # python -m tests.llm.test_live_chat
    # python -m tests.llm.test_live_chat --verbose
    # python -m tests.llm.test_live_chat --test 1
