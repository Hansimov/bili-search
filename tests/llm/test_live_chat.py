"""Live integration tests for LLM chat — requires real LLM + search backends.

Focuses on realistic user Q&A and multi-turn dialogue scenarios.
Each case validates:
  - output quality and basic formatting
  - actual tool-routing behavior via tool_events
  - token budget / runtime sanity
  - manual review checkpoints printed to stdout

Usage:
    python -m tests.llm.test_live_chat
    python -m tests.llm.test_live_chat --verbose
    python -m tests.llm.test_live_chat --test 3
    python -m tests.llm.test_live_chat --test 3 --test 5 --test 14
    python -m tests.llm.test_live_chat --search-base-url http://127.0.0.1:21001
"""

import argparse
import json
import re
import sys
import time

from tclogger import logger


TEST_CASES = [
    {
        "id": 1,
        "name": "creator_discovery_direct",
        "description": "显式找创作者，要求稳定先走 relation 工具，不允许直接编造占位推荐。",
        "messages": [
            {"role": "user", "content": "推荐几个做黑神话悟空内容的UP主"},
        ],
        "checks": {
            "expected_tools_any": ["related_owners_by_tokens", "search_owners"],
            "content_pattern": r"space\.bilibili\.com/\d+",
            "content_not_contains": ["space.bilibili.com/uid", "UP主A", "UP主B"],
            "max_tokens": 12000,
        },
        "manual_review_focus": ["作者链接是否为真实 mid", "是否避免占位式推荐"],
    },
    {
        "id": 2,
        "name": "creator_discovery_followup_dialogue",
        "description": "多轮对话中的省略跟进，要求继承上文主题并继续按创作者发现处理。",
        "messages": [
            {"role": "user", "content": "推荐几个做黑神话悟空内容的UP主"},
            {"role": "assistant", "content": "可以，我先给你找一批。"},
            {"role": "user", "content": "更偏剧情解析和世界观考据的呢？"},
        ],
        "checks": {
            "expected_tools_any": ["related_owners_by_tokens"],
            "content_pattern": r"space\.bilibili\.com/\d+",
            "content_not_contains": ["space.bilibili.com/uid", "UP主A", "UP主B"],
            "max_tokens": 14000,
        },
        "manual_review_focus": [
            "是否理解 follow-up 指向黑神话悟空",
            "是否按剧情解析/考据方向收窄",
        ],
    },
    {
        "id": 3,
        "name": "official_updates_plus_bili",
        "description": "官方更新 + B站解读，要求同轮触发 Google 和站内视频搜索。",
        "messages": [
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            },
        ],
        "checks": {
            "expected_tools_all": ["search_google", "search_videos"],
            "content_contains": ["Gemini 2.5"],
            "content_pattern": r"bilibili\.com/video/BV",
            "max_tokens": 15000,
        },
        "manual_review_focus": [
            "官方信息是否明显来自官网/官方 changelog",
            "B站部分是否给出具体视频而非空泛描述",
        ],
    },
    {
        "id": 4,
        "name": "official_updates_followup_dialogue",
        "description": "多轮 follow-up 下的官方更新查询，要求继承前文产品主题并收窄到开发者 API 侧。",
        "messages": [
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            },
            {"role": "assistant", "content": "我先查一下官方更新。"},
            {"role": "user", "content": "更偏开发者 API 侧，有没有 B 站解读？"},
        ],
        "checks": {
            "expected_tools_all": ["search_google", "search_videos"],
            "content_contains": ["Gemini 2.5", "API"],
            "content_pattern": r"bilibili\.com/video/BV",
            "max_tokens": 17000,
        },
        "manual_review_focus": [
            "Google 查询是否继承 Gemini 2.5 主题",
            "回答是否明显偏开发者/API 更新",
        ],
    },
    {
        "id": 5,
        "name": "explicit_video_request",
        "description": "显式找视频，要求稳定先走 search_videos。",
        "messages": [
            {"role": "user", "content": "找几条黑神话悟空剧情解析视频，优先高播放"},
        ],
        "checks": {
            "expected_tools_any": ["search_videos"],
            "content_pattern": r"bilibili\.com/video/BV",
            "content_contains": ["黑神话悟空"],
            "max_tokens": 12000,
        },
        "manual_review_focus": ["是否给出真实视频链接", "是否明显偏剧情解析而非泛推荐"],
    },
    {
        "id": 6,
        "name": "author_timeline_recent_posts",
        "description": "UP主时间线查询，要求用站内搜索，不应误走 Google。",
        "messages": [
            {"role": "user", "content": "影视飓风最近有什么新视频？"},
        ],
        "checks": {
            "expected_tools_any": ["search_videos"],
            "forbidden_tools": ["search_google"],
            "content_contains": ["影视飓风"],
            "content_pattern": r"bilibili\.com/video/BV",
            "max_tokens": 12000,
        },
        "manual_review_focus": ["是否按照最近时间线回答", "是否避免站外检索"],
    },
    {
        "id": 10,
        "name": "official_updates_google_only",
        "description": "只问官方更新，不要为了凑流程再搜 B 站视频。",
        "messages": [
            {"role": "user", "content": "Gemini 2.5 最近有哪些官方更新？"},
        ],
        "checks": {
            "expected_tools_any": ["search_google"],
            "forbidden_tools": ["search_videos"],
            "content_contains": ["Gemini 2.5"],
            "max_tokens": 12000,
        },
        "manual_review_focus": ["是否以官方来源为主", "是否避免无意义的 B 站扩搜"],
    },
    {
        "id": 13,
        "name": "official_updates_followup_official_only",
        "description": "首轮问官方更新 + B站解读，follow-up 改成只看官网后，不应再继承视频搜索。",
        "messages": [
            {
                "role": "user",
                "content": "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            },
            {"role": "assistant", "content": "我先查一下官方更新。"},
            {"role": "user", "content": "先只看官网就行"},
        ],
        "checks": {
            "expected_tools_any": ["search_google"],
            "forbidden_tools": ["search_videos"],
            "content_contains": ["Gemini 2.5"],
            "max_tokens": 12000,
        },
        "manual_review_focus": [
            "是否停止继承上一轮的 B 站解读需求",
            "是否以官网/官方发布信息为主",
        ],
    },
    {
        "id": 11,
        "name": "account_query_after_capability_chat",
        "description": "前一轮是能力闲聊，后一轮问关联账号，不应把口语历史塞进视频搜索。",
        "messages": [
            {"role": "user", "content": "你有什么功能？"},
            {"role": "assistant", "content": "我可以帮你搜索 B 站内容。"},
            {"role": "user", "content": "何同学有哪些关联账号？"},
        ],
        "checks": {
            "expected_tools_any": ["related_owners_by_tokens"],
            "forbidden_tools": ["search_videos"],
            "content_not_contains": ["你有什么功能 q=vwr"],
            "max_tokens": 12000,
        },
        "manual_review_focus": [
            "是否避免把能力闲聊污染成 query",
            "是否优先回答作者关系而不是视频列表",
        ],
    },
    {
        "id": 12,
        "name": "account_followup_pronoun_dialogue",
        "description": "代词 follow-up 的账号关系问题，应继承上文作者但不要补视频搜索。",
        "messages": [
            {"role": "user", "content": "何同学有哪些关联账号？"},
            {"role": "assistant", "content": "我先帮你找相关作者线索。"},
            {"role": "user", "content": "他还有别的号吗？"},
        ],
        "checks": {
            "expected_tools_any": ["related_owners_by_tokens"],
            "forbidden_tools": ["search_videos"],
            "content_not_contains": ["他还有别的号吗 q=vwr"],
            "max_tokens": 12000,
        },
        "manual_review_focus": [
            "是否正确继承何同学作为主体",
            "是否避免把代词口语转成视频 query",
        ],
    },
    {
        "id": 14,
        "name": "creator_relation_then_representative_videos",
        "description": "先问作者关系，再追问代表作，要求继承作者主体并落到 :user= 定向视频搜索。",
        "messages": [
            {"role": "user", "content": "何同学有哪些关联账号？"},
            {"role": "assistant", "content": "我先帮你找相关作者线索。"},
            {"role": "user", "content": "那他的代表作有哪些？"},
        ],
        "checks": {
            "expected_tools_any": ["search_videos"],
            "content_contains": ["何同学"],
            "content_pattern": r"bilibili\.com/video/BV",
            "max_tokens": 15000,
        },
        "manual_review_focus": [
            "是否继承何同学而不是把代词原样塞进 query",
            "是否给出这位作者自己的代表作而非泛数码热视频",
        ],
    },
    {
        "id": 7,
        "name": "seed_owner_similarity",
        "description": "以已有作者为种子找相近创作者，观察 relation 工具是否被合理使用。",
        "messages": [
            {
                "role": "user",
                "content": "和影视飓风风格接近的UP主有哪些？各给我一句推荐理由。",
            },
        ],
        "checks": {
            "expected_tools_any": [
                "related_owners_by_tokens",
                "related_owners_by_owners",
            ],
            "content_pattern": r"space\.bilibili\.com/\d+",
            "content_contains": ["影视飓风"],
            "max_tokens": 15000,
        },
        "manual_review_focus": ["推荐理由是否与频道风格相关", "是否出现明显无关作者"],
    },
    {
        "id": 8,
        "name": "multi_up_comparison",
        "description": "多 UP 对比，要求并行或分步搜索后给出比较结论。",
        "messages": [
            {
                "role": "user",
                "content": "对比一下老番茄和影视飓风最近一个月发布的视频，谁更高产？",
            },
        ],
        "checks": {
            "expected_tools_any": ["search_videos"],
            "content_contains": ["老番茄", "影视飓风"],
            "content_pattern": r"bilibili\.com/video/BV",
            "max_tokens": 18000,
        },
        "manual_review_focus": [
            "是否真的比较了两边近一个月产量",
            "结论是否和列出的视频一致",
        ],
    },
    {
        "id": 15,
        "name": "multi_creator_productivity_comparison_followup_guard",
        "description": "多作者近期产量对比，要求优先拆成每位作者自己的时间窗查询，而不是一句泛搜。",
        "messages": [
            {
                "role": "user",
                "content": "对比一下老番茄和影视飓风最近一个月发布的视频，谁更高产？",
            },
        ],
        "checks": {
            "expected_tools_any": ["search_videos"],
            "content_contains": ["老番茄", "影视飓风"],
            "content_pattern": r"bilibili\.com/video/BV",
            "max_tokens": 18000,
        },
        "manual_review_focus": [
            "是否真的按两位作者各自近一个月的视频来比较",
            "是否避免把整句口语直接作为 query",
        ],
    },
    {
        "id": 16,
        "name": "token_alias_then_video_search",
        "description": "别名/错写先做 token 补全，再落成实体化视频搜索，不应把原始错写直接塞进 search_videos。",
        "messages": [
            {"role": "user", "content": "康夫UI 有什么入门教程？"},
        ],
        "checks": {
            "expected_tools_all": ["related_tokens_by_tokens", "search_videos"],
            "content_contains": ["ComfyUI"],
            "content_pattern": r"bilibili\.com/video/BV",
            "content_not_contains": ["康夫UI q=vwr"],
            "max_tokens": 14000,
        },
        "manual_review_focus": [
            "是否先完成术语纠错，再用纠正后的实体名搜索",
            "search_videos query 是否已经去掉口语词和原始错写",
        ],
    },
    {
        "id": 17,
        "name": "abstract_mood_video_request",
        "description": "抽象情绪需求，要求先语义展开或直接落成视频搜索，不应误走 Google。",
        "messages": [
            {"role": "user", "content": "来点让我开心的视频，别太长"},
        ],
        "checks": {
            "expected_tools_any": ["related_tokens_by_tokens", "search_videos"],
            "forbidden_tools": ["search_google"],
            "content_pattern": r"bilibili\.com/video/BV",
            "max_tokens": 16000,
        },
        "manual_review_focus": [
            "是否把抽象需求落成更具体的视频结果",
            "是否明显偏轻量、好进入的内容",
        ],
    },
    {
        "id": 9,
        "name": "multi_hop_topic_drill",
        "description": "先找热门主题，再追到作者的其他热门视频，检验多跳链路。",
        "messages": [
            {
                "role": "user",
                "content": "最近一个月B站上最火的AI相关视频是谁发的？帮我找一下这个UP主的其他热门视频。",
            },
        ],
        "checks": {
            "expected_tools_any": ["search_videos"],
            "content_contains": ["AI"],
            "content_pattern": r"bilibili\.com/video/BV",
            "max_tokens": 22000,
        },
        "manual_review_focus": [
            "是否先找到了热门 AI 视频再追作者",
            "第二跳推荐是否来自同一作者",
        ],
    },
]


def _flatten_tool_names(tool_events: list[dict] | None) -> list[str]:
    names = []
    for event in tool_events or []:
        names.extend(event.get("tools") or [])
    return names


def _format_messages(messages: list[dict]) -> str:
    return " | ".join(
        f"{message.get('role')}={message.get('content', '').strip()}"
        for message in messages
    )


def _record_soft_check(
    scored_checks: list[tuple[bool, float, str]],
    *,
    passed: bool,
    weight: float,
    message: str,
) -> None:
    scored_checks.append((passed, weight, message))


def evaluate_test_case_result(
    *,
    content: str,
    used_tools: list[str],
    checks: dict,
    total_tokens: int,
    usage_trace: dict,
) -> dict:
    hard_failures: list[str] = []
    soft_failures: list[str] = []
    scored_checks: list[tuple[bool, float, str]] = []

    for keyword in checks.get("content_contains", []):
        _record_soft_check(
            scored_checks,
            passed=keyword in content,
            weight=0.9,
            message=f"content missing '{keyword}'",
        )

    for forbidden in checks.get("content_not_contains", []):
        if forbidden in content:
            hard_failures.append(f"content unexpectedly contains '{forbidden}'")

    pattern = checks.get("content_pattern")
    if pattern:
        _record_soft_check(
            scored_checks,
            passed=bool(re.search(pattern, content)),
            weight=1.1,
            message=f"content does not match pattern '{pattern}'",
        )

    for tool_name in checks.get("expected_tools_all", []):
        _record_soft_check(
            scored_checks,
            passed=tool_name in used_tools,
            weight=1.0,
            message=f"missing expected tool '{tool_name}'",
        )

    expected_any = checks.get("expected_tools_any", [])
    if expected_any:
        _record_soft_check(
            scored_checks,
            passed=any(tool_name in used_tools for tool_name in expected_any),
            weight=1.1,
            message=f"missing any expected tool in {expected_any}",
        )

    for tool_name in checks.get("forbidden_tools", []):
        if tool_name in used_tools:
            hard_failures.append(f"unexpected tool '{tool_name}' was used")

    max_tokens = checks.get("max_tokens", 30000)
    if total_tokens > int(max_tokens * 1.5):
        hard_failures.append(
            f"token budget grossly exceeded: {total_tokens} > {max_tokens}"
        )
    elif total_tokens > max_tokens:
        _record_soft_check(
            scored_checks,
            passed=False,
            weight=0.7,
            message=f"token budget exceeded: {total_tokens} > {max_tokens}",
        )
    else:
        _record_soft_check(
            scored_checks,
            passed=True,
            weight=0.7,
            message="token budget within limit",
        )

    min_content_length = checks.get("min_content_length", 50)
    if len(content) < min_content_length:
        hard_failures.append(f"content too short: {len(content)} chars")

    if not usage_trace:
        hard_failures.append("usage_trace missing")

    if "DSML" in content or "function_calls" in content:
        hard_failures.append("DSML markup leaked into content")

    if any(
        marker in content
        for marker in [
            "[Error:",
            "Client Error:",
            "LLM request error",
            "Traceback (most recent call last)",
        ]
    ):
        hard_failures.append("runtime error leaked into content")

    score_total = sum(weight for _, weight, _ in scored_checks) or 1.0
    score = sum(weight for passed, weight, _ in scored_checks if passed) / score_total

    for passed, _, message in scored_checks:
        if not passed:
            soft_failures.append(message)

    min_score = checks.get("min_score", 0.70)
    warn_score = checks.get("warn_score", 0.55)
    if hard_failures:
        status = "FAIL"
    elif score >= min_score:
        status = "PASS"
    elif score >= warn_score:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "status": status,
        "score": round(score, 3),
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
    }


def run_test_case(handler, test_case: dict, verbose: bool = False) -> dict:
    """Run a single live chat scenario and validate output + tool routing."""
    tc_id = test_case["id"]
    name = test_case["name"]
    messages = test_case["messages"]
    checks = test_case["checks"]

    logger.note(f"\n{'=' * 72}")
    logger.note(f"[TEST {tc_id}] {name}")
    logger.mesg(f"  {test_case['description']}")
    logger.mesg(f"  Messages: {_format_messages(messages)}")
    if test_case.get("manual_review_focus"):
        logger.mesg(f"  Manual review: {'；'.join(test_case['manual_review_focus'])}")

    start_time = time.perf_counter()

    try:
        result = handler.handle(
            messages=messages,
            thinking=test_case.get("thinking", False),
            max_iterations=test_case.get("max_iterations"),
        )
    except Exception as exc:
        logger.warn(f"  × Handler error: {exc}")
        return {"name": name, "status": "ERROR", "error": str(exc)}

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    perf_stats = result.get("perf_stats", {})
    usage_trace = result.get("usage_trace", {})
    tool_events = result.get("tool_events", [])
    used_tools = _flatten_tool_names(tool_events)
    total_tokens = usage.get("total_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)

    cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    prompt_details = usage.get("prompt_tokens_details", {})
    if isinstance(prompt_details, dict) and not cache_hit:
        cache_hit = prompt_details.get("cached_tokens", 0)
    cache_miss = usage.get("prompt_cache_miss_tokens", 0)
    if isinstance(prompt_details, dict) and not cache_miss:
        cache_miss = max(0, prompt_tokens - cache_hit)

    logger.mesg(
        f"  Time: {perf_stats.get('total_elapsed', f'{elapsed_ms}ms')} ({elapsed_ms}ms)"
    )
    logger.mesg(
        f"  Tokens: {total_tokens} (prompt={prompt_tokens}, cache_hit={cache_hit}, cache_miss={cache_miss})"
    )
    logger.mesg(f"  Tool events: {used_tools or ['<none>']}")
    if verbose and tool_events:
        for event in tool_events:
            logger.mesg(f"  Iteration {event.get('iteration')}")
            for call in event.get("calls") or []:
                summary_preview = json.dumps(
                    call.get("summary") or {}, ensure_ascii=False
                )[:320]
                logger.mesg(
                    f"    - {call.get('type')} args={call.get('args')} summary={summary_preview}"
                )
    if usage_trace:
        summary = usage_trace.get("summary", {})
        prompt_meta = usage_trace.get("prompt", {})
        models = usage_trace.get("models", {})
        logger.mesg(
            "  Usage trace: "
            f"llm_calls={summary.get('llm_calls', 0)}, "
            f"tool_iterations={summary.get('tool_iterations', 0)}, "
            f"prompt_chars={prompt_meta.get('total_chars', 0)}, "
            f"peak_prompt_tokens={summary.get('peak_prompt_tokens', 0)}"
        )
        if models:
            logger.mesg(
                "  Models: "
                f"planner={models.get('planner', {}).get('config', '')}, "
                f"response={models.get('response', {}).get('config', '')}, "
                f"delegate={models.get('delegate', {}).get('config', '')}"
            )

    evaluation = evaluate_test_case_result(
        content=content,
        used_tools=used_tools,
        checks=checks,
        total_tokens=total_tokens,
        usage_trace=usage_trace,
    )

    preview_limit = 900 if verbose else 320
    logger.mesg(f"  Preview ({min(len(content), preview_limit)} chars):")
    print(content[:preview_limit])
    if len(content) > preview_limit:
        print(f"  ... ({len(content) - preview_limit} chars truncated)")

    status = evaluation["status"]
    logger.mesg(f"  Score: {evaluation['score']:.3f}")
    if evaluation["hard_failures"]:
        logger.warn("  Hard failures:")
        for failure in evaluation["hard_failures"]:
            logger.warn(f"    × {failure}")
    if evaluation["soft_failures"]:
        logger.warn("  Soft misses:")
        for failure in evaluation["soft_failures"]:
            logger.warn(f"    △ {failure}")

    if status == "PASS":
        logger.success("  Quality checks passed")
    elif status == "WARN":
        logger.warn("  Acceptable with soft misses")
    else:
        logger.warn("  Quality checks failed")

    return {
        "name": name,
        "status": status,
        "score": evaluation["score"],
        "hard_failures": evaluation["hard_failures"],
        "soft_failures": evaluation["soft_failures"],
        "elapsed_ms": elapsed_ms,
        "total_tokens": total_tokens,
        "tokens_per_second": perf_stats.get("tokens_per_second", 0),
        "content_length": len(content),
        "used_tools": used_tools,
    }


def main():
    parser = argparse.ArgumentParser(description="Live LLM Chat Tests")
    from configs.envs import LLM_CONFIG
    from llms.models import DEFAULT_SMALL_MODEL_CONFIG

    parser.add_argument(
        "--llm-config", type=str, default=LLM_CONFIG, help="LLM config name"
    )
    parser.add_argument(
        "--small-llm-config",
        type=str,
        default=DEFAULT_SMALL_MODEL_CONFIG,
        help="Small-model config used for delegate tasks",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=60.0,
        help="Per-request LLM timeout in seconds for live regression runs",
    )
    parser.add_argument(
        "--elastic-index", type=str, default=None, help="Elastic videos index name"
    )
    parser.add_argument(
        "--elastic-env-name",
        type=str,
        default=None,
        help="Elastic env name in secrets.json",
    )
    parser.add_argument(
        "--search-base-url",
        type=str,
        default=None,
        help="Use a running search_app service instead of local direct search components",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show longer response previews",
    )
    parser.add_argument(
        "--test",
        type=int,
        action="append",
        help="Run one or more specific test IDs",
    )
    args = parser.parse_args()

    from configs.envs import SEARCH_APP_ENVS
    from llms.chat.handler import ChatHandler
    from llms.models import create_model_clients
    from llms.tools.executor import create_search_service

    elastic_index = args.elastic_index
    elastic_env_name = args.elastic_env_name or "elastic_dev"
    if not elastic_index:
        idx = SEARCH_APP_ENVS.get("elastic_index", {})
        elastic_index = idx.get("prod", idx) if isinstance(idx, dict) else idx

    logger.note("> Initializing live test environment...")
    logger.mesg(
        f"  LLM: large={args.llm_config}, small={args.small_llm_config}, timeout={args.llm_timeout}s, search_base_url={args.search_base_url or '<local>'}, index={elastic_index}, elastic_env={elastic_env_name}"
    )

    if args.search_base_url:
        search_client = create_search_service(
            base_url=args.search_base_url,
            verbose=args.verbose,
        )
    else:
        from elastics.relations import RelationsClient
        from elastics.videos.explorer import VideoExplorer
        from elastics.videos.searcher_v2 import VideoSearcherV2

        video_searcher = VideoSearcherV2(
            elastic_index,
            elastic_env_name=elastic_env_name,
        )
        video_explorer = VideoExplorer(
            elastic_index,
            elastic_env_name=elastic_env_name,
        )
        relations_client = RelationsClient(
            elastic_index,
            elastic_env_name=elastic_env_name,
        )
        search_client = create_search_service(
            video_searcher=video_searcher,
            video_explorer=video_explorer,
            relations_client=relations_client,
            verbose=args.verbose,
        )

    model_registry, llm_clients = create_model_clients(
        primary_large_config=args.llm_config,
        primary_small_config=args.small_llm_config,
        verbose=args.verbose,
    )
    large_client = llm_clients[model_registry.primary_large_config]
    small_client = llm_clients[model_registry.primary_small_config]
    large_client.timeout = args.llm_timeout
    small_client.timeout = args.llm_timeout

    handler = ChatHandler(
        llm_client=large_client,
        small_llm_client=small_client,
        search_client=search_client,
        model_registry=model_registry,
        verbose=args.verbose,
    )

    cases = TEST_CASES
    if args.test:
        selected_ids = set(args.test)
        cases = [case for case in TEST_CASES if case["id"] in selected_ids]
        if not cases:
            logger.warn(f"× Unknown test ID(s): {args.test}")
            sys.exit(1)

    results = [run_test_case(handler, case, verbose=args.verbose) for case in cases]

    logger.note(f"\n{'=' * 72}")
    logger.note("[SUMMARY]")
    logger.note("=" * 72)
    passed = sum(1 for result in results if result["status"] == "PASS")
    warned = sum(1 for result in results if result["status"] == "WARN")
    failed = sum(1 for result in results if result["status"] == "FAIL")
    total = len(results)
    for result in results:
        status_fn = logger.success if result["status"] == "PASS" else logger.warn
        status_fn(
            f"  {result['name']}: {result['status']} "
            f"(score={result.get('score', 0):.3f}, {result.get('elapsed_ms', 0)}ms, {result.get('total_tokens', 0)} tokens, "
            f"{result.get('tokens_per_second', 0)} tok/s, {result.get('content_length', 0)} chars, "
            f"tools={result.get('used_tools', [])})"
        )
    logger.note(f"\n  {passed}/{total} passed, {warned} warned, {failed} failed")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
