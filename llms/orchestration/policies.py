"""Coverage and nudge policies for the chat orchestrator.

Policy activation must depend on structured intent and execution results. Do not
smuggle new route decisions into this module with ad hoc text matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from llms.contracts import IntentProfile


FINAL_ANSWER_NUDGE = "请直接基于现有结果回答，不要继续规划，也不要再次调用工具。"


@dataclass(frozen=True, slots=True)
class CoverageRule:
    final_target: str
    predicate: Callable[[Any], bool]


@dataclass(frozen=True, slots=True)
class ToolLoopNudgeRule:
    name: str
    predicate: Callable[[Any, IntentProfile, list[str]], bool]
    message: str


@dataclass(frozen=True, slots=True)
class ResultNudgeRule:
    name: str
    predicate: Callable[[Any, IntentProfile, str], bool]
    message: str


@dataclass(frozen=True, slots=True)
class BlockedRequestNudgeRule:
    name: str
    predicate: Callable[[IntentProfile, bool], bool]
    message: str


def is_recent_timeline_request(intent: IntentProfile) -> bool:
    return intent.task_mode == "repeat" or "recent_only" in intent.top_labels(
        "constraints",
        limit=8,
    )


def has_successful_tool_result(result_store, tool_name: str) -> bool:
    for result_id in reversed(result_store.order):
        record = result_store.get(result_id)
        if record is None or record.request.name != tool_name:
            continue
        result = record.result or {}
        if result.get("error"):
            continue
        if tool_name == "search_google":
            return bool(result.get("results")) or int(result.get("result_count", 0)) > 0
        if tool_name == "search_videos":
            if result.get("hits"):
                return True
            if int(result.get("total_hits", 0) or 0) > 0:
                return True
            nested_results = result.get("results") or []
            for item in nested_results:
                if item.get("error"):
                    continue
                if item.get("hits") or int(item.get("total_hits", 0) or 0) > 0:
                    return True
            continue
        return True
    return False


def has_google_video_result(result_store) -> bool:
    for result_id in reversed(result_store.order):
        record = result_store.get(result_id)
        if record is None or record.request.name != "search_google":
            continue
        for row in record.result.get("results") or []:
            link = str(row.get("link", "") or "")
            domain = str(row.get("domain", "") or "")
            site_kind = str(row.get("site_kind", "") or "")
            if "bilibili.com/video/" in link:
                return True
            if domain.endswith("bilibili.com") and site_kind == "video":
                return True
    return False


def has_non_bilibili_google_result(result_store) -> bool:
    for result_id in reversed(result_store.order):
        record = result_store.get(result_id)
        if record is None or record.request.name != "search_google":
            continue
        for row in record.result.get("results") or []:
            link = str(row.get("link", "") or "")
            domain = str(row.get("domain", "") or "")
            if "bilibili.com/video/" in link:
                continue
            if domain.endswith("bilibili.com"):
                continue
            return True
    return False


def has_sufficient_mixed_coverage(result_store) -> bool:
    return has_non_bilibili_google_result(result_store) and (
        has_successful_tool_result(result_store, "search_videos")
        or has_google_video_result(result_store)
    )


def has_owner_coverage(result_store) -> bool:
    return has_successful_tool_result(result_store, "search_owners") or any(
        has_successful_tool_result(result_store, tool_name)
        for tool_name in (
            "related_owners_by_tokens",
            "related_owners_by_videos",
            "related_owners_by_owners",
        )
    )


def has_token_expansion_result(result_store) -> bool:
    return has_successful_tool_result(result_store, "related_tokens_by_tokens")


def has_video_coverage(result_store) -> bool:
    return has_successful_tool_result(
        result_store, "search_videos"
    ) or has_google_video_result(result_store)


def count_zero_hit_search_videos(result_store) -> int:
    count = 0
    for result_id in result_store.order:
        record = result_store.get(result_id)
        if record is None or record.request.name != "search_videos":
            continue
        result = record.result or {}
        if result.get("error"):
            continue
        if result.get("hits") or int(result.get("total_hits", 0) or 0) > 0:
            continue
        nested_results = result.get("results") or []
        if nested_results and any(
            item.get("hits") or int(item.get("total_hits", 0) or 0) > 0
            for item in nested_results
            if not item.get("error")
        ):
            continue
        count += 1
    return count


COVERAGE_RULES = {
    "external": CoverageRule(
        final_target="external",
        predicate=lambda store: has_successful_tool_result(store, "search_google"),
    ),
    "mixed": CoverageRule(
        final_target="mixed",
        predicate=has_sufficient_mixed_coverage,
    ),
    "videos": CoverageRule(
        final_target="videos",
        predicate=has_video_coverage,
    ),
    "owners": CoverageRule(
        final_target="owners",
        predicate=has_owner_coverage,
    ),
    "relations": CoverageRule(
        final_target="relations",
        predicate=has_owner_coverage,
    ),
}


PRE_EXECUTION_NUDGE_RULES = (
    ToolLoopNudgeRule(
        name="prefer_term_normalization_before_video_search",
        predicate=lambda store, intent, user_tool_names: (
            intent.final_target == "videos"
            and intent.needs_term_normalization
            and not store.order
            and "search_videos" in user_tool_names
            and "related_tokens_by_tokens" not in user_tool_names
        ),
        message=(
            "当前请求更像别名、错写或中英混写缩写。请先调用 related_tokens_by_tokens "
            "用 correction 或 associate 思路把原词归一化，再基于纠正后的规范词执行 search_videos；"
            "不要直接把原始错写整句塞进 search_videos。"
        ),
    ),
    ToolLoopNudgeRule(
        name="owner_results_already_sufficient",
        predicate=lambda store, intent, user_tool_names: (
            intent.final_target in {"owners", "relations"}
            and has_owner_coverage(store)
            and any(
                tool_name
                in {
                    "search_owners",
                    "related_owners_by_tokens",
                    "related_owners_by_videos",
                    "related_owners_by_owners",
                }
                for tool_name in user_tool_names
            )
        ),
        message=(
            "已经拿到一轮作者候选或关系线索。请直接基于现有 result_id 回答；"
            "不要重复 search_owners 或 relation 类工具。若只差细节，再用 inspect_tool_result。"
        ),
    ),
    ToolLoopNudgeRule(
        name="mixed_results_already_sufficient",
        predicate=lambda store, intent, user_tool_names: (
            intent.final_target == "mixed"
            and has_sufficient_mixed_coverage(store)
            and any(
                tool_name in {"search_google", "search_videos"}
                for tool_name in user_tool_names
            )
        ),
        message=(
            "官方信息和站内视频候选已经各拿到一轮。请直接基于现有 result_id 回答；"
            "不要再同时重跑 search_google 和 search_videos。若只差细节，再用 inspect_tool_result。"
        ),
    ),
)


POST_EXECUTION_NUDGE_RULES = (
    ResultNudgeRule(
        name="mixed_zero_hit_video_fallback",
        predicate=lambda store, intent, latest_user_message: (
            intent.final_target == "mixed"
            and not has_sufficient_mixed_coverage(store)
            and has_successful_tool_result(store, "search_google")
            and not has_google_video_result(store)
            and count_zero_hit_search_videos(store) >= 1
        ),
        message=(
            "站内 search_videos 已连续 0 命中。若还需要 B 站链接，请改用 search_google，"
            "并显式加上 site:bilibili.com/video；拿到具体 BV 或链接后直接回答。"
        ),
    ),
    ResultNudgeRule(
        name="token_expansion_retry",
        predicate=lambda store, intent, latest_user_message: (
            intent.final_target == "videos"
            and not has_video_coverage(store)
            and has_token_expansion_result(store)
            and count_zero_hit_search_videos(store) >= 1
        ),
        message=(
            "你已经拿到 related_tokens_by_tokens 的候选，但第一轮 search_videos 仍然没有有效结果。"
            "请先用 inspect_tool_result 查看 token options，选 1 到 2 个最可信的规范词重试 search_videos；"
            "若站内仍无结果，再用 search_google + site:bilibili.com/video 侦察，避免直接返回“没找到”。"
        ),
    ),
    ResultNudgeRule(
        name="video_zero_hit_google_fallback",
        predicate=lambda store, intent, latest_user_message: (
            intent.final_target == "videos"
            and not has_video_coverage(store)
            and count_zero_hit_search_videos(store) >= 1
            and bool(intent.explicit_entities or intent.explicit_topics)
            and not is_recent_timeline_request(intent)
        ),
        message=(
            "站内 search_videos 目前 0 命中。请改用 search_google，并显式加上 site:bilibili.com/video；"
            "优先使用用户提到的实体名、作者名或更规范的写法，拿到具体链接后直接回答。"
        ),
    ),
)


BLOCKED_REQUEST_NUDGE_RULES = (
    BlockedRequestNudgeRule(
        name="external_video_blocked",
        predicate=lambda intent, blocked: intent.final_target == "external" and blocked,
        message=(
            "当前任务只看官方或站外信息，不要调用 search_videos。"
            "请只用 search_google，必要时再 inspect_tool_result，然后直接回答。"
        ),
    ),
)


def has_target_coverage(result_store, intent: IntentProfile) -> bool:
    rule = COVERAGE_RULES.get(intent.final_target)
    return bool(rule and rule.predicate(result_store))


def _select_nudge(rules, prompted_rules: set[str], *args) -> tuple[str, str] | None:
    for rule in rules:
        if rule.name in prompted_rules:
            continue
        if rule.predicate(*args):
            return rule.name, rule.message
    return None


def select_pre_execution_nudge(
    result_store,
    intent: IntentProfile,
    user_tool_names: list[str],
    prompted_rules: set[str],
) -> tuple[str, str] | None:
    return _select_nudge(
        PRE_EXECUTION_NUDGE_RULES,
        prompted_rules,
        result_store,
        intent,
        user_tool_names,
    )


def select_post_execution_nudge(
    result_store,
    intent: IntentProfile,
    latest_user_message: str,
    prompted_rules: set[str],
) -> tuple[str, str] | None:
    return _select_nudge(
        POST_EXECUTION_NUDGE_RULES,
        prompted_rules,
        result_store,
        intent,
        latest_user_message,
    )


def select_blocked_request_nudge(
    intent: IntentProfile,
    blocked_external_video_request: bool,
    prompted_rules: set[str],
) -> tuple[str, str] | None:
    return _select_nudge(
        BLOCKED_REQUEST_NUDGE_RULES,
        prompted_rules,
        intent,
        blocked_external_video_request,
    )
