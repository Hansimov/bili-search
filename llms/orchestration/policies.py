"""Coverage and nudge policies for the chat orchestrator.

Policy activation must depend on structured intent and execution results. Do not
smuggle new route decisions into this module with ad hoc text matching.

Search-query wording, typo correction, aliasing, and conversational cleanup must
be handled by the large-model planning workflow before XML commands are emitted,
or by stable execution gates that reason over tool structure and results. Do not
add lexical examples or regex phrase catalogs here to "fix" query wording. This
module may only hold coarse routing/coverage predicates whose inputs are intent
labels and tool results, plus stable syntax checks such as explicit BV anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

from llms.contracts import IntentProfile
from llms.tools.names import canonical_tool_name


FINAL_ANSWER_NUDGE = "请直接基于现有结果回答，不要继续规划，也不要再次调用工具。"
_EXPLICIT_BVID_RE = re.compile(r"\bBV[0-9A-Za-z]{10}\b", re.IGNORECASE)
_RECENT_TIMELINE_TOKENS = (
    "最近",
    "近期",
    "近况",
    "最近还发",
    "最近还有哪些视频",
    "最近发了哪些视频",
    "还发了哪些视频",
    "还发了什么视频",
)
_INTERVIEW_THEME_TOKENS = (
    "采访",
    "专访",
    "访谈",
)


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
    query_text = re.sub(r"\s+", "", str(intent.raw_query or ""))
    return (
        intent.task_mode == "repeat"
        or "recent_only" in intent.top_labels("constraints", limit=8)
        or any(token in query_text for token in _RECENT_TIMELINE_TOKENS)
    )


def _iter_successful_tool_records(result_store, tool_name: str):
    expected_tool_name = canonical_tool_name(tool_name)
    for result_id in result_store.order:
        record = result_store.get(result_id)
        if (
            record is None
            or canonical_tool_name(record.request.name) != expected_tool_name
        ):
            continue
        result = record.result or {}
        if result.get("error"):
            continue
        yield record


def has_explicit_video_anchor(intent: IntentProfile) -> bool:
    candidates = [
        str(token or "")
        for token in [
            *(intent.explicit_entities or []),
            *(intent.explicit_topics or []),
            intent.raw_query,
        ]
        if str(token or "").strip()
    ]
    return any(_EXPLICIT_BVID_RE.search(candidate) for candidate in candidates)


def has_explicit_bvid_lookup_result(result_store) -> bool:
    for record in _iter_successful_tool_records(result_store, "search_videos"):
        args = record.request.arguments or {}
        result = record.result or {}
        lookup_by = str(result.get("lookup_by") or args.get("lookup_by") or "").lower()
        mode = str(args.get("mode") or result.get("mode") or "").lower()
        if mode != "lookup":
            continue
        if lookup_by in {"bvid", "bvids"}:
            return True
        if args.get("bv") or args.get("bvid") or args.get("bvids"):
            return True
    return False


def has_recent_owner_timeline_result(result_store) -> bool:
    for record in _iter_successful_tool_records(result_store, "search_videos"):
        args = record.request.arguments or {}
        result = record.result or {}
        lookup_by = str(result.get("lookup_by") or args.get("lookup_by") or "").lower()
        mode = str(args.get("mode") or result.get("mode") or "").lower()
        if (
            mode == "lookup"
            and (
                lookup_by in {"mid", "mids"}
                or args.get("mid")
                or args.get("mids")
                or args.get("uid")
            )
            and (args.get("date_window") or result.get("date_window"))
        ):
            return True

        queries = args.get("queries")
        if isinstance(queries, str):
            queries = [queries]
        if any(":date<=" in str(query or "") for query in queries or []) and any(
            marker in str(query or "")
            for query in queries or []
            for marker in (":uid=", ":user=")
        ):
            return True
    return False


def needs_explicit_video_lookup_followup(result_store, intent: IntentProfile) -> bool:
    return bool(
        intent.final_target == "videos"
        and intent.needs_owner_resolution
        and has_explicit_video_anchor(intent)
        and is_recent_timeline_request(intent)
        and has_explicit_bvid_lookup_result(result_store)
        and not has_recent_owner_timeline_result(result_store)
    )


def has_successful_tool_result(result_store, tool_name: str) -> bool:
    expected_tool_name = canonical_tool_name(tool_name)
    for result_id in reversed(result_store.order):
        record = result_store.get(result_id)
        if (
            record is None
            or canonical_tool_name(record.request.name) != expected_tool_name
        ):
            continue
        result = record.result or {}
        if result.get("error"):
            continue
        if expected_tool_name == "search_google":
            return bool(result.get("results")) or int(result.get("result_count", 0)) > 0
        if expected_tool_name == "search_videos":
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
        if expected_tool_name == "search_owners":
            return (
                bool(result.get("owners"))
                or int(result.get("total_owners", 0) or 0) > 0
            )
        if expected_tool_name == "expand_query":
            return (
                bool(result.get("options"))
                or int(result.get("total_options", 0) or 0) > 0
            )
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
    return has_successful_tool_result(result_store, "search_owners")


def has_token_expansion_result(result_store) -> bool:
    return has_successful_tool_result(result_store, "expand_query")


def has_video_coverage(result_store) -> bool:
    return has_successful_tool_result(
        result_store, "search_videos"
    ) or has_google_video_result(result_store)


def _iter_search_video_items(result_store):
    for record in _iter_successful_tool_records(result_store, "search_videos"):
        result = record.result or {}
        nested_results = result.get("results") or []
        if nested_results:
            for item in nested_results:
                if not isinstance(item, dict) or item.get("error"):
                    continue
                yield item
            continue
        yield result


def _has_interview_intent(result_store, latest_user_message: str) -> bool:
    if any(
        token in str(latest_user_message or "") for token in _INTERVIEW_THEME_TOKENS
    ):
        return True
    for record in _iter_successful_tool_records(result_store, "expand_query"):
        result = record.result or {}
        for option in result.get("options") or []:
            text = str(option.get("text") or "")
            if any(token in text for token in _INTERVIEW_THEME_TOKENS):
                return True
    return False


def _search_video_results_lack_interview_anchor(result_store) -> bool:
    saw_hits = False
    for item in _iter_search_video_items(result_store):
        for hit in item.get("hits") or []:
            saw_hits = True
            searchable_text = "\n".join(
                [
                    str(hit.get("title") or ""),
                    str(hit.get("desc") or ""),
                    (
                        " ".join(hit.get("tags") or [])
                        if isinstance(hit.get("tags"), list)
                        else str(hit.get("tags") or "")
                    ),
                ]
            )
            if any(token in searchable_text for token in _INTERVIEW_THEME_TOKENS):
                return False
    return saw_hits


def has_internal_answer_ready_result(result_store) -> bool:
    return has_successful_tool_result(result_store, "run_small_llm_task")


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
        predicate=lambda store: has_video_coverage(store)
        or has_internal_answer_ready_result(store),
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
        name="prefer_owner_discovery_before_video_search",
        predicate=lambda store, intent, user_tool_names: (
            intent.final_target in {"owners", "relations"}
            and not store.order
            and "search_videos" in user_tool_names
            and "search_owners" not in user_tool_names
        ),
        message=(
            "当前任务目标是找作者、矩阵号或关联作者。请先调用 search_owners 获取作者候选，"
            "它会自动聚合名字、主题、关系和空间页线索；不要先用 search_videos 兜圈子。"
        ),
    ),
    ToolLoopNudgeRule(
        name="prefer_owner_resolution_before_external_detour",
        predicate=lambda store, intent, user_tool_names: (
            intent.final_target == "videos"
            and intent.needs_owner_resolution
            and not store.order
            and "search_google" in user_tool_names
            and not any(
                tool_name in {"search_owners", "search_videos"}
                for tool_name in user_tool_names
            )
        ),
        message=(
            "当前请求更像基于上文作者找代表作、时间线或作者本人视频。请先 search_owners 确认作者，"
            "或直接发起带 :user / :uid 的 search_videos；不要先绕到 search_google。"
        ),
    ),
    ToolLoopNudgeRule(
        name="prefer_term_normalization_before_video_search",
        predicate=lambda store, intent, user_tool_names: (
            intent.final_target == "videos"
            and intent.needs_term_normalization
            and not store.order
            and "search_videos" in user_tool_names
            and "expand_query" not in user_tool_names
        ),
        message=(
            "当前请求更像别名、错写或中英混写缩写。请先调用 expand_query，"
            "默认直接用 semantic 做归一化；只有明确拼写纠错时才指定 correction。"
            "拿到规范词后再执行 search_videos；"
            "不要直接把原始错写整句塞进 search_videos。"
        ),
    ),
    ToolLoopNudgeRule(
        name="prefer_video_search_after_token_expansion",
        predicate=lambda store, intent, user_tool_names: (
            intent.final_target == "videos"
            and intent.needs_term_normalization
            and has_token_expansion_result(store)
            and not has_video_coverage(store)
            and "search_videos" not in user_tool_names
            and any(
                tool_name in {"expand_query", "search_google", "search_owners"}
                for tool_name in user_tool_names
            )
        ),
        message=(
            "你已经拿到术语归一化候选。下一步直接执行 search_videos，"
            "把 expand_query 里最可信的 1 到 2 个规范词写进 queries；"
            "不要重复 expand_query，也不要先绕到 search_google。"
        ),
    ),
    ToolLoopNudgeRule(
        name="prefer_owner_scoped_video_search_after_resolution",
        predicate=lambda store, intent, user_tool_names: (
            intent.final_target == "videos"
            and intent.needs_owner_resolution
            and has_owner_coverage(store)
            and not has_video_coverage(store)
            and "search_videos" not in user_tool_names
            and any(
                tool_name in {"search_owners", "search_google", "expand_query"}
                for tool_name in user_tool_names
            )
        ),
        message=(
            "你已经拿到作者候选。下一步直接执行 search_videos，并尽量带 :user 或 :uid "
            "把结果定向到该作者；不要重复 search_owners，也不要先绕到 search_google。"
            "如果用户问的是代表作或经典视频，不要默认加最近时间窗；只有明确问“最近”时再加 :date。"
        ),
    ),
    ToolLoopNudgeRule(
        name="owner_results_already_sufficient",
        predicate=lambda store, intent, user_tool_names: (
            intent.final_target in {"owners", "relations"}
            and has_owner_coverage(store)
            and "search_owners" in user_tool_names
        ),
        message=(
            "已经拿到一轮作者候选或关系线索。请直接基于现有 result_id 回答；"
            "不要重复 search_owners。若只差细节，再用 inspect_tool_result。"
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
        name="weak_interview_video_evidence",
        predicate=lambda store, intent, latest_user_message: (
            intent.final_target == "videos"
            and has_video_coverage(store)
            and _has_interview_intent(store, latest_user_message)
            and _search_video_results_lack_interview_anchor(store)
        ),
        message=(
            "当前 search_videos 返回的标题里没有出现采访 / 专访 / 访谈等主题锚点。"
            "不要把这些结果包装成采访命中；应明确说明当前语料里缺少高置信采访结果，"
            "现有结果最多只能当旁证或相关人物线索。"
        ),
    ),
    ResultNudgeRule(
        name="explicit_video_lookup_recent_followup",
        predicate=lambda store, intent, latest_user_message: (
            needs_explicit_video_lookup_followup(store, intent)
        ),
        message=(
            "你已经通过显式 BV lookup 拿到了这期视频和作者线索。下一步直接继续调用 search_videos，"
            "使用 mode='lookup' 和作者 mid/uid 查询最近视频，并把当前 BV 放进 exclude_bvids；"
            "不要现在就结束回答，也不要退回泛化搜索。"
        ),
    ),
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
            "你已经拿到 expand_query 的候选，但第一轮 search_videos 仍然没有有效结果。"
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
    if not rule:
        return False
    if needs_explicit_video_lookup_followup(result_store, intent):
        return False
    return bool(rule.predicate(result_store))


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
