"""Intent classifier built on top of the shared taxonomy.

The core rule is simple: route by labels and examples, not by one-off keywords.
Small cleanup heuristics are allowed for topic extraction, but final target and
task mode decisions must flow through the taxonomy matcher.
"""

from __future__ import annotations

import re

from llms.intent.taxonomy import detect_final_target
from llms.intent.taxonomy import detect_task_mode
from llms.intent.taxonomy import iter_content_tokens
from llms.intent.taxonomy import normalize_text
from llms.intent.taxonomy import rank_facet_matches
from llms.intent.taxonomy import rank_final_target_matches
from llms.intent.taxonomy import rank_task_mode_matches
from llms.protocol import FacetScore, IntentProfile


_STOPWORDS = {
    "我",
    "想",
    "想看",
    "想找",
    "帮我",
    "给我",
    "一下",
    "有没有",
    "有什么",
    "有哪些",
    "推荐",
    "几个",
    "一些",
    "这个",
    "那个",
    "内容",
    "结果",
    "b站",
}
_QUERY_BOUNDARY_MARKERS = (
    "有哪些关联账号",
    "有哪些关联作者",
    "最近有哪些官方更新",
    "最近有什么新视频",
    "最近一个月发布的视频",
    "代表作有哪些",
    "有哪些",
    "有什么",
    "是谁",
    "是什么",
    "最近",
    "最新",
    "视频",
    "教程",
    "解读",
    "作者",
    "账号",
    "内容",
)
_ENTITY_LIKE_RE = re.compile(
    r"(?:bv[0-9a-z]+|[a-z][a-z0-9.+#:/_-]+|\d+(?:\.\d+)*)", re.IGNORECASE
)


_DOC_SIGNAL_MAP = {
    "funny": {
        "promise.cover_hook": ["funny_expression", "cute_animal"],
        "promise.surface_style": ["chaotic", "meme_like"],
        "evidence.presentation_form": ["compilation", "reaction"],
    },
    "eye_candy": {
        "promise.cover_subject": ["young_female"],
        "promise.cover_hook": ["pretty_face"],
        "promise.surface_style": ["sexy", "pure_desire_style"],
    },
    "clear_explanation": {
        "promise.promise_format": ["looks_like_tutorial", "looks_like_review"],
        "evidence.presentation_form": ["step_by_step_tutorial", "commentary"],
        "evidence.audio_structure": ["clear_narration"],
    },
    "relaxing": {
        "promise.surface_style": ["cozy", "healing_style"],
        "evidence.emotion_trajectory": ["positive_low_arousal", "healing"],
    },
}


def _normalize_query(text: str) -> str:
    return normalize_text(text)


def _latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return _normalize_query(message.get("content") or "")
    return ""


def _user_history_text(messages: list[dict]) -> str:
    user_messages = [
        _normalize_query(message.get("content") or "")
        for message in messages or []
        if message.get("role") == "user"
    ]
    if len(user_messages) <= 1:
        return ""
    return " ".join(user_messages[:-1][-2:])


def _collect_facet_scores(
    text: str,
    facet_name: str,
    *,
    history_text: str,
    weight: float,
    threshold: float = 0.16,
    limit: int = 3,
) -> list[FacetScore]:
    labels = []
    for match in rank_facet_matches(text, facet_name, history_text=history_text):
        if match.score < threshold:
            continue
        labels.append(
            FacetScore(label=match.name, score=round(match.score * weight, 3))
        )
        if len(labels) >= limit:
            break
    return labels


def _cleanup_topic_candidate(token: str) -> str:
    cleaned = str(token or "").strip().lower()
    if not cleaned:
        return ""
    for marker in _QUERY_BOUNDARY_MARKERS:
        if marker in cleaned:
            prefix = cleaned.split(marker, 1)[0].strip()
            if len(prefix) >= 2:
                cleaned = prefix
                break
    cleaned = cleaned.strip(" -_/：:，。！？?；;")
    return cleaned


def _extract_topic_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for token in iter_content_tokens(text):
        cleaned = _cleanup_topic_candidate(token)
        if len(cleaned) < 2:
            continue
        if cleaned in _STOPWORDS:
            continue
        if cleaned.startswith(":") or cleaned.startswith("q="):
            continue
        if cleaned not in candidates:
            candidates.append(cleaned)
    return candidates


def _merge_followup_candidates(
    messages: list[dict], candidates: list[str]
) -> list[str]:
    merged = list(candidates)
    if len(merged) >= 2:
        return merged
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        for token in _extract_topic_candidates(message.get("content") or ""):
            if token not in merged:
                merged.append(token)
            if len(merged) >= 5:
                return merged
    return merged


def _extract_topics(messages: list[dict], latest_user_text: str) -> list[str]:
    return _merge_followup_candidates(
        messages, _extract_topic_candidates(latest_user_text)
    )[:10]


def _extract_entities(messages: list[dict], latest_user_text: str) -> list[str]:
    latest_topics = _extract_topic_candidates(latest_user_text)
    topics = _merge_followup_candidates(messages, latest_topics)
    entities: list[str] = []
    for topic in topics:
        from_history = topic not in latest_topics
        if _ENTITY_LIKE_RE.search(topic) or from_history or len(topic) <= 4:
            entities.append(topic)
    return entities[:8]


def _is_ascii_like(token: str) -> bool:
    token_text = str(token or "")
    return bool(token_text) and all(ord(char) < 128 for char in token_text)


def _looks_like_alias_compound(
    explicit_entities: list[str],
    explicit_topics: list[str],
) -> bool:
    candidates = [
        token for token in (explicit_entities or explicit_topics) if len(token) <= 6
    ]
    return (
        bool(candidates)
        and any(_is_ascii_like(token) for token in candidates)
        and any(not _is_ascii_like(token) for token in candidates)
    )


def _estimate_ambiguity(
    text: str,
    entities: list[str],
    topics: list[str],
    expected_payoff: list[FacetScore],
    top_target_score: float,
    target_margin: float,
) -> float:
    score = 0.14
    if len(text) <= 8:
        score += 0.18
    if not entities:
        score += 0.18
    if len(topics) <= 1:
        score += 0.10
    if expected_payoff and not entities:
        score += 0.10
    if top_target_score < 0.18:
        score += 0.14
    if target_margin < 0.06:
        score += 0.10
    return min(score, 0.95)


def _estimate_complexity(
    final_target: str,
    ambiguity: float,
    *,
    needs_keyword_expansion: bool,
    needs_owner_resolution: bool,
    needs_external_search: bool,
    is_followup: bool,
    task_mode: str,
) -> float:
    score = 0.15 + ambiguity * 0.35
    if final_target == "mixed":
        score += 0.30
    elif final_target in {"owners", "relations", "external"}:
        score += 0.10
    if needs_keyword_expansion:
        score += 0.20
    if needs_owner_resolution:
        score += 0.15
    if needs_external_search:
        score += 0.15
    if is_followup:
        score += 0.10
    if task_mode == "collect_compare":
        score += 0.15
    return min(score, 1.0)


def build_intent_profile(messages: list[dict]) -> IntentProfile:
    latest_user_text = _latest_user_text(messages)
    normalized = _normalize_query(latest_user_text)
    history_text = _user_history_text(messages)
    final_target_matches = rank_final_target_matches(
        normalized, history_text=history_text
    )
    final_target = detect_final_target(normalized, history_text=history_text)
    task_mode_matches = rank_task_mode_matches(
        normalized,
        final_target,
        history_text=history_text,
    )
    task_mode = detect_task_mode(normalized, final_target, history_text=history_text)
    motivation = _collect_facet_scores(
        normalized,
        "motivation",
        history_text=history_text,
        weight=0.95,
    )
    expected_payoff = _collect_facet_scores(
        normalized,
        "expected_payoff",
        history_text=history_text,
        weight=1.0,
    )
    consumption_mode = _collect_facet_scores(
        normalized,
        "consumption_mode",
        history_text=history_text,
        weight=0.9,
    )
    constraints = _collect_facet_scores(
        normalized,
        "constraints",
        history_text=history_text,
        weight=0.95,
    )
    visual_intent_hints = _collect_facet_scores(
        normalized,
        "visual_intent_hints",
        history_text=history_text,
        weight=0.75,
        threshold=0.14,
    )
    explicit_entities = _extract_entities(messages, latest_user_text)
    explicit_topics = _extract_topics(messages, latest_user_text)
    top_target_score = final_target_matches[0].score if final_target_matches else 0.0
    target_margin = (
        top_target_score - final_target_matches[1].score
        if len(final_target_matches) > 1
        else top_target_score
    )
    ambiguity = _estimate_ambiguity(
        normalized,
        explicit_entities,
        explicit_topics,
        expected_payoff,
        top_target_score,
        target_margin,
    )
    is_followup = (
        sum(1 for message in messages or [] if message.get("role") == "user") > 1
    )

    needs_keyword_expansion = bool(
        final_target == "videos"
        and task_mode == "exploration"
        and (
            (ambiguity >= 0.45 and len(explicit_entities) <= 1)
            or _looks_like_alias_compound(explicit_entities, explicit_topics)
        )
    )
    needs_owner_resolution = bool(
        final_target == "videos"
        and explicit_entities
        and not needs_keyword_expansion
        and (
            is_followup
            or task_mode in {"repeat", "known_item"}
            or any(len(entity) <= 8 for entity in explicit_entities[:2])
        )
    )
    needs_external_search = final_target in {"external", "mixed"}

    doc_signal_hints: dict[str, list[str]] = {}
    for facet in expected_payoff:
        mapped = _DOC_SIGNAL_MAP.get(facet.label, {})
        for key, values in mapped.items():
            bucket = doc_signal_hints.setdefault(key, [])
            for value in values:
                if value not in bucket:
                    bucket.append(value)

    complexity_score = _estimate_complexity(
        final_target,
        ambiguity,
        needs_keyword_expansion=needs_keyword_expansion,
        needs_owner_resolution=needs_owner_resolution,
        needs_external_search=needs_external_search,
        is_followup=is_followup,
        task_mode=task_mode,
    )

    route_reasons = []
    if needs_keyword_expansion:
        route_reasons.append("taxonomy:abstract-video-exploration")
    if needs_owner_resolution:
        route_reasons.append("taxonomy:owner-context-carry")
    if needs_external_search:
        route_reasons.append("taxonomy:external-facts")
    if task_mode == "collect_compare":
        route_reasons.append("taxonomy:compare-multiple-candidates")
    if final_target_matches:
        route_reasons.append(f"target≈{final_target_matches[0].evidence}")
    if task_mode_matches:
        route_reasons.append(f"mode≈{task_mode_matches[0].evidence}")

    if not motivation:
        if final_target in {"external", "mixed"}:
            motivation = [FacetScore(label="information", score=0.70)]
        elif needs_keyword_expansion:
            motivation = [FacetScore(label="entertainment", score=0.55)]

    return IntentProfile(
        raw_query=latest_user_text,
        normalized_query=normalized,
        final_target=final_target,
        task_mode=task_mode,
        motivation=motivation,
        consumption_mode=consumption_mode,
        expected_payoff=expected_payoff,
        constraints=constraints,
        visual_intent_hints=visual_intent_hints,
        explicit_entities=explicit_entities,
        explicit_topics=explicit_topics,
        doc_signal_hints=doc_signal_hints,
        ambiguity=ambiguity,
        complexity_score=complexity_score,
        needs_keyword_expansion=needs_keyword_expansion,
        needs_owner_resolution=needs_owner_resolution,
        needs_external_search=needs_external_search,
        is_followup=is_followup,
        route_reason=", ".join(route_reasons),
    )


def select_prompt_asset_ids(intent: IntentProfile) -> list[str]:
    asset_ids = [
        "role.brief",
        "output.brief",
        "prompt_loading.brief",
        "result_isolation.brief",
        "response_style.brief",
    ]
    if intent.final_target == "videos":
        asset_ids.extend(
            ["route.videos.brief", "dsl.quickref.brief", "tool.search_videos.brief"]
        )
    elif intent.final_target == "owners":
        asset_ids.extend(["route.owners.brief", "tool.search_owners.brief"])
    elif intent.final_target == "relations":
        asset_ids.extend(["route.relations.brief", "tool.search_owners.brief"])
    elif intent.final_target == "external":
        asset_ids.extend(["route.external.brief", "tool.search_google.brief"])
    elif intent.final_target == "mixed":
        asset_ids.extend(
            [
                "route.mixed.brief",
                "dsl.quickref.brief",
                "tool.search_google.brief",
                "tool.search_videos.brief",
            ]
        )

    if intent.needs_keyword_expansion:
        asset_ids.extend(
            ["semantic.expansion.brief", "tool.related_tokens_by_tokens.brief"]
        )
    if intent.doc_signal_hints:
        asset_ids.append("facet.mapping.brief")
    return asset_ids
