"""Structured signal extraction helpers for intent classification.

Classifier policy should live here as named signal rules instead of being spread
through llms.intent.classifier as ad hoc constants.
"""

from __future__ import annotations

import re

from dataclasses import dataclass

from llms.contracts import FacetScore
from llms.intent.taxonomy import iter_content_tokens
from llms.intent.taxonomy import normalize_text
from llms.intent.taxonomy import rank_facet_matches
from llms.intent.taxonomy import SemanticMatch


_STOPWORDS = frozenset(
    {
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
)
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
_LEADING_TOPIC_FILLER_RE = re.compile(
    r"^(?:请问|麻烦(?:帮我)?|帮我|给我|推荐(?:几个|一些)?|找(?:几个|一些)?|我想看|我想找|想看|想找|来点|看看|搜一下|搜搜|查一下|查查|做)+"
)
_TRAILING_OWNER_SUFFIX_RE = re.compile(
    r"(?:相关)?内容(?:创作)?的?(?:up主|作者|博主|账号).*$"
)
_FOLLOWUP_REFERENT_TOKENS = frozenset(
    {
        "他",
        "她",
        "它",
        "他的",
        "她的",
        "它的",
        "那他",
        "那她",
        "那它",
        "那他的",
        "那她的",
        "那它的",
        "这个",
        "那个",
    }
)

_FACET_SCORE_RULES = {
    "motivation": {"weight": 0.95, "threshold": 0.16, "limit": 3},
    "expected_payoff": {"weight": 1.0, "threshold": 0.16, "limit": 3},
    "consumption_mode": {"weight": 0.9, "threshold": 0.16, "limit": 3},
    "constraints": {"weight": 0.95, "threshold": 0.16, "limit": 3},
    "visual_intent_hints": {"weight": 0.75, "threshold": 0.14, "limit": 3},
}

_PAYOFF_DOC_HINTS = {
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

_PAYOFF_TO_MOTIVATION = {
    "clear_explanation": "utility_task",
    "decision_help": "utility_task",
    "funny": "emotion_regulation",
    "relaxing": "emotion_regulation",
    "healing": "emotion_regulation",
    "eye_candy": "aesthetic",
    "topic_for_chat": "social_shareability",
    "cozy_companionship": "companionship",
}

_KEYWORD_EXPANSION_AMBIGUITY_THRESHOLD = 0.45
_TARGET_LOW_CONFIDENCE_THRESHOLD = 0.18
_TARGET_MARGIN_THRESHOLD = 0.06
_ALIAS_TOKEN_MAX_LEN = 6
_KNOWN_TERM_ALIAS_PATTERNS = ((re.compile(r"康夫\s*u\s*i", re.IGNORECASE), "ComfyUI"),)


def _normalize_query_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def rewrite_known_term_aliases(text: str) -> str:
    rewritten = str(text or "")
    for pattern, canonical in _KNOWN_TERM_ALIAS_PATTERNS:
        rewritten = pattern.sub(canonical, rewritten)
    return rewritten


@dataclass(frozen=True, slots=True)
class ConversationWindow:
    latest_user_text: str
    normalized_query: str
    history_text: str
    user_turn_count: int

    @property
    def is_followup(self) -> bool:
        return self.user_turn_count > 1


@dataclass(frozen=True, slots=True)
class IntentSignalProfile:
    motivation: list[FacetScore]
    expected_payoff: list[FacetScore]
    consumption_mode: list[FacetScore]
    constraints: list[FacetScore]
    visual_intent_hints: list[FacetScore]
    explicit_entities: list[str]
    explicit_topics: list[str]
    doc_signal_hints: dict[str, list[str]]
    ambiguity: float
    complexity_score: float
    needs_keyword_expansion: bool
    needs_term_normalization: bool
    needs_owner_resolution: bool
    needs_external_search: bool
    route_reasons: list[str]


def build_conversation_window(messages: list[dict]) -> ConversationWindow:
    user_messages = [
        normalize_text(message.get("content") or "")
        for message in messages or []
        if message.get("role") == "user"
    ]
    latest_user_text = user_messages[-1] if user_messages else ""
    history_text = " ".join(user_messages[:-1][-2:]) if len(user_messages) > 1 else ""
    return ConversationWindow(
        latest_user_text=latest_user_text,
        normalized_query=latest_user_text,
        history_text=history_text,
        user_turn_count=len(user_messages),
    )


def collect_facet_scores(
    text: str,
    facet_name: str,
    *,
    history_text: str,
) -> list[FacetScore]:
    rule = _FACET_SCORE_RULES.get(facet_name, {})
    threshold = float(rule.get("threshold", 0.16))
    weight = float(rule.get("weight", 1.0))
    limit = int(rule.get("limit", 3))
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
    cleaned = _TRAILING_OWNER_SUFFIX_RE.sub("", cleaned)
    cleaned = _LEADING_TOPIC_FILLER_RE.sub("", cleaned)
    cleaned = _normalize_query_spaces(cleaned)
    cleaned = cleaned.strip(" -_/:：，。！？?；;")
    if cleaned in _FOLLOWUP_REFERENT_TOKENS:
        return ""
    return cleaned


def extract_topic_candidates(text: str) -> list[str]:
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


def merge_followup_candidates(messages: list[dict], candidates: list[str]) -> list[str]:
    merged = list(candidates)
    if len(merged) >= 2:
        return merged
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        for token in extract_topic_candidates(message.get("content") or ""):
            if token not in merged:
                merged.append(token)
            if len(merged) >= 5:
                return merged
    return merged


def extract_topics(messages: list[dict], latest_user_text: str) -> list[str]:
    return merge_followup_candidates(
        messages, extract_topic_candidates(latest_user_text)
    )[:10]


def extract_entities(messages: list[dict], latest_user_text: str) -> list[str]:
    latest_topics = extract_topic_candidates(latest_user_text)
    topics = merge_followup_candidates(messages, latest_topics)
    entities: list[str] = []
    for topic in topics:
        from_history = topic not in latest_topics
        if _ENTITY_LIKE_RE.search(topic) or from_history or len(topic) <= 4:
            entities.append(topic)
    return entities[:8]


def is_ascii_like(token: str) -> bool:
    token_text = str(token or "")
    return bool(token_text) and all(ord(char) < 128 for char in token_text)


def needs_term_normalization(
    explicit_entities: list[str],
    explicit_topics: list[str],
) -> bool:
    candidates = [
        token
        for token in (explicit_entities or explicit_topics)
        if 1 < len(token) <= _ALIAS_TOKEN_MAX_LEN
    ]
    return (
        bool(candidates)
        and any(is_ascii_like(token) for token in candidates)
        and any(not is_ascii_like(token) for token in candidates)
    )


def estimate_ambiguity(
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
    if top_target_score < _TARGET_LOW_CONFIDENCE_THRESHOLD:
        score += 0.14
    if target_margin < _TARGET_MARGIN_THRESHOLD:
        score += 0.10
    return min(score, 0.95)


def estimate_complexity(
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


def build_doc_signal_hints(expected_payoff: list[FacetScore]) -> dict[str, list[str]]:
    hints: dict[str, list[str]] = {}
    for facet in expected_payoff:
        mapped = _PAYOFF_DOC_HINTS.get(facet.label, {})
        for key, values in mapped.items():
            bucket = hints.setdefault(key, [])
            for value in values:
                if value not in bucket:
                    bucket.append(value)
    return hints


def default_motivation(
    expected_payoff: list[FacetScore],
    final_target: str,
) -> list[FacetScore]:
    if final_target in {"external", "mixed"}:
        return [FacetScore(label="information", score=0.70)]

    for payoff in expected_payoff:
        motivation = _PAYOFF_TO_MOTIVATION.get(payoff.label)
        if motivation:
            return [FacetScore(label=motivation, score=0.60)]
    return []


def build_route_reasons(
    *,
    needs_keyword_expansion: bool,
    needs_term_normalization_flag: bool,
    needs_owner_resolution: bool,
    needs_external_search: bool,
    task_mode: str,
    final_target_matches: list[SemanticMatch],
    task_mode_matches: list[SemanticMatch],
) -> list[str]:
    route_reasons: list[str] = []
    if needs_keyword_expansion:
        route_reasons.append("taxonomy:abstract-video-exploration")
    if needs_term_normalization_flag:
        route_reasons.append("taxonomy:term-normalization")
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
    return route_reasons


def derive_intent_signals(
    *,
    messages: list[dict],
    window: ConversationWindow,
    final_target: str,
    task_mode: str,
    final_target_matches: list[SemanticMatch],
    task_mode_matches: list[SemanticMatch],
) -> IntentSignalProfile:
    motivation = collect_facet_scores(
        window.normalized_query,
        "motivation",
        history_text=window.history_text,
    )
    expected_payoff = collect_facet_scores(
        window.normalized_query,
        "expected_payoff",
        history_text=window.history_text,
    )
    consumption_mode = collect_facet_scores(
        window.normalized_query,
        "consumption_mode",
        history_text=window.history_text,
    )
    constraints = collect_facet_scores(
        window.normalized_query,
        "constraints",
        history_text=window.history_text,
    )
    visual_intent_hints = collect_facet_scores(
        window.normalized_query,
        "visual_intent_hints",
        history_text=window.history_text,
    )
    explicit_entities = extract_entities(messages, window.latest_user_text)
    explicit_topics = extract_topics(messages, window.latest_user_text)

    top_target_score = final_target_matches[0].score if final_target_matches else 0.0
    target_margin = (
        top_target_score - final_target_matches[1].score
        if len(final_target_matches) > 1
        else top_target_score
    )
    ambiguity = estimate_ambiguity(
        window.normalized_query,
        explicit_entities,
        explicit_topics,
        expected_payoff,
        top_target_score,
        target_margin,
    )
    needs_term_normalization_flag = needs_term_normalization(
        explicit_entities,
        explicit_topics,
    )
    needs_keyword_expansion = bool(
        final_target == "videos"
        and task_mode == "exploration"
        and (
            (
                ambiguity >= _KEYWORD_EXPANSION_AMBIGUITY_THRESHOLD
                and len(explicit_entities) <= 1
            )
            or needs_term_normalization_flag
        )
    )
    needs_owner_resolution = bool(
        final_target == "videos"
        and explicit_entities
        and not needs_keyword_expansion
        and (
            window.is_followup
            or task_mode in {"repeat", "known_item"}
            or any(len(entity) <= 8 for entity in explicit_entities[:2])
        )
    )
    needs_external_search = final_target in {"external", "mixed"}
    complexity_score = estimate_complexity(
        final_target,
        ambiguity,
        needs_keyword_expansion=needs_keyword_expansion,
        needs_owner_resolution=needs_owner_resolution,
        needs_external_search=needs_external_search,
        is_followup=window.is_followup,
        task_mode=task_mode,
    )

    if not motivation:
        motivation = default_motivation(expected_payoff, final_target)

    return IntentSignalProfile(
        motivation=motivation,
        expected_payoff=expected_payoff,
        consumption_mode=consumption_mode,
        constraints=constraints,
        visual_intent_hints=visual_intent_hints,
        explicit_entities=explicit_entities,
        explicit_topics=explicit_topics,
        doc_signal_hints=build_doc_signal_hints(expected_payoff),
        ambiguity=ambiguity,
        complexity_score=complexity_score,
        needs_keyword_expansion=needs_keyword_expansion,
        needs_term_normalization=needs_term_normalization_flag,
        needs_owner_resolution=needs_owner_resolution,
        needs_external_search=needs_external_search,
        route_reasons=build_route_reasons(
            needs_keyword_expansion=needs_keyword_expansion,
            needs_term_normalization_flag=needs_term_normalization_flag,
            needs_owner_resolution=needs_owner_resolution,
            needs_external_search=needs_external_search,
            task_mode=task_mode,
            final_target_matches=final_target_matches,
            task_mode_matches=task_mode_matches,
        ),
    )


__all__ = [
    "ConversationWindow",
    "IntentSignalProfile",
    "build_conversation_window",
    "derive_intent_signals",
]
