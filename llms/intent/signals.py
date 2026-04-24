"""Structured signal extraction helpers for intent classification.

Classifier policy should live here as named signal rules instead of being spread
through llms.intent.classifier as ad hoc constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from llms.contracts import FacetScore
from llms.messages import extract_bvids, extract_owner_mids
from llms.messages import extract_message_text
from llms.intent.focus import compact_focus_key
from llms.intent.focus import extract_focus_spans
from llms.intent.taxonomy import normalize_text
from llms.intent.taxonomy import rank_facet_matches
from llms.intent.taxonomy import SemanticMatch


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

_KEYWORD_EXPANSION_AMBIGUITY_THRESHOLD = 0.40
_TARGET_LOW_CONFIDENCE_THRESHOLD = 0.18
_TARGET_MARGIN_THRESHOLD = 0.06
_ALIAS_TOKEN_MAX_LEN = 6
_QUESTION_TAIL_RE = re.compile(
    r"(是谁|是什么|讲了什么|讲啥|说了什么|说啥|最近还?发了哪些视频|最近还有哪些视频|最近视频|代表作有哪些?)"
)
_OWNER_TOPIC_QUERY_RE = re.compile(
    r"^\s*(?P<owner>[\u4e00-\u9fffA-Za-z0-9_.\-]{2,24})\s*(?:关于|有关|相关|聊聊|讲讲|说说)\s*(?P<topic>.+?)\s*$"
)
_VIDEO_TOPIC_TAIL_RE = re.compile(
    r"(?:有哪些(?:值得看|推荐|相关)?(?:的)?视频.*|有什么(?:值得看|推荐|相关)?(?:的)?视频.*|值得看的视频.*|相关的视频.*|视频.*)$"
)


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
    raw_user_messages = [
        extract_message_text(message)
        for message in messages or []
        if message.get("role") == "user"
    ]
    normalized_user_messages = [
        normalize_text(text) for text in raw_user_messages if str(text or "").strip()
    ]
    latest_user_text = raw_user_messages[-1] if raw_user_messages else ""
    normalized_query = normalize_text(latest_user_text)
    history_text = (
        " ".join(normalized_user_messages[:-1][-2:])
        if len(normalized_user_messages) > 1
        else ""
    )
    return ConversationWindow(
        latest_user_text=latest_user_text,
        normalized_query=normalized_query,
        history_text=history_text,
        user_turn_count=len(raw_user_messages),
    )


def has_distinct_video_topic(
    explicit_entities: list[str],
    explicit_topics: list[str],
) -> bool:
    entity_keys = {
        compact_focus_key(entity)
        for entity in explicit_entities or []
        if compact_focus_key(entity)
    }
    for topic in explicit_topics or []:
        topic_key = compact_focus_key(topic)
        if len(topic_key) < 4 or topic_key in entity_keys:
            continue
        return True
    return False


def is_stable_owner_topic_video_query(
    *,
    final_target: str,
    task_mode: str,
    window: ConversationWindow,
    explicit_entities: list[str],
    explicit_topics: list[str],
    explicit_video_anchors: list[str],
    explicit_owner_ids: list[str],
    needs_term_normalization: bool,
) -> bool:
    return bool(
        final_target == "videos"
        and task_mode == "exploration"
        and not window.is_followup
        and explicit_entities
        and not explicit_video_anchors
        and not explicit_owner_ids
        and not needs_term_normalization
        and has_distinct_video_topic(explicit_entities, explicit_topics)
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


def extract_owner_topic_pair(text: str) -> tuple[str, str] | None:
    normalized = normalize_text(text)
    match = _OWNER_TOPIC_QUERY_RE.match(normalized)
    if not match:
        return None
    owner = str(match.group("owner") or "").strip()
    topic = str(match.group("topic") or "").strip()
    topic = _VIDEO_TOPIC_TAIL_RE.sub("", topic).strip(" ，。！？?；;：:")
    if len(compact_focus_key(owner)) < 2 or len(compact_focus_key(topic)) < 4:
        return None
    return owner, topic


def extract_topic_candidates(
    text: str,
    *,
    history_text: str = "",
    final_target_matches: list[SemanticMatch] | None = None,
    task_mode_matches: list[SemanticMatch] | None = None,
) -> list[str]:
    return extract_focus_spans(
        text,
        history_text=history_text,
        final_target_matches=final_target_matches,
        task_mode_matches=task_mode_matches,
        limit=10,
    )


def merge_followup_candidates(messages: list[dict], candidates: list[str]) -> list[str]:
    merged = list(candidates)
    if len(merged) >= 2:
        return merged
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        for token in extract_topic_candidates(extract_message_text(message)):
            if token not in merged:
                merged.append(token)
            if len(merged) >= 5:
                return merged
    return merged


def collect_history_candidates(messages: list[dict], limit: int = 5) -> list[str]:
    history_candidates: list[str] = []
    user_messages = [
        message for message in messages or [] if message.get("role") == "user"
    ]
    for message in reversed(user_messages[:-1]):
        for token in extract_topic_candidates(extract_message_text(message)):
            if token not in history_candidates:
                history_candidates.append(token)
            if len(history_candidates) >= limit:
                return history_candidates
    return history_candidates


def _collect_explicit_anchor_tokens(latest_user_text: str) -> list[str]:
    anchors: list[str] = []
    for bvid in extract_bvids({"content": latest_user_text}):
        if bvid not in anchors:
            anchors.append(bvid)
    for mid in extract_owner_mids({"content": latest_user_text}):
        token = f"uid={mid}"
        if token not in anchors:
            anchors.append(token)
    return anchors


def _drop_anchored_question_tail_candidates(
    candidates: list[str],
    *,
    anchor_tokens: list[str],
) -> list[str]:
    if not anchor_tokens:
        return list(candidates)

    filtered: list[str] = []
    for candidate in candidates:
        normalized = normalize_text(candidate).replace(" ", "")
        if normalized and _QUESTION_TAIL_RE.search(normalized):
            continue
        if candidate not in filtered:
            filtered.append(candidate)
    return filtered


def _merge_anchor_tokens(candidates: list[str], latest_user_text: str) -> list[str]:
    anchor_tokens = _collect_explicit_anchor_tokens(latest_user_text)
    merged = _drop_anchored_question_tail_candidates(
        candidates,
        anchor_tokens=anchor_tokens,
    )
    if not anchor_tokens:
        return merged

    anchored: list[str] = []
    for candidate in [*anchor_tokens, *merged]:
        if candidate not in anchored:
            anchored.append(candidate)
    return anchored


def extract_topics(
    messages: list[dict],
    latest_user_text: str,
    *,
    history_text: str = "",
    final_target_matches: list[SemanticMatch] | None = None,
    task_mode_matches: list[SemanticMatch] | None = None,
) -> list[str]:
    structured_topics: list[str] = []
    owner_topic_pair = extract_owner_topic_pair(latest_user_text)
    if owner_topic_pair:
        structured_topics.append(owner_topic_pair[1])

    topics = merge_followup_candidates(
        messages,
        extract_topic_candidates(
            latest_user_text,
            history_text=history_text,
            final_target_matches=final_target_matches,
            task_mode_matches=task_mode_matches,
        ),
    )
    merged_topics: list[str] = []
    for topic in [*structured_topics, *_merge_anchor_tokens(topics, latest_user_text)]:
        if topic not in merged_topics:
            merged_topics.append(topic)
    return merged_topics[:10]


def _looks_entity_like(
    text: str,
    *,
    from_history: bool,
    task_mode: str,
) -> bool:
    compact = compact_focus_key(text)
    if not compact:
        return False
    if from_history:
        return True
    if task_mode not in {"lookup_entity", "known_item", "repeat"}:
        return False
    if any(char.isascii() and char.isalnum() for char in compact):
        return True
    return len(compact) <= 4


def extract_entities(
    messages: list[dict],
    latest_user_text: str,
    *,
    history_text: str = "",
    task_mode: str = "exploration",
    final_target_matches: list[SemanticMatch] | None = None,
    task_mode_matches: list[SemanticMatch] | None = None,
) -> list[str]:
    owner_topic_pair = extract_owner_topic_pair(latest_user_text)
    latest_topics = _merge_anchor_tokens(
        extract_topic_candidates(
            latest_user_text,
            history_text=history_text,
            final_target_matches=final_target_matches,
            task_mode_matches=task_mode_matches,
        ),
        latest_user_text,
    )
    topics = merge_followup_candidates(messages, latest_topics)
    explicit_anchor_keys = {
        compact_focus_key(token)
        for token in _collect_explicit_anchor_tokens(latest_user_text)
        if compact_focus_key(token)
    }
    entities: list[str] = []
    if owner_topic_pair:
        entities.append(owner_topic_pair[0])
    for topic in topics:
        if compact_focus_key(topic) in explicit_anchor_keys:
            if topic not in entities:
                entities.append(topic)
            continue
        from_history = topic not in latest_topics
        if _looks_entity_like(
            topic,
            from_history=from_history,
            task_mode=task_mode,
        ):
            entities.append(topic)
    return entities[:8]


def is_ascii_like(token: str) -> bool:
    token_text = str(token or "")
    return bool(token_text) and all(ord(char) < 128 for char in token_text)


def _has_mixed_script(token: str) -> bool:
    token_text = str(token or "")
    has_ascii_alnum = any(char.isascii() and char.isalnum() for char in token_text)
    has_non_ascii = any(not char.isascii() for char in token_text)
    return has_ascii_alnum and has_non_ascii


def needs_term_normalization(
    explicit_entities: list[str],
    explicit_topics: list[str],
) -> bool:
    candidates = [
        token
        for token in [*(explicit_entities or []), *(explicit_topics or [])]
        if 1 < len(compact_focus_key(token)) <= _ALIAS_TOKEN_MAX_LEN
    ]
    return bool(candidates) and (
        any(_has_mixed_script(token) for token in candidates)
        or (
            any(is_ascii_like(token) for token in candidates)
            and any(not is_ascii_like(token) for token in candidates)
        )
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
    has_explicit_video_anchor: bool,
    has_explicit_owner_id: bool,
    task_mode: str,
    final_target_matches: list[SemanticMatch],
    task_mode_matches: list[SemanticMatch],
) -> list[str]:
    route_reasons: list[str] = []
    if has_explicit_video_anchor:
        route_reasons.append("signal:explicit-video-anchor")
    if has_explicit_owner_id:
        route_reasons.append("signal:explicit-owner-id")
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
    explicit_entities = extract_entities(
        messages,
        window.latest_user_text,
        history_text=window.history_text,
        task_mode=task_mode,
        final_target_matches=final_target_matches,
        task_mode_matches=task_mode_matches,
    )
    explicit_topics = extract_topics(
        messages,
        window.latest_user_text,
        history_text=window.history_text,
        final_target_matches=final_target_matches,
        task_mode_matches=task_mode_matches,
    )
    explicit_video_anchors = extract_bvids({"content": window.latest_user_text})
    explicit_owner_ids = extract_owner_mids({"content": window.latest_user_text})
    if window.is_followup and not explicit_entities:
        for topic in collect_history_candidates(messages):
            if topic not in explicit_topics:
                explicit_topics.append(topic)
            if topic not in explicit_entities:
                explicit_entities.append(topic)
            if len(explicit_entities) >= 8:
                break

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
    stable_owner_topic_video_query = is_stable_owner_topic_video_query(
        final_target=final_target,
        task_mode=task_mode,
        window=window,
        explicit_entities=explicit_entities,
        explicit_topics=explicit_topics,
        explicit_video_anchors=explicit_video_anchors,
        explicit_owner_ids=explicit_owner_ids,
        needs_term_normalization=needs_term_normalization_flag,
    )
    needs_keyword_expansion = bool(
        final_target == "videos"
        and not stable_owner_topic_video_query
        and task_mode == "exploration"
        and not explicit_video_anchors
        and not explicit_owner_ids
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
        and not needs_keyword_expansion
        and (
            explicit_video_anchors
            or (
                explicit_entities
                and not explicit_owner_ids
                and not stable_owner_topic_video_query
                and (
                    window.is_followup
                    or task_mode in {"repeat", "known_item"}
                    or any(len(entity) <= 8 for entity in explicit_entities[:2])
                )
            )
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
            has_explicit_video_anchor=bool(explicit_video_anchors),
            has_explicit_owner_id=bool(explicit_owner_ids),
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
