from __future__ import annotations

import re

from llms.protocol import FacetScore, IntentProfile


_STOPWORDS = {
    "我",
    "想",
    "想看",
    "想找",
    "帮我",
    "给我",
    "一下",
    "一下子",
    "有没有",
    "有无",
    "有什么",
    "有哪些",
    "推荐",
    "几个",
    "一些",
    "这个",
    "那个",
    "内容",
    "视频",
    "b站",
    "B站",
}

_MOTIVATION_PATTERNS = {
    "information": re.compile(
        r"教程|攻略|原理|解读|评测|对比|解析|是什么|怎么|文档|API|更新"
    ),
    "emotion_regulation": re.compile(r"开心|放松|治愈|解压|下饭|轻松|助眠"),
    "aesthetic": re.compile(r"养眼|氛围感|颜值|性感|好看|美感|漂亮|小姐姐"),
    "social_shareability": re.compile(r"适合发给朋友|梗|整活|话题|能聊|适合分享"),
    "utility_task": re.compile(r"入门|上手|工作流|实践|实战|配置|安装|排错"),
    "companionship": re.compile(r"陪伴|背景音|挂着听|睡前|陪我"),
}

_PAYOFF_PATTERNS = {
    "funny": re.compile(r"搞笑|开心|笑|整活|沙雕"),
    "relaxing": re.compile(r"放松|轻松|下饭|摸鱼|舒服"),
    "healing": re.compile(r"治愈|温柔|安静|舒缓|陪伴"),
    "stimulating": re.compile(r"刺激|高能|燃|爽|炸裂"),
    "eye_candy": re.compile(r"养眼|性感|颜值|美女|小姐姐|帅哥|氛围感"),
    "clear_explanation": re.compile(r"讲得清楚|讲清楚|清晰|入门|教程|解析|原理"),
    "decision_help": re.compile(r"选哪个|推荐买|值不值|对比|怎么选"),
    "topic_for_chat": re.compile(r"适合聊|话题|新鲜事|梗"),
    "cozy_companionship": re.compile(r"陪伴|挂着听|睡前|背景"),
}

_CONSUMPTION_PATTERNS = {
    "quick_scroll": re.compile(r"来点|刷|随便看|短点|摸鱼|下饭"),
    "deep_watch": re.compile(r"系统|深度|完整|全流程|长一点|详细"),
    "background_play": re.compile(r"挂着|背景|听着|睡前|陪伴"),
    "share_first": re.compile(r"发给朋友|转发|分享"),
    "learn_first": re.compile(r"教程|入门|原理|实战|工作流"),
}

_CONSTRAINT_PATTERNS = {
    "recent_only": re.compile(r"最近|最新|近期|这几天|刚发|更新"),
    "popular_only": re.compile(r"热门|高播放|高赞|爆款|代表作"),
    "duration_short": re.compile(r"短一点|短视频|几分钟|速看"),
    "duration_long": re.compile(r"长一点|长视频|完整|全流程|系统"),
    "safe_only": re.compile(r"安全|正经|别太擦边|别太刺激|别吓人"),
    "authoritative_only": re.compile(r"官方|权威|靠谱|准确信息|文档"),
    "low_cognitive_load": re.compile(r"轻松|不用动脑|随便刷"),
}

_VISUAL_HINT_PATTERNS = {
    "cute_animal": re.compile(r"萌宠|猫|狗|可爱"),
    "funny_expression": re.compile(r"搞笑|整活|表情|抽象"),
    "pretty_face": re.compile(r"颜值|美女|小姐姐|帅哥|养眼"),
    "professional_setup": re.compile(r"测评|工作流|搭建|专业|设备"),
}

_RELATION_RE = re.compile(r"关联账号|矩阵号|主号|副号|别的号|小号|还有别的号|关联作者")
_OWNER_RE = re.compile(
    r"UP主|up主|作者|博主|创作者|谁在做|有哪些作者|推荐几个作者|推荐几个UP主"
)
_EXTERNAL_RE = re.compile(
    r"官网|官方|release|更新日志|release notes|公告|文档|API", re.IGNORECASE
)
_VIDEO_RE = re.compile(r"视频|教程|攻略|解说|剧情解析|代表作|最近.*视频|找几条|热门")
_MIXED_RE = re.compile(
    r"(官方更新|官网|release|公告).*(B站|视频|解读)|(B站|视频|解读).*(官方更新|官网|release|公告)",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(r"对比|比较|谁更|哪个好|哪家")
_ABSTRACT_RE = re.compile(r"来点|这种|这类|vibe|氛围|感觉|口语|黑话|抽象|偏.*的")
_OWNER_ALIAS_RE = re.compile(r"^[A-Za-z0-9\u4e00-\u9fff]{2,8}$")
_ENTITY_RE = re.compile(
    r"(?:[A-Z][A-Za-z0-9.+-]{1,}|[\u4e00-\u9fff]{2,8}|BV[0-9A-Za-z]+)"
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
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    normalized = re.sub(r"^[\s，。！？?：:]+", "", normalized)
    normalized = re.sub(r"[，。！？?：:]+$", "", normalized)
    return normalized


def _latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return _normalize_query(message.get("content") or "")
    return ""


def _collect_labels(
    text: str, patterns: dict[str, re.Pattern], weight: float = 1.0
) -> list[FacetScore]:
    labels = []
    for label, pattern in patterns.items():
        if pattern.search(text):
            labels.append(FacetScore(label=label, score=weight))
    return labels


def _extract_entities(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _ENTITY_RE.findall(text or ""):
        token = match.strip()
        if len(token) < 2:
            continue
        if token in _STOPWORDS or token.lower() in {
            item.lower() for item in _STOPWORDS
        }:
            continue
        if token not in candidates:
            candidates.append(token)
    return candidates[:8]


def _extract_topics(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9.+-]+|[\u4e00-\u9fff]{2,8}", text or "")
    topics: list[str] = []
    for token in tokens:
        if token in _STOPWORDS or token.lower() in {
            item.lower() for item in _STOPWORDS
        }:
            continue
        if token.startswith(":") or token.startswith("q="):
            continue
        if token not in topics:
            topics.append(token)
    return topics[:10]


def _detect_final_target(text: str) -> str:
    if _MIXED_RE.search(text):
        return "mixed"
    if _RELATION_RE.search(text):
        return "relations"
    if (
        _EXTERNAL_RE.search(text)
        and not _VIDEO_RE.search(text)
        and not _OWNER_RE.search(text)
    ):
        return "external"
    if _OWNER_RE.search(text) and not _VIDEO_RE.search(text):
        return "owners"
    return "videos"


def _detect_task_mode(text: str, final_target: str) -> str:
    if _COMPARE_RE.search(text):
        return "collect_compare"
    if re.search(r"最近|最新|最近发了什么|有哪些新视频", text):
        return "repeat" if final_target == "videos" else "lookup_entity"
    if re.search(r"是谁|是什么|哪个|哪位", text):
        return "lookup_entity"
    if re.search(r"推荐|来点|找几条|随便看看|发现", text):
        return "exploration"
    if re.search(r"BV[0-9A-Za-z]+|这一条|这个视频|那条视频", text):
        return "known_item"
    return "lookup_entity" if final_target in {"owners", "relations"} else "exploration"


def _estimate_ambiguity(
    text: str, entities: list[str], expected_payoff: list[FacetScore]
) -> float:
    score = 0.15
    if len(text) <= 8:
        score += 0.25
    if not entities:
        score += 0.20
    if _ABSTRACT_RE.search(text):
        score += 0.25
    if expected_payoff and not entities:
        score += 0.10
    if re.search(r"什么|哪些|来点|这种|这类", text):
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
    final_target = _detect_final_target(normalized)
    task_mode = _detect_task_mode(normalized, final_target)
    motivation = _collect_labels(normalized, _MOTIVATION_PATTERNS, weight=0.85)
    expected_payoff = _collect_labels(normalized, _PAYOFF_PATTERNS, weight=0.90)
    consumption_mode = _collect_labels(normalized, _CONSUMPTION_PATTERNS, weight=0.80)
    constraints = _collect_labels(normalized, _CONSTRAINT_PATTERNS, weight=0.85)
    visual_intent_hints = _collect_labels(
        normalized, _VISUAL_HINT_PATTERNS, weight=0.70
    )
    explicit_entities = _extract_entities(normalized)
    explicit_topics = _extract_topics(normalized)
    ambiguity = _estimate_ambiguity(normalized, explicit_entities, expected_payoff)
    is_followup = (
        sum(1 for message in messages or [] if message.get("role") == "user") > 1
    )

    needs_keyword_expansion = bool(
        final_target == "videos"
        and ambiguity >= 0.45
        and (_ABSTRACT_RE.search(normalized) or not explicit_entities)
    )
    needs_owner_resolution = bool(
        final_target == "videos"
        and explicit_entities
        and any(_OWNER_ALIAS_RE.match(entity or "") for entity in explicit_entities[:2])
        and re.search(r"最近|视频|代表作|作者|up主|UP主", normalized)
    )
    needs_external_search = bool(
        final_target in {"external", "mixed"}
        or (
            _EXTERNAL_RE.search(normalized)
            and re.search(r"官网|官方|文档|API", normalized)
        )
    )

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
        route_reasons.append("abstract-video-query")
    if needs_owner_resolution:
        route_reasons.append("owner-alias-resolution")
    if needs_external_search:
        route_reasons.append("external-facts")
    if task_mode == "collect_compare":
        route_reasons.append("compare-multiple-candidates")

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
        asset_ids.extend(["route.videos.brief", "dsl.quickref.brief"])
    elif intent.final_target == "owners":
        asset_ids.append("route.owners.brief")
    elif intent.final_target == "relations":
        asset_ids.append("route.relations.brief")
    elif intent.final_target == "external":
        asset_ids.append("route.external.brief")
    elif intent.final_target == "mixed":
        asset_ids.extend(["route.mixed.brief", "dsl.quickref.brief"])

    if intent.needs_keyword_expansion:
        asset_ids.append("semantic.expansion.brief")
    if intent.doc_signal_hints:
        asset_ids.append("facet.mapping.brief")
    return asset_ids
