"""Intent taxonomy and similarity-based label matching.

Route selection in llms must be expressed as labels, descriptions, and examples.
Do not add ad hoc keyword gates or regex routing here. If routing drifts, extend
the taxonomy examples or the classifier scoring logic instead of hard-coding
new branches for individual phrases.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from functools import lru_cache


_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9.+#:/_-]+|[\u4e00-\u9fff]+")
_PUNCT_TRANSLATION = str.maketrans(
    {
        char: " "
        for char in "，。！？；：、,.!?;:()[]{}<>《》“”‘’\"'`~!@#$%^&*-+=|\\/\n\t"
    }
)


@dataclass(frozen=True, slots=True)
class SemanticLabel:
    name: str
    description: str
    examples: tuple[str, ...]
    allowed_targets: tuple[str, ...] = ()
    default_score: float = 0.0


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    name: str
    score: float
    evidence: str


FINAL_TARGET_LABELS: tuple[SemanticLabel, ...] = (
    SemanticLabel(
        name="mixed",
        description="用户同时需要官方站外信息和 B 站视频或解读。",
        examples=(
            "Gemini 2.5 最近有哪些官方更新，B站上有没有相关解读视频？",
            "先查官方公告，再找几条 B 站解读。",
            "官网更新和站内视频一起看看。",
        ),
        default_score=0.02,
    ),
    SemanticLabel(
        name="relations",
        description="用户要查作者、账号之间的关联、矩阵或别名关系。",
        examples=(
            "何同学有哪些关联账号？",
            "他还有别的号吗？",
            "这个作者还有没有别的账号？",
        ),
        default_score=0.02,
    ),
    SemanticLabel(
        name="owners",
        description="用户想发现创作者、UP 主、作者或相似账号。",
        examples=(
            "推荐几个做黑神话悟空内容的UP主。",
            "和影视飓风风格接近的UP主有哪些？",
            "推荐几个专门做 AI 绘图教程的UP主。",
            "“这里是小天啊” 是一个 UP主名字，帮我查一下。",
            "搜索作者 红警08。",
        ),
        default_score=0.02,
    ),
    SemanticLabel(
        name="external",
        description="用户只想看官网、官方文档、release notes 或其他站外权威信息。",
        examples=(
            "Gemini 2.5 最近有哪些官方更新？",
            "先只看官网就行。",
            "官方文档怎么说？",
        ),
        default_score=0.02,
    ),
    SemanticLabel(
        name="videos",
        description="用户最终想拿到具体视频、教程、解读、代表作或时间线内容。",
        examples=(
            "找几条黑神话悟空剧情解析视频，优先高播放。",
            "影视飓风最近有什么新视频？",
            "那他的代表作有哪些？",
            "来点让我开心的视频。",
            "某个软件有什么入门教程？",
            "推荐几个 AI 工具入门教程视频。",
        ),
        default_score=0.05,
    ),
)


TASK_MODE_LABELS: tuple[SemanticLabel, ...] = (
    SemanticLabel(
        name="collect_compare",
        description="用户要比较两个或多个人、作品或来源，并形成结论。",
        examples=(
            "对比一下老番茄和影视飓风最近一个月发布的视频，谁更高产？",
            "Gemini 和 Claude 哪个更适合写代码？",
            "比较一下这两个作者谁更适合入门。",
        ),
        default_score=0.02,
    ),
    SemanticLabel(
        name="repeat",
        description="用户要看近期更新、时间线或最近一段时间的连续产出。",
        examples=(
            "影视飓风最近有什么新视频？",
            "老番茄最近一个月发了什么？",
            "这位作者近期更新如何？",
            "这个视频的作者还发了哪些视频？",
            "红警08是谁，最近发了哪些视频？",
        ),
        allowed_targets=("videos",),
        default_score=0.02,
    ),
    SemanticLabel(
        name="lookup_entity",
        description="用户要确认某个对象是谁、是什么，或查清它的身份与关系。",
        examples=(
            "何同学有哪些关联账号？",
            "Gemini 2.5 是什么？",
            "这个视频是谁做的？",
        ),
        default_score=0.02,
    ),
    SemanticLabel(
        name="known_item",
        description="用户已经指向某个具体视频、对象或条目，要继续追问这个已知项。",
        examples=(
            "BV1abc 这条视频怎么样？",
            "这个视频讲了什么？",
            "那条视频是谁做的？",
        ),
        default_score=0.01,
    ),
    SemanticLabel(
        name="exploration",
        description="用户在探索候选项，希望拿到一批推荐、发现结果或主题样例。",
        examples=(
            "推荐几条黑神话悟空剧情解析视频。",
            "来点让我开心的视频。",
            "推荐几个做这类内容的作者。",
        ),
        default_score=0.05,
    ),
)


FACET_TAXONOMIES: dict[str, tuple[SemanticLabel, ...]] = {
    "motivation": (
        SemanticLabel(
            name="information",
            description="更关注信息增量、教程、文档、原理、更新说明。",
            examples=(
                "Gemini 2.5 最近有哪些官方更新？",
                "给我找点 AI 工具入门教程。",
                "这个 API 怎么用？",
            ),
        ),
        SemanticLabel(
            name="emotion_regulation",
            description="主要想放松、开心、解压、舒缓情绪。",
            examples=(
                "来点让我开心的视频。",
                "想看点轻松下饭的内容。",
                "给我一些解压视频。",
            ),
        ),
        SemanticLabel(
            name="aesthetic",
            description="追求颜值、氛围感、好看、画面风格。",
            examples=(
                "来点很有氛围感的视频。",
                "想看点养眼的内容。",
                "画面漂亮一点。",
            ),
        ),
        SemanticLabel(
            name="social_shareability",
            description="想拿到适合分享、聊天、整活、带梗的话题素材。",
            examples=(
                "来点适合发给朋友看的。",
                "有没有适合聊天的话题视频？",
                "想找点能分享的整活内容。",
            ),
        ),
        SemanticLabel(
            name="utility_task",
            description="为了完成具体任务或工作流，需要上手、实践、配置、排错。",
            examples=(
                "某个工具有什么入门教程？",
                "AI 工具有什么入门教程？",
                "给我找点实战配置教程。",
                "怎么排这个问题？",
            ),
        ),
        SemanticLabel(
            name="companionship",
            description="更像是想找陪伴、背景音、睡前或挂着听的内容。",
            examples=(
                "找点能挂着听的。",
                "睡前适合看什么？",
                "来点背景音内容。",
            ),
        ),
    ),
    "expected_payoff": (
        SemanticLabel(
            name="funny",
            description="预期回报是搞笑、开心、整活。",
            examples=(
                "来点让我开心的视频。",
                "想看点搞笑整活的内容。",
                "有没有沙雕一点的。",
            ),
        ),
        SemanticLabel(
            name="relaxing",
            description="预期回报是放松、轻松、下饭、摸鱼。",
            examples=(
                "想看点轻松下饭的内容。",
                "摸鱼的时候看什么？",
                "来点不用动脑的。",
            ),
        ),
        SemanticLabel(
            name="healing",
            description="预期回报是温柔、安静、治愈。",
            examples=(
                "想看点治愈的视频。",
                "来点安静温柔的内容。",
                "给我一些舒缓一点的。",
            ),
        ),
        SemanticLabel(
            name="stimulating",
            description="预期回报是高能、刺激、燃、炸裂。",
            examples=(
                "来点高能一点的。",
                "有没有燃一点的视频？",
                "想看刺激一些的内容。",
            ),
        ),
        SemanticLabel(
            name="eye_candy",
            description="预期回报是养眼、颜值、好看。",
            examples=(
                "来点养眼的视频。",
                "想看点颜值高的内容。",
                "有没有画面好看的。",
            ),
        ),
        SemanticLabel(
            name="clear_explanation",
            description="预期回报是讲得清楚、结构化、适合入门。",
            examples=(
                "给我讲得清楚一点的教程。",
                "有没有适合入门的解析？",
                "想看解释清晰的内容。",
            ),
        ),
        SemanticLabel(
            name="decision_help",
            description="预期回报是帮助比较、选择、做判断。",
            examples=(
                "帮我比较一下哪个好。",
                "这个值不值得买？",
                "推荐买哪个更合适？",
            ),
        ),
        SemanticLabel(
            name="topic_for_chat",
            description="预期回报是拿到可聊、可传播的新鲜话题。",
            examples=(
                "来点适合聊天的话题。",
                "有没有最近可以和朋友聊的内容？",
                "想找点能分享的新鲜事。",
            ),
        ),
        SemanticLabel(
            name="cozy_companionship",
            description="预期回报是陪伴感或舒服的背景播放体验。",
            examples=(
                "找点能陪我挂着听的。",
                "睡前放着听什么好？",
                "来点背景陪伴感强的内容。",
            ),
        ),
    ),
    "consumption_mode": (
        SemanticLabel(
            name="quick_scroll",
            description="更适合快速刷、随便看、短平快消费。",
            examples=(
                "来点随便刷刷的。",
                "短一点，摸鱼看看。",
                "下饭刷几条。",
            ),
        ),
        SemanticLabel(
            name="deep_watch",
            description="更适合系统性、完整、长时间观看。",
            examples=(
                "想看一套完整流程。",
                "给我系统一点的长视频。",
                "来点深度讲解。",
            ),
        ),
        SemanticLabel(
            name="background_play",
            description="更适合挂着听、背景播放。",
            examples=(
                "找点能挂着听的。",
                "背景放着就行。",
                "睡前开着听。",
            ),
        ),
        SemanticLabel(
            name="share_first",
            description="主要用于分享、转发、发给朋友。",
            examples=(
                "来点适合发给朋友看的。",
                "我想转给同事。",
                "有没有适合分享的。",
            ),
        ),
        SemanticLabel(
            name="learn_first",
            description="主要为了学习、上手、掌握。",
            examples=(
                "我想学这个怎么做。",
                "找点入门教程。",
                "来点实战工作流。",
            ),
        ),
    ),
    "constraints": (
        SemanticLabel(
            name="recent_only",
            description="需要最近、近期、最新的结果。",
            examples=(
                "最近一个月发布的视频。",
                "最近有哪些官方更新？",
                "这几天有没有新视频？",
                "还发了哪些视频？",
            ),
        ),
        SemanticLabel(
            name="popular_only",
            description="偏向热门、高播放、高赞或代表作。",
            examples=(
                "优先高播放。",
                "给我代表作。",
                "来点热门的。",
            ),
        ),
        SemanticLabel(
            name="duration_short",
            description="偏向短一点、几分钟、速看。",
            examples=(
                "短一点。",
                "几分钟能看完的。",
                "来点速看。",
            ),
        ),
        SemanticLabel(
            name="duration_long",
            description="偏向长一点、完整、全流程。",
            examples=(
                "长一点的完整讲解。",
                "全流程视频。",
                "来点系统性的。",
            ),
        ),
        SemanticLabel(
            name="safe_only",
            description="需要安全、正经、不要过于刺激。",
            examples=(
                "别太刺激。",
                "安全一点。",
                "正经一点。",
            ),
        ),
        SemanticLabel(
            name="authoritative_only",
            description="需要官方、权威、靠谱来源。",
            examples=(
                "先只看官网就行。",
                "只看官方更新。",
                "文档里怎么写的？",
            ),
        ),
        SemanticLabel(
            name="low_cognitive_load",
            description="需要轻松、不费脑、随便刷。",
            examples=(
                "来点不用动脑的。",
                "轻松一点。",
                "摸鱼看看就行。",
            ),
        ),
    ),
    "visual_intent_hints": (
        SemanticLabel(
            name="cute_animal",
            description="视觉上偏萌宠、动物、可爱。",
            examples=(
                "来点可爱猫猫狗狗。",
                "有没有萌宠视频？",
                "想看小动物。",
            ),
        ),
        SemanticLabel(
            name="funny_expression",
            description="视觉上偏夸张表情、抽象整活。",
            examples=(
                "来点表情很有戏的。",
                "想看抽象整活。",
                "搞笑反应类。",
            ),
        ),
        SemanticLabel(
            name="pretty_face",
            description="视觉上偏颜值、人物出镜。",
            examples=(
                "来点颜值高的内容。",
                "想看养眼一点的。",
                "人物出镜好看的。",
            ),
        ),
        SemanticLabel(
            name="professional_setup",
            description="视觉上偏设备、工作流、专业搭建场景。",
            examples=(
                "想看设备搭建流程。",
                "专业工作流展示。",
                "偏测评搭建类。",
            ),
        ),
    ),
}


def normalize_text(text: str) -> str:
    normalized = str(text or "").translate(_PUNCT_TRANSLATION).lower()
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized


def iter_content_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in _TOKEN_RE.findall(normalize_text(text)) if token)


@lru_cache(maxsize=2048)
def _fingerprint(text: str) -> tuple[frozenset[str], frozenset[str]]:
    normalized = normalize_text(text)
    tokens = frozenset(iter_content_tokens(normalized))
    compact = normalized.replace(" ", "")
    ngrams: set[str] = set()
    for size in (2, 3):
        if len(compact) < size:
            continue
        ngrams.update(
            compact[index : index + size] for index in range(len(compact) - size + 1)
        )
    return tokens, frozenset(ngrams)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _score_text_similarity(left: str, right: str) -> float:
    left_tokens, left_ngrams = _fingerprint(left)
    right_tokens, right_ngrams = _fingerprint(right)
    token_score = _jaccard(left_tokens, right_tokens)
    ngram_score = _jaccard(left_ngrams, right_ngrams)
    if token_score == 0.0 and ngram_score == 0.0:
        return 0.0
    return token_score * 0.35 + ngram_score * 0.65


def _score_label(
    text: str,
    history_text: str,
    label: SemanticLabel,
) -> SemanticMatch:
    primary_text = normalize_text(text)
    contextual_text = normalize_text(history_text)
    best_score = _score_text_similarity(primary_text, label.description)
    best_evidence = label.description
    for example in label.examples:
        example_score = _score_text_similarity(primary_text, example)
        if example_score > best_score:
            best_score = example_score
            best_evidence = example

    if contextual_text:
        context_score = _score_text_similarity(contextual_text, label.description)
        context_evidence = label.description
        for example in label.examples:
            example_score = _score_text_similarity(contextual_text, example)
            if example_score > context_score:
                context_score = example_score
                context_evidence = example
        if context_score > 0:
            merged_score = best_score * 0.8 + context_score * 0.2
            if context_score > best_score:
                best_evidence = context_evidence
            best_score = merged_score

    return SemanticMatch(
        name=label.name,
        score=min(1.0, best_score + label.default_score),
        evidence=best_evidence,
    )


def rank_labels(
    text: str,
    labels: tuple[SemanticLabel, ...],
    *,
    history_text: str = "",
    final_target: str | None = None,
) -> list[SemanticMatch]:
    matches = []
    for label in labels:
        if label.allowed_targets and final_target not in label.allowed_targets:
            continue
        matches.append(_score_label(text, history_text, label))
    return sorted(matches, key=lambda item: item.score, reverse=True)


def rank_final_target_matches(
    text: str, *, history_text: str = ""
) -> list[SemanticMatch]:
    return rank_labels(text, FINAL_TARGET_LABELS, history_text=history_text)


def rank_task_mode_matches(
    text: str,
    final_target: str,
    *,
    history_text: str = "",
) -> list[SemanticMatch]:
    return rank_labels(
        text,
        TASK_MODE_LABELS,
        history_text=history_text,
        final_target=final_target,
    )


def rank_facet_matches(
    text: str,
    facet_name: str,
    *,
    history_text: str = "",
) -> list[SemanticMatch]:
    return rank_labels(
        text, FACET_TAXONOMIES.get(facet_name, ()), history_text=history_text
    )


def detect_final_target(text: str, history_text: str = "") -> str:
    matches = rank_final_target_matches(text, history_text=history_text)
    if not matches:
        return "videos"
    score_map = {match.name: match.score for match in matches}
    if (
        score_map.get("mixed", 0.0) >= 0.18
        and score_map.get("external", 0.0) >= 0.16
        and score_map.get("videos", 0.0) >= 0.16
    ):
        return "mixed"
    if matches[0].score < 0.1:
        return "videos"
    return matches[0].name


def detect_task_mode(text: str, final_target: str, history_text: str = "") -> str:
    matches = rank_task_mode_matches(
        text,
        final_target,
        history_text=history_text,
    )
    if not matches or matches[0].score < 0.1:
        return (
            "lookup_entity"
            if final_target in {"owners", "relations"}
            else "exploration"
        )
    return matches[0].name
