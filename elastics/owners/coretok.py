"""Runtime coretok query encoder for owner search."""

import json
import math
import re

from pathlib import Path


LATIN_CHAR_RE = re.compile(r"[a-z0-9]")
LATIN_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-\.]*")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")
CORETEXT_SPLIT_RE = re.compile(r"[，,。.!！?？、;；:：()（）\[\]【】<>《》\-\|/\\\s]+")
LOW_INFO_TERMS = {
    "日常",
    "生活",
    "视频",
    "分享",
    "记录",
    "合集",
    "更新",
    "原创",
    "官方",
    "频道",
}


def normalize_core_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def count_mixed_units(text: str) -> int:
    normalized = normalize_core_text(text)
    if not normalized:
        return 0
    cjk_chars = len(CJK_CHAR_RE.findall(normalized))
    latin_chars = len(LATIN_CHAR_RE.findall(normalized))
    return cjk_chars + (math.ceil(latin_chars / 3) if latin_chars else 0)


def is_low_info_text(text: str) -> bool:
    normalized = normalize_core_text(text)
    if not normalized:
        return True
    if normalized in LOW_INFO_TERMS:
        return True
    unique_chars = {char for char in normalized if not char.isspace()}
    return len(unique_chars) <= 1


def is_valid_stage1_tag(tag: str) -> bool:
    normalized = normalize_core_text(tag)
    if not normalized or is_low_info_text(normalized):
        return False
    if count_mixed_units(normalized) > 8:
        return False
    if len(CJK_CHAR_RE.findall(normalized)) > 8:
        return False
    if len(LATIN_CHAR_RE.findall(normalized)) > 24:
        return False
    return True


def suggest_token_budget(text: str) -> int:
    units = count_mixed_units(text)
    if units <= 0:
        return 0
    if units <= 4:
        return 1
    if units <= 6:
        return 2
    return 3


def _balanced_cjk_parts(span: str) -> list[str]:
    compact = span.strip()
    if len(compact) <= 4:
        return [compact]
    if len(compact) <= 6:
        pivot = len(compact) // 2
        return [compact[:pivot], compact[pivot:]]
    part_size = math.ceil(len(compact) / 3)
    return [
        compact[index : index + part_size]
        for index in range(0, len(compact), part_size)
        if compact[index : index + part_size]
    ]


def extract_core_candidates(text: str, *, for_stage1: bool) -> list[str]:
    normalized = normalize_core_text(text)
    if not normalized:
        return []

    candidates = []
    seen = set()

    def add(value: str):
        candidate = normalize_core_text(value)
        if not candidate or candidate in seen or is_low_info_text(candidate):
            return
        seen.add(candidate)
        candidates.append(candidate)

    add(normalized)
    for token in LATIN_TOKEN_RE.findall(normalized):
        if len(token) >= 2:
            add(token)

    cjk_source = [normalized] if for_stage1 else CORETEXT_SPLIT_RE.split(normalized)
    for chunk in cjk_source:
        for span in CJK_SPAN_RE.findall(chunk):
            if len(span) < 2:
                continue
            if len(span) <= 8:
                add(span)
            for part in _balanced_cjk_parts(span):
                if 2 <= len(part) <= 8:
                    add(part)
    return candidates


def _char_ngrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", normalize_core_text(text))
    if len(compact) <= 2:
        return {compact} if compact else set()
    grams = {compact[index : index + 2] for index in range(len(compact) - 1)}
    grams.add(compact)
    return grams


def _surface_overlap_score(left: str, right: str) -> float:
    if left == right:
        return 1.0
    compact_left = re.sub(r"\s+", "", normalize_core_text(left))
    compact_right = re.sub(r"\s+", "", normalize_core_text(right))
    containment = 0.0
    if (
        compact_left
        and compact_right
        and (compact_left in compact_right or compact_right in compact_left)
    ):
        containment = min(len(compact_left), len(compact_right)) / max(
            len(compact_left), len(compact_right), 1
        )
    left_grams = _char_ngrams(left)
    right_grams = _char_ngrams(right)
    if not left_grams or not right_grams:
        return containment
    inter = len(left_grams & right_grams)
    union = len(left_grams | right_grams)
    jaccard = inter / union if union else 0.0
    return max(jaccard, containment)


class _RuntimeLexicon:
    def __init__(self, payload: dict | None = None):
        payload = payload or {}
        self.token_to_id = {
            str(token): int(token_id)
            for token, token_id in (payload.get("token_to_id") or {}).items()
        }
        self.id_to_token = {
            int(token_id): str(token)
            for token_id, token in (payload.get("id_to_token") or {}).items()
        }

    def get_token_id(self, token: str) -> int | None:
        return self.token_to_id.get(normalize_core_text(token))

    def find_best_match(self, token: str) -> tuple[int | None, float]:
        normalized = normalize_core_text(token)
        exact_id = self.get_token_id(normalized)
        if exact_id is not None:
            return exact_id, 1.0
        best_token_id = None
        best_score = 0.0
        for token_id, existing in self.id_to_token.items():
            score = _surface_overlap_score(normalized, existing)
            if score > best_score:
                best_token_id = token_id
                best_score = score
        return best_token_id, best_score


class OwnerQueryCoreTokEncoder:
    def __init__(self, bundle_path: str | Path):
        self.bundle_path = Path(bundle_path)
        payload = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        self.bundle_version = payload.get("bundle_version", "coretok-v1")
        self.lexicon = _RuntimeLexicon(payload.get("lexicon"))
        tag_cfg = payload.get("tag_tokenizer") or {}
        text_cfg = payload.get("text_tokenizer") or {}
        self.tag_reuse_threshold = float(tag_cfg.get("reuse_threshold", 0.5))
        self.text_reuse_threshold = float(text_cfg.get("reuse_threshold", 0.6))

    def _encode_candidates(
        self,
        candidates: list[str],
        *,
        budget: int,
        reuse_threshold: float,
    ) -> list[int]:
        token_ids = []
        for candidate in candidates[: max(budget * 2, 1)]:
            exact_id = self.lexicon.get_token_id(candidate)
            if exact_id is not None:
                if exact_id not in token_ids:
                    token_ids.append(exact_id)
            else:
                best_token_id, best_score = self.lexicon.find_best_match(candidate)
                if best_token_id is not None and best_score >= reuse_threshold:
                    if best_token_id not in token_ids:
                        token_ids.append(best_token_id)
            if len(token_ids) >= budget:
                break
        return token_ids

    def encode_query(self, query: str) -> dict:
        tag_token_ids = []
        if is_valid_stage1_tag(query):
            tag_token_ids = self._encode_candidates(
                extract_core_candidates(query, for_stage1=True),
                budget=suggest_token_budget(query),
                reuse_threshold=self.tag_reuse_threshold,
            )

        text_budget = 0
        units = count_mixed_units(query)
        if units > 0:
            text_budget = max(1, min(6, math.ceil(units / 3)))
        text_token_ids = self._encode_candidates(
            extract_core_candidates(query, for_stage1=False),
            budget=text_budget,
            reuse_threshold=self.text_reuse_threshold,
        )

        return {
            "core_tokenizer_version": self.bundle_version,
            "core_tag_token_ids": tag_token_ids,
            "core_text_token_ids": text_token_ids,
        }
