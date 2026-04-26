from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import re

from llms.intent.semantic_assets import get_term_alias_rules
from llms.intent.taxonomy import FACET_TAXONOMIES
from llms.intent.taxonomy import FINAL_TARGET_LABELS
from llms.intent.taxonomy import TASK_MODE_LABELS
from llms.intent.taxonomy import SemanticLabel
from llms.intent.taxonomy import SemanticMatch
from llms.intent.taxonomy import detect_final_target
from llms.intent.taxonomy import rank_facet_matches
from llms.intent.taxonomy import rank_final_target_matches
from llms.intent.taxonomy import rank_task_mode_matches


_EDGE_PUNCTUATION = " ，。！？?；;：:、()[]{}<>《》\"'`~!@#$%^&*-+=|\\/"
_QUOTED_FOCUS_RE = re.compile(r"[\"“《](?P<text>[^\"”》]{2,64})[\"”》]")
_REFERENCE_FACETS: tuple[str, ...] = (
    "expected_payoff",
    "constraints",
    "motivation",
    "consumption_mode",
)
_FINAL_TARGET_LABELS_BY_NAME = {label.name: label for label in FINAL_TARGET_LABELS}
_TASK_MODE_LABELS_BY_NAME = {label.name: label for label in TASK_MODE_LABELS}
_FACET_LABELS_BY_NAME = {
    facet_name: {label.name: label for label in labels}
    for facet_name, labels in FACET_TAXONOMIES.items()
}


@dataclass(frozen=True, slots=True)
class TextUnit:
    text: str
    normalized: str
    start: int
    end: int
    kind: str


@lru_cache(maxsize=1)
def _term_alias_entries() -> tuple[tuple[tuple[str, ...], str], ...]:
    entries: list[tuple[tuple[str, ...], str]] = []
    for alias_key, replacement in get_term_alias_rules():
        alias_units = _scan_text_units(alias_key)
        if not alias_units:
            continue
        entries.append((tuple(unit.normalized for unit in alias_units), replacement))
    return tuple(entries)


def _is_cjk(char: str) -> bool:
    return bool(char) and 0x4E00 <= ord(char) <= 0x9FFF


def _is_word_char(char: str) -> bool:
    return bool(char) and (char.isalnum() or char in ".+#/_-")


def _scan_text_units(text: str) -> tuple[TextUnit, ...]:
    source = str(text or "")
    units: list[TextUnit] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char.isspace():
            index += 1
            continue
        if _is_cjk(char):
            units.append(
                TextUnit(
                    text=char,
                    normalized=char,
                    start=index,
                    end=index + 1,
                    kind="cjk",
                )
            )
            index += 1
            continue
        if _is_word_char(char):
            start = index
            while index < len(source) and _is_word_char(source[index]):
                index += 1
            token = source[start:index]
            units.append(
                TextUnit(
                    text=token,
                    normalized=token.lower(),
                    start=start,
                    end=index,
                    kind="word",
                )
            )
            continue
        index += 1
    return tuple(units)


def compact_focus_key(text: str) -> str:
    return "".join(unit.normalized for unit in _scan_text_units(text))


def _collect_reference_texts(
    text: str,
    *,
    history_text: str,
    final_target_matches: list[SemanticMatch],
    task_mode_matches: list[SemanticMatch],
) -> list[str]:
    source_key = compact_focus_key(text)
    references: list[str] = []
    seen_keys: set[str] = set()

    def add_label(label: SemanticLabel | None) -> None:
        if label is None:
            return
        for candidate in (label.description, *label.examples):
            candidate_key = compact_focus_key(candidate)
            if (
                not candidate_key
                or candidate_key == source_key
                or candidate_key in seen_keys
            ):
                continue
            seen_keys.add(candidate_key)
            references.append(candidate)

    for match in final_target_matches[:2]:
        add_label(_FINAL_TARGET_LABELS_BY_NAME.get(match.name))
    for match in task_mode_matches[:2]:
        add_label(_TASK_MODE_LABELS_BY_NAME.get(match.name))
    for facet_name in _REFERENCE_FACETS:
        facet_matches = rank_facet_matches(text, facet_name, history_text=history_text)
        if not facet_matches:
            continue
        add_label(_FACET_LABELS_BY_NAME.get(facet_name, {}).get(facet_matches[0].name))
    return references


def _collect_unit_coverages(
    units: tuple[TextUnit, ...],
    references: list[str],
) -> list[float]:
    if not units or not references:
        return [0.0 for _ in units]
    reference_unit_sets = [
        frozenset(unit.normalized for unit in _scan_text_units(reference))
        for reference in references
    ]
    coverages: list[float] = []
    total = float(len(reference_unit_sets))
    for unit in units:
        covered = sum(
            unit.normalized in reference_set for reference_set in reference_unit_sets
        )
        coverages.append(covered / total if total else 0.0)
    return coverages


def _is_informative_unit(unit: TextUnit, coverage: float) -> bool:
    if not unit.normalized:
        return False
    if unit.kind == "cjk":
        return coverage <= 0.34
    if unit.normalized.isdigit():
        return False
    if len(unit.normalized) == 1 and unit.normalized.isalpha():
        return False
    return coverage <= 0.4


def _emit_focus_span(source: str, units: list[TextUnit]) -> str:
    if not units:
        return ""
    span = source[units[0].start : units[-1].end].strip()
    span = span.strip(_EDGE_PUNCTUATION)
    if len(compact_focus_key(span)) < 2:
        return ""
    return " ".join(span.split())


def _extract_template_focus_spans(
    source: str,
    units: tuple[TextUnit, ...],
    references: list[str],
    coverages: list[float],
    limit: int,
) -> list[str]:
    if not units or not references:
        return []

    source_tokens = [unit.normalized for unit in units]
    reference_keys = [compact_focus_key(reference) for reference in references]
    candidate_scores: dict[str, tuple[float, str]] = {}
    for reference in references:
        reference_units = _scan_text_units(reference)
        reference_tokens = [unit.normalized for unit in reference_units]
        if not reference_tokens:
            continue

        matcher = SequenceMatcher(a=source_tokens, b=reference_tokens, autojunk=False)
        blocks = [block for block in matcher.get_matching_blocks() if block.size]
        matched_tokens = sum(block.size for block in blocks)
        if matched_tokens < 2:
            continue

        coverage_score = matched_tokens / max(len(source_tokens), 1)
        cursor = 0
        for block in blocks:
            if block.a > cursor:
                candidate_units = list(units[cursor : block.a])
                candidate = _emit_focus_span(source, candidate_units)
                candidate_key = compact_focus_key(candidate)
                if candidate_key:
                    average_coverage = sum(coverages[cursor : block.a]) / max(
                        len(candidate_units),
                        1,
                    )
                    reference_hits = sum(
                        candidate_key in reference_key
                        for reference_key in reference_keys
                    )
                    if reference_hits and len(candidate_key) <= 3:
                        cursor = block.a + block.size
                        continue
                    if (
                        cursor == 0
                        and len(candidate_key) <= 6
                        and average_coverage >= 0.15
                    ):
                        cursor = block.a + block.size
                        continue
                    reference_penalty = reference_hits / max(len(reference_keys), 1)
                    edge_penalty = 0.03 if cursor == 0 or block.a == len(units) else 0.0
                    score = (
                        coverage_score
                        + len(candidate_key) * 0.01
                        - average_coverage
                        - reference_penalty
                        - edge_penalty
                    )
                    previous = candidate_scores.get(candidate_key)
                    if previous is None or score > previous[0]:
                        candidate_scores[candidate_key] = (score, candidate)
            cursor = block.a + block.size
        if cursor < len(units):
            candidate_units = list(units[cursor:])
            candidate = _emit_focus_span(source, candidate_units)
            candidate_key = compact_focus_key(candidate)
            if candidate_key:
                average_coverage = sum(coverages[cursor:]) / max(
                    len(candidate_units), 1
                )
                reference_hits = sum(
                    candidate_key in reference_key for reference_key in reference_keys
                )
                if reference_hits and len(candidate_key) <= 3:
                    continue
                if cursor == 0 and len(candidate_key) <= 6 and average_coverage >= 0.15:
                    continue
                reference_penalty = reference_hits / max(len(reference_keys), 1)
                edge_penalty = 0.03 if cursor == 0 else 0.0
                score = (
                    coverage_score
                    + len(candidate_key) * 0.01
                    - average_coverage
                    - reference_penalty
                    - edge_penalty
                )
                previous = candidate_scores.get(candidate_key)
                if previous is None or score > previous[0]:
                    candidate_scores[candidate_key] = (score, candidate)

    ranked = sorted(
        candidate_scores.values(),
        key=lambda item: (-item[0], len(compact_focus_key(item[1])), item[1]),
    )
    return _prune_nested_spans([candidate for _, candidate in ranked], limit)


def _prune_nested_spans(spans: list[str], limit: int) -> list[str]:
    selected: list[str] = []
    selected_keys: list[str] = []
    for span in spans:
        span_key = compact_focus_key(span)
        if not span_key:
            continue
        if any(
            existing_key in span_key or span_key in existing_key
            for existing_key in selected_keys
        ):
            continue
        selected.append(span)
        selected_keys.append(span_key)
        if len(selected) >= limit:
            break
    return selected


def _fallback_focus_spans(
    source: str, units: tuple[TextUnit, ...], limit: int
) -> list[str]:
    fallback: list[str] = []
    seen: set[str] = set()
    for unit in units:
        candidate = unit.text.strip(_EDGE_PUNCTUATION)
        candidate_key = compact_focus_key(candidate)
        if len(candidate_key) < 2 or candidate_key in seen:
            continue
        if unit.kind == "cjk":
            continue
        seen.add(candidate_key)
        fallback.append(candidate)
        if len(fallback) >= limit:
            return fallback
    trimmed = " ".join(str(source or "").strip(_EDGE_PUNCTUATION).split())
    if trimmed and compact_focus_key(trimmed) not in seen:
        fallback.append(trimmed)
    return fallback[:limit]


def _extract_quoted_focus_spans(source: str, limit: int) -> list[str]:
    spans: list[str] = []
    seen_keys: set[str] = set()
    for match in _QUOTED_FOCUS_RE.finditer(source):
        candidate = " ".join(str(match.group("text") or "").split()).strip(
            _EDGE_PUNCTUATION
        )
        candidate_key = compact_focus_key(candidate)
        if len(candidate_key) < 2 or candidate_key in seen_keys:
            continue
        seen_keys.add(candidate_key)
        spans.append(candidate)
        if len(spans) >= limit:
            break
    return spans


def extract_focus_spans(
    text: str,
    *,
    history_text: str = "",
    final_target_matches: list[SemanticMatch] | None = None,
    task_mode_matches: list[SemanticMatch] | None = None,
    limit: int = 5,
) -> list[str]:
    source = str(text or "").strip()
    if not source:
        return []

    quoted_spans = _extract_quoted_focus_spans(source, limit)
    if quoted_spans:
        return quoted_spans

    resolved_final_target_matches = final_target_matches or rank_final_target_matches(
        source,
        history_text=history_text,
    )
    final_target = (
        resolved_final_target_matches[0].name
        if resolved_final_target_matches
        else detect_final_target(source, history_text=history_text)
    )
    resolved_task_mode_matches = task_mode_matches or rank_task_mode_matches(
        source,
        final_target,
        history_text=history_text,
    )
    references = _collect_reference_texts(
        source,
        history_text=history_text,
        final_target_matches=resolved_final_target_matches,
        task_mode_matches=resolved_task_mode_matches,
    )
    units = _scan_text_units(source)
    if not units:
        return []

    coverages = _collect_unit_coverages(units, references)
    template_spans = _extract_template_focus_spans(
        source,
        units,
        references,
        coverages,
        limit,
    )
    if template_spans:
        return template_spans

    informative_flags = [
        _is_informative_unit(unit, coverage) for unit, coverage in zip(units, coverages)
    ]

    spans: list[str] = []
    seen_keys: set[str] = set()
    current_units: list[TextUnit] = []
    for unit, is_informative in zip(units, informative_flags):
        if is_informative:
            if current_units and unit.start - current_units[-1].end > 1:
                candidate = _emit_focus_span(source, current_units)
                candidate_key = compact_focus_key(candidate)
                if candidate and candidate_key not in seen_keys:
                    seen_keys.add(candidate_key)
                    spans.append(candidate)
                current_units = []
            current_units.append(unit)
            continue
        if not current_units:
            continue
        candidate = _emit_focus_span(source, current_units)
        candidate_key = compact_focus_key(candidate)
        if candidate and candidate_key not in seen_keys:
            seen_keys.add(candidate_key)
            spans.append(candidate)
        current_units = []

    if current_units:
        candidate = _emit_focus_span(source, current_units)
        candidate_key = compact_focus_key(candidate)
        if candidate and candidate_key not in seen_keys:
            seen_keys.add(candidate_key)
            spans.append(candidate)

    if spans:
        return _prune_nested_spans(spans, limit)
    return _fallback_focus_spans(source, units, limit)


def build_focus_query(
    text: str,
    *,
    history_text: str = "",
    final_target_matches: list[SemanticMatch] | None = None,
    task_mode_matches: list[SemanticMatch] | None = None,
    limit: int = 4,
) -> str:
    spans = extract_focus_spans(
        text,
        history_text=history_text,
        final_target_matches=final_target_matches,
        task_mode_matches=task_mode_matches,
        limit=limit,
    )
    if spans:
        return " ".join(spans[:limit]).strip()
    return " ".join(str(text or "").strip(_EDGE_PUNCTUATION).split())


def select_primary_focus_term(candidates: list[str] | tuple[str, ...]) -> str:
    for candidate in candidates or []:
        text = " ".join(str(candidate or "").strip(_EDGE_PUNCTUATION).split())
        if len(compact_focus_key(text)) >= 2:
            return text
    return ""


def rewrite_known_term_aliases(text: str) -> str:
    source = str(text or "")
    if not source:
        return ""
    units = _scan_text_units(source)
    if not units:
        return source

    replacements: list[tuple[int, int, str]] = []
    index = 0
    while index < len(units):
        best_match: tuple[int, str] | None = None
        for alias_tokens, replacement in _term_alias_entries():
            window_tokens: list[str] = []
            end_index = index
            while end_index < len(units) and len(window_tokens) < len(alias_tokens):
                window_tokens.append(units[end_index].normalized)
                if tuple(window_tokens) == alias_tokens:
                    best_match = (end_index, replacement)
                    break
                if tuple(alias_tokens[: len(window_tokens)]) != tuple(window_tokens):
                    break
                end_index += 1
            if best_match is not None:
                break
        if best_match is None:
            index += 1
            continue
        end_index, replacement = best_match
        replacements.append((units[index].start, units[end_index].end, replacement))
        index = end_index + 1

    if not replacements:
        return source

    pieces: list[str] = []
    cursor = 0
    for start, end, replacement in replacements:
        pieces.append(source[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(source[cursor:])
    return "".join(pieces)


__all__ = [
    "build_focus_query",
    "compact_focus_key",
    "extract_focus_spans",
    "rewrite_known_term_aliases",
    "select_primary_focus_term",
]
