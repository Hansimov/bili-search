import re

from llms.intent import build_intent_profile
from llms.intent.focus import build_focus_query
from llms.intent.focus import compact_focus_key
from llms.intent.focus import rewrite_known_term_aliases
from llms.intent.focus import select_primary_focus_term
from llms.chat.content import _anchor_subject_key, _canonical_response_subject


class ChatResponseContextMixin:
    @classmethod
    def _ensure_author_timeline_context(
        cls,
        messages: list[dict],
        content: str,
        *,
        intent=None,
    ) -> str:
        final_content = (content or "").strip()
        if not final_content:
            return final_content

        resolved_intent = intent or build_intent_profile(messages)
        if (
            resolved_intent.final_target != "videos"
            or resolved_intent.task_mode != "repeat"
        ):
            return final_content

        author_name = select_primary_focus_term(
            [
                *(resolved_intent.explicit_entities or []),
                *(resolved_intent.explicit_topics or []),
            ]
        )
        leading_subject = cls._extract_leading_subject_phrase(
            cls._get_latest_user_text(messages)
        )
        author_name_key = _anchor_subject_key(author_name)
        leading_subject_key = _anchor_subject_key(leading_subject)
        if leading_subject_key and (
            not author_name_key
            or author_name_key in leading_subject_key
            or len(leading_subject) < len(author_name)
        ):
            author_name = leading_subject
        author_name = " ".join(str(author_name or "").split()).strip()
        if not author_name or author_name in final_content:
            return final_content
        return f"{author_name}最近视频：\n{final_content}"

    @staticmethod
    def _extract_subject_from_latest_user_text(
        latest_user_text: str,
        candidate_keys: set[str],
    ) -> str:
        source = str(latest_user_text or "").strip()
        if not source or not candidate_keys:
            return ""

        segments = [
            (
                match.group(0),
                compact_focus_key(match.group(0)),
                match.start(),
                match.end(),
            )
            for match in re.finditer(
                r"[A-Za-z]+|\d+(?:[./]\d+)*|[\u4e00-\u9fff]+|[^A-Za-z0-9\u4e00-\u9fff\s]+",
                source,
            )
        ]
        if not segments:
            return ""

        best_subject = ""
        best_score = -1
        for index, (_, segment_key, start, end) in enumerate(segments):
            if segment_key not in candidate_keys:
                continue

            matched_keys = [segment_key]
            subject_end = end
            scan_index = index + 1
            while scan_index < len(segments):
                _, next_key, _, next_end = segments[scan_index]
                if not next_key:
                    scan_index += 1
                    continue
                if next_key not in candidate_keys:
                    break
                matched_keys.append(next_key)
                subject_end = next_end
                scan_index += 1

            subject = source[start:subject_end].strip(
                " ，。！？?；;：:、()[]{}<>《》\"'`~!@#$%^&*-+=|\\/"
            )
            score = sum(len(key) for key in dict.fromkeys(matched_keys))
            if score > best_score or (
                score == best_score and len(subject) > len(best_subject)
            ):
                best_subject = subject
                best_score = score

        return best_subject

    @staticmethod
    def _extract_leading_subject_phrase(latest_user_text: str) -> str:
        source = str(latest_user_text or "").strip()
        if not source:
            return ""

        segments = [
            (
                match.group(0),
                compact_focus_key(match.group(0)),
                match.start(),
                match.end(),
            )
            for match in re.finditer(
                r"[A-Za-z]+|\d+(?:[./]\d+)*|[\u4e00-\u9fff]+|[^A-Za-z0-9\u4e00-\u9fff\s]+",
                source,
            )
        ]
        if not segments:
            return ""

        subject_start = None
        subject_end = None
        saw_content = False
        for segment_text, segment_key, start, end in segments:
            is_long_cjk_clause = (
                bool(re.fullmatch(r"[\u4e00-\u9fff]+", segment_text))
                and len(segment_text) >= 5
            )
            normalized_segment = "".join(str(segment_text or "").split())
            is_short_question_clause = normalized_segment in {
                "是谁",
                "是什么",
                "谁",
                "什么",
                "哪个",
                "哪位",
                "哪种",
                "吗",
                "呢",
                "嘛",
                "么",
            }
            if not saw_content:
                if not segment_key:
                    continue
                subject_start = start
                subject_end = end
                saw_content = True
                continue
            if not segment_key:
                subject_end = end
                continue
            if is_long_cjk_clause or is_short_question_clause:
                break
            subject_end = end

        if subject_start is None or subject_end is None:
            return ""
        return source[subject_start:subject_end].strip(
            " ，。！？?；;：:、()[]{}<>《》\"'`~!@#$%^&*-+=|\\/"
        )

    @classmethod
    def _ensure_primary_subject_context(
        cls,
        messages: list[dict],
        content: str,
        *,
        intent=None,
    ) -> str:
        final_content = (content or "").strip()
        if not final_content:
            return final_content

        resolved_intent = intent or build_intent_profile(messages)
        if resolved_intent.final_target not in {"external", "mixed"} and not (
            resolved_intent.final_target == "videos"
            and resolved_intent.needs_term_normalization
        ):
            return final_content

        candidate_texts = [
            *(resolved_intent.explicit_entities or []),
            *(resolved_intent.explicit_topics or []),
        ]
        candidate_keys = {
            compact_focus_key(candidate)
            for candidate in candidate_texts
            if len(compact_focus_key(candidate)) >= 2
        }
        subject = cls._extract_subject_from_latest_user_text(
            cls._get_latest_user_text(messages),
            candidate_keys,
        ) or select_primary_focus_term(candidate_texts)
        leading_subject = cls._extract_leading_subject_phrase(
            cls._get_latest_user_text(messages)
        )
        subject_key = _anchor_subject_key(subject)
        leading_subject_key = _anchor_subject_key(leading_subject)
        if leading_subject_key and (
            not subject_key or subject_key in leading_subject_key
        ):
            subject = leading_subject
        if resolved_intent.needs_term_normalization and subject:
            subject = rewrite_known_term_aliases(subject) or subject
            subject = _canonical_response_subject(subject)
        subject = " ".join(str(subject or "").split()).strip()
        subject_key = _anchor_subject_key(subject)
        if not subject_key or len(subject) > 48:
            return final_content
        if subject_key in _anchor_subject_key(final_content):
            return final_content
        return f"{subject}：\n{final_content}"

    @classmethod
    def _ensure_response_context(cls, messages: list[dict], content: str) -> str:
        resolved_intent = build_intent_profile(messages)
        final_content = cls._ensure_author_timeline_context(
            messages,
            content,
            intent=resolved_intent,
        )
        latest_user_text = cls._get_latest_user_text(messages)
        if latest_user_text:
            primary_context_intent = build_intent_profile(
                [{"role": "user", "content": latest_user_text}]
            )
        else:
            primary_context_intent = resolved_intent
        return cls._ensure_primary_subject_context(
            messages,
            final_content,
            intent=primary_context_intent,
        )

    @staticmethod
    def _normalize_entity_focused_query_text(text: str) -> str:
        return _canonical_response_subject(build_focus_query(text))
