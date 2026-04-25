import re

from llms.intent.focus import compact_focus_key, rewrite_known_term_aliases
from llms.orchestration.tool_markup import EXTERNAL_TOOL_NAMES, EXTERNAL_TOOL_PREFIXES
from llms.orchestration.tool_markup import sanitize_generated_content


_SUPPORTED_TOOL_NAMES: tuple[str, ...] = EXTERNAL_TOOL_NAMES
_SUPPORTED_TOOL_NAME_PATTERN = "|".join(_SUPPORTED_TOOL_NAMES)
_TOOL_PREFIXES: tuple[str, ...] = EXTERNAL_TOOL_PREFIXES

_RESULTS_HEADER_RE = re.compile(r"\[搜索结果\][ \t]*\n?")
_RESULTS_ECHO_RE = re.compile(
    rf"(?:{_SUPPORTED_TOOL_NAME_PATTERN})\([^\n)]*\):[ \t]*\n?\{{[^\n]*\}}",
)


def _find_tool_command_start(text: str) -> int | None:
    """Return the earliest index of any tool command prefix in *text*, or None."""
    pos = None
    for prefix in _TOOL_PREFIXES:
        idx = text.find(prefix)
        if idx >= 0 and (pos is None or idx < pos):
            pos = idx
    return pos


def _has_partial_tool_prefix(text: str) -> bool:
    """Return True if *text* ends with a partial match for any tool prefix.

    Used as a look-ahead guard: if the last few characters of the accumulated
    content *could* be the start of a tool tag, we withhold them from the
    client until more data arrives to confirm or deny the match.
    """
    for prefix in _TOOL_PREFIXES:
        for length in range(1, len(prefix)):
            if text.endswith(prefix[:length]):
                return True
    return False


def _shared_prefix_len(left: str, right: str) -> int:
    """Return the length of the shared prefix between *left* and *right*."""
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def _leading_duplicate_prefix_len(text: str, *candidates: str) -> int:
    """Return the longest leading prefix in *text* duplicated by any candidate.

    Used while streaming content alongside reasoning_content: many providers
    start the content channel by echoing the same analysis text that is already
    available in reasoning_content or was shown in earlier tool iterations.
    Hiding that shared prefix lets us keep answer tokens incremental without
    flashing duplicate analysis text into the answer area.
    """
    max_len = 0
    for candidate in candidates:
        if not candidate:
            continue
        prefix_len = _shared_prefix_len(text, candidate)
        if prefix_len > max_len:
            max_len = prefix_len
    return max_len


def _sanitize_content(content: str) -> str:
    """Strip leaked markup and echoed tool results from content.

    Handles:
    - DeepSeek DSML tool-call tags
    - Inline XML tool commands (<search_videos/>, <search_owners/>, ...)
    - Echoed tool results in _format_results_message format
    """
    content = sanitize_generated_content(content, tool_names=_SUPPORTED_TOOL_NAMES)
    # Remove echoed tool results (e.g. search_videos(queries=[...]):\n{...})
    content = _RESULTS_HEADER_RE.sub("", content)
    content = _RESULTS_ECHO_RE.sub("", content)
    # Collapse runs of 3+ newlines left by stripped blocks
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = _dedupe_repeated_output(content)
    return content.strip()


def _dedupe_repeated_output(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if paragraphs:
        for repeat in (3, 2):
            if len(paragraphs) >= repeat and len(paragraphs) % repeat == 0:
                unit = len(paragraphs) // repeat
                chunks = [
                    "\n\n".join(paragraphs[index * unit : (index + 1) * unit]).strip()
                    for index in range(repeat)
                ]
                if len(set(chunks)) == 1:
                    return chunks[0]

        deduped: list[str] = []
        previous_norm = None
        for paragraph in paragraphs:
            paragraph_norm = re.sub(r"\s+", " ", paragraph)
            if paragraph_norm == previous_norm:
                continue
            deduped.append(paragraph)
            previous_norm = paragraph_norm
        return "\n\n".join(deduped)

    return text


def _anchor_subject_key(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", compact_focus_key(text))


def _canonical_response_subject(text: str) -> str:
    subject = " ".join(str(text or "").split()).strip()
    if not subject:
        return ""
    return rewrite_known_term_aliases(subject) or subject
