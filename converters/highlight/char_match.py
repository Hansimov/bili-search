"""Char-level keyword matching highlighter for vector search.

This module provides highlighting for KNN/vector search results where
Elasticsearch keyword highlighting is not available.

The approach:
1. Extract keywords from the query (excluding DSL expressions)
2. Find all occurrences of keywords in target text fields
3. Wrap matches with <tag>...</tag> for display

This is simpler than ES highlighting but effective for vector search
where we just want to show which query terms appear in results.
"""

import re
from typing import Union


class CharMatchHighlighter:
    """Highlighter using char-level keyword matching.

    Example:
        highlighter = CharMatchHighlighter()
        result = highlighter.highlight(
            text="红警HBK08 游戏视频",
            keywords=["红警", "08"],
            tag="hit"
        )
        # Returns: "<hit>红警</hit>HBK<hit>08</hit> 游戏视频"
    """

    def __init__(self, tag: str = "hit"):
        """Initialize the highlighter.

        Args:
            tag: HTML tag name for highlighting (default: "hit").
        """
        self.tag = tag

    def extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query string, excluding DSL expressions.

        Args:
            query: Query string, possibly with DSL expressions like q=v, date=2024.

        Returns:
            List of keywords for matching.
        """
        if not query or not query.strip():
            return []

        # Remove common DSL patterns: key=value, key<value, key>value, etc.
        # Patterns to remove:
        # - qmod/q/qm expressions: q=w, q=v, q=wv, q=wvr, etc.
        # - date expressions: date=xxx, d=xxx, d>xxx, d<xxx
        # - stat expressions: v>1000, like>100, etc.
        # - user expressions: u=xxx, @xxx
        # - uid expressions: uid=xxx, mid=xxx
        # - bvid expressions: bv=xxx, av=xxx
        # - region expressions: rg=xxx, rid=xxx
        # - duration expressions: t>xxx, dura=xxx
        cleaned = query

        # Remove DSL key-value expressions (key=value, key>value, key<value, etc.)
        # This pattern matches: word followed by operator followed by value
        dsl_pattern = r"\b(?:q|qm|qmod|date|dt|d|rq|view|vw|v|like|lk|l|coin|cn|c|fav|favorite|fv|sc|reply|rp|pl|danmaku|dm|share|sh|fx|user|up|u|uid|mid|ud|bvid|bv|avid|av|region|rid|rg|fq|duration|dura|dr|time|t|ugid|g)\s*[=<>!]+\s*[^\s,;，；]+\s*"
        cleaned = re.sub(dsl_pattern, " ", cleaned, flags=re.IGNORECASE)

        # Remove @ mentions
        cleaned = re.sub(r"@[^\s]+", " ", cleaned)

        # Remove quoted strings markers but keep content
        # "xxx" -> xxx, 《xxx》-> xxx
        cleaned = re.sub(r'[""《》【】（）\[\]()]', " ", cleaned)

        # Split by whitespace and common punctuation
        tokens = re.split(r"[\s,;，；、:：]+", cleaned)

        # Filter: keep tokens with length >= 1 that are meaningful
        # Remove very short tokens and pure punctuation/symbols
        keywords = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            # Skip pure punctuation/operators
            if re.match(r"^[+\-!<>=|&?~]+$", token):
                continue
            # Skip very short non-Chinese tokens (likely noise)
            # Chinese characters are meaningful even at length 1
            has_chinese = bool(re.search(r"[\u4e00-\u9fff]", token))
            if not has_chinese and len(token) < 2:
                continue
            keywords.append(token)

        return keywords

    def highlight(
        self,
        text: str,
        keywords: Union[str, list[str]],
        tag: str = None,
        case_sensitive: bool = False,
    ) -> str:
        """Highlight keywords in text using char-level matching.

        Args:
            text: The text to highlight.
            keywords: A single keyword string or list of keywords.
            tag: Tag name for highlighting (overrides instance default).
            case_sensitive: Whether matching is case-sensitive.

        Returns:
            Text with keywords wrapped in <tag>...</tag>.
            Returns original text if no keywords or no matches.
        """
        if not text or not keywords:
            return text

        if tag is None:
            tag = self.tag

        # Normalize keywords to list
        if isinstance(keywords, str):
            keywords = self.extract_keywords(keywords)

        if not keywords:
            return text

        # Find all keyword occurrences and their positions
        # Track (start, end) tuples for each match
        matches = []

        for keyword in keywords:
            if not keyword:
                continue

            # Escape special regex chars in keyword
            escaped_keyword = re.escape(keyword)

            # Use case-insensitive matching by default
            flags = 0 if case_sensitive else re.IGNORECASE

            try:
                for match in re.finditer(escaped_keyword, text, flags=flags):
                    matches.append((match.start(), match.end()))
            except re.error:
                # Skip invalid patterns
                continue

        if not matches:
            return text

        # Sort matches by start position
        matches.sort(key=lambda x: x[0])

        # Merge overlapping/adjacent matches
        merged = []
        for start, end in matches:
            if merged and merged[-1][1] >= start:
                # Overlapping or adjacent - extend the previous match
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Build highlighted text
        result = []
        last_pos = 0

        for start, end in merged:
            # Add text before match
            result.append(text[last_pos:start])
            # Add highlighted match
            result.append(f"<{tag}>{text[start:end]}</{tag}>")
            last_pos = end

        # Add remaining text
        result.append(text[last_pos:])

        return "".join(result)

    def highlight_fields(
        self,
        hit: dict,
        keywords: Union[str, list[str]],
        fields: list[str] = None,
        tag: str = None,
    ) -> dict:
        """Highlight keywords in multiple fields of a search hit.

        Args:
            hit: Search hit document.
            keywords: Query string or list of keywords.
            fields: Fields to highlight. Defaults to ["title", "tags", "desc", "owner.name"].
            tag: Tag name for highlighting.

        Returns:
            Dict with field names as keys and highlighted text as values.
            Only includes fields that have matches.
        """
        if fields is None:
            fields = ["title", "tags", "desc", "owner.name"]

        if isinstance(keywords, str):
            keywords = self.extract_keywords(keywords)

        if not keywords:
            return {}

        highlights = {}

        for field in fields:
            # Handle nested fields like "owner.name"
            value = hit
            for key in field.split("."):
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    value = None
                    break

            if not value or not isinstance(value, str):
                continue

            highlighted = self.highlight(value, keywords, tag=tag)

            # Only include if there was actually a highlight
            if highlighted != value:
                highlights[field] = highlighted

        return highlights

    def add_highlights_to_hits(
        self,
        hits: list[dict],
        keywords: Union[str, list[str]],
        fields: list[str] = None,
        tag: str = None,
    ) -> list[dict]:
        """Add highlight information to a list of search hits.

        This modifies hits in-place and adds a 'highlights' field
        compatible with the existing highlight format.

        Args:
            hits: List of search hit documents.
            keywords: Query string or list of keywords.
            fields: Fields to highlight.
            tag: Tag name for highlighting.

        Returns:
            The same list with highlights added.
        """
        if isinstance(keywords, str):
            keywords = self.extract_keywords(keywords)

        if not keywords:
            return hits

        for hit in hits:
            field_highlights = self.highlight_fields(
                hit, keywords, fields=fields, tag=tag
            )

            if field_highlights:
                # Add to existing highlights or create new
                if "highlights" not in hit:
                    hit["highlights"] = {}

                # Add char-matched highlights in merged format
                if "merged" not in hit["highlights"]:
                    hit["highlights"]["merged"] = {}

                for field, highlighted in field_highlights.items():
                    hit["highlights"]["merged"][field] = [highlighted]

        return hits


# Singleton instance
_char_highlighter: CharMatchHighlighter = None


def get_char_highlighter() -> CharMatchHighlighter:
    """Get or create a singleton CharMatchHighlighter instance."""
    global _char_highlighter
    if _char_highlighter is None:
        _char_highlighter = CharMatchHighlighter()
    return _char_highlighter


if __name__ == "__main__":
    from tclogger import logger, dict_to_str

    highlighter = CharMatchHighlighter()

    # Test extract_keywords
    test_queries = [
        "红警08",
        "黑神话 悟空",
        "q=v 红警",
        "红警 v>1000 q=wv",
        "date=2024 黑神话 悟空",
        "@UP主 视频",
        '"精彩剪辑" 游戏',
    ]

    logger.note("Test extract_keywords:")
    for query in test_queries:
        keywords = highlighter.extract_keywords(query)
        logger.mesg(f"  [{query}] -> {keywords}")

    # Test highlight
    logger.note("\nTest highlight:")
    test_cases = [
        ("红警HBK08 游戏视频", ["红警", "08"]),
        ("黑神话悟空精彩剪辑", ["黑神话", "悟空"]),
        ("APEX传奇高手操作", ["apex", "传奇"]),  # case insensitive
    ]

    for text, keywords in test_cases:
        result = highlighter.highlight(text, keywords)
        logger.mesg(f"  [{text}] + {keywords}")
        logger.success(f"    -> {result}")

    # python -m converters.highlight.char_match
