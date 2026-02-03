"""Char-level keyword matching highlighter for vector search.

This module provides highlighting for KNN/vector search results where
Elasticsearch keyword highlighting is not available.

The approach:
1. Extract keywords from the query (excluding DSL expressions)
2. Tokenize keywords into units: each Chinese char is a unit, 
   continuous alphanumeric sequences are units
3. Find all occurrences of these units in target text
4. Merge adjacent matches and wrap with <tag>...</tag>

Example:
    For query "红警hbk08", the units are: ["红", "警", "hbk", "08"]
    For text "红警08", matches: "红", "警", "08"
    Result: "<hit>红警08</hit>" (merged because adjacent)
"""

import re
from typing import Union


def tokenize_to_units(text: str) -> list[str]:
    """Tokenize text into matching units.

    - Each Chinese character is an independent unit
    - Continuous alphanumeric sequences are units
    - Punctuation and whitespace are ignored

    Args:
        text: Input text to tokenize.

    Returns:
        List of units for matching.

    Example:
        "红警hbk08" -> ["红", "警", "hbk", "08"]
        "黑神话悟空" -> ["黑", "神", "话", "悟", "空"]
        "apex2024" -> ["apex", "2024"]
    """
    if not text:
        return []

    units = []
    i = 0
    n = len(text)

    while i < n:
        char = text[i]

        # Check if Chinese character (CJK Unified Ideographs)
        if "\u4e00" <= char <= "\u9fff":
            units.append(char)
            i += 1
        # Check if alphanumeric
        elif char.isalnum():
            # Collect continuous alphanumeric sequence
            # But separate letters from digits
            if char.isdigit():
                # Collect digits
                start = i
                while i < n and text[i].isdigit():
                    i += 1
                units.append(text[start:i])
            else:
                # Collect letters
                start = i
                while (
                    i < n
                    and text[i].isalpha()
                    and not ("\u4e00" <= text[i] <= "\u9fff")
                ):
                    i += 1
                units.append(text[start:i])
        else:
            # Skip punctuation, whitespace, etc.
            i += 1

    return units


def merge_adjacent_tags(text: str, tag: str = "hit") -> str:
    """Merge adjacent highlight tags.

    Converts: <hit>红</hit><hit>警</hit><hit>08</hit>
    To:       <hit>红警08</hit>

    Args:
        text: Text with potentially adjacent tags.
        tag: The tag name used for highlighting.

    Returns:
        Text with merged adjacent tags.
    """
    # Pattern to match: </tag><tag> (with optional whitespace between)
    # We merge tags that are directly adjacent
    pattern = f"</{tag}><{tag}>"

    # Keep merging until no more adjacent tags
    while pattern in text:
        text = text.replace(pattern, "")

    return text


class CharMatchHighlighter:
    """Highlighter using char-level keyword matching.

    For Chinese characters, each character is matched independently.
    For alphanumeric, continuous sequences are matched as units.
    Adjacent matches are merged into a single highlight span.

    Example:
        highlighter = CharMatchHighlighter()

        # Query "红警hbk08" matches "红警08"
        # because units ["红", "警", "08"] all match
        result = highlighter.highlight(
            text="红警08 游戏视频",
            keywords=["红警hbk08"],
            tag="hit"
        )
        # Returns: "<hit>红警08</hit> 游戏视频"
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
        cleaned = query

        # Remove DSL key-value expressions
        dsl_pattern = r"\b(?:q|qm|qmod|date|dt|d|rq|view|vw|v|like|lk|l|coin|cn|c|fav|favorite|fv|sc|reply|rp|pl|danmaku|dm|share|sh|fx|user|up|u|uid|mid|ud|bvid|bv|avid|av|region|rid|rg|fq|duration|dura|dr|time|t|ugid|g)\s*[=<>!]+\s*[^\s,;，；]+\s*"
        cleaned = re.sub(dsl_pattern, " ", cleaned, flags=re.IGNORECASE)

        # Remove @ mentions
        cleaned = re.sub(r"@[^\s]+", " ", cleaned)

        # Remove quoted strings markers but keep content
        cleaned = re.sub(r'[""《》【】（）\[\]()]', " ", cleaned)

        # Split by whitespace and common punctuation
        tokens = re.split(r"[\s,;，；、:：]+", cleaned)

        # Filter: keep tokens with length >= 1 that are meaningful
        keywords = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            # Skip pure punctuation/operators
            if re.match(r"^[+\-!<>=|&?~]+$", token):
                continue
            keywords.append(token)

        return keywords

    def extract_units_from_keywords(self, keywords: list[str]) -> list[str]:
        """Extract all matching units from keywords.

        Args:
            keywords: List of keyword strings.

        Returns:
            List of unique units for matching (lowercase for case-insensitive).
        """
        all_units = set()
        for keyword in keywords:
            units = tokenize_to_units(keyword)
            for unit in units:
                # Store lowercase for case-insensitive matching
                all_units.add(unit.lower())
        return list(all_units)

    def highlight(
        self,
        text: str,
        keywords: Union[str, list[str]],
        tag: str = None,
        case_sensitive: bool = False,
    ) -> str:
        """Highlight keywords in text using char-level matching.

        For each keyword:
        - Chinese characters are matched individually
        - Alphanumeric sequences are matched as units
        - Adjacent matches are merged into single highlight spans

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

        # Extract all units from keywords
        units = self.extract_units_from_keywords(keywords)
        if not units:
            return text

        # Find all unit occurrences and their positions
        matches = []  # List of (start, end) tuples

        for unit in units:
            if not unit:
                continue

            # Escape special regex chars
            escaped_unit = re.escape(unit)

            # Use case-insensitive matching by default
            flags = 0 if case_sensitive else re.IGNORECASE

            try:
                for match in re.finditer(escaped_unit, text, flags=flags):
                    matches.append((match.start(), match.end()))
            except re.error:
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

        highlighted = "".join(result)

        # Merge adjacent tags (should already be merged, but just in case)
        highlighted = merge_adjacent_tags(highlighted, tag)

        return highlighted

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
    from tclogger import logger

    highlighter = CharMatchHighlighter()

    # Test tokenize_to_units
    logger.note("Test tokenize_to_units:")
    test_texts = [
        "红警hbk08",
        "黑神话悟空",
        "apex2024",
        "HBK08小块地",
        "test123abc",
    ]
    for text in test_texts:
        units = tokenize_to_units(text)
        logger.mesg(f"  [{text}] -> {units}")

    # Test extract_keywords
    logger.note("\nTest extract_keywords:")
    test_queries = [
        "红警08",
        "黑神话 悟空",
        "q=v 红警",
        "红警 v>1000 q=wv",
        "date=2024 黑神话 悟空",
    ]
    for query in test_queries:
        keywords = highlighter.extract_keywords(query)
        logger.mesg(f"  [{query}] -> {keywords}")

    # Test highlight with char-level matching
    logger.note("\nTest highlight (char-level):")
    test_cases = [
        # Query keywords, text, expected behavior
        ("红警HBK08 游戏视频", ["红警", "08"], "红警 and 08 should match"),
        ("红警08", ["红警hbk08"], "红警 and 08 should match (partial)"),
        ("红警HBK08 游戏视频", ["红警hbk08"], "红, 警, hbk, 08 should match"),
        ("黑神话悟空精彩剪辑", ["黑神话", "悟空"], "黑神话悟空 should match"),
        ("APEX传奇高手操作", ["apex", "传奇"], "apex and 传奇 should match"),
        ("测试abc123数据", ["abc", "123"], "abc and 123 should match"),
    ]

    for text, keywords, description in test_cases:
        result = highlighter.highlight(text, keywords)
        logger.mesg(f"  [{text}] + {keywords}")
        logger.success(f"    -> {result}")
        logger.hint(f"    ({description})")

    # Test that adjacent tags are merged
    logger.note("\nTest adjacent tag merging:")
    text = "红警08"
    keywords = ["红", "警", "08"]
    result = highlighter.highlight(text, keywords)
    logger.mesg(f"  [{text}] + {keywords}")
    logger.success(f"    -> {result}")
    expected = "<hit>红警08</hit>"
    if result == expected:
        logger.okay(f"    ✓ Correctly merged to {expected}")
    else:
        logger.warn(f"    × Expected {expected}")

    # python -m converters.highlight.char_match
