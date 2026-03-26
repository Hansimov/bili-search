"""Utility functions for processing search results for LLM consumption."""

import re
from urllib.parse import urlparse

from tclogger import ts_to_str, get_now_ts, dt_to_zh_str


# Source fields to keep when formatting hits for the LLM.
# Minimized to reduce token consumption — only include what the LLM needs
# to generate a useful response. desc/coin/danmaku are omitted
# because the LLM only lists title, author, view count, and pubdate.
LLM_HIT_FIELDS = [
    "title",
    "bvid",
    "owner",
    "pubdate",
    "duration",
    "pic",
    "tags",
    "stat.view",
]


def extract_field(hit: dict, field: str):
    """Extract a possibly nested field (e.g., 'stat.view') from a dict."""
    parts = field.split(".")
    val = hit
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return None
    return val


def shrink_hit(hit: dict, fields: list[str] = None) -> dict:
    """Reduce a hit dict to only the specified fields.

    Handles nested fields like 'stat.view' by preserving the nesting.
    """
    if fields is None:
        fields = LLM_HIT_FIELDS
    result = {}
    for field in fields:
        val = extract_field(hit, field)
        if val is None:
            continue
        parts = field.split(".")
        target = result
        for p in parts[:-1]:
            if p not in target:
                target[p] = {}
            target = target[p]
        target[parts[-1]] = val
    return result


def add_link(hit: dict) -> dict:
    """Add a bilibili video link from bvid."""
    bvid = hit.get("bvid", "")
    if bvid:
        hit["link"] = f"https://www.bilibili.com/video/{bvid}"
    return hit


def add_pubdate_str(hit: dict) -> dict:
    """Add human-readable pubdate_str field."""
    pubdate = hit.get("pubdate")
    if pubdate is not None:
        hit["pubdate_str"] = ts_to_str(int(pubdate))
    return hit


def add_pub_to_now_str(hit: dict) -> dict:
    """Add relative time string (e.g., '3天前')."""
    now_ts = get_now_ts()
    pubdate = hit.get("pubdate")
    if pubdate is not None:
        delta_seconds = now_ts - int(pubdate)
        hit["pub_to_now_str"] = dt_to_zh_str(delta_seconds)
    return hit


def add_duration_str(hit: dict) -> dict:
    """Add human-readable duration_str and drop raw duration later if needed."""
    duration = hit.get("duration")
    if duration is None:
        return hit

    total_seconds = int(duration)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        hit["duration_str"] = f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        hit["duration_str"] = f"{minutes}:{seconds:02d}"
    return hit


# Maximum number of tags to keep per hit (to limit token consumption)
MAX_TAGS_PER_HIT = 5


def format_hit_for_llm(hit: dict, fields: list[str] = None) -> dict:
    """Format a single hit for LLM consumption.

    Shrinks to relevant fields, adds link + time strings,
    truncates tags, and removes raw pubdate timestamp.
    """
    result = shrink_hit(hit, fields)
    add_link(result)
    add_pubdate_str(result)
    add_pub_to_now_str(result)
    add_duration_str(result)
    result.pop("duration", None)
    # Truncate tags to save tokens
    if "tags" in result and isinstance(result["tags"], str):
        tags_list = [t.strip() for t in result["tags"].split(",") if t.strip()]
        result["tags"] = ",".join(tags_list[:MAX_TAGS_PER_HIT])
    return result


def format_hits_for_llm(
    hits: list[dict],
    fields: list[str] = None,
    max_hits: int = 15,
) -> list[dict]:
    """Format a list of hits for LLM consumption.

    Args:
        hits: Raw hit dicts from search/explore results.
        fields: Fields to keep. Defaults to LLM_HIT_FIELDS.
        max_hits: Maximum hits to return (to limit LLM context usage).

    Returns:
        List of formatted hit dicts.
    """
    return [format_hit_for_llm(hit, fields) for hit in hits[:max_hits]]


def extract_explore_hits(explore_result: dict) -> tuple[list[dict], int]:
    """Extract hits from an explore response.

    The explore response has a nested structure:
        data[0].output.hits → list of video hits
        data[0].output.total_hits → total hit count

    Returns:
        Tuple of (hits_list, total_hits).
    """
    data = explore_result.get("data", [])
    if not data:
        return [], 0

    # The first step contains the search results
    first_step = data[0]
    output = first_step.get("output", {})
    hits = output.get("hits", [])
    total_hits = output.get("total_hits", len(hits))
    return hits, total_hits


def extract_explore_authors(explore_result: dict) -> list[dict]:
    """Extract author groups from an explore response.

    The explore response may have a group_hits_by_owner step:
        data[N].name == "group_hits_by_owner" → data[N].output.authors

    Returns:
        List of author group dicts.
    """
    data = explore_result.get("data", [])
    for step in data:
        if step.get("name") == "group_hits_by_owner":
            return step.get("output", {}).get("authors", [])
    return []


def format_related_token_options(options: list[dict], max_hits: int = 10) -> list[dict]:
    return [
        {
            "text": option.get("text", ""),
            "score": option.get("score", 0),
        }
        for option in options[:max_hits]
    ]


def format_related_owners(owners: list[dict], max_hits: int = 10) -> list[dict]:
    formatted = []
    for owner in owners[:max_hits]:
        item = {
            "mid": owner.get("mid"),
            "name": owner.get("name", ""),
            "score": owner.get("score", 0),
        }
        if owner.get("face"):
            item["face"] = owner.get("face")
        if owner.get("sources"):
            item["sources"] = owner.get("sources")
        if owner.get("sample_title"):
            item["sample_title"] = owner.get("sample_title")
        if owner.get("sample_bvid"):
            item["sample_bvid"] = owner.get("sample_bvid")
        if owner.get("sample_view") is not None:
            item["sample_view"] = owner.get("sample_view")
        sample_pic = (
            owner.get("sample_pic")
            or owner.get("sample_cover")
            or owner.get("sample_cover_url")
        )
        if sample_pic:
            item["sample_pic"] = sample_pic
        formatted.append(item)
    return formatted


def format_related_videos(videos: list[dict], max_hits: int = 10) -> list[dict]:
    formatted = []
    for video in videos[:max_hits]:
        formatted.append(
            {
                "bvid": video.get("bvid", ""),
                "title": video.get("title", ""),
                "score": video.get("score", 0),
                "owner": {
                    "mid": video.get("owner_mid"),
                    "name": video.get("owner_name", ""),
                },
            }
        )
    return formatted


def format_google_results(results: list[dict], max_hits: int = 5) -> list[dict]:
    def extract_google_link_metadata(link: str) -> dict:
        parsed = urlparse(str(link or ""))
        domain = (parsed.netloc or "").lower()
        if domain.startswith("www."):
            domain = domain[4:]
        path = parsed.path or ""

        metadata = {"domain": domain}
        if not domain.endswith("bilibili.com"):
            return metadata

        metadata["is_bilibili"] = True
        if domain == "space.bilibili.com":
            metadata["site_kind"] = "space"
            match = re.search(r"/(\d+)(?:/|$)", path)
            if match:
                metadata["mid"] = int(match.group(1))
            return metadata

        if path.startswith("/video/"):
            metadata["site_kind"] = "video"
            match = re.search(r"/video/(BV[0-9A-Za-z]+)", path)
            if match:
                metadata["bvid"] = match.group(1)
            return metadata

        if path.startswith("/read/"):
            metadata["site_kind"] = "read"
            match = re.search(r"/read/(cv\d+|\d+)", path)
            if match:
                metadata["article_id"] = match.group(1)
            return metadata

        metadata["site_kind"] = "bilibili"
        return metadata

    formatted = []
    for result in results[:max_hits]:
        link = result.get("link", result.get("url", ""))
        formatted_result = {
            "title": result.get("title", ""),
            "link": link,
            "snippet": result.get("snippet", result.get("body", "")),
            "display_link": result.get("display_link", result.get("source", "")),
        }
        formatted_result.update(extract_google_link_metadata(link))
        formatted.append(formatted_result)
    return formatted


def analyze_suggest_for_authors(
    suggest_result: dict,
    query: str,
) -> dict:
    """Analyze suggest results to detect author intent.

    Processes suggest hits to determine if the query matches a video author.
    Returns keyword highlights and related authors with ratios.

    Args:
        suggest_result: Response from /suggest endpoint.
        query: Original query text.

    Returns:
        Dict with query, total_hits, highlighted_keywords, related_authors.
    """
    hits = suggest_result.get("hits", [])
    total_hits = suggest_result.get("total_hits", len(hits))
    query_lower = query.lower().strip()

    if not hits:
        return {
            "query": query,
            "total_hits": 0,
            "highlighted_keywords": {},
            "related_authors": {},
        }

    # Count keyword highlights from merged highlight field
    keyword_counts = {}
    for hit in hits:
        highlights = hit.get("highlights", {})
        merged = highlights.get("merged", [])
        if isinstance(merged, list):
            for h in merged:
                found = re.findall(r"<em>(.*?)</em>", str(h))
                for kw in found:
                    keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

    # If no merged highlights, try common and segged
    if not keyword_counts:
        for hit in hits:
            highlights = hit.get("highlights", {})
            for field_name in ["common", "segged"]:
                field_highlights = highlights.get(field_name, {})
                if isinstance(field_highlights, dict):
                    for _field, frags in field_highlights.items():
                        if isinstance(frags, list):
                            for frag in frags:
                                found = re.findall(r"<em>(.*?)</em>", str(frag))
                                for kw in found:
                                    keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

    # Count and analyze related authors
    author_counts = {}
    for hit in hits:
        owner = hit.get("owner", {})
        if not owner:
            continue
        name = owner.get("name", "")
        mid = owner.get("mid", 0)
        if name:
            if name not in author_counts:
                author_counts[name] = {"uid": mid, "count": 0}
            author_counts[name]["count"] += 1

    # Calculate ratios and check highlighting
    num_hits = max(len(hits), 1)
    related_authors = {}
    for name, info in sorted(author_counts.items(), key=lambda x: -x[1]["count"]):
        ratio = round(info["count"] / num_hits, 2)
        # Author name is "highlighted" if the query text appears in the name
        # or the name appears in query
        highlighted = query_lower in name.lower() or name.lower() in query_lower
        entry = {
            "uid": info["uid"],
            "ratio": ratio,
        }
        if highlighted:
            entry["highlighted"] = True
        related_authors[name] = entry

    # Determine the best matching author (highlighted + highest ratio)
    best_author = None
    best_ratio = 0.0
    for name, info in related_authors.items():
        if info.get("highlighted") and info.get("ratio", 0) > best_ratio:
            best_author = name
            best_ratio = info["ratio"]

    result = {
        "query": query,
        "total_hits": total_hits,
        "highlighted_keywords": keyword_counts,
        "related_authors": related_authors,
    }

    # Top-level found/name/mid for easy frontend consumption
    if best_author:
        result["found"] = True
        result["name"] = best_author
        result["mid"] = related_authors[best_author].get("uid")
        result["ratio"] = best_ratio
    else:
        result["found"] = False

    return result


def format_view_count(view: int) -> str:
    """Format view count for display (e.g., 123456 → '12.3万')."""
    if view is None:
        return ""
    if view >= 100_000_000:
        return f"{view / 100_000_000:.1f}亿"
    if view >= 10_000:
        return f"{view / 10_000:.1f}万"
    return str(view)
