from tclogger import ts_to_str, get_now_ts, dt_to_zh_str

from converters.query.field import is_field_in_fields


def add_links(hits: list[dict]) -> list[dict]:
    for hit in hits:
        bvid = hit.get("bvid", "")
        if bvid:
            hit["link"] = f"https://www.bilibili.com/video/{bvid}"
    return hits


def shrink_hits_by_source_fields(
    hits: list[dict], source_fields: list[str]
) -> list[dict]:
    new_hits = [
        {k: v for k, v in hit.items() if is_field_in_fields(k, source_fields)}
        for hit in hits
    ]
    return new_hits


def add_pubdate_str(hits: list[dict]) -> list[dict]:
    """Add `pubdate_str` (YYYY-MM-DD HH:mm:SS) to each hit that has `pubdate`."""
    for hit in hits:
        pubdate = hit.get("pubdate", None)
        if pubdate is not None:
            hit["pubdate_str"] = ts_to_str(int(pubdate))
    return hits


def add_pub_to_now_str(hits: list[dict]) -> list[dict]:
    """Add `pub_to_now_str` (relative time in Chinese) to each hit that has `pubdate`."""
    now_ts = get_now_ts()
    for hit in hits:
        pubdate = hit.get("pubdate", None)
        if pubdate is not None:
            delta_seconds = now_ts - int(pubdate)
            hit["pub_to_now_str"] = dt_to_zh_str(delta_seconds)
    return hits


def shrink_results(results: dict, source_fields: list[str]) -> dict:
    """Shrink search/suggest results to only include specified source fields,
    and add links and time strings."""
    hits = results.get("hits", [])
    hits = shrink_hits_by_source_fields(hits, source_fields)
    hits = add_links(hits)
    hits = add_pubdate_str(hits)
    hits = add_pub_to_now_str(hits)
    res = {"query": results.get("query", ""), "hits": hits}
    return res
