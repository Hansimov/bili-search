"""Result store, summaries, and inspection helpers for llms orchestration."""

from __future__ import annotations

import json

from typing import Any

from llms.protocol import ToolExecutionRecord


def make_video_url(bvid: str) -> str:
    bvid_text = str(bvid or "").strip()
    return f"https://www.bilibili.com/video/{bvid_text}" if bvid_text else ""


def make_space_url(mid: Any) -> str:
    mid_text = str(mid or "").strip()
    return f"https://space.bilibili.com/{mid_text}" if mid_text else ""


def compact_video_hit(hit: dict) -> dict:
    owner = hit.get("owner") or {}
    owner_name = owner.get("name", "") if isinstance(owner, dict) else str(owner or "")
    owner_mid = owner.get("mid") if isinstance(owner, dict) else None
    bvid = str(hit.get("bvid", "") or "")
    return {
        "title": hit.get("title", ""),
        "bvid": bvid,
        "url": make_video_url(bvid),
        "owner": owner_name,
        "owner_mid": owner_mid,
        "view": ((hit.get("stat") or {}).get("view")),
    }


def compact_owner(owner: dict) -> dict:
    mid = owner.get("mid")
    return {
        "name": owner.get("name", ""),
        "mid": mid,
        "url": make_space_url(mid),
        "score": owner.get("score"),
    }


def compact_google_row(row: dict) -> dict:
    return {
        "title": row.get("title", ""),
        "link": row.get("link", ""),
        "domain": row.get("domain", ""),
        "site_kind": row.get("site_kind", ""),
    }


def compact_token_option(option: dict) -> dict:
    return {
        "text": option.get("text", ""),
        "score": option.get("score"),
    }


class ResultStore:
    def __init__(self):
        self.records: dict[str, ToolExecutionRecord] = {}
        self.order: list[str] = []

    def add(self, record: ToolExecutionRecord) -> None:
        self.records[record.result_id] = record
        self.order.append(record.result_id)

    def get(self, result_id: str) -> ToolExecutionRecord | None:
        return self.records.get(result_id)

    def latest_ids(self, limit: int = 5) -> list[str]:
        return self.order[-limit:]

    def render_observation(self, result_ids: list[str]) -> str:
        lines = ["[TOOL_OBSERVATIONS]"]
        for result_id in result_ids:
            record = self.records.get(result_id)
            if record is None:
                continue
            lines.append(
                f"- {result_id} {record.request.name}: {record.summary.get('summary_text', '')}"
            )
        lines.append(
            "如需更细结果，可调用 inspect_tool_result(result_ids=[...])；如需压缩/对比，可调用 run_small_llm_task。"
        )
        lines.append("[/TOOL_OBSERVATIONS]")
        return "\n".join(lines)


def summarize_result(result_id: str, tool_name: str, result: dict) -> dict:
    summary_text = ""
    if tool_name == "search_videos":
        if result.get("results"):
            query_summaries = []
            for item in result.get("results", [])[:3]:
                hits = item.get("hits") or []
                top_hits = [compact_video_hit(hit) for hit in hits[:3]]
                query_summaries.append(
                    {
                        "query": item.get("query", ""),
                        "resolved_query": item.get("resolved_query", ""),
                        "total_hits": item.get("total_hits", len(hits)),
                        "top_hits": top_hits,
                    }
                )
            summary_text = "; ".join(
                (
                    f"{item['query']}"
                    + (
                        f" (fallback={item['resolved_query']})"
                        if item.get("resolved_query")
                        else ""
                    )
                    + f" -> {item['total_hits']} hits, top="
                )
                + ", ".join(
                    f"{hit['title']}({hit['bvid']})"
                    for hit in item["top_hits"]
                    if hit.get("title")
                )
                for item in query_summaries
            )
            return {
                "result_id": result_id,
                "tool": tool_name,
                "queries": query_summaries,
                "summary_text": summary_text,
            }

        hits = result.get("hits") or []
        top_hits = [compact_video_hit(hit) for hit in hits[:5]]
        summary_text = (
            f"query={result.get('query', '')}"
            + (
                f", fallback={result.get('resolved_query', '')}"
                if result.get("resolved_query")
                else ""
            )
            + f", total_hits={result.get('total_hits', len(hits))}, top_hits="
            + ", ".join(
                f"{hit['title']}({hit['bvid']})" for hit in top_hits if hit.get("title")
            )
        )
        return {
            "result_id": result_id,
            "tool": tool_name,
            "query": result.get("query", ""),
            "resolved_query": result.get("resolved_query", ""),
            "total_hits": result.get("total_hits", len(hits)),
            "top_hits": top_hits,
            "summary_text": summary_text,
        }

    if tool_name == "search_owners":
        owners = result.get("owners") or []
        owner_rows = [compact_owner(owner) for owner in owners[:6]]
        summary_text = f"text={result.get('text', '')}, owners=" + ", ".join(
            owner["name"] for owner in owner_rows if owner.get("name")
        )
        return {
            "result_id": result_id,
            "tool": tool_name,
            "text": result.get("text", ""),
            "owners": owner_rows,
            "summary_text": summary_text,
        }

    if tool_name == "search_google":
        rows = result.get("results") or []
        top_results = [compact_google_row(row) for row in rows[:5]]
        summary_text = f"query={result.get('query', '')}, top_results=" + "; ".join(
            row["title"] for row in top_results if row.get("title")
        )
        return {
            "result_id": result_id,
            "tool": tool_name,
            "query": result.get("query", ""),
            "results": top_results,
            "summary_text": summary_text,
        }

    if tool_name == "related_tokens_by_tokens":
        options = result.get("options") or []
        top_options = [compact_token_option(item) for item in options[:8]]
        summary_text = f"text={result.get('text', '')}, token_options=" + ", ".join(
            option["text"] for option in top_options if option.get("text")
        )
        return {
            "result_id": result_id,
            "tool": tool_name,
            "text": result.get("text", ""),
            "options": top_options,
            "summary_text": summary_text,
        }

    if isinstance(result.get("owners"), list):
        owner_rows = [
            compact_owner(owner) for owner in (result.get("owners") or [])[:8]
        ]
        summary_text = ", ".join(
            f"{owner['name']}({owner['mid']})"
            for owner in owner_rows
            if owner.get("name")
        )
        return {
            "result_id": result_id,
            "tool": tool_name,
            "owners": owner_rows,
            "summary_text": summary_text,
        }

    if isinstance(result.get("videos"), list):
        video_rows = [
            compact_video_hit(video) for video in (result.get("videos") or [])[:8]
        ]
        summary_text = ", ".join(
            f"{video['title']}({video['bvid']})"
            for video in video_rows
            if video.get("title")
        )
        return {
            "result_id": result_id,
            "tool": tool_name,
            "videos": video_rows,
            "summary_text": summary_text,
        }

    summary_text = json.dumps(result, ensure_ascii=False)[:480]
    return {
        "result_id": result_id,
        "tool": tool_name,
        "summary_text": summary_text,
    }


def inspect_results(result_store: ResultStore, args: dict) -> dict:
    result_ids = list(args.get("result_ids") or [])
    focus = str(args.get("focus", "summary") or "summary")
    max_items = int(args.get("max_items", 5) or 5)
    inspected = []
    for result_id in result_ids:
        record = result_store.get(result_id)
        if record is None:
            continue
        if record.request.name == "search_videos":
            payload = {"result_id": result_id, "focus": focus}
            if record.result.get("results"):
                payload["queries"] = []
                for item in record.result.get("results", [])[:max_items]:
                    payload["queries"].append(
                        {
                            "query": item.get("query", ""),
                            "resolved_query": item.get("resolved_query", ""),
                            "hits": [
                                compact_video_hit(hit)
                                for hit in (item.get("hits") or [])[:max_items]
                            ],
                        }
                    )
            else:
                payload["query"] = record.result.get("query", "")
                if record.result.get("resolved_query"):
                    payload["resolved_query"] = record.result.get("resolved_query", "")
                payload["hits"] = [
                    compact_video_hit(hit)
                    for hit in (record.result.get("hits") or [])[:max_items]
                ]
            inspected.append(payload)
            continue

        if record.request.name == "search_google":
            inspected.append(
                {
                    "result_id": result_id,
                    "focus": focus,
                    "query": record.result.get("query", ""),
                    "results": [
                        compact_google_row(row)
                        for row in (record.result.get("results") or [])[:max_items]
                    ],
                }
            )
            continue

        if record.request.name == "search_owners" or isinstance(
            record.result.get("owners"), list
        ):
            inspected.append(
                {
                    "result_id": result_id,
                    "focus": focus,
                    "text": record.result.get("text", ""),
                    "owners": [
                        compact_owner(owner)
                        for owner in (record.result.get("owners") or [])[:max_items]
                    ],
                }
            )
            continue

        if record.request.name == "related_tokens_by_tokens":
            inspected.append(
                {
                    "result_id": result_id,
                    "focus": focus,
                    "text": record.result.get("text", ""),
                    "options": [
                        compact_token_option(option)
                        for option in (record.result.get("options") or [])[:max_items]
                    ],
                }
            )
            continue

        inspected.append(
            {
                "result_id": result_id,
                "focus": focus,
                "summary": record.summary,
            }
        )

    return {"focus": focus, "results": inspected}


__all__ = [
    "ResultStore",
    "compact_google_row",
    "compact_owner",
    "compact_token_option",
    "compact_video_hit",
    "inspect_results",
    "summarize_result",
]
