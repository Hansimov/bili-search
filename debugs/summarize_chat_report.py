from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def iter_tools(row: dict[str, Any]) -> list[str]:
    chat = row.get("chat")
    if not isinstance(chat, dict):
        return []
    tools = chat.get("tools")
    if not isinstance(tools, list):
        return []
    return [tool for tool in tools if isinstance(tool, str)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_path", type=Path)
    parser.add_argument("--slow-limit", type=int, default=8)
    args = parser.parse_args()

    rows = json.loads(args.report_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("report JSON must be a list")

    tool_counts: Counter[str] = Counter()
    errors = 0
    google_calls = 0
    elapsed_values: list[float] = []
    category_counts: dict[str, int] = defaultdict(int)
    slow_rows: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("id") or "")
        category = case_id.rsplit("_", 1)[0] if "_" in case_id else "unknown"
        category_counts[category] += 1
        tools = iter_tools(row)
        tool_counts.update(tools)
        if "search_google" in tools:
            google_calls += 1
        if row.get("chat_error") is not None:
            errors += 1
        elapsed = row.get("chat_elapsed_s")
        if isinstance(elapsed, int | float):
            elapsed_values.append(float(elapsed))
            slow_rows.append(row)

    print(f"completed={len(rows)}")
    print(f"errors={errors}")
    print(f"google_calls={google_calls}")
    if elapsed_values:
        print(f"avg_elapsed_s={mean(elapsed_values):.2f}")
        print(f"max_elapsed_s={max(elapsed_values):.2f}")
    print(
        "tool_counts="
        + ",".join(f"{tool}:{count}" for tool, count in sorted(tool_counts.items()))
    )
    print(
        "category_counts="
        + ",".join(
            f"{category}:{count}" for category, count in sorted(category_counts.items())
        )
    )

    for row in sorted(
        slow_rows, key=lambda item: item.get("chat_elapsed_s") or 0, reverse=True
    )[: args.slow_limit]:
        print(
            "SLOW "
            f"id={row.get('id')} "
            f"elapsed_s={row.get('chat_elapsed_s')} "
            f"tools={','.join(iter_tools(row))} "
            f"error={row.get('chat_error')}"
        )


if __name__ == "__main__":
    main()
