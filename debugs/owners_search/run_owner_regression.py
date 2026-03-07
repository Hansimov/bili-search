import argparse
import datetime as dt
import json
from pathlib import Path

from debugs.owners_search.eval_owner_panel import (
    DEFAULT_PANEL_PATH,
    evaluate_case,
    load_panel,
    run_case,
    summarize_results,
)
from elastics.owners.searcher import OwnerSearcher

DEFAULT_PANELS = [
    DEFAULT_PANEL_PATH,
    Path(__file__).with_name("owner_query_panel_hardneg.json"),
]
DEFAULT_RECORD_DIR = Path("docs/owner_explore/experiments")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", required=True)
    parser.add_argument("-ev", "--elastic-env", default="elastic_dev")
    parser.add_argument(
        "-p",
        "--panel",
        action="append",
        default=None,
        help="Panel JSON path; may be repeated. Defaults to balanced + hardneg.",
    )
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument(
        "--record-dir",
        default=str(DEFAULT_RECORD_DIR),
        help="Directory for persisted regression experiment records.",
    )
    parser.add_argument(
        "--record-tag",
        default="",
        help="Optional label appended to the experiment record stem.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Skip writing JSON/Markdown experiment records.",
    )
    return parser


def _normalize_panel_paths(panel_args: list[str] | None) -> list[Path]:
    if panel_args:
        return [Path(item) for item in panel_args]
    return [Path(item) for item in DEFAULT_PANELS]


def _merge_check_buckets(aggregate: dict, summary: dict) -> None:
    for kind, bucket in (summary.get("by_check_kind") or {}).items():
        target = aggregate.setdefault(kind, {"total": 0, "passed": 0})
        target["total"] += int(bucket.get("total") or 0)
        target["passed"] += int(bucket.get("passed") or 0)


def run_panel_suite(
    searcher: OwnerSearcher,
    panel_path: Path,
    default_top_k: int,
) -> dict:
    panel = load_panel(panel_path)
    results = []
    for case in panel:
        result = run_case(searcher, case)
        results.append(evaluate_case(case, result, default_top_k=default_top_k))
    return {
        "panel": str(panel_path),
        "summary": summarize_results(results),
        "results": results,
    }


def summarize_regression_runs(runs: list[dict]) -> dict:
    total_cases = sum(
        int(run.get("summary", {}).get("total_cases") or 0) for run in runs
    )
    passed_cases = sum(
        int(run.get("summary", {}).get("passed_cases") or 0) for run in runs
    )
    by_check_kind: dict[str, dict] = {}
    by_dimension: dict[str, dict] = {}
    for run in runs:
        _merge_check_buckets(by_check_kind, run.get("summary") or {})
        for dimension, bucket in (
            (run.get("summary") or {}).get("by_dimension") or {}
        ).items():
            target = by_dimension.setdefault(dimension, {"total": 0, "passed": 0})
            target["total"] += int(bucket.get("total") or 0)
            target["passed"] += int(bucket.get("passed") or 0)
    return {
        "panel_count": len(runs),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "pass_rate": round((passed_cases / total_cases), 4) if total_cases else 0.0,
        "by_dimension": by_dimension,
        "by_check_kind": by_check_kind,
    }


def build_record_stem(index: str, record_tag: str = "") -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem_parts = [timestamp, index]
    if record_tag:
        stem_parts.append(record_tag)
    return "__".join(
        part.strip().replace("/", "_").replace(" ", "_")
        for part in stem_parts
        if part and part.strip()
    )


def build_record_paths(record_dir: str | Path, stem: str) -> dict[str, Path]:
    base_dir = Path(record_dir)
    return {
        "json": base_dir / f"{stem}.json",
        "md": base_dir / f"{stem}.md",
    }


def format_summary_markdown(payload: dict) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# Owner Regression Record",
        "",
        "## Run",
        "",
        f"- generated_at: {payload.get('generated_at') or ''}",
        f"- index: {payload.get('index') or ''}",
        f"- elastic_env: {payload.get('elastic_env') or ''}",
        f"- panel_count: {summary.get('panel_count') or 0}",
        f"- total_cases: {summary.get('total_cases') or 0}",
        f"- passed_cases: {summary.get('passed_cases') or 0}",
        f"- failed_cases: {summary.get('failed_cases') or 0}",
        f"- pass_rate: {summary.get('pass_rate') or 0.0}",
        "",
        "## Panels",
        "",
    ]
    for run in payload.get("runs") or []:
        run_summary = run.get("summary") or {}
        lines.append(
            f"- {run.get('panel')}: {run_summary.get('passed_cases') or 0}/{run_summary.get('total_cases') or 0} passed, pass_rate={run_summary.get('pass_rate') or 0.0}"
        )

    lines.extend(["", "## Check Buckets", ""])
    for kind, bucket in sorted((summary.get("by_check_kind") or {}).items()):
        lines.append(
            f"- {kind}: {bucket.get('passed') or 0}/{bucket.get('total') or 0}"
        )

    lines.extend(["", "## Dimension Buckets", ""])
    for dimension, bucket in sorted((summary.get("by_dimension") or {}).items()):
        lines.append(
            f"- {dimension}: {bucket.get('passed') or 0}/{bucket.get('total') or 0}"
        )

    lines.extend(
        ["", "## Notes", "", "- Source payload is stored in the sibling JSON file."]
    )
    return "\n".join(lines) + "\n"


def write_record_files(
    payload: dict, record_dir: str | Path, record_tag: str = ""
) -> dict:
    stem = build_record_stem(payload.get("index") or "owner_regression", record_tag)
    paths = build_record_paths(record_dir, stem)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["md"].write_text(format_summary_markdown(payload), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def main() -> None:
    args = build_parser().parse_args()
    panel_paths = _normalize_panel_paths(args.panel)
    searcher = OwnerSearcher(index_name=args.index, elastic_env_name=args.elastic_env)
    runs = [
        run_panel_suite(searcher, panel_path=panel_path, default_top_k=args.top_k)
        for panel_path in panel_paths
    ]
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "index": args.index,
        "elastic_env": args.elastic_env,
        "panels": [str(path) for path in panel_paths],
        "summary": summarize_regression_runs(runs),
        "runs": runs,
    }
    if not args.no_record:
        payload["record_paths"] = write_record_files(
            payload,
            record_dir=args.record_dir,
            record_tag=args.record_tag,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
