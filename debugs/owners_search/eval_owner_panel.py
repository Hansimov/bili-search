import argparse
import json
from pathlib import Path

from elastics.owners.searcher import OwnerSearcher

DEFAULT_PANEL_PATH = Path(__file__).with_name("owner_query_panel.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", required=True)
    parser.add_argument("-ev", "--elastic-env", default="elastic_dev")
    parser.add_argument("-p", "--panel", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("-k", "--top-k", type=int, default=5)
    return parser


def load_panel(panel_path: str | Path) -> list[dict]:
    path = Path(panel_path)
    panel = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(panel, list):
        raise ValueError("owner query panel must be a list")

    normalized = []
    for item in panel:
        if not isinstance(item, dict):
            raise ValueError("each owner query panel item must be an object")
        case = dict(item)
        case.setdefault("method", "search")
        case.setdefault("sort_by", "influence")
        case.setdefault("top_k", 5)
        case.setdefault("expected_any_name", [])
        case.setdefault("forbidden_any_name", [])
        case.setdefault("min_hits", 0)
        case.setdefault("notes", "")
        required = ["id", "tier", "intent", "query"]
        missing = [key for key in required if not case.get(key)]
        if missing:
            raise ValueError(f"panel item missing required fields: {missing}")
        normalized.append(case)
    return normalized


def run_case(searcher: OwnerSearcher, case: dict) -> dict:
    method = case.get("method", "search")
    query = case["query"]
    sort_by = case.get("sort_by", "influence")
    limit = max(int(case.get("top_k", 5)), 1)

    if method == "search_by_domain":
        result = searcher.search_by_domain(
            query,
            sort_by=sort_by,
            limit=limit,
            compact=True,
        )
    else:
        result = searcher.search(
            query,
            sort_by=sort_by,
            limit=limit,
            compact=True,
        )
    return result or {}


def evaluate_case(case: dict, result: dict, default_top_k: int = 5) -> dict:
    top_k = max(int(case.get("top_k", default_top_k)), 1)
    hits = (result.get("hits") or [])[:top_k]
    names = [hit.get("name") for hit in hits if hit.get("name")]
    total = result.get("total")
    if total is None:
        total = len(result.get("hits") or [])

    checks = []
    expected_route = case.get("expected_route")
    actual_route = result.get("query_route")
    if expected_route:
        checks.append(
            {
                "kind": "route",
                "expected": expected_route,
                "actual": actual_route,
                "passed": actual_route == expected_route,
            }
        )

    expected_names = case.get("expected_any_name") or []
    if expected_names:
        matched = [name for name in expected_names if name in names]
        checks.append(
            {
                "kind": "name_hit",
                "expected": expected_names,
                "actual": names,
                "passed": bool(matched),
            }
        )

    forbidden_names = case.get("forbidden_any_name") or []
    if forbidden_names:
        forbidden_hits = [name for name in names if name in forbidden_names]
        checks.append(
            {
                "kind": "forbidden_name",
                "expected": forbidden_names,
                "actual": names,
                "passed": not forbidden_hits,
            }
        )

    min_hits = int(case.get("min_hits") or 0)
    if min_hits:
        checks.append(
            {
                "kind": "min_hits",
                "expected": min_hits,
                "actual": total,
                "passed": int(total or 0) >= min_hits,
            }
        )

    passed = all(check["passed"] for check in checks) if checks else True
    return {
        "id": case["id"],
        "tier": case["tier"],
        "dimension": case.get("dimension") or "unknown",
        "intent": case["intent"],
        "query": case["query"],
        "method": case.get("method", "search"),
        "sort_by": case.get("sort_by", "influence"),
        "top_names": names,
        "top_mids": [hit.get("mid") for hit in hits],
        "total": total,
        "query_route": actual_route,
        "checks": checks,
        "passed": passed,
        "notes": case.get("notes", ""),
    }


def summarize_results(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item.get("passed"))
    by_tier: dict[str, dict] = {}
    by_dimension: dict[str, dict] = {}
    by_check_kind: dict[str, dict] = {}
    for item in results:
        tier = item.get("tier") or "unknown"
        bucket = by_tier.setdefault(tier, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += 1 if item.get("passed") else 0
        dimension = item.get("dimension") or "unknown"
        dimension_bucket = by_dimension.setdefault(
            dimension,
            {"total": 0, "passed": 0},
        )
        dimension_bucket["total"] += 1
        dimension_bucket["passed"] += 1 if item.get("passed") else 0
        for check in item.get("checks") or []:
            kind = check.get("kind") or "unknown"
            check_bucket = by_check_kind.setdefault(kind, {"total": 0, "passed": 0})
            check_bucket["total"] += 1
            check_bucket["passed"] += 1 if check.get("passed") else 0
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": round((passed / total), 4) if total else 0.0,
        "by_tier": by_tier,
        "by_dimension": by_dimension,
        "by_check_kind": by_check_kind,
    }


def main() -> None:
    args = build_parser().parse_args()
    panel = load_panel(args.panel)
    searcher = OwnerSearcher(index_name=args.index, elastic_env_name=args.elastic_env)
    results = []
    for case in panel:
        result = run_case(searcher, case)
        results.append(evaluate_case(case, result, default_top_k=args.top_k))

    payload = {
        "index": args.index,
        "elastic_env": args.elastic_env,
        "panel": str(Path(args.panel)),
        "summary": summarize_results(results),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
