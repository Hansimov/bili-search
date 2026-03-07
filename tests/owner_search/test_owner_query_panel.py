import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from debugs.owners_search.eval_owner_panel import evaluate_case, load_panel
from debugs.owners_search.run_owner_regression import (
    build_record_paths,
    format_summary_markdown,
    summarize_regression_runs,
)


def test_owner_query_panel_has_head_and_tail_coverage():
    panel_path = ROOT / "debugs" / "owners_search" / "owner_query_panel.json"
    panel = load_panel(panel_path)

    assert len(panel) >= 8
    assert any(item["tier"] == "head" and item["intent"] == "name" for item in panel)
    assert any(item["tier"] == "head" and item["intent"] == "domain" for item in panel)
    assert any(item["tier"] == "tail" and item["intent"] == "phrase" for item in panel)

    hardneg_panel_path = (
        ROOT / "debugs" / "owners_search" / "owner_query_panel_hardneg.json"
    )
    hardneg_panel = load_panel(hardneg_panel_path)
    assert len(hardneg_panel) >= 8
    assert all(item.get("forbidden_any_name") for item in hardneg_panel)

    broad_panel_path = (
        ROOT / "debugs" / "owners_search" / "owner_query_panel_broad.json"
    )
    broad_panel = load_panel(broad_panel_path)
    assert len(broad_panel) >= 9
    assert any(item.get("dimension") == "photography" for item in broad_panel)
    assert any(item.get("dimension") == "movie_analysis" for item in broad_panel)
    assert any(item.get("dimension") == "3c" for item in broad_panel)


def test_evaluate_case_checks_route_name_and_min_hits():
    case = {
        "id": "demo",
        "tier": "tail",
        "intent": "phrase",
        "query": "原神纳塔剧情解析",
        "expected_route": "phrase",
        "expected_any_name": ["某个创作者"],
        "min_hits": 1,
        "top_k": 3,
    }
    result = {
        "query_route": "phrase",
        "total": 2,
        "hits": [
            {"mid": 1, "name": "别的创作者"},
            {"mid": 2, "name": "某个创作者"},
        ],
    }

    report = evaluate_case(case, result)

    assert report["passed"] is True
    assert report["top_names"] == ["别的创作者", "某个创作者"]
    assert len(report["checks"]) == 3


def test_evaluate_case_checks_forbidden_names():
    case = {
        "id": "hardneg",
        "tier": "tail",
        "intent": "phrase",
        "query": "希区柯克镜头语言",
        "expected_route": "phrase",
        "forbidden_any_name": ["圆桌动漫"],
        "top_k": 3,
    }
    result = {
        "query_route": "phrase",
        "total": 2,
        "hits": [
            {"mid": 1, "name": "DrRedTheRed"},
            {"mid": 2, "name": "圆桌动漫"},
        ],
    }

    report = evaluate_case(case, result)

    assert report["passed"] is False
    assert any(check["kind"] == "forbidden_name" for check in report["checks"])


def test_summarize_regression_runs_merges_check_kinds():
    summary = summarize_regression_runs(
        [
            {
                "summary": {
                    "total_cases": 2,
                    "passed_cases": 2,
                    "by_dimension": {"game": {"total": 2, "passed": 2}},
                    "by_check_kind": {
                        "route": {"total": 2, "passed": 2},
                        "forbidden_name": {"total": 1, "passed": 1},
                    },
                }
            },
            {
                "summary": {
                    "total_cases": 3,
                    "passed_cases": 2,
                    "by_dimension": {"photography": {"total": 1, "passed": 1}},
                    "by_check_kind": {
                        "route": {"total": 3, "passed": 2},
                        "min_hits": {"total": 2, "passed": 2},
                    },
                }
            },
        ]
    )

    assert summary["panel_count"] == 2
    assert summary["total_cases"] == 5
    assert summary["passed_cases"] == 4
    assert summary["by_dimension"]["game"] == {"total": 2, "passed": 2}
    assert summary["by_check_kind"]["route"] == {"total": 5, "passed": 4}


def test_summarize_results_includes_dimension_bucket():
    summary = summarize_regression_runs(
        [
            {
                "summary": {
                    "total_cases": 2,
                    "passed_cases": 2,
                    "by_check_kind": {"route": {"total": 2, "passed": 2}},
                }
            }
        ]
    )

    assert summary["by_check_kind"]["route"] == {"total": 2, "passed": 2}


def test_regression_record_helpers_build_paths_and_markdown():
    paths = build_record_paths("docs/owner_explore/experiments", "demo_record")

    assert paths["json"].name == "demo_record.json"
    assert paths["md"].name == "demo_record.md"

    markdown = format_summary_markdown(
        {
            "generated_at": "2026-03-07T10:00:00",
            "index": "demo_index",
            "elastic_env": "elastic_dev",
            "summary": {
                "panel_count": 2,
                "total_cases": 5,
                "passed_cases": 4,
                "failed_cases": 1,
                "pass_rate": 0.8,
                "by_check_kind": {
                    "route": {"total": 5, "passed": 4},
                    "forbidden_name": {"total": 2, "passed": 2},
                },
            },
            "runs": [
                {
                    "panel": "debugs/owners_search/owner_query_panel.json",
                    "summary": {"total_cases": 3, "passed_cases": 3, "pass_rate": 1.0},
                }
            ],
        }
    )

    assert "# Owner Regression Record" in markdown
    assert "- route: 4/5" in markdown
    assert "## Dimension Buckets" in markdown
    assert "demo_index" in markdown
