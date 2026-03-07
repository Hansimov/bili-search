import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from debugs.owners_search.eval_owner_panel import evaluate_case, load_panel


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
    assert len(hardneg_panel) >= 4
    assert all(item.get("forbidden_any_name") for item in hardneg_panel)


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
