from tests.llm.test_live_chat import evaluate_test_case_result


def test_evaluate_test_case_result_can_warn_on_soft_miss():
    evaluation = evaluate_test_case_result(
        content="Gemini 2.5 最近有几项官方更新。",
        used_tools=[],
        checks={
            "content_contains": ["Gemini 2.5"],
            "expected_tools_any": ["search_google"],
            "min_content_length": 10,
            "warn_score": 0.40,
        },
        total_tokens=1200,
        usage_trace={"summary": {}},
    )

    assert evaluation["status"] == "WARN"
    assert not evaluation["hard_failures"]
    assert evaluation["soft_failures"]


def test_evaluate_test_case_result_keeps_forbidden_tool_as_hard_failure():
    evaluation = evaluate_test_case_result(
        content="Gemini 2.5 最近有几项官方更新。",
        used_tools=["search_videos"],
        checks={
            "forbidden_tools": ["search_videos"],
            "min_content_length": 10,
        },
        total_tokens=500,
        usage_trace={"summary": {}},
    )

    assert evaluation["status"] == "FAIL"
    assert evaluation["hard_failures"] == ["unexpected tool 'search_videos' was used"]


def test_evaluate_test_case_result_rejects_runtime_error_content():
    evaluation = evaluate_test_case_result(
        content="Gemini 2.5 最近视频：\n[Error: 400 Client Error for url]",
        used_tools=["search_google"],
        checks={"min_content_length": 10},
        total_tokens=500,
        usage_trace={"summary": {}},
    )

    assert evaluation["status"] == "FAIL"
    assert evaluation["hard_failures"] == ["runtime error leaked into content"]
