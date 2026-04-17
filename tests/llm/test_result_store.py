from llms.contracts import ToolCallRequest, ToolExecutionRecord
from llms.orchestration.result_store import ResultStore, summarize_result


def test_summarize_result_keeps_full_run_small_llm_task_output():
    result = {
        "task": "把视频转写压成 5 条要点",
        "model": "doubao-seed-2-0-mini",
        "result": "\n".join(
            [
                "这期视频主要展示影视飓风员工包里的随身物品。",
                "- 黛黛分享穿搭相关帽子、头巾和项链。",
                "- 奥斯卡展示 6x8 画幅相机与四拉片。",
                "- 主持人天介绍生活包和工作包。",
                "- 后半段继续展示老白、OO、大臣等人的物品。",
            ]
        ),
    }

    summary = summarize_result("R2", "run_small_llm_task", result)

    assert summary["tool"] == "run_small_llm_task"
    assert summary["result_text"] == result["result"]
    assert "后半段继续展示老白" in summary["summary_text"]


def test_render_observation_indents_multiline_small_task_output():
    store = ResultStore()
    record = ToolExecutionRecord(
        result_id="R2",
        request=ToolCallRequest(
            id="call_small_1",
            name="run_small_llm_task",
            arguments={"task": "总结要点"},
            visibility="internal",
        ),
        result={
            "task": "总结要点",
            "result": "主题句\n- 要点1\n- 要点2",
        },
        summary={
            "summary_text": "主题句\n- 要点1\n- 要点2",
        },
        visibility="internal",
    )
    store.add(record)

    observation = store.render_observation(["R2"])

    assert "- R2 run_small_llm_task: 主题句\n  - 要点1\n  - 要点2" in observation
