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


def test_summarize_video_results_keeps_tags_as_evidence():
    result = {
        "results": [
            {
                "query": "红警HBK08 月亮3 决赛对局",
                "total_hits": 1,
                "hits": [
                    {
                        "title": "2026年4月16日晚 红警HBK08直播回放",
                        "bvid": "BV1jmdvBYEPr",
                        "owner": {"mid": 284671271, "name": "SleepTight睡个好觉"},
                        "tags": "红色警戒2,红色警戒,HBK08,红警08,红警月亮3",
                        "stat": {"view": 2730},
                    }
                ],
            }
        ]
    }

    summary = summarize_result("R1", "search_videos", result)

    assert "tags=红色警戒2,红色警戒,HBK08,红警08,红警月亮3" in summary["summary_text"]
    assert summary["queries"][0]["top_hits"][0]["tags"] == (
        "红色警戒2,红色警戒,HBK08,红警08,红警月亮3"
    )


def test_summarize_mid_lookup_marks_owner_mid_as_match_basis():
    result = {
        "mode": "lookup",
        "lookup_by": "mids",
        "mids": ["267298820", "3546906370247281"],
        "total_hits": 2,
        "hits": [
            {
                "title": "玩机器深夜闲聊解答世间万物第五期",
                "bvid": "BV1FroRBAEmG",
                "owner": {"mid": 267298820, "name": "勤奋的yoke"},
                "tags": "杂谈,直播切片",
                "stat": {"view": 2571},
            },
            {
                "title": "玩机器直言自己会理光头的",
                "bvid": "BV1zpo9BxEPP",
                "owner": {"mid": 3546906370247281, "name": "懒惰的yoke"},
                "tags": "解说,直播切片",
                "stat": {"view": 3957},
            },
        ],
    }

    summary = summarize_result("R1", "search_videos", result)

    assert summary["match_basis"] == "owner_mid"
    assert "match_basis=owner_mid" in summary["summary_text"]
    assert "owner_groups=勤奋的yoke" in summary["summary_text"]
    assert "玩机器直言自己会理光头的(BV1zpo9BxEPP)" in summary["summary_text"]
    assert summary["owner_groups"][0]["owner"] == "勤奋的yoke"
    assert summary["owner_groups"][1]["owner_mid"] == 3546906370247281


def test_summarize_transcript_keeps_page_part_title():
    result = {
        "bvid": "BV1jmdvBYEPr",
        "title": "2026年4月16日晚 红警HBK08直播回放",
        "page_index": 1,
        "page": {
            "page": 1,
            "part": "第一部分：08 红警阿V vs 月亮3 国米  2V2 抢7",
        },
        "selection": {"selected_text_length": 8000, "full_text_length": 22157},
        "transcript": {
            "text": "我们打个抢七吧。",
            "segment_count": 52,
        },
    }

    summary = summarize_result("R1", "get_video_transcript", result)

    assert summary["part"] == "第一部分：08 红警阿V vs 月亮3 国米  2V2 抢7"
    assert "part=第一部分：08 红警阿V vs 月亮3 国米  2V2 抢7" in summary["summary_text"]
