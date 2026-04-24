from __future__ import annotations


STEP_ZH_NAMES = {
    "most_relevant_search": {
        "name_zh": "搜索相关",
        "output_type": "hits",
    },
    "knn_search": {
        "name_zh": "向量搜索",
        "output_type": "hits",
    },
    "rerank": {
        "name_zh": "精排重排",
        "output_type": "info",
    },
    "hybrid_search": {
        "name_zh": "混合搜索",
        "output_type": "hits",
    },
    "group_hits_by_owner": {
        "name_zh": "UP主聚合",
        "output_type": "info",
    },
}


class StepBuilder:
    """Helper to build step result dicts with consistent format."""

    def __init__(self):
        self.steps: list[dict] = []
        self.step_idx: int = -1
        self.final_status: str = "finished"

    def add_step(
        self,
        name: str,
        status: str = "finished",
        input_data: dict = None,
        output: dict = None,
        comment: str = "",
    ) -> dict:
        self.step_idx += 1
        step = {
            "step": self.step_idx,
            "name": name,
            "name_zh": STEP_ZH_NAMES.get(name, {}).get("name_zh", name),
            "status": status,
            "input": input_data or {},
            "output_type": STEP_ZH_NAMES.get(name, {}).get("output_type", "info"),
            "output": output or {},
            "comment": comment,
        }
        self.steps.append(step)
        return step

    def update_step(self, step: dict, output: dict, status: str = "finished"):
        step["output"] = output
        if isinstance(output, dict) and output.get("timed_out"):
            step["status"] = "timedout"
            self.final_status = "timedout"
        else:
            step["status"] = status

    def finalize(self, query: str, **extra) -> dict:
        result = {
            "query": query,
            "status": self.final_status,
            "data": self.steps,
        }
        result.update(extra)
        return result
