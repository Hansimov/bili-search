"""Compare q=vw vs q=vwr search quality on a set of representative queries."""

import json
import requests

BASE_URL = "http://127.0.0.1:21001"

TEST_QUERIES = [
    "Python 深度学习入门",
    "黑神话悟空 攻略",
    "影视飓风 最近视频",
    "GPT-5 最新消息",
    "做饭教程 新手",
    "独立游戏 推荐",
    "数码测评 手机",
    "AI 绘画 Stable Diffusion",
]


def run_search(query: str, qmod: str) -> dict:
    full_query = f"{query} q={qmod}"
    resp = requests.post(
        f"{BASE_URL}/explore",
        json={"query": full_query},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def extract_top_hits(result: dict, n: int = 5) -> list[dict]:
    data = result.get("data", [])
    if not data:
        return []
    output = data[0].get("output", {})
    hits = output.get("hits", [])
    top = []
    for h in hits[:n]:
        top.append(
            {
                "bvid": h.get("bvid", ""),
                "title": h.get("title", "")[:60],
                "owner": h.get("owner", {}).get("name", ""),
                "view": h.get("stat", {}).get("view", 0),
                "score": round(h.get("score", 0), 4),
                "rank_score": round(h.get("rank_score", 0), 4),
            }
        )
    return top


def extract_perf(result: dict) -> dict:
    data = result.get("data", [])
    if not data:
        return {}
    output = data[0].get("output", {})
    perf = output.get("perf", {})
    return {
        "total_ms": perf.get("total_ms", "?"),
        "recall_ms": perf.get("recall_ms", "?"),
        "rank_ms": perf.get("rank_ms", "?"),
        "rerank_ms": perf.get("rerank_ms", "?"),
        "total_hits": output.get("total_hits", "?"),
        "return_hits": output.get("return_hits", "?"),
        "rank_method": output.get("rank_method", "?"),
        "qmod": output.get("qmod", "?"),
    }


def main():
    for query in TEST_QUERIES:
        print(f"\n{'='*80}")
        print(f"  QUERY: {query}")
        print(f"{'='*80}")

        for qmod in ["vw", "vwr"]:
            try:
                result = run_search(query, qmod)
            except Exception as e:
                print(f"  [{qmod}] ERROR: {e}")
                continue

            perf = extract_perf(result)
            hits = extract_top_hits(result, n=5)

            print(f"\n  --- q={qmod} ---")
            print(
                f"  perf: total={perf.get('total_ms')}ms  recall={perf.get('recall_ms')}ms  "
                f"rank={perf.get('rank_ms')}ms  rerank={perf.get('rerank_ms')}ms"
            )
            print(
                f"  total_hits={perf.get('total_hits')}  return={perf.get('return_hits')}  "
                f"rank_method={perf.get('rank_method')}  qmod={perf.get('qmod')}"
            )
            for i, h in enumerate(hits, 1):
                print(f"    {i}. [{h['bvid']}] {h['title']}")
                print(
                    f"       by {h['owner']}  view={h['view']}  score={h['score']}  rank={h['rank_score']}"
                )


if __name__ == "__main__":
    main()
