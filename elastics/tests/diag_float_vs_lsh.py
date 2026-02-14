"""Diagnostic: Compare float-vector vs LSH-vector rankings.

Tests whether the low KNN overlap is due to LSH compression
or due to the embedding model's semantics.

If float rankings differ from LSH rankings → LSH is the problem.
If float rankings match LSH rankings → embedding model is the problem.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from converters.embed.embed_client import TextEmbedClient


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


def hex_to_bits(hex_str: str) -> list[int]:
    byte_data = bytes.fromhex(hex_str)
    bits = []
    for b in byte_data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def hamming_similarity(hex1: str, hex2: str, total_bits: int = 2048) -> float:
    bits1 = hex_to_bits(hex1)
    bits2 = hex_to_bits(hex2)
    dist = sum(b1 != b2 for b1, b2 in zip(bits1, bits2))
    return (total_bits - dist) / total_bits


def compare_float_vs_lsh():
    """Compare float cosine similarity with LSH hamming similarity."""
    client = TextEmbedClient(lazy_init=False)

    # Test queries
    queries = ["影视飓风", "deepseek", "原神"]

    # For each query, compare against relevant and irrelevant docs
    test_cases = {
        "影视飓风": {
            "UP主_videos": [
                "【影视飓风】 你的相册正在泄露你的秘密 (科技,数码,影视飓风,Yige,Tim) 你的相册正在泄露你的秘密",
                "【影视飓风】 日本超市vs中国超市 (日常,vlog,日本超市,中国超市,对比) 中国超市和日本超市到底有什么不同",
                "【影视飓风】 我花了几百万，给你们建了一个工作室 (科技,工作室,影视飓风) 花了几百万装修",
                "【影视飓风】 不要satisfying！我们才是强迫症的最终解决方案! (科技,强迫症,satisfying)",
            ],
            "hurricane_videos": [
                "【ItIsEnderman】 【闲聊飓风】迷你飓风费利西娅 巅峰及前后部分时段 (飓风,气象,风暴)",
                "【Transylvaniaball】 历来最强飓风 (飓风,气象,自然灾害)",
                "【螺旋老DJ】 这属于龙卷风吗？ (龙卷风,风,天气)",
                "【军立方】 宇宙最强的两个飓风 (飓风,宇宙,科普)",
            ],
        },
        "deepseek": {
            "deepseek_videos": [
                "【量子位】 DeepSeek R1 开源发布 (AI,DeepSeek,大模型,LLM) DeepSeek正式开源R1推理模型",
                "【机器之心】 DeepSeek V3技术报告解读 (AI,LLM,DeepSeek) 详解DeepSeek V3的技术细节",
                "【二次元的中科院物理所】 DeepSeek是什么 (科普,AI,DeepSeek)",
            ],
            "other_llm_videos": [
                "【samwitteveen】 ChatGPT API Announcement & Code Walkthrough with LangChain (ChatGPT,API,LangChain)",
                "【coolcloud86】 ChatGPT about Business (ChatGPT,AI,business)",
                "【洛河神茶007】 Google Gemini (Google,Gemini,AI)",
            ],
        },
        "原神": {
            "genshin_videos": [
                "【原神】 丽莎角色演示 (原神,Genshin Impact,角色,丽莎) 雷元素法器角色丽莎",
                "【原神攻略】 深渊12层全三星攻略 (原神,深渊,攻略) 最新深渊攻略",
                "【半夜是游戏】 原神4.0版本前瞻 (原神,游戏,枫丹)",
            ],
            "phone_benchmark": [
                "【风中的老梧桐】 鸿蒙os下华为matepad Pro 玩原神 (华为,鸿蒙,平板) 麒麟990原神帧数测试",
                "【乃贝上大分】 属于米粉的Nova11se来了 (华为,手机,评测)",
            ],
        },
    }

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # Get query embeddings
        query_hex = client.text_to_hex(query, use_cache=False)
        query_float = client.embed([query])[0]

        if not query_hex or not query_float:
            print("  FAILED to get embeddings!")
            continue

        print(f"  Float embedding dim: {len(query_float)}")
        print(f"  LSH dim: {len(query_hex)*4} bits")

        cases = test_cases.get(query, {})
        all_docs = []
        all_labels = []
        for category, docs in cases.items():
            for doc in docs:
                all_docs.append(doc)
                all_labels.append(category)

        if not all_docs:
            continue

        # Get float embeddings for all docs
        doc_floats = client.embed(all_docs)
        # Get LSH embeddings for all docs
        doc_hexes = client.texts_to_hex(all_docs)

        print(
            f"\n  {'Category':<25} {'Float cos':>10} {'LSH sim':>10} {'Δ':>8} | Doc text"
        )
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} | {'-'*60}")

        results = []
        for i, (doc, label) in enumerate(zip(all_docs, all_labels)):
            cos_sim = (
                cosine_similarity(query_float, doc_floats[i]) if doc_floats[i] else 0
            )
            ham_sim = hamming_similarity(query_hex, doc_hexes[i]) if doc_hexes[i] else 0
            delta = cos_sim - ham_sim
            results.append((label, cos_sim, ham_sim, delta, doc[:70]))

        # Sort by float cosine similarity
        results.sort(key=lambda x: -x[1])

        for label, cos_sim, ham_sim, delta, doc in results:
            print(
                f"  {label:<25} {cos_sim:>10.4f} {ham_sim:>10.4f} {delta:>+8.4f} | {doc}"
            )

        # Check ranking consistency
        float_ranking = [r[4] for r in sorted(results, key=lambda x: -x[1])]
        lsh_ranking = [r[4] for r in sorted(results, key=lambda x: -x[2])]

        print(f"\n  Float rank:  {[r[:30] for r in float_ranking]}")
        print(f"  LSH rank:    {[r[:30] for r in lsh_ranking]}")

        # Compute rank correlation
        float_scores = {r[4]: r[1] for r in results}
        lsh_scores = {r[4]: r[2] for r in results}

        # Check if the top-ranked categories match
        float_top_cat = [r[0] for r in sorted(results, key=lambda x: -x[1])][:3]
        lsh_top_cat = [r[0] for r in sorted(results, key=lambda x: -x[2])][:3]
        print(f"\n  Float top-3 categories: {float_top_cat}")
        print(f"  LSH top-3 categories:   {lsh_top_cat}")


if __name__ == "__main__":
    compare_float_vs_lsh()

    # python -m elastics.tests.diag_float_vs_lsh
