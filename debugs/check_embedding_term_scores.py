from converters.embed.embed_client import get_embed_client


def main():
    client = get_embed_client()
    query = "显卡"
    terms = [
        "gpu",
        "GPU",
        "英伟达",
        "NVIDIA",
        "芯片",
        "算力",
        "服务器",
        "洗地机",
        "H20Ultra",
        "机械键盘",
        "游戏",
        "火龙",
    ]
    rankings = client.rerank(query, terms)
    scored = []
    for term, (_, score) in zip(terms, rankings):
        scored.append((score, term))
    for score, term in sorted(scored, reverse=True):
        print(f"{score:.6f}\t{term}")


if __name__ == "__main__":
    main()
