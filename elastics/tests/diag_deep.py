"""Deep diagnostic: check actual KNN vector quality.

Investigates WHY KNN recall is low by:
1. Fetching documents with stored text_emb from ES
2. Generating fresh embeddings for document text and query text
3. Computing Hamming distances to verify embedding quality
4. Checking score distributions in KNN results
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from converters.embed.embed_client import TextEmbedClient
from configs.envs import ELASTIC_DEV_ENVS
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX


ES_ENVS = ELASTIC_DEV_ENVS
ES_INDEX = ELASTIC_VIDEOS_DEV_INDEX


def get_es_client():
    from elasticsearch import Elasticsearch

    return Elasticsearch(
        hosts=[{"host": ES_ENVS["host"], "port": ES_ENVS["port"], "scheme": "https"}],
        basic_auth=(ES_ENVS["username"], ES_ENVS["password"]),
        ca_certs=ES_ENVS.get("ca_certs"),
        verify_certs=bool(ES_ENVS.get("ca_certs")),
    )


def hex_to_bits(hex_str: str) -> list[int]:
    """Convert hex string to list of bits."""
    byte_data = bytes.fromhex(hex_str)
    bits = []
    for b in byte_data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def byte_array_to_bits(byte_array: list[int]) -> list[int]:
    """Convert signed int8 array (from ES) to list of bits."""
    bits = []
    for b_signed in byte_array:
        b = b_signed if b_signed >= 0 else b_signed + 256
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def hamming_distance(a, b) -> int:
    """Compute Hamming distance. a and b can be hex strings or byte arrays."""
    if isinstance(a, str):
        bits_a = hex_to_bits(a)
    else:
        bits_a = byte_array_to_bits(a)
    if isinstance(b, str):
        bits_b = hex_to_bits(b)
    else:
        bits_b = byte_array_to_bits(b)
    return sum(x != y for x, y in zip(bits_a, bits_b))


def hamming_similarity(a, b, total_bits: int = 2048) -> float:
    """Compute Hamming similarity = (total_bits - hamming_dist) / total_bits."""
    dist = hamming_distance(a, b)
    return (total_bits - dist) / total_bits


def check_hex_format(hex_str: str, label: str = ""):
    """Check basic properties of a hex string."""
    if not hex_str:
        print(f"  [{label}] EMPTY hex string!")
        return
    print(
        f"  [{label}] length={len(hex_str)} chars = {len(hex_str)//2} bytes = {len(hex_str)*4} bits"
    )
    # Check bit distribution (should be ~50% ones for random-ish LSH)
    bits = hex_to_bits(hex_str)
    ones = sum(bits)
    total = len(bits)
    print(f"  [{label}] bit distribution: {ones}/{total} ones ({ones/total*100:.1f}%)")


def check_emb_format(emb, label: str = ""):
    """Check basic properties of an embedding (hex string or byte array)."""
    if isinstance(emb, str):
        check_hex_format(emb, label)
    elif isinstance(emb, list):
        total_bits = len(emb) * 8
        bits = byte_array_to_bits(emb)
        ones = sum(bits)
        print(
            f"  [{label}] {len(emb)} bytes = {total_bits} bits, {ones}/{total_bits} ones ({ones/total_bits*100:.1f}%)"
        )
    else:
        print(f"  [{label}] Unknown type: {type(emb)}")


def diag_1_check_stored_vectors():
    """Check if text_emb vectors are actually stored and valid in ES."""
    print("=" * 60)
    print("DIAG 1: Check stored text_emb vectors in ES")
    print("=" * 60)

    es = get_es_client()
    index = ES_INDEX

    # Count docs with text_emb
    total_res = es.count(index=index)
    total = total_res["count"]
    print(f"\nTotal docs in index: {total}")

    has_emb_res = es.count(
        index=index, body={"query": {"exists": {"field": "text_emb"}}}
    )
    has_emb = has_emb_res["count"]
    print(f"Docs with text_emb: {has_emb} ({has_emb/total*100:.1f}%)")

    no_emb = total - has_emb
    print(f"Docs WITHOUT text_emb: {no_emb} ({no_emb/total*100:.1f}%)")

    # Fetch some docs WITH text_emb to check format
    res = es.search(
        index=index,
        body={
            "query": {"exists": {"field": "text_emb"}},
            "_source": ["title", "owner.name", "tags", "desc", "text_emb"],
            "size": 5,
        },
    )
    print(f"\nSample docs with text_emb:")
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        title = src.get("title", "")
        owner = src.get("owner", {}).get("name", "")
        text_emb = src.get("text_emb", "")
        emb_type = type(text_emb).__name__
        print(f"\n  bvid={hit['_id']}: {owner} - {title[:50]}")
        print(f"  text_emb type={emb_type}, repr[:80]={repr(text_emb)[:80]}")

        if isinstance(text_emb, str):
            check_hex_format(text_emb, "stored")
        elif isinstance(text_emb, list):
            print(f"  text_emb is list, len={len(text_emb)}, first 10: {text_emb[:10]}")


def diag_2_embedding_consistency():
    """Check if query embedding matches document embedding for same text."""
    print("\n" + "=" * 60)
    print("DIAG 2: Embedding consistency check")
    print("=" * 60)

    client = TextEmbedClient(lazy_init=False)

    # Test: embed the same text twice - should get same result
    test_text = "影视飓风 我为什么坚决反对在海鲜市场买海鲜"
    hex1 = client.text_to_hex(test_text, use_cache=False)
    hex2 = client.text_to_hex(test_text, use_cache=False)

    print(f"\nSame text embedded twice:")
    print(f"  text: {test_text}")
    check_hex_format(hex1, "attempt1")
    check_hex_format(hex2, "attempt2")
    if hex1 and hex2:
        dist = hamming_distance(hex1, hex2)
        sim = hamming_similarity(hex1, hex2)
        print(f"  Hamming distance: {dist} / 2048")
        print(f"  Hamming similarity: {sim:.4f}")
        if dist == 0:
            print("  ✓ DETERMINISTIC: Same text → same embedding")
        else:
            print(
                f"  ⚠ NON-DETERMINISTIC: Same text → different embedding (dist={dist})"
            )


def diag_3_query_vs_document_distance():
    """Check Hamming distance between query embedding and stored doc embedding."""
    print("\n" + "=" * 60)
    print("DIAG 3: Query vs Document embedding distance")
    print("=" * 60)

    es = get_es_client()
    index = ES_INDEX
    client = TextEmbedClient(lazy_init=False)

    queries = ["影视飓风", "deepseek", "原神"]

    for query in queries:
        print(f"\n--- Query: {query} ---")

        # Get query embedding
        query_hex = client.text_to_hex(query, use_cache=False)
        if not query_hex:
            print("  FAILED to get query embedding!")
            continue
        check_hex_format(query_hex, "query")

        # Use KNN to find closest docs, then verify distances manually
        query_vector = client.hex_to_byte_array(query_hex)
        knn_res = es.search(
            index=index,
            body={
                "knn": {
                    "field": "text_emb",
                    "query_vector": query_vector,
                    "k": 10,
                    "num_candidates": 100,
                },
                "_source": ["title", "owner.name", "tags", "desc", "text_emb"],
                "size": 10,
            },
        )

        print(f"\n  KNN top-10 hits + manual Hamming distance verification:")
        for hit in knn_res["hits"]["hits"]:
            src = hit["_source"]
            title = src.get("title", "")[:60]
            owner = src.get("owner", {}).get("name", "")
            text_emb = src.get("text_emb")
            score = hit["_score"]

            if not text_emb:
                print(f"  score={score:.6f} | {hit['_id']}: {owner} - {title}")
                print(f"    NO text_emb!")
                continue

            dist = hamming_distance(query_hex, text_emb)
            sim = hamming_similarity(query_hex, text_emb)
            print(
                f"  score={score:.6f} dist={dist}/2048 sim={sim:.4f} | {owner} - {title}"
            )

            # Also re-embed the document text and check distances
            doc_sentence = build_doc_sentence(
                title=src.get("title", ""),
                tags=src.get("tags", ""),
                desc=src.get("desc", ""),
                owner_name=src.get("owner", {}).get("name", ""),
            )
            fresh_doc_hex = client.text_to_hex(doc_sentence, use_cache=False)
            if fresh_doc_hex:
                stored_vs_fresh_dist = hamming_distance(text_emb, fresh_doc_hex)
                stored_vs_fresh_sim = hamming_similarity(text_emb, fresh_doc_hex)
                query_vs_fresh_dist = hamming_distance(query_hex, fresh_doc_hex)
                query_vs_fresh_sim = hamming_similarity(query_hex, fresh_doc_hex)
                print(
                    f"    Stored vs fresh-recomputed: dist={stored_vs_fresh_dist}, sim={stored_vs_fresh_sim:.4f}"
                )
                print(
                    f"    Query vs fresh-recomputed:  dist={query_vs_fresh_dist}, sim={query_vs_fresh_sim:.4f}"
                )
                if stored_vs_fresh_dist > 100:
                    print(
                        f"    ⚠ LARGE DRIFT between stored and re-computed embedding!"
                    )
                    print(f"    Doc sentence: {doc_sentence[:120]}")
            else:
                print(f"    Could not re-embed document text")

        # Also check distance to random docs for baseline
        random_res = es.search(
            index=index,
            body={
                "query": {
                    "function_score": {"query": {"match_all": {}}, "random_score": {}}
                },
                "_source": ["title", "owner.name", "text_emb"],
                "size": 5,
            },
        )
        print(f"\n  Random docs (baseline) distance to query:")
        for hit in random_res["hits"]["hits"]:
            src = hit["_source"]
            title = src.get("title", "")[:60]
            owner = src.get("owner", {}).get("name", "")
            text_emb = src.get("text_emb")
            if text_emb:
                dist = hamming_distance(query_hex, text_emb)
                sim = hamming_similarity(query_hex, text_emb)
                print(f"  dist={dist}/2048 sim={sim:.4f} | {owner} - {title}")


def build_doc_sentence(title="", tags="", desc="", owner_name=""):
    """Replicate the build_sentence logic from blux.text_doc."""
    sentence = ""
    owner_name_strip = (owner_name or "").strip()
    if owner_name_strip:
        sentence += f"【{owner_name_strip}】"
    title_strip = (title or "").strip()
    if title_strip:
        if sentence:
            sentence += " "
        sentence += title_strip
    tags_strip = (tags or "").strip()
    if tags_strip:
        if sentence:
            sentence += " "
        sentence += f"({tags_strip})"
    desc_strip = (desc or "").strip()
    if desc_strip and desc_strip != "-":
        if sentence:
            sentence += " "
        sentence += desc_strip
    return sentence


def diag_4_knn_score_distribution():
    """Check the distribution of KNN scores to understand the scoring."""
    print("\n" + "=" * 60)
    print("DIAG 4: KNN score distribution")
    print("=" * 60)

    es = get_es_client()
    index = ES_INDEX
    client = TextEmbedClient(lazy_init=False)

    query = "影视飓风"
    query_hex = client.text_to_hex(query, use_cache=False)
    if not query_hex:
        print("  FAILED to get query embedding!")
        return

    query_vector = client.hex_to_byte_array(query_hex)
    print(
        f"\n  Query vector length: {len(query_vector)} bytes = {len(query_vector)*8} bits"
    )
    print(f"  Query vector first 20 values: {query_vector[:20]}")

    # KNN search
    knn_res = es.search(
        index=index,
        body={
            "knn": {
                "field": "text_emb",
                "query_vector": query_vector,
                "k": 20,
                "num_candidates": 200,
            },
            "_source": ["title", "owner.name", "text_emb"],
            "size": 20,
        },
    )

    print(f"\n  KNN search results (k=20):")
    scores = []
    for hit in knn_res["hits"]["hits"]:
        src = hit["_source"]
        title = src.get("title", "")[:50]
        owner = src.get("owner", {}).get("name", "")
        score = hit["_score"]
        scores.append(score)

        text_emb = src.get("text_emb")
        if text_emb and query_hex:
            dist = hamming_distance(query_hex, text_emb)
            sim = hamming_similarity(query_hex, text_emb)
            print(
                f"  score={score:.6f} dist={dist}/2048 sim={sim:.4f} | {owner} - {title}"
            )
        else:
            print(f"  score={score:.6f} NO_EMB | {owner} - {title}")

    if scores:
        print(
            f"\n  Score stats: min={min(scores):.6f} max={max(scores):.6f} "
            f"range={max(scores)-min(scores):.6f} "
            f"mean={sum(scores)/len(scores):.6f}"
        )

    # Also do an EXPLAIN query on the top result to understand score formula
    if knn_res["hits"]["hits"]:
        top_id = knn_res["hits"]["hits"][0]["_id"]
        explain_res = es.search(
            index=index,
            body={
                "knn": {
                    "field": "text_emb",
                    "query_vector": query_vector,
                    "k": 1,
                    "num_candidates": 10,
                },
                "_source": False,
                "size": 1,
                "explain": True,
            },
        )
        if explain_res["hits"]["hits"]:
            explain = explain_res["hits"]["hits"][0].get("_explanation", {})
            print(f"\n  Score explanation for top hit:")
            print_explain(explain, indent=4)


def print_explain(explain: dict, indent: int = 0):
    """Pretty print ES explanation."""
    prefix = " " * indent
    desc = explain.get("description", "")
    value = explain.get("value", "")
    print(f"{prefix}{value} : {desc}")
    for detail in explain.get("details", []):
        print_explain(detail, indent + 2)


def diag_5_byte_array_format():
    """Check if byte array format matches what ES expects for bit vectors."""
    print("\n" + "=" * 60)
    print("DIAG 5: Byte array format check")
    print("=" * 60)

    client = TextEmbedClient(lazy_init=False)
    hex_str = client.text_to_hex("test", use_cache=False)
    if not hex_str:
        print("  FAILED to get hex!")
        return

    byte_array = client.hex_to_byte_array(hex_str)
    raw_bytes = bytes.fromhex(hex_str)

    print(f"\n  Hex string length: {len(hex_str)} chars")
    print(f"  Raw bytes length: {len(raw_bytes)} bytes")
    print(f"  Byte array length: {len(byte_array)} elements")
    print(f"  Expected for 2048-bit: 256 bytes / 512 hex chars")

    # Check conversion consistency
    # hex_to_byte_array should produce signed int8 values
    for i in range(min(10, len(raw_bytes))):
        unsigned = raw_bytes[i]
        signed = byte_array[i]
        expected = unsigned if unsigned < 128 else unsigned - 256
        match = "✓" if signed == expected else "✗"
        print(
            f"  byte[{i}]: unsigned={unsigned:3d} signed={signed:4d} expected={expected:4d} {match}"
        )

    # Key check: do the bits match?
    hex_bits = hex_to_bits(hex_str)
    byte_bits = []
    for b_signed in byte_array:
        b_unsigned = b_signed if b_signed >= 0 else b_signed + 256
        for i in range(7, -1, -1):
            byte_bits.append((b_unsigned >> i) & 1)

    if hex_bits == byte_bits:
        print(f"\n  ✓ Bit representation matches between hex and byte array")
    else:
        diffs = sum(1 for a, b in zip(hex_bits, byte_bits) if a != b)
        print(f"\n  ✗ BIT MISMATCH! {diffs} bits differ!")


if __name__ == "__main__":
    diag_1_check_stored_vectors()
    diag_2_embedding_consistency()
    diag_3_query_vs_document_distance()
    diag_4_knn_score_distribution()
    diag_5_byte_array_format()

    # python -m elastics.tests.diag_deep
