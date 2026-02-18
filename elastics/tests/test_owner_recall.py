"""Quick test for owner-focused recall debugging."""

from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV

s = VideoSearcherV2(index_name=ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV)

# Test 1: Direct search with owner filter
owner_filter = {
    "bool": {
        "should": [
            {"term": {"owner.name.keyword": "红警HBK08"}},
            {"match_phrase": {"owner.name": "红警HBK08"}},
            {"match": {"owner.name": "红警HBK08"}},
        ],
        "minimum_should_match": 1,
    }
}

print("=== Test 1: search(query='红警08', owner_filter=红警HBK08) ===")
res = s.search(
    query="红警08",
    extra_filters=[owner_filter],
    parse_hits=True,
    add_region_info=False,
    add_highlights_info=False,
    is_highlight=False,
    boost=False,
    rank_method="heads",
    limit=10,
    rank_top_k=10,
    timeout=5,
    verbose=True,
)
hits = res.get("hits", [])
print(f"Results: {len(hits)}")
for h in hits[:5]:
    print(f"  {h.get('title', '')} | UP: {h.get('owner', {}).get('name', '')}")

# Test 2: Just the owner filter with match_all
print("\n=== Test 2: Raw ES query with owner filter ===")
body = {
    "query": {"bool": {"filter": [{"term": {"owner.name.keyword": "红警HBK08"}}]}},
    "_source": ["bvid", "title", "owner.name"],
    "size": 5,
}
raw = s.submit_to_es(body, context="test")
raw_hits = raw.get("hits", {}).get("hits", [])
print(f"Raw results: {len(raw_hits)}")
for h in raw_hits:
    src = h.get("_source", {})
    print(f"  {src.get('title', '')} | UP: {src.get('owner', {}).get('name', '')}")

# Test 3: filter_only_search
print("\n=== Test 3: filter_only_search with owner filter ===")
try:
    fres = s.filter_only_search(
        query="",
        extra_filters=[owner_filter],
        parse_hits=True,
        add_region_info=False,
        add_highlights_info=False,
        limit=10,
        timeout=5,
        verbose=True,
    )
    fhits = fres.get("hits", [])
    print(f"Filter-only results: {len(fhits)}")
    for h in fhits[:5]:
        print(f"  {h.get('title', '')} | UP: {h.get('owner', {}).get('name', '')}")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
