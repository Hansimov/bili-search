"""Test owner intent strength computation."""

from ranks.diversified import DiversifiedRanker
import re

# Simulate owner intent for 红警08
hits = []
for i in range(50):
    if i < 30:
        hits.append({"owner": {"name": "红警HBK08"}, "title": f"红警08 test {i}"})
    elif i < 35:
        hits.append({"owner": {"name": "红警V神"}, "title": f"红警 v{i}"})
    elif i < 40:
        hits.append({"owner": {"name": "红警魔鬼蓝天"}, "title": f"红警08 test {i}"})
    else:
        hits.append({"owner": {"name": f"user{i}"}, "title": f"test {i}"})

query_lower = "红警08"
query_cjk = re.sub(r"[^\u4e00-\u9fff]", "", query_lower)
q_terms = [t for t in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", query_lower) if t]
print(f"q_terms={q_terms}, query_cjk={query_cjk}")

# Manual check
for name in ["红警HBK08", "红警V神", "红警魔鬼蓝天"]:
    name_lower = name.lower()
    name_tokens = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", name_lower))
    all_in = all(t in name_lower for t in q_terms)
    subset = set(q_terms).issubset(name_tokens)
    print(f"  {name}: tokens={name_tokens}, all_in={all_in}, subset={subset}")

strength = DiversifiedRanker._analyze_owner_intent_strength(
    hits, query_lower, query_cjk, q_terms
)
print(f"Owner intent strength for 红警08: {strength}")

# Simulate 米娜
hits2 = []
owners = [
    "伊朗女人米娜",
    "大聪明罗米娜",
    "米娜Minana呀",
    "米娜那",
    "甜心米娜",
    "小米娜",
]
for i in range(60):
    owner = owners[i % len(owners)]
    hits2.append({"owner": {"name": owner}, "title": f"米娜 content {i}"})
for i in range(40):
    hits2.append({"owner": {"name": f"user{i}"}, "title": f"content {i} 米娜"})

query_lower2 = "米娜"
query_cjk2 = re.sub(r"[^\u4e00-\u9fff]", "", query_lower2)
q_terms2 = [t for t in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", query_lower2) if t]
strength2 = DiversifiedRanker._analyze_owner_intent_strength(
    hits2, query_lower2, query_cjk2, q_terms2
)
print(f"Owner intent strength for 米娜: {strength2}")
