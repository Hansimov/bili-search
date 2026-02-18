"""Test KNN search with es_tok_constraints filtering.

Tests 5 queries with various constraint configurations to validate
the es-tok plugin's constraint system for KNN pre-filtering.

Test queries:
1. 飓风营救 - movie/rescue topic
2. 通义实验室 - AI lab topic
3. 红警08 - game topic (Red Alert + number)
4. 小红书推荐系统 - recommendation system topic
5. 吴恩达大模型 - Andrew Ng + large models topic
"""

from tclogger import logger, logstr, dict_to_str

from elastics.structure import construct_constraint_filter
from elastics.videos.constants import ELASTIC_VIDEOS_DEV_INDEX, ELASTIC_DEV
from elastics.videos.searcher_v2 import VideoSearcherV2
from elastics.videos.explorer import VideoExplorer

# The es_tok tokenizer indexes into .words sub-fields.
# Constraint fields must target these sub-fields, not the base fields.
TITLE = "title.words"
TAGS = "tags.words"
DESC = "desc.words"
OWNER = "owner.name.words"
TITLE_TAGS = [TITLE, TAGS]
TITLE_TAGS_OWNER = [TITLE, TAGS, OWNER]

# Shared instances (lazy-initialized)
_searcher = None
_explorer = None


def get_searcher() -> VideoSearcherV2:
    global _searcher
    if _searcher is None:
        _searcher = VideoSearcherV2(
            ELASTIC_VIDEOS_DEV_INDEX, elastic_env_name=ELASTIC_DEV
        )
    return _searcher


def get_explorer() -> VideoExplorer:
    global _explorer
    if _explorer is None:
        _explorer = VideoExplorer(
            index_name=ELASTIC_VIDEOS_DEV_INDEX,
            elastic_env_name=ELASTIC_DEV,
        )
    return _explorer


def format_hit(hit: dict, idx: int = 0) -> str:
    """Format a single hit for display."""
    title = hit.get("title", "")
    bvid = hit.get("bvid", "")
    owner = hit.get("owner", {})
    owner_name = owner.get("name", "") if isinstance(owner, dict) else ""
    tags = hit.get("tags", "")
    score = hit.get("score", 0)
    stat = hit.get("stat", {})
    view = stat.get("view", 0) if isinstance(stat, dict) else 0
    return (
        f"  [{idx+1:2d}] score={score:.4f} view={view:>8d} "
        f"| {title} | @{owner_name} | tags={tags[:60]}"
    )


def print_hits(hits: list[dict], max_show: int = 10, label: str = ""):
    """Print formatted hit results."""
    if label:
        logger.note(f"  {label}: {len(hits)} hits")
    for i, hit in enumerate(hits[:max_show]):
        logger.mesg(format_hit(hit, i))
    if len(hits) > max_show:
        logger.hint(f"  ... and {len(hits) - max_show} more hits")


def run_knn_search(
    searcher: VideoSearcherV2,
    query: str,
    constraint_filter: dict = None,
    label: str = "",
    k: int = 100,
    limit: int = 20,
) -> list[dict]:
    """Run a KNN search and return hits."""
    logger.note(f"\n>>> KNN: [{query}] {label}")
    if constraint_filter:
        logger.hint(f"  constraint: {dict_to_str(constraint_filter)}")

    res = searcher.knn_search(
        query=query,
        source_fields=["bvid", "title", "tags", "owner", "stat", "pubdate"],
        constraint_filter=constraint_filter,
        k=k,
        num_candidates=10000,
        parse_hits=True,
        add_region_info=False,
        skip_ranking=True,
        limit=limit,
        timeout=8.0,
        verbose=False,
    )
    hits = res.get("hits", [])
    total = res.get("total_hits", 0)
    timed_out = res.get("timed_out", False)
    logger.success(f"  total_hits={total}, returned={len(hits)}, timed_out={timed_out}")
    print_hits(hits, max_show=10)
    return hits


def run_explore(
    explorer: VideoExplorer,
    query: str,
    constraint_filter: dict = None,
    mode: str = "v",
    label: str = "",
) -> list[dict]:
    """Run an explore search and extract the main step's hits."""
    full_query = f"{query} q={mode}"
    logger.note(f"\n>>> Explore: [{full_query}] {label}")
    if constraint_filter:
        logger.hint(f"  constraint: {dict_to_str(constraint_filter)}")

    res = explorer.unified_explore(
        query=full_query,
        constraint_filter=constraint_filter,
        verbose=False,
    )

    # Extract hits from the main search step
    hits = []
    step_name = ""
    for step in res.get("data", []):
        name = step.get("name", "")
        if name in ("knn_search", "hybrid_search", "most_relevant_search"):
            output = step.get("output", {})
            hits = output.get("hits", [])
            step_name = name
            break

    total = 0
    for step in res.get("data", []):
        output = step.get("output", {})
        if "total_hits" in output:
            total = output["total_hits"]
            break

    logger.success(f"  [{step_name}] total_hits={total}, returned={len(hits)}")
    print_hits(hits, max_show=10)
    return hits


# =====================================================================
# Test cases for each query
# =====================================================================


def test_飓风营救():
    """Test: 飓风营救 (Hurricane Rescue)

    Expected: Videos about the movie/rescue topic.
    Tokens: 飓风, 飓风营救, 营救
    """
    searcher = get_searcher()
    query = "飓风营救"

    logger.file(f"\n{'='*60}")
    logger.file(f"TEST: {query}")
    logger.file(f"{'='*60}")

    hits_baseline = run_knn_search(searcher, query, label="[baseline]")

    # Constraint 1: must have token "飓风营救" in title.words
    cf1 = construct_constraint_filter(
        constraints=[{"have_token": ["飓风营救"], "fields": [TITLE]}],
    )
    hits_c1 = run_knn_search(
        searcher, query, constraint_filter=cf1, label='[have_token "飓风营救" in title]'
    )

    # Constraint 2: must have token "飓风营救" in title or tags
    cf2 = construct_constraint_filter(
        constraints=[{"have_token": ["飓风营救"]}],
        fields=TITLE_TAGS,
    )
    hits_c2 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf2,
        label='[have_token "飓风营救" in title+tags]',
    )

    # Constraint 3: with_prefixes "飓风" in title
    cf3 = construct_constraint_filter(
        constraints=[{"with_prefixes": ["飓风"], "fields": [TITLE]}],
    )
    hits_c3 = run_knn_search(
        searcher, query, constraint_filter=cf3, label='[with_prefixes "飓风" in title]'
    )

    # Constraint 4: must have "飓风" prefix AND "营救" prefix
    cf4 = construct_constraint_filter(
        constraints=[
            {"with_prefixes": ["飓风"], "fields": [TITLE]},
            {"with_prefixes": ["营救"], "fields": TITLE_TAGS},
        ],
    )
    hits_c4 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf4,
        label='[prefix "飓风" in title AND "营救" in title+tags]',
    )

    # Constraint 5: OR - either "飓风营救" or "飓风" prefix
    cf5 = construct_constraint_filter(
        constraints=[
            {
                "OR": [{"have_token": ["飓风营救"]}, {"with_prefixes": ["飓风"]}],
                "fields": TITLE_TAGS,
            },
        ],
    )
    hits_c5 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf5,
        label='[OR: token "飓风营救" / prefix "飓风"]',
    )

    logger.file(f"\nSUMMARY for {query}:")
    logger.mesg(f"  Baseline:        {len(hits_baseline)} hits")
    logger.mesg(f"  have_token(title): {len(hits_c1)} hits")
    logger.mesg(f"  have_token(t+t):   {len(hits_c2)} hits")
    logger.mesg(f"  prefix(title):     {len(hits_c3)} hits")
    logger.mesg(f"  prefix+prefix:     {len(hits_c4)} hits")
    logger.mesg(f"  OR(token/prefix):  {len(hits_c5)} hits")


def test_通义实验室():
    """Test: 通义实验室 (Tongyi Lab)

    Tokens: 通义, 通义实验室, 义实, 实验, 实验室, 验室
    """
    searcher = get_searcher()
    query = "通义实验室"

    logger.file(f"\n{'='*60}")
    logger.file(f"TEST: {query}")
    logger.file(f"{'='*60}")

    hits_baseline = run_knn_search(searcher, query, label="[baseline]")

    cf1 = construct_constraint_filter(
        constraints=[{"have_token": ["通义实验室"]}],
        fields=TITLE_TAGS,
    )
    hits_c1 = run_knn_search(
        searcher, query, constraint_filter=cf1, label='[have_token "通义实验室"]'
    )

    cf2 = construct_constraint_filter(
        constraints=[{"with_prefixes": ["通义"]}],
        fields=TITLE_TAGS,
    )
    hits_c2 = run_knn_search(
        searcher, query, constraint_filter=cf2, label='[with_prefixes "通义"]'
    )

    cf3 = construct_constraint_filter(
        constraints=[
            {"with_prefixes": ["通义"], "fields": TITLE_TAGS},
            {"with_contains": ["实验"], "fields": TITLE_TAGS},
        ],
    )
    hits_c3 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf3,
        label='[prefix "通义" AND contains "实验"]',
    )

    cf4 = construct_constraint_filter(
        constraints=[
            {"OR": [{"have_token": ["通义实验室"]}, {"with_prefixes": ["通义"]}]},
        ],
        fields=TITLE_TAGS,
    )
    hits_c4 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf4,
        label='[OR: token "通义实验室" / prefix "通义"]',
    )

    logger.file(f"\nSUMMARY for {query}:")
    logger.mesg(f"  Baseline:          {len(hits_baseline)} hits")
    logger.mesg(f"  have_token:        {len(hits_c1)} hits")
    logger.mesg(f"  prefix:            {len(hits_c2)} hits")
    logger.mesg(f"  prefix+contains:   {len(hits_c3)} hits")
    logger.mesg(f"  OR(token/prefix):  {len(hits_c4)} hits")


def test_红警08():
    """Test: 红警08 (Red Alert 08)

    Tokens: 红警, 红警08
    """
    searcher = get_searcher()
    query = "红警08"

    logger.file(f"\n{'='*60}")
    logger.file(f"TEST: {query}")
    logger.file(f"{'='*60}")

    hits_baseline = run_knn_search(searcher, query, label="[baseline]")

    cf1 = construct_constraint_filter(
        constraints=[{"have_token": ["红警08"]}],
        fields=TITLE_TAGS,
    )
    hits_c1 = run_knn_search(
        searcher, query, constraint_filter=cf1, label='[have_token "红警08"]'
    )

    cf2 = construct_constraint_filter(
        constraints=[{"have_token": ["红警08"], "fields": [OWNER]}],
    )
    hits_c2 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf2,
        label='[have_token "红警08" in owner.name]',
    )

    cf3 = construct_constraint_filter(
        constraints=[{"with_prefixes": ["红警"]}],
        fields=TITLE_TAGS,
    )
    hits_c3 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf3,
        label='[with_prefixes "红警" in title+tags]',
    )

    cf4 = construct_constraint_filter(
        constraints=[
            {
                "OR": [
                    {"with_prefixes": ["红警"]},
                    {"have_token": ["红警08"]},
                ],
                "fields": [TITLE, TAGS, OWNER],
            },
        ],
    )
    hits_c4 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf4,
        label='[OR: prefix "红警" / token "红警08" in t+t+o]',
    )

    cf5 = construct_constraint_filter(
        constraints=[{"with_contains": ["红警"]}],
        fields=TITLE_TAGS,
    )
    hits_c5 = run_knn_search(
        searcher, query, constraint_filter=cf5, label='[with_contains "红警"]'
    )

    logger.file(f"\nSUMMARY for {query}:")
    logger.mesg(f"  Baseline:            {len(hits_baseline)} hits")
    logger.mesg(f"  have_token exact:    {len(hits_c1)} hits")
    logger.mesg(f"  have_token owner:    {len(hits_c2)} hits")
    logger.mesg(f"  prefix title+tags:   {len(hits_c3)} hits")
    logger.mesg(f"  OR(prefix/token):    {len(hits_c4)} hits")
    logger.mesg(f"  contains:            {len(hits_c5)} hits")


def test_小红书推荐系统():
    """Test: 小红书推荐系统 (Xiaohongshu Recommendation System)

    Tokens: 小红, 小红书, 红书, 书推, 书推荐, 推荐, 推荐系统, 系统
    """
    searcher = get_searcher()
    query = "小红书推荐系统"

    logger.file(f"\n{'='*60}")
    logger.file(f"TEST: {query}")
    logger.file(f"{'='*60}")

    hits_baseline = run_knn_search(searcher, query, label="[baseline]")

    cf1 = construct_constraint_filter(
        constraints=[
            {"have_token": ["小红书"]},
            {"have_token": ["推荐系统"]},
        ],
        fields=TITLE_TAGS,
    )
    hits_c1 = run_knn_search(
        searcher, query, constraint_filter=cf1, label='[have "小红书" AND "推荐系统"]'
    )

    cf2 = construct_constraint_filter(
        constraints=[{"have_token": ["小红书"]}],
        fields=TITLE_TAGS,
    )
    hits_c2 = run_knn_search(
        searcher, query, constraint_filter=cf2, label='[have_token "小红书"]'
    )

    cf3 = construct_constraint_filter(
        constraints=[
            {"have_token": ["小红书"], "fields": TITLE_TAGS},
            {"with_prefixes": ["推荐"], "fields": TITLE_TAGS},
        ],
    )
    hits_c3 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf3,
        label='[token "小红书" AND prefix "推荐"]',
    )

    cf4 = construct_constraint_filter(
        constraints=[{"with_contains": ["推荐系统"]}],
        fields=TITLE_TAGS,
    )
    hits_c4 = run_knn_search(
        searcher, query, constraint_filter=cf4, label='[contains "推荐系统"]'
    )

    cf5 = construct_constraint_filter(
        constraints=[
            {"with_prefixes": ["小红书"], "fields": TITLE_TAGS},
            {"with_prefixes": ["推荐"], "fields": [TITLE, TAGS, DESC]},
        ],
    )
    hits_c5 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf5,
        label='[prefix "小红书" AND prefix "推荐"]',
    )

    logger.file(f"\nSUMMARY for {query}:")
    logger.mesg(f"  Baseline:               {len(hits_baseline)} hits")
    logger.mesg(f"  both tokens:            {len(hits_c1)} hits")
    logger.mesg(f"  token 小红书:           {len(hits_c2)} hits")
    logger.mesg(f"  token+prefix:           {len(hits_c3)} hits")
    logger.mesg(f"  contains 推荐系统:      {len(hits_c4)} hits")
    logger.mesg(f"  prefix+prefix:          {len(hits_c5)} hits")


def test_吴恩达大模型():
    """Test: 吴恩达大模型 (Andrew Ng + Large Models)

    Tokens: 吴恩, 吴恩达, 吴恩达大模型, 恩达, 大模, 大模型, 模型
    """
    searcher = get_searcher()
    query = "吴恩达大模型"

    logger.file(f"\n{'='*60}")
    logger.file(f"TEST: {query}")
    logger.file(f"{'='*60}")

    hits_baseline = run_knn_search(searcher, query, label="[baseline]")

    cf1 = construct_constraint_filter(
        constraints=[
            {"have_token": ["吴恩达"]},
            {"have_token": ["大模型"]},
        ],
        fields=TITLE_TAGS,
    )
    hits_c1 = run_knn_search(
        searcher, query, constraint_filter=cf1, label='[have "吴恩达" AND "大模型"]'
    )

    cf2 = construct_constraint_filter(
        constraints=[{"have_token": ["吴恩达"]}],
        fields=TITLE_TAGS,
    )
    hits_c2 = run_knn_search(
        searcher, query, constraint_filter=cf2, label='[have_token "吴恩达"]'
    )

    cf3 = construct_constraint_filter(
        constraints=[
            {"with_prefixes": ["吴恩达"], "fields": TITLE_TAGS},
            {"with_contains": ["模型"], "fields": TITLE_TAGS},
        ],
    )
    hits_c3 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf3,
        label='[prefix "吴恩达" AND contains "模型"]',
    )

    cf4 = construct_constraint_filter(
        constraints=[
            {"have_token": ["吴恩达"], "fields": [TITLE]},
            {"with_prefixes": ["大模型"], "fields": TITLE_TAGS},
        ],
    )
    hits_c4 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf4,
        label='[token "吴恩达" in title + prefix "大模型"]',
    )

    cf5 = construct_constraint_filter(
        constraints=[
            {
                "OR": [
                    {"have_token": ["吴恩达大模型"]},
                    {"have_token": ["吴恩达"]},
                ],
                "fields": TITLE_TAGS,
            },
        ],
    )
    hits_c5 = run_knn_search(
        searcher,
        query,
        constraint_filter=cf5,
        label='[OR: exact "吴恩达大模型" / token "吴恩达"]',
    )

    logger.file(f"\nSUMMARY for {query}:")
    logger.mesg(f"  Baseline:          {len(hits_baseline)} hits")
    logger.mesg(f"  both tokens:       {len(hits_c1)} hits")
    logger.mesg(f"  token 吴恩达:      {len(hits_c2)} hits")
    logger.mesg(f"  prefix+contains:   {len(hits_c3)} hits")
    logger.mesg(f"  token+prefix:      {len(hits_c4)} hits")
    logger.mesg(f"  OR:                {len(hits_c5)} hits")


# =====================================================================
# Compare explore modes with constraints
# =====================================================================


def test_explore_with_constraints():
    """Run explore-level tests with constraints across all 5 queries.

    Compare baseline (no constraints) vs constrained for each query,
    using both vector (q=v) and hybrid (q=wv) modes.
    """
    explorer = get_explorer()

    test_cases = [
        {
            "query": "飓风营救",
            "constraints": [{"have_token": ["飓风营救"], "fields": TITLE_TAGS}],
            "fields": None,
        },
        {
            "query": "通义实验室",
            "constraints": [{"with_prefixes": ["通义"]}],
            "fields": TITLE_TAGS,
        },
        {
            "query": "红警08",
            "constraints": [
                {
                    "OR": [
                        {"with_prefixes": ["红警"]},
                        {"have_token": ["红警08"]},
                    ],
                    "fields": [TITLE, TAGS, OWNER],
                },
            ],
            "fields": None,
        },
        {
            "query": "小红书推荐系统",
            "constraints": [
                {"have_token": ["小红书"], "fields": TITLE_TAGS},
                {"with_prefixes": ["推荐"], "fields": TITLE_TAGS},
            ],
            "fields": None,
        },
        {
            "query": "吴恩达大模型",
            "constraints": [
                {"have_token": ["吴恩达"]},
                {"have_token": ["大模型"]},
            ],
            "fields": TITLE_TAGS,
        },
    ]

    for tc in test_cases:
        query = tc["query"]
        cf = construct_constraint_filter(
            constraints=tc["constraints"],
            fields=tc.get("fields"),
        )

        logger.file(f"\n{'='*60}")
        logger.file(f"EXPLORE TEST: {query}")
        logger.file(f"{'='*60}")

        # Baseline: vector search without constraints
        run_explore(explorer, query, mode="v", label="[baseline q=v]")

        # With constraints: vector search
        run_explore(
            explorer, query, constraint_filter=cf, mode="v", label="[constrained q=v]"
        )

        # Hybrid: with constraints
        run_explore(
            explorer, query, constraint_filter=cf, mode="wv", label="[constrained q=wv]"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        test_map = {
            "飓风营救": test_飓风营救,
            "通义实验室": test_通义实验室,
            "红警08": test_红警08,
            "小红书推荐系统": test_小红书推荐系统,
            "吴恩达大模型": test_吴恩达大模型,
            "explore": test_explore_with_constraints,
            "all": None,
        }
        if test_name in test_map and test_map[test_name]:
            test_map[test_name]()
        elif test_name == "all":
            test_飓风营救()
            test_通义实验室()
            test_红警08()
            test_小红书推荐系统()
            test_吴恩达大模型()
        else:
            logger.warn(f"Unknown test: {test_name}")
            logger.mesg(f"Available: {list(test_map.keys())}")
    else:
        # Default: run all individual KNN tests
        test_飓风营救()
        test_通义实验室()
        test_红警08()
        test_小红书推荐系统()
        test_吴恩达大模型()
