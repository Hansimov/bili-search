from copy import deepcopy

from elastics.videos.constants import SEARCH_BOOSTED_FIELDS
from elastics.videos.searcher_v2 import VideoSearcherV2


QUERY = "袁启 采访"


def make_profiles() -> dict[str, dict]:
    base = deepcopy(SEARCH_BOOSTED_FIELDS)

    title_owner_focus = deepcopy(base)
    title_owner_focus.update(
        {
            "title.words": 5.0,
            "owner.name.words": 4.0,
            "tags.words": 1.2,
            "desc.words": 0.02,
        }
    )

    title_focus = deepcopy(base)
    title_focus.update(
        {
            "title.words": 6.0,
            "owner.name.words": 3.0,
            "tags.words": 1.0,
            "desc.words": 0.01,
        }
    )

    return {
        "baseline": base,
        "title_owner_focus": title_owner_focus,
        "title_focus": title_focus,
    }


def summarize_result(result: dict, label: str):
    print(f"=== {label} ===")
    print("retry_info:", result.get("retry_info"))
    query_text = (
        result.get("search_body", {})
        .get("query", {})
        .get("bool", {})
        .get("must", {})
        .get("es_tok_query_string", {})
        .get("query")
    )
    print("query:", query_text)
    for index, hit in enumerate(result.get("hits", [])[:10], start=1):
        owner = (hit.get("owner") or {}).get("name")
        print(f"{index:02d}. {owner} :: {hit.get('title')}")
    print()


def main():
    searcher = VideoSearcherV2(
        index_name="bili_videos_dev6",
        elastic_env_name="elastic_dev",
    )

    for label, boosted_fields in make_profiles().items():
        result = searcher.search(
            QUERY,
            boosted_fields=boosted_fields,
            limit=10,
            verbose=False,
        )
        summarize_result(result, label)


if __name__ == "__main__":
    main()
