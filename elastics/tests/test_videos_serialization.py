from fastapi.encoders import jsonable_encoder
from tclogger import logger, dict_to_str

from elastics.tests.test_videos import make_explorer


def test_json_serialization(explorer: VideoExplorer = None):
    """Test that filter_only_search results can be serialized by FastAPI."""
    if explorer is None:
        explorer = make_explorer()

    test_queries = [
        'u="红警HBK08"',
        'u="红警HBK08" q=v',
        "d>2024-01-01 v>10000",
    ]

    logger.note("> Testing JSON serialization (simulating FastAPI)...")
    for query in test_queries:
        logger.hint(f"\n  Query: [{query}]")
        try:
            res = explorer.filter_only_search(
                query, limit=10, rank_top_k=10, verbose=False
            )
            # This is what FastAPI does internally
            encoded = jsonable_encoder(res)
            logger.success("    ✓ jsonable_encoder succeeded")
            logger.mesg(
                f'      total_hits={res.get("total_hits")}, filter_only={res.get("filter_only")}'
            )
            has_tree = "query_expr_tree" in res.get("query_info", {})
            logger.mesg(f"      query_expr_tree in query_info: {has_tree}")
        except Exception as e:
            logger.warn(f"    ✗ Error: {type(e).__name__}: {e}")

    logger.success("\n> JSON serialization test completed!")
