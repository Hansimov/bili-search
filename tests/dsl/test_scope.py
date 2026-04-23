from dsl.elastic import DslExprToElasticConverter
from dsl.filter import QueryDslDictFilterMerger
from dsl.rewrite import DslExprRewriter
from dsl.fields.scope import get_scope_constraint_fields
from elastics.videos.constants import CONSTRAINT_FIELDS_DEFAULT
from elastics.videos.searcher_v2 import VideoSearcherV2


def make_scope_searcher() -> VideoSearcherV2:
    searcher = VideoSearcherV2.__new__(VideoSearcherV2)
    searcher.query_rewriter = DslExprRewriter()
    searcher.elastic_converter = DslExprToElasticConverter()
    searcher.filter_merger = QueryDslDictFilterMerger()
    return searcher


def test_scope_is_extracted_from_query_info():
    rewriter = DslExprRewriter()

    query_info = rewriter.get_query_info("黑神话 :scope=tg")

    assert query_info["scope_fields"] == ["title", "tags"]
    assert query_info["scope_info"]["mode"] == "include"


def test_scope_limits_match_fields_in_query_dsl():
    searcher = make_scope_searcher()

    query_info, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
        query="红警 :scope=n",
        boosted_match_fields=["title.words^3", "tags.words^2.5", "owner.name.words^2"],
        boosted_date_fields=["title.words"],
        extra_filters=[],
    )

    def find_es_tok_fields(node):
        if isinstance(node, dict):
            if "es_tok_query_string" in node:
                return node["es_tok_query_string"]["fields"]
            for value in node.values():
                found = find_es_tok_fields(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = find_es_tok_fields(item)
                if found is not None:
                    return found
        return None

    fields = find_es_tok_fields(query_dsl_dict)

    assert fields == ["owner.name.words^2"]
    assert query_info["effective_match_fields"] == ["owner.name.words^2"]
    assert query_info["effective_date_match_fields"] == ["title.words"]


def test_scope_exclusion_filters_constraint_fields():
    rewriter = DslExprRewriter()

    query_info = rewriter.get_query_info("黑神话 :scope!=d")

    assert get_scope_constraint_fields(
        query_info["scope_info"],
        CONSTRAINT_FIELDS_DEFAULT,
    ) == ["title.words", "tags.words", "owner.name.words"]
