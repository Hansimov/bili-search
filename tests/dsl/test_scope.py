from dsl.elastic import DslExprToElasticConverter
from dsl.filter import QueryDslDictFilterMerger
from dsl.rewrite import DslExprRewriter
from dsl.fields.scope import get_scope_constraint_fields
from dsl.fields.word import override_auto_require_short_han_exact
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
    assert query_info["effective_date_match_fields"] == []


def test_scope_empty_filtered_lane_does_not_fall_back_to_default_fields():
    searcher = make_scope_searcher()

    query_info, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
        query="红警 :scope=n",
        boosted_match_fields=["title.words^5", "tags.words^3"],
        boosted_date_fields=["title.words"],
        extra_filters=[],
    )

    assert query_info["effective_match_fields"] == []
    assert query_dsl_dict == {"bool": {"must": {"match_none": {}}}}


def test_scope_exclusion_filters_constraint_fields():
    rewriter = DslExprRewriter()

    query_info = rewriter.get_query_info("黑神话 :scope!=d")

    assert get_scope_constraint_fields(
        query_info["scope_info"],
        CONSTRAINT_FIELDS_DEFAULT,
    ) == ["title.words", "tags.words", "owner.name.words"]


def test_exact_like_mixed_ascii_segment_is_auto_promoted_to_required_exact():
    searcher = make_scope_searcher()

    _, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
        query="b200",
        boosted_match_fields=["title.words^3"],
        boosted_date_fields=[],
        extra_filters=[],
    )

    assert query_dsl_dict == {
        "bool": {
            "must": {
                "es_tok_query_string": {
                    "query": "+b200",
                    "fields": ["title.words^3"],
                    "max_freq": 1000000,
                }
            }
        }
    }


def test_auto_exact_can_be_disabled_for_low_recall_retry():
    searcher = make_scope_searcher()

    with override_auto_require_short_han_exact("none"):
        _, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
            query="b200 价格",
            boosted_match_fields=["title.words^3"],
            boosted_date_fields=[],
            extra_filters=[],
        )

    assert query_dsl_dict == {
        "bool": {
            "must": {
                "es_tok_query_string": {
                    "query": "b200 价格",
                    "fields": ["title.words^3"],
                    "max_freq": 1000000,
                }
            }
        }
    }


def test_auto_exact_can_keep_only_model_code_for_low_recall_retry():
    searcher = make_scope_searcher()

    with override_auto_require_short_han_exact("model_code"):
        _, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
            query="b200 价格",
            boosted_match_fields=["title.words^3"],
            boosted_date_fields=[],
            extra_filters=[],
        )

    assert query_dsl_dict == {
        "bool": {
            "must": {
                "es_tok_query_string": {
                    "query": "+b200 价格",
                    "fields": ["title.words^3"],
                    "max_freq": 1000000,
                }
            }
        }
    }


def test_short_cjk_segment_is_auto_promoted_to_required_exact():
    searcher = make_scope_searcher()

    _, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
        query="袁启",
        boosted_match_fields=["owner.name.words^2"],
        boosted_date_fields=[],
        extra_filters=[],
    )

    assert query_dsl_dict == {
        "bool": {
            "must": {
                "es_tok_query_string": {
                    "query": "+袁启",
                    "fields": ["owner.name.words^2"],
                    "max_freq": 1000000,
                }
            }
        }
    }


def test_plain_longer_chinese_segment_keeps_loose_query_semantics():
    searcher = make_scope_searcher()

    _, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
        query="影视飓风",
        boosted_match_fields=["title.words^3"],
        boosted_date_fields=[],
        extra_filters=[],
    )

    assert query_dsl_dict == {
        "bool": {
            "must": {
                "es_tok_query_string": {
                    "query": "影视飓风",
                    "fields": ["title.words^3"],
                    "max_freq": 1000000,
                }
            }
        }
    }


def test_short_cjk_retry_mode_keeps_first_segment_exact_only():
    searcher = make_scope_searcher()

    with override_auto_require_short_han_exact("first"):
        _, _, query_dsl_dict = searcher.get_info_of_query_rewrite_dsl(
            query="袁启 采访",
            boosted_match_fields=["title.words^3"],
            boosted_date_fields=[],
            extra_filters=[],
        )

    assert query_dsl_dict == {
        "bool": {
            "must": {
                "es_tok_query_string": {
                    "query": "+袁启 采访",
                    "fields": ["title.words^3"],
                    "max_freq": 1000000,
                }
            }
        }
    }
