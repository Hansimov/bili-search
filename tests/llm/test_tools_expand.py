"""Query-expansion tests for llms.tools executor."""

import json
from unittest.mock import MagicMock

from tclogger import logger

from llms.models import ToolCall
from llms.tools.executor import ToolExecutor


def test_execute_expand_query_prefers_semantic_mode():
    logger.note("=" * 60)
    logger.note("[TEST] execute expand_query prefers semantic")

    mock_client = MagicMock()
    mock_client.capabilities.return_value = {
        "relation_endpoints": ["related_tokens_by_tokens"],
        "supports_google_search": False,
    }
    mock_client.related_tokens_by_tokens.return_value = {
        "text": "袁启 专访",
        "mode": "semantic",
        "options": [
            {
                "text": "袁启 采访",
                "doc_freq": 1,
                "score": 128.0,
                "type": "rewrite",
                "shard_count": 1,
            }
        ],
    }

    executor = ToolExecutor(search_client=mock_client)
    tc = ToolCall(
        id="call_expand_semantic",
        name="expand_query",
        arguments=json.dumps({"text": "袁启 专访"}),
    )

    result_msg = executor.execute(tc)
    result_data = json.loads(result_msg["content"])

    assert result_data["mode"] == "semantic"
    assert result_data["options"][0]["text"] == "袁启 采访"
    mock_client.related_tokens_by_tokens.assert_called_once_with(
        text="袁启 专访",
        mode="semantic",
        size=8,
    )

    logger.success("[PASS] execute expand_query prefers semantic")

    def test_execute_expand_query_keeps_text_without_alias_rules(self):
        logger.note("=" * 60)
        logger.note("[TEST] execute expand_query normalizes aliases")

        mock_client = MagicMock()
        mock_client.capabilities.return_value = {
            "relation_endpoints": ["related_tokens_by_tokens"],
            "supports_google_search": False,
        }
        mock_client.related_tokens_by_tokens.return_value = {
            "text": "康夫 UI 工作流",
            "mode": "semantic",
            "options": [
                {
                    "text": "comfyui教程工作流",
                    "doc_freq": 1,
                    "score": 256.0,
                    "type": "rewrite",
                    "shard_count": 1,
                }
            ],
        }

        executor = ToolExecutor(search_client=mock_client)
        tc = ToolCall(
            id="call_expand_alias",
            name="expand_query",
            arguments=json.dumps({"text": "康夫 UI 工作流"}),
        )

        result_msg = executor.execute(tc)
        result_data = json.loads(result_msg["content"])

        assert result_data["text"] == "康夫 UI 工作流"
        assert result_data["normalized_text"] == "康夫 UI 工作流"
        assert result_data["options"][0]["text"] == "comfyui教程工作流"
        mock_client.related_tokens_by_tokens.assert_called_once_with(
            text="康夫 UI 工作流",
            mode="semantic",
            size=8,
        )

        logger.success("[PASS] execute expand_query normalizes aliases")


def test_execute_expand_query_falls_back_to_auto_when_semantic_unsupported():
    logger.note("=" * 60)
    logger.note("[TEST] execute expand_query fallback to auto")

    mock_client = MagicMock()
    mock_client.capabilities.return_value = {
        "relation_endpoints": ["related_tokens_by_tokens"],
        "supports_google_search": False,
    }
    mock_client.related_tokens_by_tokens.side_effect = [
        {
            "text": "袁启 专访",
            "error": "mode must be 'prefix', 'associate', 'correction' or 'auto'; semantic unsupported",
            "options": [],
        },
        {
            "text": "袁启 专访",
            "mode": "auto",
            "options": [
                {
                    "text": "袁启 采访",
                    "doc_freq": 1,
                    "score": 96.0,
                    "type": "associate",
                    "shard_count": 1,
                }
            ],
        },
    ]

    executor = ToolExecutor(search_client=mock_client)
    tc = ToolCall(
        id="call_expand_fallback",
        name="expand_query",
        arguments=json.dumps({"text": "袁启 专访"}),
    )

    result_msg = executor.execute(tc)
    result_data = json.loads(result_msg["content"])

    assert result_data["mode"] == "auto"
    assert result_data["options"][0]["text"] == "袁启 采访"
    assert mock_client.related_tokens_by_tokens.call_count == 2
    assert mock_client.related_tokens_by_tokens.call_args_list[0].kwargs == {
        "text": "袁启 专访",
        "mode": "semantic",
        "size": 8,
    }
    assert mock_client.related_tokens_by_tokens.call_args_list[1].kwargs == {
        "text": "袁启 专访",
        "mode": "auto",
        "size": 8,
    }

    logger.success("[PASS] execute expand_query fallback to auto")
