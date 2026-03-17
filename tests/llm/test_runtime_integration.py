"""Runtime integration checks for a live search_app service.

These tests are skipped unless BILI_SEARCH_RUNTIME_URL is set.
Set BILI_SEARCH_RUNTIME_LLM=1 to include the live /chat/completions check.
"""

from __future__ import annotations

import os
import requests


BASE_URL = os.environ.get("BILI_SEARCH_RUNTIME_URL", "").strip()
RUN_LLM = os.environ.get("BILI_SEARCH_RUNTIME_LLM", "0").strip() == "1"


def _skip_if_disabled():
    if not BASE_URL:
        import pytest

        pytest.skip("BILI_SEARCH_RUNTIME_URL is not set")


def test_runtime_health_and_capabilities():
    _skip_if_disabled()

    health = requests.get(f"{BASE_URL}/health", timeout=5)
    capabilities = requests.get(f"{BASE_URL}/capabilities", timeout=5)

    health.raise_for_status()
    capabilities.raise_for_status()

    health_data = health.json()
    capabilities_data = capabilities.json()

    assert health_data["status"] == "ok"
    assert capabilities_data["supports_multi_query"] is True
    assert "/explore" in capabilities_data["available_endpoints"]


def test_runtime_suggest():
    _skip_if_disabled()

    response = requests.post(
        f"{BASE_URL}/suggest",
        json={"query": "黑神话", "limit": 1},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    assert data["total_hits"] >= 1
    assert len(data["hits"]) == 1


def test_runtime_chat_completion():
    _skip_if_disabled()
    if not RUN_LLM:
        import pytest

        pytest.skip("BILI_SEARCH_RUNTIME_LLM is not enabled")

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "请用一句话介绍黑神话悟空是什么。"}
            ],
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    assert data["choices"][0]["message"]["content"]
