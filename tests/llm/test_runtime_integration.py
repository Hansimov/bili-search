"""Runtime integration checks for a live search_app service.

These tests are skipped unless BILI_SEARCH_RUNTIME_URL is set.
Set BILI_SEARCH_RUNTIME_LLM=1 to include the live /chat/completions check.
"""

from __future__ import annotations

import os
import requests
import pytest


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


def _get_runtime_seed_hit() -> dict:
    response = requests.post(
        f"{BASE_URL}/suggest",
        json={"query": "黑神话", "limit": 1},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("hits"):
        pytest.skip("No runtime seed hit available for relation tests")
    return data["hits"][0]


def test_runtime_relation_endpoints():
    _skip_if_disabled()

    capabilities = requests.get(f"{BASE_URL}/capabilities", timeout=5)
    capabilities.raise_for_status()
    capabilities_data = capabilities.json()

    assert "related_owners_by_tokens" in capabilities_data.get("relation_endpoints", [])

    token_response = requests.post(
        f"{BASE_URL}/related_tokens_by_tokens",
        json={"text": "黑神话", "size": 3},
        timeout=20,
    )
    token_response.raise_for_status()
    token_data = token_response.json()
    assert "options" in token_data
    assert isinstance(token_data["options"], list)

    owner_response = requests.post(
        f"{BASE_URL}/related_owners_by_tokens",
        json={"text": "黑神话悟空", "size": 3},
        timeout=20,
    )
    owner_response.raise_for_status()
    owner_data = owner_response.json()
    assert "owners" in owner_data
    assert isinstance(owner_data["owners"], list)


def test_runtime_graph_relation_endpoints():
    _skip_if_disabled()

    seed_hit = _get_runtime_seed_hit()
    bvid = seed_hit.get("bvid")
    owner = seed_hit.get("owner") or {}
    mid = owner.get("mid")
    if not bvid or mid is None:
        pytest.skip("Seed hit is missing bvid or owner mid for graph relation tests")

    related_videos = requests.post(
        f"{BASE_URL}/related_videos_by_videos",
        json={"bvids": [bvid], "size": 3},
        timeout=25,
    )
    related_videos.raise_for_status()
    related_videos_data = related_videos.json()
    assert "videos" in related_videos_data
    assert isinstance(related_videos_data["videos"], list)

    related_owners = requests.post(
        f"{BASE_URL}/related_owners_by_videos",
        json={"bvids": [bvid], "size": 3},
        timeout=25,
    )
    related_owners.raise_for_status()
    related_owners_data = related_owners.json()
    assert "owners" in related_owners_data
    assert isinstance(related_owners_data["owners"], list)

    owner_videos = requests.post(
        f"{BASE_URL}/related_videos_by_owners",
        json={"mids": [mid], "size": 3},
        timeout=25,
    )
    owner_videos.raise_for_status()
    owner_videos_data = owner_videos.json()
    assert "videos" in owner_videos_data
    assert isinstance(owner_videos_data["videos"], list)

    owner_owners = requests.post(
        f"{BASE_URL}/related_owners_by_owners",
        json={"mids": [mid], "size": 3},
        timeout=25,
    )
    owner_owners.raise_for_status()
    owner_owners_data = owner_owners.json()
    assert "owners" in owner_owners_data
    assert isinstance(owner_owners_data["owners"], list)


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
