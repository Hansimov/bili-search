"""Unit tests for ChatHandler perf_stats and usage accumulation."""

from llms.chat.handler import ChatHandler


def test_accumulate_usage_gpt_nested():
    """Test that GPT nested usage (prompt_tokens_details.cached_tokens) is accumulated."""
    total = {}
    gpt_usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 800,
        },
        "completion_tokens_details": {
            "reasoning_tokens": 50,
        },
    }
    ChatHandler._accumulate_usage(total, gpt_usage)
    assert total["prompt_tokens"] == 1000
    assert total["completion_tokens"] == 200
    assert total["prompt_tokens_details"]["cached_tokens"] == 800
    assert total["completion_tokens_details"]["reasoning_tokens"] == 50


def test_accumulate_usage_deepseek_flat():
    """Test that DeepSeek flat usage fields are accumulated."""
    total = {}
    ds_usage = {
        "prompt_tokens": 500,
        "completion_tokens": 100,
        "total_tokens": 600,
        "prompt_cache_hit_tokens": 300,
        "prompt_cache_miss_tokens": 200,
    }
    ChatHandler._accumulate_usage(total, ds_usage)
    assert total["prompt_cache_hit_tokens"] == 300
    assert total["prompt_cache_miss_tokens"] == 200


def test_accumulate_usage_multi_iteration():
    """Test accumulation across multiple iterations."""
    total = {}
    ChatHandler._accumulate_usage(
        total,
        {
            "prompt_tokens": 500,
            "completion_tokens": 100,
            "total_tokens": 600,
            "prompt_tokens_details": {"cached_tokens": 400},
        },
    )
    ChatHandler._accumulate_usage(
        total,
        {
            "prompt_tokens": 800,
            "completion_tokens": 150,
            "total_tokens": 950,
            "prompt_tokens_details": {"cached_tokens": 600},
        },
    )
    assert total["prompt_tokens"] == 1300
    assert total["completion_tokens"] == 250
    assert total["total_tokens"] == 1550
    assert total["prompt_tokens_details"]["cached_tokens"] == 1000


def test_compute_perf_stats_gpt():
    """Test perf stats computation with GPT nested format."""
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "prompt_tokens_details": {"cached_tokens": 800},
    }
    stats = ChatHandler._compute_perf_stats(usage, elapsed_seconds=5.0)
    assert stats["tokens_per_second"] == 40.0
    assert stats["prompt_cache_hit_tokens"] == 800
    assert stats["prompt_cache_miss_tokens"] == 200
    assert "total_elapsed" in stats
    assert stats["total_elapsed_ms"] == 5000.0


def test_compute_perf_stats_deepseek():
    """Test perf stats computation with DeepSeek flat format."""
    usage = {
        "prompt_tokens": 500,
        "completion_tokens": 100,
        "total_tokens": 600,
        "prompt_cache_hit_tokens": 300,
        "prompt_cache_miss_tokens": 200,
    }
    stats = ChatHandler._compute_perf_stats(usage, elapsed_seconds=2.0)
    assert stats["tokens_per_second"] == 50.0
    assert stats["prompt_cache_hit_tokens"] == 300
    assert stats["prompt_cache_miss_tokens"] == 200


def test_compute_perf_stats_no_cache():
    """Test perf stats when no cache info is present."""
    usage = {
        "prompt_tokens": 500,
        "completion_tokens": 100,
        "total_tokens": 600,
    }
    stats = ChatHandler._compute_perf_stats(usage, elapsed_seconds=2.0)
    assert stats["tokens_per_second"] == 50.0
    assert "prompt_cache_hit_tokens" not in stats
    assert "prompt_cache_miss_tokens" not in stats


if __name__ == "__main__":
    test_accumulate_usage_gpt_nested()
    test_accumulate_usage_deepseek_flat()
    test_accumulate_usage_multi_iteration()
    test_compute_perf_stats_gpt()
    test_compute_perf_stats_deepseek()
    test_compute_perf_stats_no_cache()
    print("All tests passed!")
