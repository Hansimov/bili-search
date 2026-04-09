from __future__ import annotations

from typing import Mapping

from tclogger import dt_to_str


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_usage(usage: Mapping[str, object] | None) -> dict[str, int | float]:
    raw = dict(usage or {})
    result: dict[str, int | float] = {
        key: value for key, value in raw.items() if _is_numeric(value)
    }

    prompt_tokens = result.get("prompt_tokens", 0)
    completion_tokens = result.get("completion_tokens", 0)

    prompt_details = raw.get("prompt_tokens_details")
    cache_hit_tokens = result.get("prompt_cache_hit_tokens", 0)
    has_cache_signal = any(
        key in result for key in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
    ) or isinstance(prompt_details, dict)
    if isinstance(prompt_details, dict):
        nested_cached_tokens = prompt_details.get("cached_tokens", 0)
        if _is_numeric(nested_cached_tokens):
            cache_hit_tokens = max(cache_hit_tokens, nested_cached_tokens)
    if has_cache_signal:
        result["prompt_cache_hit_tokens"] = cache_hit_tokens
        result["prompt_cache_miss_tokens"] = max(0, prompt_tokens - cache_hit_tokens)

    completion_details = raw.get("completion_tokens_details")
    reasoning_tokens = result.get("reasoning_tokens", 0)
    if isinstance(completion_details, dict):
        nested_reasoning_tokens = completion_details.get("reasoning_tokens", 0)
        if _is_numeric(nested_reasoning_tokens):
            reasoning_tokens = max(reasoning_tokens, nested_reasoning_tokens)
    if reasoning_tokens or "reasoning_tokens" in result:
        result["reasoning_tokens"] = reasoning_tokens

    if prompt_tokens or completion_tokens or "total_tokens" in result:
        result["total_tokens"] = prompt_tokens + completion_tokens

    return result


def accumulate_usage(total: dict[str, int | float], usage: Mapping[str, object] | None):
    for key, value in normalize_usage(usage).items():
        if _is_numeric(value):
            total[key] = total.get(key, 0) + value


def compute_perf_stats(
    usage: Mapping[str, object] | None, elapsed_seconds: float
) -> dict:
    normalized_usage = normalize_usage(usage)
    completion_tokens = normalized_usage.get("completion_tokens", 0)

    tokens_per_second = (
        int(completion_tokens / elapsed_seconds)
        if elapsed_seconds > 0 and completion_tokens > 0
        else 0
    )

    result = {
        "tokens_per_second": tokens_per_second,
        "total_elapsed": dt_to_str(elapsed_seconds, precision=0),
        "total_elapsed_ms": round(elapsed_seconds * 1000, 1),
    }

    for key in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        if key in normalized_usage:
            result[key] = normalized_usage[key]

    return result


__all__ = ["accumulate_usage", "normalize_usage", "compute_perf_stats"]
