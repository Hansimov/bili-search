from llms.models import (
    DEFAULT_LARGE_MODEL_CONFIG,
    DEFAULT_SMALL_MODEL_CONFIG,
    ModelRegistry,
)


def test_model_registry_uses_fixed_small_and_large_defaults():
    registry = ModelRegistry.from_envs()

    assert registry.primary_large_config == DEFAULT_LARGE_MODEL_CONFIG
    assert registry.primary_small_config == DEFAULT_SMALL_MODEL_CONFIG
    assert registry.primary("large").config_name == DEFAULT_LARGE_MODEL_CONFIG
    assert registry.primary("small").config_name == DEFAULT_SMALL_MODEL_CONFIG


def test_model_registry_public_dict_exposes_available_models():
    registry = ModelRegistry.from_envs()
    payload = registry.public_dict()

    assert payload["primary_large"] == DEFAULT_LARGE_MODEL_CONFIG
    assert payload["primary_small"] == DEFAULT_SMALL_MODEL_CONFIG
    assert DEFAULT_LARGE_MODEL_CONFIG in payload["available"]
    assert DEFAULT_SMALL_MODEL_CONFIG in payload["available"]
    assert payload["available"][DEFAULT_LARGE_MODEL_CONFIG]["role"] == "large"
    assert payload["available"][DEFAULT_SMALL_MODEL_CONFIG]["role"] == "small"
    assert payload["available"][DEFAULT_LARGE_MODEL_CONFIG]["provider"] == "minimax"
    assert payload["available"][DEFAULT_SMALL_MODEL_CONFIG]["provider"] == "doubao"


def test_default_small_model_is_doubao_seed_2_0_mini():
    assert DEFAULT_SMALL_MODEL_CONFIG == "doubao-seed-2-0-mini"


def test_default_large_model_is_minimax_m2_7():
    assert DEFAULT_LARGE_MODEL_CONFIG == "minimax-m2.7"
