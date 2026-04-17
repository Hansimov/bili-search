from __future__ import annotations

from dataclasses import asdict

from configs.envs import LLMS_ENVS

from llms.contracts import ModelSpec


DEFAULT_SMALL_MODEL_CONFIG = "doubao-seed-2-0-mini"
DEFAULT_LARGE_MODEL_CONFIG = "deepseek"


def _supports_tools(model_envs: dict) -> bool:
    return model_envs.get("api_format", "openai") in {"openai", "ollama"}


class ModelRegistry:
    def __init__(
        self,
        specs: dict[str, ModelSpec],
        *,
        primary_large_config: str,
        primary_small_config: str,
    ):
        self.specs = specs
        self.primary_large_config = primary_large_config
        self.primary_small_config = primary_small_config

    @classmethod
    def from_envs(
        cls,
        *,
        llm_envs: dict | None = None,
        primary_large_config: str | None = None,
        primary_small_config: str | None = None,
    ) -> "ModelRegistry":
        envs = llm_envs or LLMS_ENVS
        large_config = primary_large_config or DEFAULT_LARGE_MODEL_CONFIG
        small_config = primary_small_config or DEFAULT_SMALL_MODEL_CONFIG

        specs: dict[str, ModelSpec] = {}
        for config_name, model_envs in envs.items():
            if config_name == small_config:
                role = "small"
            elif config_name == large_config:
                role = "large"
            else:
                role = "large" if model_envs.get("tasks") else "small"

            specs[config_name] = ModelSpec(
                config_name=config_name,
                model_name=model_envs.get("model", config_name),
                role=role,
                description=model_envs.get("description", ""),
                supports_tools=_supports_tools(model_envs),
                supports_streaming=True,
                max_iterations=4 if role == "large" else 2,
            )

        if large_config not in specs:
            raise ValueError(f"Unknown large model config: {large_config}")
        if small_config not in specs:
            raise ValueError(f"Unknown small model config: {small_config}")

        return cls(
            specs,
            primary_large_config=large_config,
            primary_small_config=small_config,
        )

    def get(self, config_name: str) -> ModelSpec | None:
        return self.specs.get(config_name)

    def primary(self, role: str) -> ModelSpec:
        config_name = (
            self.primary_small_config if role == "small" else self.primary_large_config
        )
        spec = self.get(config_name)
        if spec is None:
            raise KeyError(f"Missing model spec for role={role}: {config_name}")
        return spec

    def public_dict(self) -> dict:
        return {
            "primary_large": self.primary_large_config,
            "primary_small": self.primary_small_config,
            "available": {
                name: asdict(spec) for name, spec in sorted(self.specs.items())
            },
        }


def create_model_clients(
    *,
    primary_large_config: str | None = None,
    primary_small_config: str | None = None,
    verbose: bool = False,
) -> tuple[ModelRegistry, dict[str, object]]:
    from llms.models.client import create_llm_client

    registry = ModelRegistry.from_envs(
        primary_large_config=primary_large_config,
        primary_small_config=primary_small_config,
    )
    clients: dict[str, object] = {}
    for config_name in {
        registry.primary_large_config,
        registry.primary_small_config,
    }:
        clients[config_name] = create_llm_client(
            model_config=config_name,
            verbose=verbose,
        )
    return registry, clients
