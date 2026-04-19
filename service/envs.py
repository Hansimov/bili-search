from __future__ import annotations

import os

from copy import deepcopy


SEARCH_APP_ENV_PREFIX = "BILI_SEARCH_APP_"
SEARCH_APP_CONFIG_PATH = "configs/envs.json"
SEARCH_APP_CONFIG_SECTION = "search_app"
SEARCH_APP_RUNTIME_FIELDS = (
    "host",
    "port",
    "elastic_index",
    "elastic_env_name",
    "llm_config",
)
SEARCH_APP_ENV_KEYS = {
    "host": f"{SEARCH_APP_ENV_PREFIX}HOST",
    "port": f"{SEARCH_APP_ENV_PREFIX}PORT",
    "elastic_index": f"{SEARCH_APP_ENV_PREFIX}ELASTIC_INDEX",
    "elastic_env_name": f"{SEARCH_APP_ENV_PREFIX}ELASTIC_ENV_NAME",
    "llm_config": f"{SEARCH_APP_ENV_PREFIX}LLM_CONFIG",
}


def _default_search_app_envs() -> dict:
    from configs.envs import SEARCH_APP_ENVS

    return SEARCH_APP_ENVS


def default_search_app_envs() -> dict:
    return deepcopy(_default_search_app_envs())


def search_app_config_field_path(name: str) -> str:
    return f"{SEARCH_APP_CONFIG_PATH}.{SEARCH_APP_CONFIG_SECTION}.{name}"


def search_app_env_source_summary() -> str:
    return (
        f"CLI args > {SEARCH_APP_ENV_PREFIX}* environment variables > "
        f"{SEARCH_APP_CONFIG_PATH}.{SEARCH_APP_CONFIG_SECTION}"
    )


def search_app_default_value(name: str):
    return default_search_app_envs().get(name)


def resolve_search_app_envs(
    app_envs: dict | None = None,
    *,
    overrides: dict | None = None,
) -> dict:
    resolved_envs = default_search_app_envs()

    for key, value in (app_envs or {}).items():
        resolved_envs[key] = value

    for key, value in (overrides or {}).items():
        if value is not None:
            resolved_envs[key] = value

    return resolved_envs


def _get_env_override(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_search_app_env_overrides_from_env() -> dict:
    overrides = {
        "host": _get_env_override(SEARCH_APP_ENV_KEYS["host"]),
        "elastic_index": _get_env_override(SEARCH_APP_ENV_KEYS["elastic_index"]),
        "elastic_env_name": _get_env_override(SEARCH_APP_ENV_KEYS["elastic_env_name"]),
        "llm_config": _get_env_override(SEARCH_APP_ENV_KEYS["llm_config"]),
    }
    port_value = _get_env_override(SEARCH_APP_ENV_KEYS["port"])
    if port_value is not None:
        overrides["port"] = int(port_value)
    return {key: value for key, value in overrides.items() if value is not None}


def apply_search_app_envs_to_environment(app_envs: dict):
    for key, env_name in SEARCH_APP_ENV_KEYS.items():
        value = app_envs.get(key)
        if value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = str(value)
