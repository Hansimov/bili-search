from service.envs import SEARCH_APP_ENV_KEYS
from service.envs import apply_search_app_envs_to_environment
from service.envs import get_search_app_env_overrides_from_env
from service.envs import resolve_search_app_envs


__all__ = [
    "SEARCH_APP_ENV_KEYS",
    "ChatCompletionRequest",
    "ChatMessage",
    "SearchApp",
    "apply_search_app_envs_to_environment",
    "create_app",
    "create_app_from_env",
    "get_search_app_env_overrides_from_env",
    "resolve_search_app_envs",
]


def __getattr__(name: str):
    if name in {
        "ChatCompletionRequest",
        "ChatMessage",
        "SearchApp",
        "create_app",
        "create_app_from_env",
    }:
        from service.app import (
            ChatCompletionRequest,
            ChatMessage,
            SearchApp,
            create_app,
            create_app_from_env,
        )

        return {
            "ChatCompletionRequest": ChatCompletionRequest,
            "ChatMessage": ChatMessage,
            "SearchApp": SearchApp,
            "create_app": create_app,
            "create_app_from_env": create_app_from_env,
        }[name]
    raise AttributeError(name)
