"""CLI interactive and single-shot chat mode for testing.

Provides a command-line interface for interacting with the search copilot.
Uses direct search (in-process) instead of HTTP, so no separate search server needed.

Usage:
    # Interactive mode
    python -m llms.runtime.cli
    python -m llms.runtime.cli --llm-config volcengine
    python -m llms.runtime.cli --verbose

    # Single-shot mode (for quick testing)
    python -m llms.runtime.cli --query "影视飓风最近有什么新视频？"
    python -m llms.runtime.cli --query "老番茄和影视飓风最近30天的视频" --verbose
    python -m llms.runtime.cli --elastic-index bili_videos_dev6 --elastic-env-name elastic_dev
"""

import argparse
import json
import sys
import time

from tclogger import logger


def _print_usage_summary(usage: dict):
    """Print a compact token usage summary."""
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", 0)
    cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    cache_miss = usage.get("prompt_cache_miss_tokens", 0)

    parts = [f"tokens: {total} (prompt={prompt}, completion={completion})"]
    if cache_hit or cache_miss:
        rate = round(cache_hit / max(prompt, 1) * 100) if prompt else 0
        parts.append(f"cache: hit={cache_hit}, miss={cache_miss} ({rate}%)")
    logger.mesg(f"  {', '.join(parts)}")


def _create_handler(args):
    """Create ChatHandler with all dependencies initialized."""
    from configs.envs import SEARCH_APP_ENVS
    from elastics.relations import RelationsClient
    from elastics.videos.searcher_v2 import VideoSearcherV2
    from elastics.videos.explorer import VideoExplorer
    from llms.models import create_model_clients
    from llms.tools.executor import create_search_service
    from llms.chat.handler import ChatHandler

    # Resolve elastic index
    elastic_index = args.elastic_index
    elastic_env_name = args.elastic_env_name
    if not elastic_index:
        idx = SEARCH_APP_ENVS.get("elastic_index", {})
        elastic_index = idx.get("prod", idx) if isinstance(idx, dict) else idx

    logger.note("> Initializing Bili Search Copilot CLI...")
    logger.mesg(f"  LLM config: {args.llm_config}")
    if args.search_base_url:
        logger.mesg(f"  Search service: {args.search_base_url}")
        search_service = create_search_service(
            base_url=args.search_base_url,
            timeout=args.search_timeout,
            verbose=args.verbose,
        )
    else:
        logger.mesg(f"  Elastic index: {elastic_index}")
        if elastic_env_name:
            logger.mesg(f"  Elastic env: {elastic_env_name}")

        video_searcher = VideoSearcherV2(
            elastic_index,
            elastic_env_name=elastic_env_name,
        )
        video_explorer = VideoExplorer(
            elastic_index,
            elastic_env_name=elastic_env_name,
        )
        relations_client = RelationsClient(
            elastic_index,
            elastic_env_name=elastic_env_name,
        )
        search_service = create_search_service(
            video_searcher=video_searcher,
            video_explorer=video_explorer,
            relations_client=relations_client,
            verbose=args.verbose,
        )

    model_registry, llm_clients = create_model_clients(
        primary_large_config=args.llm_config,
        verbose=args.verbose,
    )
    llm_client = llm_clients[model_registry.primary_large_config]
    small_llm_client = llm_clients[model_registry.primary_small_config]
    handler = ChatHandler(
        llm_client=llm_client,
        small_llm_client=small_llm_client,
        search_client=search_service,
        model_registry=model_registry,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    capabilities = handler.search_capabilities
    logger.mesg(
        "  Tool capabilities: "
        f"mode=q={capabilities.get('default_query_mode', 'wv')}, "
        f"rerank=q={capabilities.get('rerank_query_mode', 'vwr')}, "
        f"multi_query={capabilities.get('supports_multi_query', True)}, "
        f"google={capabilities.get('supports_google_search', False)}, "
        f"relations={','.join(capabilities.get('relation_endpoints', [])) or 'none'}"
    )
    logger.mesg(
        "  Models: "
        f"large={model_registry.primary_large_config} ({llm_client.model}), "
        f"small={model_registry.primary_small_config} ({small_llm_client.model})"
    )

    return handler


def _run_single_query(handler, query: str, temperature: float = None):
    """Run a single query and print the result with usage stats."""
    messages = [{"role": "user", "content": query}]

    logger.note(f"\n> Query: {query}")
    logger.hint("> Thinking...")

    start_time = time.perf_counter()
    result = handler.handle(messages=messages, temperature=temperature)
    elapsed = round((time.perf_counter() - start_time) * 1000, 1)

    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})

    print(f"\n{content}\n")
    logger.note(f"> Stats ({elapsed}ms):")
    _print_usage_summary(usage)


def _run_interactive(handler, temperature: float = None):
    """Run interactive chat loop."""
    logger.success("> Ready! Type your question (Ctrl+C to exit)\n")
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                logger.note("Goodbye!")
                break

            if user_input.lower() in ("/clear", "/reset"):
                messages.clear()
                logger.note("> Conversation cleared")
                continue

            if user_input.lower() == "/history":
                for msg in messages:
                    role = msg["role"]
                    content = (
                        msg["content"][:80] + "..."
                        if len(msg["content"]) > 80
                        else msg["content"]
                    )
                    logger.mesg(f"  [{role}] {content}")
                continue

            if user_input.lower() == "/help":
                logger.note("Commands:")
                logger.mesg("  /clear  - Clear conversation history")
                logger.mesg("  /history - Show conversation history")
                logger.mesg("  /quit   - Exit")
                logger.mesg("  /help   - Show this help")
                continue

            messages.append({"role": "user", "content": user_input})

            logger.hint("> Thinking...")
            start_time = time.perf_counter()
            result = handler.handle(
                messages=messages,
                temperature=temperature,
            )
            elapsed = round((time.perf_counter() - start_time) * 1000, 1)

            content = result["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": content})
            usage = result.get("usage", {})

            print(f"\nCopilot: {content}\n")
            _print_usage_summary(usage)
            logger.mesg(f"  time: {elapsed}ms")
            print()

        except KeyboardInterrupt:
            print()
            logger.note("Goodbye!")
            break
        except EOFError:
            logger.note("Goodbye!")
            break
        except Exception as e:
            logger.warn(f"× Error: {e}")
            if messages and messages[-1]["role"] == "user":
                messages.pop()


def main():
    parser = argparse.ArgumentParser(description="Bili Search Copilot CLI")
    from configs.envs import LLM_CONFIG

    parser.add_argument(
        "--llm-config",
        type=str,
        default=LLM_CONFIG,
        help="LLM config name (default: deepseek)",
    )
    parser.add_argument(
        "--elastic-index",
        type=str,
        default=None,
        help="Elastic videos index name (default: from envs.json)",
    )
    parser.add_argument(
        "--elastic-env-name",
        type=str,
        default=None,
        help="Elastic env name in secrets.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature",
    )
    parser.add_argument(
        "--search-base-url",
        type=str,
        default=None,
        help="Use a remote search_app service instead of in-process searchers",
    )
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=30.0,
        help="Timeout for remote search service requests",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Single-shot query (non-interactive mode). Prints result and exits.",
    )
    args = parser.parse_args()

    handler = _create_handler(args)

    if args.query:
        _run_single_query(handler, args.query, temperature=args.temperature)
    else:
        _run_interactive(handler, temperature=args.temperature)


if __name__ == "__main__":
    main()

    # Interactive mode:
    #   python -m llms.runtime.cli
    #   python -m llms.runtime.cli --llm-config deepseek --verbose
    #   python -m llms.runtime.cli --elastic-index bili_videos_dev6 --elastic-env-name elastic_dev
    #
    # Single-shot mode:
    #   python -m llms.runtime.cli -q "影视飓风最近有什么新视频？"
    #   python -m llms.runtime.cli -q "老番茄和影视飓风最近的视频" --verbose
    #   python -m llms.runtime.cli -q "黑神话 播放量最高的视频"
