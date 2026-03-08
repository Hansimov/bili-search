"""CLI interactive and single-shot chat mode for testing.

Provides a command-line interface for interacting with the search copilot.
Uses direct search (in-process) instead of HTTP, so no separate search server needed.

Usage:
    # Interactive mode
    python -m llms.cli
    python -m llms.cli --llm-config volcengine
    python -m llms.cli --verbose

    # Single-shot mode (for quick testing)
    python -m llms.cli --query "影视飓风最近有什么新视频？"
    python -m llms.cli --query "老番茄和影视飓风最近30天的视频" --verbose
    python -m llms.cli --elastic-index bili_videos_dev6 --elastic-env-name elastic_dev
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
    from elastics.videos.searcher_v2 import VideoSearcherV2
    from elastics.videos.explorer import VideoExplorer
    from llms.llm_client import create_llm_client
    from llms.tools.executor import SearchService
    from llms.chat.handler import ChatHandler

    # Resolve elastic index
    elastic_index = args.elastic_index
    elastic_env_name = args.elastic_env_name
    if not elastic_index:
        idx = SEARCH_APP_ENVS.get("elastic_index", {})
        elastic_index = idx.get("prod", idx) if isinstance(idx, dict) else idx

    logger.note("> Initializing Bili Search Copilot CLI...")
    logger.mesg(f"  LLM config: {args.llm_config}")
    logger.mesg(f"  Elastic index: {elastic_index}")
    if elastic_env_name:
        logger.mesg(f"  Elastic env: {elastic_env_name}")

    video_searcher = VideoSearcherV2(elastic_index, elastic_env_name=elastic_env_name)
    video_explorer = VideoExplorer(elastic_index, elastic_env_name=elastic_env_name)

    # Optional owner searcher
    owner_searcher = None
    try:
        from elastics.owners.searcher import OwnerSearcher
        from elastics.owners.constants import ELASTIC_OWNERS_INDEX

        owner_searcher = OwnerSearcher(
            index_name=ELASTIC_OWNERS_INDEX,
            elastic_env_name=elastic_env_name,
            coretok_bundle_path=args.owner_coretok_bundle_path,
        )
        video_explorer.owner_searcher = owner_searcher
        logger.mesg(f"  Owner searcher: {ELASTIC_OWNERS_INDEX}")
    except Exception as e:
        logger.hint(f"  Owner searcher unavailable: {e}")

    llm_client = create_llm_client(
        model_config=args.llm_config,
        verbose=args.verbose,
    )
    search_service = SearchService(
        video_searcher=video_searcher,
        video_explorer=video_explorer,
        owner_searcher=owner_searcher,
        verbose=args.verbose,
    )
    handler = ChatHandler(
        llm_client=llm_client,
        search_client=search_service,
        temperature=args.temperature,
        verbose=args.verbose,
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
        help="LLM config name (e.g. deepseek, gpt, volcengine)",
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
        "--owner-coretok-bundle-path",
        type=str,
        default=None,
        help="Path to serialized owner coretok bundle for query-time token encoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature",
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
    #   python -m llms.cli
    #   python -m llms.cli --llm-config deepseek --verbose
    #   python -m llms.cli --elastic-index bili_videos_dev6 --elastic-env-name elastic_dev
    #
    # Single-shot mode:
    #   python -m llms.cli -q "影视飓风最近有什么新视频？"
    #   python -m llms.cli -q "老番茄和影视飓风最近的视频" --verbose
    #   python -m llms.cli -q "黑神话 播放量最高的视频"
