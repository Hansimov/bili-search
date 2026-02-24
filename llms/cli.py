"""CLI interactive chat mode for testing.

Provides a command-line interface for interacting with the search copilot.
Uses direct search (in-process) instead of HTTP, so no separate search server needed.

Usage:
    python -m llms.cli
    python -m llms.cli --llm-config volcengine
    python -m llms.cli --verbose
    python -m llms.cli --elastic-index bili_videos_dev6 --elastic-env-name elastic_dev
"""

import argparse
import json
import sys

from tclogger import logger


def main():
    parser = argparse.ArgumentParser(description="Bili Search Copilot CLI")
    parser.add_argument(
        "--llm-config",
        type=str,
        default="deepseek",
        help="LLM config name (e.g. deepseek, volcengine)",
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
    args = parser.parse_args()

    # Lazy imports to avoid slow startup for --help
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
        # Use default from envs.json (prod mode)
        idx = SEARCH_APP_ENVS.get("elastic_index", {})
        elastic_index = idx.get("prod", idx) if isinstance(idx, dict) else idx

    # Initialize components
    logger.note("> Initializing Bili Search Copilot CLI...")
    logger.mesg(f"  LLM config: {args.llm_config}")
    logger.mesg(f"  Elastic index: {elastic_index}")
    if elastic_env_name:
        logger.mesg(f"  Elastic env: {elastic_env_name}")

    video_searcher = VideoSearcherV2(elastic_index, elastic_env_name=elastic_env_name)
    video_explorer = VideoExplorer(elastic_index, elastic_env_name=elastic_env_name)

    llm_client = create_llm_client(
        model_config=args.llm_config,
        verbose=args.verbose,
    )
    search_service = SearchService(
        video_searcher=video_searcher,
        video_explorer=video_explorer,
        verbose=args.verbose,
    )
    handler = ChatHandler(
        llm_client=llm_client,
        search_client=search_service,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    logger.success("> Ready! Type your question (Ctrl+C to exit)\\n")

    # Conversation history
    messages = []

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Special commands
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

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Get response
            logger.hint("> Thinking...")
            result = handler.handle(
                messages=messages,
                temperature=args.temperature,
            )

            # Extract and display response
            content = result["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": content})

            print(f"\nCopilot: {content}\n")

        except KeyboardInterrupt:
            print()
            logger.note("Goodbye!")
            break
        except EOFError:
            logger.note("Goodbye!")
            break
        except Exception as e:
            logger.warn(f"× Error: {e}")
            # Remove the failed user message
            if messages and messages[-1]["role"] == "user":
                messages.pop()


if __name__ == "__main__":
    main()

    # python -m llms.cli
    # python -m llms.cli --llm-config deepseek --verbose
    # python -m llms.cli --elastic-index bili_videos_dev6 --elastic-env-name elastic_dev
