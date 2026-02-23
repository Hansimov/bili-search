"""CLI interactive chat mode for testing.

Provides a command-line interface for interacting with the search copilot.
Useful for development and testing without running a full HTTP server.

Usage:
    python -m llms.cli
    python -m llms.cli --llm-config volcengine
    python -m llms.cli --search-url http://localhost:21001
    python -m llms.cli --verbose
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
        "--search-url",
        type=str,
        default="http://localhost:20001",
        help="Search App URL",
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
    from llms.llm_client import create_llm_client
    from llms.search_service import SearchServiceClient
    from llms.chat.handler import ChatHandler

    # Initialize components
    logger.note("> Initializing Bili Search Copilot CLI...")
    logger.mesg(f"  LLM config: {args.llm_config}")
    logger.mesg(f"  Search URL: {args.search_url}")

    llm_client = create_llm_client(
        model_config=args.llm_config,
        verbose=args.verbose,
    )
    search_client = SearchServiceClient(
        base_url=args.search_url,
        verbose=args.verbose,
    )
    handler = ChatHandler(
        llm_client=llm_client,
        search_client=search_client,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    # Check search service availability
    if search_client.is_available():
        logger.success("> Search service is available")
    else:
        logger.warn(
            f"× Search service at {args.search_url} is not reachable. "
            "Make sure the search app is running."
        )

    logger.success("> Ready! Type your question (Ctrl+C to exit)\n")

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
    # python -m llms.cli --search-url http://localhost:21001
