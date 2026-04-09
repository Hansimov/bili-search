"""Compatibility shim for llms.runtime.cli."""

from llms.runtime.cli import _create_handler
from llms.runtime.cli import main

__all__ = ["_create_handler", "main"]


if __name__ == "__main__":
    main()
