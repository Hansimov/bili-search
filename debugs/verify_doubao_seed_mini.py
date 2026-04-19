from __future__ import annotations

import sys

from verify_model_client import main


DEFAULT_MODEL_CONFIG = "doubao-seed-2-0-mini"


if __name__ == "__main__":
    sys.exit(main(default_model_config=DEFAULT_MODEL_CONFIG))
