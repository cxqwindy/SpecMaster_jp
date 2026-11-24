"""Load the NMR gateway API key from the environment.

Evaluation and distillation scripts used to carry a literal key. They now call
``require_gateway_api_key()`` and fail immediately when ``NMR_GATEWAY_API_KEY``
is missing or blank. There is no baked-in default and no fallback to an old key.
"""
from __future__ import annotations

import os

ENV_NAME = "NMR_GATEWAY_API_KEY"


def require_gateway_api_key() -> str:
    raw = os.environ.get(ENV_NAME)
    if raw is None or raw.strip() == "":
        raise RuntimeError(
            f"Missing required environment variable {ENV_NAME}. "
            "Export it before calling the gateway. "
            "This program does not fall back to a hardcoded key."
        )
    return raw.strip()
