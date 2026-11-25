"""Lossless NMR task adapter for newly generated data.

Historical files keep their existing text. ``scripts/build_test_2000.py``
``format_alpaca`` remains the reproduction path for that old text and is not
used here. This module does not rewrite a registered training file or test_2000.
"""

from __future__ import annotations

import sys
from pathlib import Path

OA_ROOT = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
if str(OA_ROOT) not in sys.path:
    sys.path.insert(0, str(OA_ROOT))

from recipe.nmr_matching_agent.task_input import (  # noqa: E402
    SCHEMA_VERSION,
    normalize_task_input,
    parse_task_text,
    task_to_lossless_text,
)

__all__ = [
    "SCHEMA_VERSION",
    "normalize_task_input",
    "parse_task_text",
    "task_to_lossless_text",
]
