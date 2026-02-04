"""
Legacy cleaner (kept for backwards-compat).

Prefer: from project root:
    python ingest/build_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure project root (with build_corpus.py and config.py) is on sys.path,
# even when running this file directly as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest.build_corpus import build_corpus  # type: ignore[import]


if __name__ == "__main__":
    build_corpus()
